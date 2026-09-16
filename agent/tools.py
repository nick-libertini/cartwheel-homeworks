"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from rapidfuzz import fuzz, utils

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20
FIND_ORDER_LIMIT = 5
# WRatio scales its score by the length ratio of the two strings, so a wordy
# query ("earmuffs I bought last week") still scores ~85 against a short title
# ("Wool Earmuffs"). 75 keeps that match while rejecting unrelated titles.
FIND_ORDER_MATCH_CUTOFF = 75


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id.

    Args:
        ctx: The caller's auth context.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {
        "ok": False,
        "error": "not_found",
        "reason": f"no policy doc with id '{policy_id}'",
    }


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog by keyword.
    
    - Returns matching products, sorted by (price, id), up to MAX_SEARCH_LIMIT.
    - Matching is deterministic keyword matching, not semantic search: a product matches
    when every whitespace token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.


    """
    tokens = query.lower().split()
    if not tokens:
        return {"ok": False, "error": "invalid_argument", "reason": "empty query"}
    if max_price_usd is not None and max_price_usd <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "max_price_usd must be positive",
        }
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    with db.connection() as conn:
        store_id = None
        if store is not None:
            matched = db.get_store_by_name(conn, store)
            if matched is None:
                return {
                    "ok": False,
                    "error": "not_found",
                    "reason": f"no store named '{store}'",
                }
            store_id = matched.id
        candidates = db.list_products(conn, store_id)
    matches = []
    for product in candidates:
        if max_price_usd is not None and product.price_usd > max_price_usd:
            continue
        haystack = f"{product.title} {product.description}".lower()
        if all(token in haystack for token in tokens):
            matches.append(product)
    matches.sort(key=lambda p: (p.price_usd, p.id))
    products = [
        {
            "product_id": p.id,
            "store_id": p.store_id,
            "title": p.title,
            "price_usd": p.price_usd,
        }
        for p in matches[:limit]
    ]
    return {"ok": True, "products": products, "count": len(products)}


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders, newest first, up to DEFAULT_ORDER_LIMIT orders.

    Role behavior:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders.
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead.

    Args:
        ctx: The caller's auth context.

    Returns:
        {"ok": True, "orders": [...], "count": <len(orders)>} if listing orders
        is a valid operation otherwise an error message.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": (
                "support staff have no orders of their own; "
                "look up a specific order with get_order instead"
            ),
        }
    with db.connection() as conn:
        if ctx.role == "merchant":
            store_id = ctx.store_id
            if store_id is None:
                return {
                    "ok": False,
                    "error": "invalid_argument",
                    "reason": "merchant session is missing a store_id",
                }
            orders = db.list_orders_for_store(conn, store_id, DEFAULT_ORDER_LIMIT)
        else:
            orders = db.list_orders_for_user(conn, ctx.user_id, DEFAULT_ORDER_LIMIT)
    payload = [order.to_public_dict() for order in orders]
    return {"ok": True, "orders": payload, "count": len(payload)}


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {
                "ok": False,
                "error": "not_found",
                "reason": f"no order #{order_id}",
            }
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )
        if order.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": (
                    f"order #{order_id} has status '{order.status}'; "
                    f"orders can be cancelled only before shipment"
                ),
            }
        db.set_order_status(conn, order_id, "cancelled")
        return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search orders by product name.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and uses fuzzy string matching to find orders whose product name is close 
    to the query.

    Role behavior:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders.
        - support: Any order.

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), newest first, each as the dict. If no orders match, 
        return {"ok": True, "orders": []}.
    """
    query = query.strip()
    if not query:
        return {"ok": False, "error": "invalid_argument", "reason": "empty query"}
    with db.connection() as conn:
        if ctx.role == "shopper":
            candidates = db.list_order_search_candidates(conn, user_id=ctx.user_id)
        elif ctx.role == "merchant":
            store_id = ctx.store_id
            if store_id is None:
                return {
                    "ok": False,
                    "error": "invalid_argument",
                    "reason": "merchant session is missing a store_id",
                }
            candidates = db.list_order_search_candidates(conn, store_id=store_id)
        elif ctx.role == "support":
            candidates = db.list_order_search_candidates(conn, all_orders=True)
        else:
            return permission_denied(f"role '{ctx.role}' may not search orders")
        titles = {product.id: product.title for product in db.list_products(conn)}
    # The candidates already arrive newest first, so filtering in place keeps
    # that order and the first five matches are the five most recent ones.
    orders = []
    for order in candidates:
        title = titles.get(order.product_id)
        if title is None:
            continue
        score = fuzz.WRatio(query, title, processor=utils.default_process)
        if score >= FIND_ORDER_MATCH_CUTOFF:
            orders.append(order.to_public_dict())
            if len(orders) == FIND_ORDER_LIMIT:
                break
    return {"ok": True, "orders": orders}


def _return_record_inconsistency(order: db.Order) -> str | None:
    """Name the defect that makes a return deadline impossible to trust, or None.

    Two seeded data-quality cases land here: an order marked delivered with no
    delivery date (dq-order-missing-delivery-date) and an order whose shipment
    date follows its delivery date (dq-order-reversed-chronology). Both mean the
    timeline is wrong, so the tool reports the defect instead of computing a
    deadline from it.
    """
    if order.status == "delivered" and order.delivered_at is None:
        return f"order #{order.id} is marked delivered but records no delivery date"
    if (
        order.shipped_at is not None
        and order.delivered_at is not None
        and order.shipped_at > order.delivered_at
    ):
        return (
            f"order #{order.id} records shipment on {order.shipped_at}, "
            f"after delivery on {order.delivered_at}"
        )
    return None


def check_return_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Report the return deadline for one order, with the dates it was computed from.

    Args:
        ctx: The caller's auth context.
        order_id: The order to check.

    Returns:
        On success: {"ok": True, "order_id": int, "status": str,
        "delivered_at": str | None, "as_of": str, "return_window_days": int,
        "store_return_window_days": int | None, "return_deadline": str | None,
        "days_remaining": int | None, "eligible": bool, "reason_code": str,
        "policy_ids": list[str]}, plus "inconsistency" when the record's dates
        contradict each other. Dates are ISO strings and `days_remaining` is
        negative once the window has passed.

        `reason_code` is one of "within_window", "window_expired",
        "not_delivered", "delivery_in_future", or "inconsistent_record".

        {"ok": False, "error": "not_found", ...} for an unknown order and
        {"ok": False, "error": "permission_denied", ...} for an order outside
        the caller's scope, checked before any order detail is returned.
    """
    facts = load_facts()
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {
                "ok": False,
                "error": "not_found",
                "reason": f"no order #{order_id}",
            }
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        as_of = db.world_asof(conn)
        store = db.get_store(conn, order.store_id)
    override = store.return_window_days_override if store else None
    window_days = effective_return_window_days(facts["return_window_days"], override)
    policy_ids = ["cw-returns"]
    if store is not None and override is not None:
        store_policy_id = f"store-{store.slug}-policy"
        if any(doc.policy_id == store_policy_id for doc in load_policy_docs()):
            policy_ids.append(store_policy_id)

    result: dict[str, Any] = {
        "ok": True,
        "order_id": order.id,
        "status": order.status,
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "as_of": as_of.isoformat(),
        "return_window_days": window_days,
        "store_return_window_days": override,
        "return_deadline": None,
        "days_remaining": None,
        "eligible": order.refund_eligible,
        "reason_code": "",
        "policy_ids": policy_ids,
    }

    inconsistency = _return_record_inconsistency(order)
    if inconsistency is not None:
        result["eligible"] = False
        result["reason_code"] = "inconsistent_record"
        result["inconsistency"] = inconsistency
        return result

    if order.status != "delivered" or order.delivered_at is None:
        result["reason_code"] = "not_delivered"
        return result

    deadline = order.delivered_at + timedelta(days=window_days)
    result["return_deadline"] = deadline.isoformat()
    result["days_remaining"] = (deadline - as_of).days
    if order.delivered_at > as_of:
        result["reason_code"] = "delivery_in_future"
    elif as_of <= deadline:
        result["reason_code"] = "within_window"
    else:
        result["reason_code"] = "window_expired"
    return result
