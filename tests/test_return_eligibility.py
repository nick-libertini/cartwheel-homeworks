"""Contract tests for the Homework 1 additional tool, check_return_eligibility.

Offline: the tool only reads the seeded world, so every case uses the shared
`world` fixture. The dates below come from that world, whose fixed today is
2026-07-01.
"""

from __future__ import annotations

from agent.auth import AuthContext
from agent.tools import check_return_eligibility

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
SHOPPER_2 = AuthContext(user_id=2, role="shopper")
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
SUPPORT = AuthContext(user_id=9501, role="support")


def test_reports_the_deadline_for_an_order_inside_the_window(world: dict) -> None:
    result = check_return_eligibility(SHOPPER_1, 4127)
    assert result["ok"] is True
    assert result["as_of"] == "2026-07-01"
    assert result["delivered_at"] == "2026-06-19"
    assert result["return_window_days"] == 30
    assert result["store_return_window_days"] is None
    assert result["return_deadline"] == "2026-07-19"
    assert result["days_remaining"] == 18
    assert result["eligible"] is True
    assert result["reason_code"] == "within_window"
    assert result["policy_ids"] == ["cw-returns"]


def test_reports_an_expired_window_with_negative_days_remaining(world: dict) -> None:
    result = check_return_eligibility(SHOPPER_1, 3980)
    assert result["reason_code"] == "window_expired"
    assert result["return_deadline"] == "2026-06-16"
    assert result["days_remaining"] == -15
    assert result["eligible"] is False


def test_store_override_replaces_the_platform_window(world: dict) -> None:
    result = check_return_eligibility(SUPPORT, 9951)
    assert result["return_window_days"] == 7
    assert result["store_return_window_days"] == 7
    assert result["delivered_at"] == "2025-08-23"
    assert result["return_deadline"] == "2025-08-30"
    assert result["policy_ids"] == ["cw-returns", "store-saltbox-pantry-policy"]


def test_unshipped_order_has_no_deadline(world: dict) -> None:
    result = check_return_eligibility(SUPPORT, 8081)
    assert result["status"] == "placed"
    assert result["reason_code"] == "not_delivered"
    assert result["return_deadline"] is None
    assert result["days_remaining"] is None
    assert result["eligible"] is False


def test_missing_delivery_date_is_reported_not_computed(world: dict) -> None:
    result = check_return_eligibility(SUPPORT, 8002)
    assert result["reason_code"] == "inconsistent_record"
    assert result["return_deadline"] is None
    assert result["days_remaining"] is None
    assert result["eligible"] is False
    assert "no delivery date" in result["inconsistency"]


def test_reversed_chronology_is_reported_not_computed(world: dict) -> None:
    result = check_return_eligibility(SUPPORT, 8001)
    assert result["reason_code"] == "inconsistent_record"
    assert result["return_deadline"] is None
    assert "after delivery" in result["inconsistency"]


def test_denies_another_shoppers_order(world: dict) -> None:
    result = check_return_eligibility(SHOPPER_2, 4127)
    assert result == {
        "ok": False,
        "error": "permission_denied",
        "reason": "role 'shopper' (user 2) may not view order #4127",
    }


def test_denies_another_stores_order_and_allows_support(world: dict) -> None:
    assert (
        check_return_eligibility(MERCHANT_STORE_2, 4127)["error"] == "permission_denied"
    )
    assert check_return_eligibility(SUPPORT, 4127)["ok"] is True


def test_unknown_order_is_not_found(world: dict) -> None:
    result = check_return_eligibility(SHOPPER_1, 999999)
    assert result == {"ok": False, "error": "not_found", "reason": "no order #999999"}
