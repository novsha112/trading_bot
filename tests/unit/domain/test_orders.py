"""Order and OrderUpdate invariants."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.orders import Order, OrderUpdate

D = Decimal
S = OrderStatus
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NAIVE = datetime(2026, 1, 15, 12, 0)  # noqa: DTZ001 - deliberately naive

ORDER: dict[str, Any] = {
    "client_order_id": "grid1-buy-0001",
    "exchange_order_id": None,
    "strategy_id": "grid-1",
    "symbol": "BTCUSDT",
    "side": Side.BUY,
    "order_type": OrderType.LIMIT,
    "price": D("65000"),
    "qty": D("1.0"),
    "time_in_force": TimeInForce.POST_ONLY,
    "reduce_only": False,
    "status": S.NEW,
    "filled_qty": D("0"),
    "avg_fill_price": None,
    "created_at": TS,
    "updated_at": TS,
    "last_exchange_update_ts": None,
    "version": 0,
}
UPDATE: dict[str, Any] = {
    "client_order_id": "grid1-buy-0001",
    "exchange_order_id": "o-1",
    "status": S.PARTIALLY_FILLED,
    "cum_filled_qty": D("0.3"),
    "avg_fill_price": D("65000"),
    "reject_reason": None,
    "exchange_ts": TS,
}


def order(**overrides: Any) -> Order:
    return Order(**{**ORDER, **overrides})


def update(**overrides: Any) -> OrderUpdate:
    return OrderUpdate(**{**UPDATE, **overrides})


def filled(status: OrderStatus, qty: str) -> dict[str, Any]:
    return {
        "status": status,
        "filled_qty": D(qty),
        "avg_fill_price": D("65000") if D(qty) > 0 else None,
    }


@pytest.mark.parametrize(("model", "values"), [(Order, ORDER), (OrderUpdate, UPDATE)])
def test_structure_frozen_slotted_keyword_only(model: Any, values: dict[str, Any]) -> None:
    instance = model(**values)
    for name, value in values.items():
        assert getattr(instance, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.status = S.FILLED
    assert not hasattr(instance, "__dict__")
    with pytest.raises(TypeError):
        model(*values.values())


# --- Order: order terms ------------------------------------------------------------------


def test_market_order_without_price() -> None:
    o = order(order_type=OrderType.MARKET, price=None, time_in_force=TimeInForce.IOC)
    assert o.price is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"price": None}, "price must be a Decimal"),
        ({"price": D("0")}, "price must be > 0"),
        ({"order_type": OrderType.MARKET, "time_in_force": TimeInForce.IOC}, "price must be None"),
        (
            {"order_type": OrderType.MARKET, "price": None, "time_in_force": TimeInForce.POST_ONLY},
            "post_only",
        ),
        ({"qty": D("0")}, "qty must be > 0"),
        ({"qty": 1.0}, "qty must be a Decimal"),
        ({"side": "buy"}, "side must be a Side"),
        ({"order_type": "limit"}, "order_type must be a OrderType"),
        ({"time_in_force": "gtc"}, "time_in_force must be a TimeInForce"),
        ({"status": "new"}, "status must be a OrderStatus"),
        ({"reduce_only": 0}, "reduce_only must be a bool"),
    ],
)
def test_order_terms(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{message}"):
        order(**overrides)


@pytest.mark.parametrize("field", ["client_order_id", "exchange_order_id", "strategy_id", "symbol"])
def test_order_text_fields(field: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        order(**{field: " "})


def test_exchange_order_id_optional() -> None:
    assert order(exchange_order_id="o-1").exchange_order_id == "o-1"


# --- Order: fill / avg price -------------------------------------------------------------


@pytest.mark.parametrize("value", [D("-0.1"), D("NaN"), 0, 0.0, None])
def test_filled_qty_must_be_non_negative_decimal(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^filled_qty must be"):
        order(status=S.UNKNOWN, filled_qty=value)


def test_filled_qty_cannot_exceed_qty() -> None:
    with pytest.raises(DomainValidationError, match=r"^filled_qty .* must be <= qty"):
        order(**filled(S.UNKNOWN, "1.1"))


def test_zero_fill_must_not_have_avg_price() -> None:
    with pytest.raises(DomainValidationError, match=r"^avg_fill_price must be None"):
        order(status=S.UNKNOWN, avg_fill_price=D("65000"))


@pytest.mark.parametrize(("avg", "message"), [(None, "a Decimal"), (D("0"), "> 0")])
def test_positive_fill_requires_positive_avg(avg: Any, message: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^avg_fill_price must be {message}"):
        order(status=S.UNKNOWN, filled_qty=D("0.3"), avg_fill_price=avg)


# --- Order: status / fill invariants -----------------------------------------------------

VALID_STATUS_FILLS = [
    (S.NEW, "0"),
    (S.SUBMITTING, "0"),
    (S.OPEN, "0"),
    (S.REJECTED, "0"),
    (S.FAILED, "0"),
    (S.PARTIALLY_FILLED, "0.3"),
    (S.PARTIALLY_FILLED, "0.999"),
    (S.FILLED, "1.0"),
    (S.FILLED, "1"),
    (S.CANCELING, "0"),
    (S.CANCELING, "0.3"),
    (S.CANCELING, "1.0"),
    (S.CANCELED, "0"),
    (S.CANCELED, "0.6"),
    (S.EXPIRED, "0"),
    (S.EXPIRED, "0.6"),
    (S.UNKNOWN, "0"),
    (S.UNKNOWN, "0.3"),
    (S.UNKNOWN, "1.0"),
]
INVALID_STATUS_FILLS = [
    (S.NEW, "0.3"),
    (S.SUBMITTING, "0.3"),
    (S.OPEN, "0.3"),
    (S.REJECTED, "0.3"),
    (S.FAILED, "0.3"),
    (S.PARTIALLY_FILLED, "0"),
    (S.PARTIALLY_FILLED, "1.0"),
    (S.FILLED, "0"),
    (S.FILLED, "0.999"),
    (S.CANCELED, "1.0"),  # fully executed order is FILLED, not CANCELED
    (S.EXPIRED, "1.0"),
]


@pytest.mark.parametrize(("status", "qty"), VALID_STATUS_FILLS)
def test_valid_status_fill_combinations(status: OrderStatus, qty: str) -> None:
    assert order(**filled(status, qty)).status is status


@pytest.mark.parametrize(("status", "qty"), INVALID_STATUS_FILLS)
def test_invalid_status_fill_combinations(status: OrderStatus, qty: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^filled_qty .* {status.value}"):
        order(**filled(status, qty))


# --- Order: timestamps / version ---------------------------------------------------------


def test_updated_at_before_created_at_rejected() -> None:
    with pytest.raises(DomainValidationError, match=r"^updated_at .* must be >= created_at"):
        order(updated_at=TS - timedelta(microseconds=1))


@pytest.mark.parametrize("field", ["created_at", "updated_at", "last_exchange_update_ts"])
def test_timestamps_must_be_utc(field: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        order(**{field: NAIVE})


def test_last_exchange_update_ts_optional_utc() -> None:
    assert order(last_exchange_update_ts=TS).last_exchange_update_ts == TS


@pytest.mark.parametrize("value", [True, False, -1, 1.0, D("1"), "1", None])
def test_version_must_be_non_negative_int(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^version must be"):
        order(version=value)


# --- OrderUpdate -------------------------------------------------------------------------


def test_update_zero_fill_without_avg() -> None:
    u = update(status=S.OPEN, cum_filled_qty=D("0"), avg_fill_price=None, exchange_order_id=None)
    assert u.avg_fill_price is None


def test_update_does_not_know_order_qty() -> None:
    # Consistency with the order quantity is checked when the update is applied.
    assert update(status=S.PARTIALLY_FILLED, cum_filled_qty=D("1000000")).cum_filled_qty


def test_update_reject_reason() -> None:
    u = update(status=S.REJECTED, cum_filled_qty=D("0"), avg_fill_price=None, reject_reason="x")
    assert u.reject_reason == "x"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cum_filled_qty": D("-0.1")}, "cum_filled_qty must be >= 0"),
        ({"cum_filled_qty": 0.3}, "cum_filled_qty must be a Decimal"),
        ({"cum_filled_qty": D("NaN")}, "cum_filled_qty must be finite"),
        ({"cum_filled_qty": D("0")}, "avg_fill_price must be None"),
        ({"avg_fill_price": None}, "avg_fill_price must be a Decimal"),
        ({"avg_fill_price": D("0")}, "avg_fill_price must be > 0"),
        ({"status": "partially_filled"}, "status must be a OrderStatus"),
        ({"client_order_id": ""}, "client_order_id "),
        ({"exchange_order_id": " o-1"}, "exchange_order_id "),
        ({"reject_reason": ""}, "reject_reason "),
        ({"exchange_ts": NAIVE}, "exchange_ts "),
    ],
)
def test_update_invariants(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{message}"):
        update(**overrides)
