"""Strategy output: place / cancel intents."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import CancelOrderIntent, Intent, PlaceOrderIntent

D = Decimal
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

PLACE: dict[str, Any] = {
    "intent_id": "i-1",
    "strategy_id": "grid-1",
    "symbol": "BTCUSDT",
    "side": Side.BUY,
    "order_type": OrderType.LIMIT,
    "price": D("65000.5"),
    "qty": D("0.001"),
    "time_in_force": TimeInForce.POST_ONLY,
    "reduce_only": False,
    "tag": "grid:L07:buy",
    "created_at": TS,
}
CANCEL: dict[str, Any] = {
    "intent_id": "i-2",
    "strategy_id": "grid-1",
    "symbol": "BTCUSDT",
    "client_order_id": "grid1-buy-0001",
    "reason": "out of range",
    "created_at": TS,
}


def place(**overrides: Any) -> PlaceOrderIntent:
    return PlaceOrderIntent(**{**PLACE, **overrides})


def cancel(**overrides: Any) -> CancelOrderIntent:
    return CancelOrderIntent(**{**CANCEL, **overrides})


@pytest.mark.parametrize(
    ("model", "values"), [(PlaceOrderIntent, PLACE), (CancelOrderIntent, CANCEL)]
)
def test_structure_frozen_slotted_keyword_only(model: Any, values: dict[str, Any]) -> None:
    instance = model(**values)
    for name, value in values.items():
        assert getattr(instance, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.symbol = "ETHUSDT"
    assert not hasattr(instance, "__dict__")
    with pytest.raises(TypeError):
        model(*values.values())


def test_intent_alias_covers_both_types() -> None:
    intents: list[Intent] = [place(), cancel()]
    assert [type(i) for i in intents] == [PlaceOrderIntent, CancelOrderIntent]


# --- PlaceOrderIntent --------------------------------------------------------------------


@pytest.mark.parametrize("tif", [TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK])
def test_market_order_without_price(tif: TimeInForce) -> None:
    intent = place(order_type=OrderType.MARKET, price=None, time_in_force=tif, reduce_only=True)
    assert intent.price is None


def test_tag_optional() -> None:
    assert place(tag=None).tag is None


def test_limit_requires_price() -> None:
    with pytest.raises(DomainValidationError, match=r"^price must be a Decimal"):
        place(price=None)


@pytest.mark.parametrize("price", [D("0"), D("-1")])
def test_limit_price_must_be_positive(price: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=r"^price must be > 0"):
        place(price=price)


def test_market_must_not_have_price() -> None:
    with pytest.raises(DomainValidationError, match=r"^price must be None for a market order"):
        place(order_type=OrderType.MARKET, time_in_force=TimeInForce.IOC)


def test_post_only_only_for_limit() -> None:
    with pytest.raises(DomainValidationError, match=r"^post_only .* limit"):
        place(order_type=OrderType.MARKET, price=None, time_in_force=TimeInForce.POST_ONLY)


@pytest.mark.parametrize("qty", [D("0"), D("-0.001"), 0.001, 1, True, D("NaN"), D("Infinity")])
def test_qty_must_be_positive_decimal(qty: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^qty must be"):
        place(qty=qty)


@pytest.mark.parametrize("price", [65000.5, 65000, D("sNaN")])
def test_price_must_be_decimal(price: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^price must be"):
        place(price=price)


@pytest.mark.parametrize(
    ("field", "value", "enum_name"),
    [
        ("side", "buy", "Side"),
        ("order_type", "limit", "OrderType"),
        ("time_in_force", "post_only", "TimeInForce"),
        ("side", OrderType.LIMIT, "Side"),
    ],
)
def test_enum_fields_require_exact_enum(field: str, value: Any, enum_name: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be a {enum_name}"):
        place(**{field: value})


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_reduce_only_must_be_bool(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^reduce_only must be a bool"):
        place(reduce_only=value)


@pytest.mark.parametrize("field", ["intent_id", "strategy_id", "symbol", "tag"])
@pytest.mark.parametrize("value", ["", " x", 1])
def test_place_text_fields(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        place(**{field: value})


def test_place_created_at_utc() -> None:
    with pytest.raises(DomainValidationError, match=r"^created_at "):
        place(created_at=datetime(2026, 1, 15))  # noqa: DTZ001 - deliberately naive


# --- CancelOrderIntent -------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["intent_id", "strategy_id", "symbol", "client_order_id", "reason"]
)
@pytest.mark.parametrize("value", ["", "  ", None])
def test_cancel_text_fields(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        cancel(**{field: value})


def test_cancel_created_at_utc() -> None:
    with pytest.raises(DomainValidationError, match=r"^created_at "):
        cancel(created_at=datetime(2026, 1, 15))  # noqa: DTZ001 - deliberately naive
