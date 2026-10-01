"""Exchange boundary DTOs: OrderRequest, OrderRef, OrderAck."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.exchanges.models import OrderAck, OrderRef, OrderRequest

D = Decimal
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

REQUEST: dict[str, Any] = {
    "client_order_id": "grid1-buy-0001",
    "symbol": "BTCUSDT",
    "side": Side.BUY,
    "order_type": OrderType.LIMIT,
    "price": D("65000.5"),
    "qty": D("0.001"),
    "time_in_force": TimeInForce.POST_ONLY,
    "reduce_only": False,
}
ACK: dict[str, Any] = {
    "client_order_id": "grid1-buy-0001",
    "exchange_order_id": "o-1",
    "exchange_ts": TS,
}


def request(**overrides: Any) -> OrderRequest:
    return OrderRequest(**{**REQUEST, **overrides})


def ack(**overrides: Any) -> OrderAck:
    return OrderAck(**{**ACK, **overrides})


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (OrderRequest, REQUEST),
        (OrderRef, {"symbol": "BTCUSDT", "client_order_id": "c-1", "exchange_order_id": "o-1"}),
        (OrderAck, ACK),
    ],
)
def test_structure_frozen_slotted_keyword_only(model: Any, values: dict[str, Any]) -> None:
    instance = model(**values)
    for name, value in values.items():
        assert getattr(instance, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.client_order_id = "other"
    assert not hasattr(instance, "__dict__")
    with pytest.raises(TypeError):
        model(*values.values())


# --- OrderRequest ------------------------------------------------------------------------


def test_request_has_no_lifecycle_fields() -> None:
    assert {f.name for f in dataclasses.fields(OrderRequest)} == set(REQUEST)


def test_valid_limit_request() -> None:
    assert request().price == D("65000.5")


@pytest.mark.parametrize("tif", [TimeInForce.IOC, TimeInForce.FOK, TimeInForce.GTC])
def test_valid_market_request(tif: TimeInForce) -> None:
    r = request(order_type=OrderType.MARKET, price=None, time_in_force=tif, reduce_only=True)
    assert r.price is None


@pytest.mark.parametrize("value", ["", " c-1", None, 1])
def test_client_order_id_required(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^client_order_id "):
        request(client_order_id=value)


@pytest.mark.parametrize("value", ["", "BTCUSDT ", None])
def test_symbol_validated(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^symbol "):
        request(symbol=value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"qty": D("0")}, "qty must be > 0"),
        ({"qty": 0.001}, "qty must be a Decimal"),
        ({"qty": D("NaN")}, "qty must be finite"),
        ({"price": None}, "price must be a Decimal"),
        ({"price": D("-1")}, "price must be > 0"),
        ({"order_type": OrderType.MARKET, "time_in_force": TimeInForce.IOC}, "price must be None"),
        (
            {"order_type": OrderType.MARKET, "price": None, "time_in_force": TimeInForce.POST_ONLY},
            "post_only",
        ),
        ({"side": "buy"}, "side must be a Side"),
        ({"order_type": "limit"}, "order_type must be a OrderType"),
        ({"time_in_force": "gtc"}, "time_in_force must be a TimeInForce"),
        ({"reduce_only": 0}, "reduce_only must be a bool"),
        ({"reduce_only": "false"}, "reduce_only must be a bool"),
    ],
)
def test_request_invariants(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{message}"):
        request(**overrides)


# --- OrderRef ----------------------------------------------------------------------------


def test_ref_before_ack_has_no_exchange_id() -> None:
    ref = OrderRef(symbol="BTCUSDT", client_order_id="c-1")
    assert ref.exchange_order_id is None


def test_ref_after_ack() -> None:
    a = ack()
    ref = OrderRef(
        symbol="BTCUSDT", client_order_id=a.client_order_id, exchange_order_id=a.exchange_order_id
    )
    assert (ref.client_order_id, ref.exchange_order_id) == ("grid1-buy-0001", "o-1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", ""),
        ("symbol", None),
        ("client_order_id", ""),
        ("client_order_id", None),
        ("client_order_id", " c-1"),
        ("exchange_order_id", ""),
        ("exchange_order_id", "o-1 "),
        ("exchange_order_id", 1),
    ],
)
def test_ref_text_validated(field: str, value: Any) -> None:
    values: dict[str, Any] = {"symbol": "BTCUSDT", "client_order_id": "c-1", field: value}
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        OrderRef(**values)


# --- OrderAck ----------------------------------------------------------------------------


def test_ack_has_no_status() -> None:
    assert {f.name for f in dataclasses.fields(OrderAck)} == set(ACK)


def test_ack_exchange_ts_optional() -> None:
    assert ack(exchange_ts=None).exchange_ts is None


@pytest.mark.parametrize("field", ["client_order_id", "exchange_order_id"])
@pytest.mark.parametrize("value", ["", " o-1", None, 1])
def test_ack_identifiers_required_and_clean(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        ack(**{field: value})


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 15),  # noqa: DTZ001 - deliberately naive
        datetime(2026, 1, 15, tzinfo=timezone(timedelta(hours=3))),
        1768478400000,
    ],
)
def test_ack_exchange_ts_must_be_utc(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^exchange_ts "):
        ack(exchange_ts=value)
