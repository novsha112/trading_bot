"""OrderAck: acceptance of a placement request, not an order status."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from app.domain.errors import DomainValidationError
from app.exchanges.models import OrderAck

TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
ACK: dict[str, Any] = {
    "client_order_id": "grid1-buy-0001",
    "exchange_order_id": "o-1",
    "exchange_ts": TS,
}


def ack(**overrides: Any) -> OrderAck:
    return OrderAck(**{**ACK, **overrides})


def test_structure_frozen_slotted_keyword_only() -> None:
    a = ack()
    assert (a.client_order_id, a.exchange_order_id, a.exchange_ts) == ("grid1-buy-0001", "o-1", TS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.exchange_order_id = "o-2"  # type: ignore[misc]
    assert not hasattr(a, "__dict__")
    with pytest.raises(TypeError):
        OrderAck("grid1-buy-0001", "o-1", TS)  # type: ignore[call-arg]


def test_ack_has_no_status() -> None:
    assert {f.name for f in dataclasses.fields(OrderAck)} == {
        "client_order_id",
        "exchange_order_id",
        "exchange_ts",
    }


def test_exchange_ts_optional() -> None:
    assert ack(exchange_ts=None).exchange_ts is None


@pytest.mark.parametrize("field", ["client_order_id", "exchange_order_id"])
@pytest.mark.parametrize("value", ["", " o-1", None, 1])
def test_identifiers_required_and_clean(field: str, value: Any) -> None:
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
def test_exchange_ts_must_be_utc(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^exchange_ts "):
        ack(exchange_ts=value)
