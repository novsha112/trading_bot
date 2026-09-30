"""Fill: a confirmed execution reported by the exchange."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import Side
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill

D = Decimal
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

FILL: dict[str, Any] = {
    "exec_id": "e-1",
    "exchange_order_id": "o-1",
    "client_order_id": "grid1-buy-0001",
    "symbol": "BTCUSDT",
    "side": Side.BUY,
    "price": D("65000.5"),
    "qty": D("0.001"),
    "fee": D("0.0130001"),
    "fee_asset": "USDT",
    "is_maker": True,
    "exchange_ts": TS,
}


def fill(**overrides: Any) -> Fill:
    return Fill(**{**FILL, **overrides})


def test_structure_frozen_slotted_keyword_only() -> None:
    f = fill()
    for name, value in FILL.items():
        assert getattr(f, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.qty = D("1")  # type: ignore[misc]
    assert not hasattr(f, "__dict__")
    with pytest.raises(TypeError):
        Fill(*FILL.values())  # type: ignore[call-arg]


def test_client_order_id_optional() -> None:
    # Fills of orders not placed by the bot (e.g. manual) have no client order id.
    assert fill(client_order_id=None).client_order_id is None


@pytest.mark.parametrize(
    "field", ["exec_id", "exchange_order_id", "client_order_id", "symbol", "fee_asset"]
)
@pytest.mark.parametrize("value", ["", " x", 1])
def test_text_fields_validated(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        fill(**{field: value})


@pytest.mark.parametrize("field", ["exec_id", "exchange_order_id", "symbol", "fee_asset"])
def test_required_text_fields_reject_none(field: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be a str"):
        fill(**{field: None})


@pytest.mark.parametrize("value", ["buy", "Buy", None])
def test_side_must_be_domain_enum(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^side must be a Side"):
        fill(side=value)


@pytest.mark.parametrize("field", ["price", "qty"])
@pytest.mark.parametrize("value", [D("0"), D("-1"), 1.5, 1, True, D("NaN"), D("Infinity")])
def test_price_and_qty_must_be_positive_decimal(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be"):
        fill(**{field: value})


@pytest.mark.parametrize("fee", [D("-0.001"), D("0"), D("0.013")])
def test_fee_any_sign(fee: Decimal) -> None:
    # Negative fee = maker rebate.
    assert fill(fee=fee).fee == fee


@pytest.mark.parametrize("value", [0.01, 0, D("NaN"), D("sNaN"), D("-Infinity"), None])
def test_fee_must_be_finite_decimal(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^fee must be"):
        fill(fee=value)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_is_maker_must_be_bool(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^is_maker must be a bool"):
        fill(is_maker=value)


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 15, 12, 0),  # noqa: DTZ001 - deliberately naive
        datetime(2026, 1, 15, 12, 0, tzinfo=timezone(timedelta(hours=-5))),
        1768478400000,
    ],
)
def test_exchange_ts_must_be_utc(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^exchange_ts "):
        fill(exchange_ts=value)
