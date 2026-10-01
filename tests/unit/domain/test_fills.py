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


@pytest.mark.parametrize("field", ["exec_id", "exchange_order_id", "symbol"])
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
def test_known_fee_any_sign(fee: Decimal) -> None:
    # Negative fee = maker rebate; zero is a real, known fee of zero.
    f = fill(fee=fee, fee_asset="USDT")
    assert f.fee == fee
    assert f.fee_asset == "USDT"


@pytest.mark.parametrize("value", [0.01, 0, 1, "0.01", True, D("NaN"), D("sNaN"), D("-Infinity")])
def test_known_fee_must_be_finite_decimal(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^fee must be"):
        fill(fee=value)


# --- unknown metadata ---------------------------------------------------------


def test_unknown_fee_is_a_none_pair() -> None:
    f = fill(fee=None, fee_asset=None)
    assert f.fee is None
    assert f.fee_asset is None


def test_unknown_liquidity_role() -> None:
    assert fill(is_maker=None).is_maker is None


def test_all_metadata_unknown_keeps_the_execution_facts() -> None:
    f = fill(fee=None, fee_asset=None, is_maker=None)

    assert (f.fee, f.fee_asset, f.is_maker) == (None, None, None)
    # The execution itself is confirmed and fully known.
    assert f.exec_id == "e-1"
    assert f.exchange_order_id == "o-1"
    assert f.client_order_id == "grid1-buy-0001"
    assert f.side is Side.BUY
    assert f.price == D("65000.5")
    assert f.qty == D("0.001")
    assert f.exchange_ts == TS


@pytest.mark.parametrize(
    ("fee", "fee_asset"),
    [(None, "USDT"), (D("0"), None), (D("0.013"), None), (D("-0.001"), None)],
)
def test_fee_and_fee_asset_are_known_or_unknown_together(
    fee: Decimal | None, fee_asset: str | None
) -> None:
    with pytest.raises(DomainValidationError, match=r"^fee and fee_asset must be"):
        fill(fee=fee, fee_asset=fee_asset)


@pytest.mark.parametrize("value", ["", " USDT", "USDT ", 1])
def test_known_fee_asset_is_validated_text(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^fee_asset "):
        fill(fee_asset=value)


def test_zero_is_never_unknown() -> None:
    known_zero = fill(fee=D("0"), fee_asset="USDT")
    unknown = fill(fee=None, fee_asset=None)

    assert known_zero.fee == 0
    assert unknown.fee is None
    assert known_zero != unknown


# --- liquidity role -----------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, None])
def test_is_maker_true_false_or_unknown(value: bool | None) -> None:
    assert fill(is_maker=value).is_maker is value


@pytest.mark.parametrize("value", [1, 0, "true", "maker", D("1")])
def test_is_maker_must_be_bool_when_known(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^is_maker must be a bool"):
        fill(is_maker=value)


@pytest.mark.parametrize(
    ("fee", "fee_asset", "is_maker"),
    [
        (D("0.001"), "USDT", None),
        (None, None, True),
        (None, None, False),
        (D("0"), "USDT", False),
    ],
)
def test_liquidity_role_is_independent_of_fee_data(
    fee: Decimal | None, fee_asset: str | None, is_maker: bool | None
) -> None:
    f = fill(fee=fee, fee_asset=fee_asset, is_maker=is_maker)

    assert (f.fee, f.fee_asset, f.is_maker) == (fee, fee_asset, is_maker)


@pytest.mark.parametrize("field", ["fee", "fee_asset", "is_maker"])
def test_metadata_fields_have_no_defaults(field: str) -> None:
    # Callers must state "unknown" explicitly.
    values = {k: v for k, v in FILL.items() if k != field}
    with pytest.raises(TypeError):
        Fill(**values)


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
