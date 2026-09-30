"""Position snapshot with signed quantity."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import PositionSide
from app.domain.errors import DomainValidationError
from app.domain.positions import Position

D = Decimal
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

POSITION: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "qty": D("0.01"),
    "entry_price": D("65000"),
    "mark_price": D("65100"),
    "unrealized_pnl": D("1"),
    "realized_pnl": D("-0.5"),
    "updated_at": TS,
}


def position(**overrides: Any) -> Position:
    return Position(**{**POSITION, **overrides})


def test_structure_frozen_slotted_keyword_only() -> None:
    p = position()
    for name, value in POSITION.items():
        assert getattr(p, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.qty = D("1")  # type: ignore[misc]
    assert not hasattr(p, "__dict__")
    with pytest.raises(TypeError):
        Position(*POSITION.values())  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("qty", "entry", "side"),
    [
        (D("0.01"), D("65000"), PositionSide.LONG),
        (D("0.00000001"), D("1"), PositionSide.LONG),
        (D("-0.01"), D("65000"), PositionSide.SHORT),
        (D("0"), None, PositionSide.FLAT),
        (D("-0"), None, PositionSide.FLAT),
        (D("0.000"), None, PositionSide.FLAT),
    ],
)
def test_side_is_derived_from_signed_qty(
    qty: Decimal, entry: Decimal | None, side: PositionSide
) -> None:
    assert position(qty=qty, entry_price=entry).side is side


def test_side_is_not_a_stored_field() -> None:
    assert "side" not in {f.name for f in dataclasses.fields(Position)}
    with pytest.raises(TypeError):
        position(side=PositionSide.LONG)


def test_flat_position_with_entry_price_rejected() -> None:
    with pytest.raises(DomainValidationError, match=r"^entry_price must be None"):
        position(qty=D("0"), entry_price=D("65000"))


@pytest.mark.parametrize("qty", [D("0.01"), D("-0.01")])
def test_open_position_without_entry_price_rejected(qty: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=r"^entry_price must be a Decimal"):
        position(qty=qty, entry_price=None)


@pytest.mark.parametrize("value", [D("0"), D("-1")])
def test_entry_price_must_be_positive(value: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=r"^entry_price must be > 0"):
        position(entry_price=value)


def test_mark_price_optional_but_positive() -> None:
    assert position(mark_price=None).mark_price is None
    with pytest.raises(DomainValidationError, match=r"^mark_price must be > 0"):
        position(mark_price=D("0"))


@pytest.mark.parametrize(
    "field", ["qty", "entry_price", "mark_price", "unrealized_pnl", "realized_pnl"]
)
@pytest.mark.parametrize("value", [1.5, 1, True, D("NaN"), D("sNaN"), D("Infinity")])
def test_numeric_fields_reject_invalid(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be"):
        position(**{field: value})


@pytest.mark.parametrize("pnl", [D("-100"), D("0"), D("100")])
def test_pnl_any_sign(pnl: Decimal) -> None:
    p = position(unrealized_pnl=pnl, realized_pnl=pnl)
    assert p.unrealized_pnl == pnl == p.realized_pnl


def test_flat_position_may_keep_realized_pnl() -> None:
    p = position(qty=D("0"), entry_price=None, unrealized_pnl=D("0"), realized_pnl=D("12.5"))
    assert p.side is PositionSide.FLAT


@pytest.mark.parametrize(
    "value",
    [datetime(2026, 1, 15), None],  # noqa: DTZ001 - deliberately naive
)
def test_updated_at_must_be_utc(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^updated_at "):
        position(updated_at=value)


def test_symbol_validated() -> None:
    with pytest.raises(DomainValidationError, match=r"^symbol "):
        position(symbol="")
