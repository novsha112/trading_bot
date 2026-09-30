"""InstrumentSpec: exchange trading constraints of one instrument."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest

from app.domain.errors import DomainValidationError
from app.domain.instrument import InstrumentSpec

VALID: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "base_asset": "BTC",
    "quote_asset": "USDT",
    "tick_size": Decimal("0.1"),
    "qty_step": Decimal("0.001"),
    "min_qty": Decimal("0.001"),
    "max_qty": Decimal("100"),
    "min_notional": Decimal("5"),
}


def make(**overrides: Any) -> InstrumentSpec:
    return InstrumentSpec(**{**VALID, **overrides})


def test_valid_spec_keeps_values_unchanged() -> None:
    spec = make()
    for name, value in VALID.items():
        assert getattr(spec, name) == value
    # No hidden rounding or normalization.
    assert str(make(tick_size=Decimal("0.10")).tick_size) == "0.10"


def test_spec_is_immutable_and_slotted() -> None:
    spec = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tick_size = Decimal("1")  # type: ignore[misc]
    assert not hasattr(spec, "__dict__")


def test_spec_requires_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        InstrumentSpec(*VALID.values())  # type: ignore[call-arg]


def test_spec_equality_and_hash() -> None:
    assert make() == make()
    assert hash(make()) == hash(make())


def test_boundary_values_accepted() -> None:
    spec = make(max_qty=Decimal("0.001"), min_notional=Decimal("0"))
    assert spec.max_qty == spec.min_qty
    assert spec.min_notional == 0


@pytest.mark.parametrize("field", ["symbol", "base_asset", "quote_asset"])
@pytest.mark.parametrize("value", ["", "   ", " BTCUSDT", "BTC\n", None, 1, b"BTC"])
def test_text_fields_must_be_non_empty_strings(field: str, value: object) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        make(**{field: value})


@pytest.mark.parametrize("field", ["tick_size", "qty_step", "min_qty"])
@pytest.mark.parametrize("value", [Decimal("0"), Decimal("-0"), Decimal("-0.1")])
def test_steps_and_min_qty_must_be_positive(field: str, value: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be > 0"):
        make(**{field: value})


def test_max_qty_below_min_qty_rejected() -> None:
    with pytest.raises(DomainValidationError, match=r"^max_qty .* must be >= min_qty"):
        make(min_qty=Decimal("1"), max_qty=Decimal("0.999"))


def test_negative_min_notional_rejected() -> None:
    with pytest.raises(DomainValidationError, match=r"^min_notional must be >= 0"):
        make(min_notional=Decimal("-1"))


@pytest.mark.parametrize("field", ["tick_size", "qty_step", "min_qty", "max_qty", "min_notional"])
@pytest.mark.parametrize(
    "value", [0.1, 1, True, "0.1", None, Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")]
)
def test_decimal_fields_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be"):
        make(**{field: value})
