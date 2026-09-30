"""Rounding to tick_size / qty_step and exchange minimum checks.

Rounding is based on multiples of the step, never on a number of decimal places.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from decimal import Decimal, getcontext, localcontext
from typing import Any

import pytest

from app.domain.enums import RoundingDirection
from app.domain.errors import DomainValidationError
from app.domain.instrument import InstrumentSpec
from app.domain.rounding import (
    is_price_aligned,
    is_qty_aligned,
    meets_min_notional,
    meets_min_qty,
    round_price,
    round_qty,
)

DOWN = RoundingDirection.DOWN
UP = RoundingDirection.UP
D = Decimal

STEPS = [D("5"), D("0.5"), D("0.25"), D("0.01"), D("0.001"), D("0.0001")]


def spec(
    tick_size: str = "0.01",
    qty_step: str = "0.001",
    min_qty: str = "0.001",
    max_qty: str = "1000000",
    min_notional: str = "5",
) -> InstrumentSpec:
    return InstrumentSpec(
        symbol="TESTUSDT",
        base_asset="TEST",
        quote_asset="USDT",
        tick_size=D(tick_size),
        qty_step=D(qty_step),
        min_qty=D(min_qty),
        max_qty=D(max_qty),
        min_notional=D(min_notional),
    )


def spec_with_step(step: Decimal) -> InstrumentSpec:
    return spec(tick_size=str(step), qty_step=str(step), min_qty=str(step))


# --- Known examples ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tick", "price", "down", "up"),
    [
        # Multiple of 0.25, not "two decimal places".
        ("0.25", "100.37", "100.25", "100.50"),
        ("0.25", "100.13", "100.00", "100.25"),
        ("0.25", "100.26", "100.25", "100.50"),
        ("5", "103", "100", "105"),
        ("5", "100.01", "100", "105"),
        ("5", "99.99", "95", "100"),
        ("5", "7", "5", "10"),
        ("0.5", "10.26", "10.0", "10.5"),
        ("0.5", "10.74", "10.5", "11.0"),
        ("0.01", "1.234", "1.23", "1.24"),
        ("0.001", "0.12345", "0.123", "0.124"),
        ("0.0001", "65000.12345", "65000.1234", "65000.1235"),
    ],
)
def test_round_price_examples(tick: str, price: str, down: str, up: str) -> None:
    s = spec(tick_size=tick)
    assert round_price(D(price), s, DOWN) == D(down)
    assert round_price(D(price), s, UP) == D(up)


@pytest.mark.parametrize(
    ("step", "qty", "down", "up"),
    [
        ("0.001", "0.0125", "0.012", "0.013"),
        ("0.25", "3.3", "3.25", "3.50"),
        ("5", "12", "10", "15"),
        ("0.0001", "1.00005", "1.0000", "1.0001"),
    ],
)
def test_round_qty_examples(step: str, qty: str, down: str, up: str) -> None:
    s = spec(qty_step=step, min_qty=step)
    assert round_qty(D(qty), s, DOWN) == D(down)
    assert round_qty(D(qty), s, UP) == D(up)


# --- Boundaries --------------------------------------------------------------------------


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("multiple", ["1", "2", "7", "1000", "123457"])
@pytest.mark.parametrize("direction", [DOWN, UP])
def test_aligned_value_is_unchanged(
    step: Decimal, multiple: str, direction: RoundingDirection
) -> None:
    value = step * D(multiple)
    s = spec_with_step(step)
    assert round_price(value, s, direction) == value
    assert round_qty(value, s, direction) == value
    assert is_price_aligned(value, s)
    assert is_qty_aligned(value, s)


@pytest.mark.parametrize("step", STEPS)
def test_just_above_and_below_a_multiple(step: Decimal) -> None:
    s = spec_with_step(step)
    level = step * 40
    epsilon = D("0.00000001")

    assert round_price(level + epsilon, s, DOWN) == level
    assert round_price(level + epsilon, s, UP) == level + step
    assert round_price(level - epsilon, s, DOWN) == level - step
    assert round_price(level - epsilon, s, UP) == level
    assert not is_price_aligned(level + epsilon, s)
    assert not is_price_aligned(level - epsilon, s)


def test_trailing_zeros_do_not_matter() -> None:
    assert round_price(D("100.3700"), spec(tick_size="0.250"), DOWN) == D("100.25")
    assert round_price(D("100.3700"), spec(tick_size="0.25"), UP) == D("100.5")
    assert round_price(D("100.2500"), spec(tick_size="0.25"), UP) == D("100.25")
    assert is_price_aligned(D("100.2500"), spec(tick_size="0.2500"))
    assert is_price_aligned(D("5.000"), spec(tick_size="5"))
    assert is_price_aligned(D("1E+1"), spec(tick_size="5"))


def test_aligned_value_is_returned_as_is() -> None:
    value = D("100.2500")
    assert round_price(value, spec(tick_size="0.25"), UP) is value


def test_very_small_values() -> None:
    s = spec(tick_size="0.00000001", qty_step="0.00000001", min_qty="0.00000001")
    assert round_price(D("0.000000015"), s, DOWN) == D("0.00000001")
    assert round_price(D("0.000000015"), s, UP) == D("0.00000002")
    assert round_qty(D("1E-8"), s, DOWN) == D("0.00000001")


def test_value_below_one_step_rounds_up_to_one_step() -> None:
    assert round_price(D("0.1"), spec(tick_size="0.25"), UP) == D("0.25")
    assert round_qty(D("0.0001"), spec(qty_step="0.001"), UP) == D("0.001")


@pytest.mark.parametrize("value", ["0.1", "0.24999999"])
def test_rounding_down_to_zero_is_rejected(value: str) -> None:
    with pytest.raises(DomainValidationError, match=r"^price .* rounds down to zero"):
        round_price(D(value), spec(tick_size="0.25"), DOWN)
    with pytest.raises(DomainValidationError, match=r"^qty .* rounds down to zero"):
        round_qty(D(value), spec(qty_step="0.25", min_qty="0.25"), DOWN)


def test_large_values_are_exact() -> None:
    s = spec(tick_size="0.0001", qty_step="0.0001", min_qty="0.0001")
    big = D("12345678901234567890.12345")
    assert round_price(big, s, DOWN) == D("12345678901234567890.1234")
    assert round_price(big, s, UP) == D("12345678901234567890.1235")
    assert round_qty(D("99999999999999999999"), s, UP) == D("99999999999999999999")


def test_value_beyond_exact_precision_is_rejected_not_rounded() -> None:
    s = spec(tick_size="0.0001")
    too_precise = D("1" * 45 + ".12345")
    with pytest.raises(DomainValidationError, match="cannot be computed exactly"):
        round_price(too_precise, s, DOWN)


def test_global_decimal_context_is_neither_used_nor_changed() -> None:
    before = getcontext().copy()
    with localcontext() as ctx:
        ctx.prec = 3  # a hostile global context must not affect results
        assert round_price(D("65000.12345"), spec(tick_size="0.0001"), DOWN) == D("65000.1234")
        assert meets_min_notional(D("12345.67"), D("0.001"), spec(min_notional="12.34567"))
    after = getcontext()
    assert (after.prec, after.rounding, after.flags, after.traps) == (
        before.prec,
        before.rounding,
        before.flags,
        before.traps,
    )


# --- Property-like checks ----------------------------------------------------------------

PROPERTY_VALUES = [
    D(v)
    for v in [
        "0.00000001",
        "0.0001",
        "0.3",
        "1",
        "4.99999999",
        "5.00000001",
        "100.37",
        "65000.12345",
        "99999.99999999",
        "123456789.123456789",
        "98765432109876.54321",
    ]
]


@pytest.mark.parametrize(
    ("step", "value", "direction"), list(itertools.product(STEPS, PROPERTY_VALUES, [DOWN, UP]))
)
def test_rounding_properties(step: Decimal, value: Decimal, direction: RoundingDirection) -> None:
    s = spec_with_step(step)
    if direction is DOWN and value < step:
        with pytest.raises(DomainValidationError, match="rounds down to zero"):
            round_price(value, s, direction)
        return

    for rounded in (round_price(value, s, direction), round_qty(value, s, direction)):
        assert rounded % step == 0
        assert rounded > 0
        assert abs(rounded - value) < step
        if direction is DOWN:
            assert rounded <= value
        else:
            assert rounded >= value
        assert is_price_aligned(rounded, s)
        # Idempotent: rounding an aligned value changes nothing.
        assert round_price(rounded, s, DOWN) == rounded
        assert round_price(rounded, s, UP) == rounded


# --- min_qty / min_notional --------------------------------------------------------------


@pytest.mark.parametrize(
    ("qty", "expected"),
    [("0.009", False), ("0.00999999", False), ("0.01", True), ("0.0100", True), ("0.011", True)],
)
def test_meets_min_qty_inclusive(qty: str, expected: bool) -> None:
    assert meets_min_qty(D(qty), spec(qty_step="0.001", min_qty="0.01")) is expected


@pytest.mark.parametrize(
    ("price", "qty", "expected"),
    [
        ("5", "1", True),  # exactly equal
        ("2.5", "2", True),  # exactly equal, different factors
        ("0.1", "50", True),  # 0.1 * 50 == 5 exactly (would be 5.000000000000001-ish in float)
        ("4.99999999", "1", False),
        ("5.00000001", "1", True),
        ("1", "4.999", False),
    ],
)
def test_meets_min_notional_inclusive_and_exact(price: str, qty: str, expected: bool) -> None:
    assert meets_min_notional(D(price), D(qty), spec(min_notional="5")) is expected


def test_min_notional_zero_always_met() -> None:
    assert meets_min_notional(D("0.0001"), D("0.001"), spec(min_notional="0"))


def test_min_notional_uses_full_precision_product() -> None:
    s = spec(min_notional="1.000000000000000000000000000001")
    assert not meets_min_notional(D("1"), D("1"), s)
    assert meets_min_notional(D("1.000000000000000000000000000001"), D("1"), s)


# --- Invalid input -----------------------------------------------------------------------

INVALID_VALUES: list[Any] = [
    1.5,
    1,
    True,
    "1.5",
    None,
    D("NaN"),
    D("sNaN"),
    D("Infinity"),
    D("-Infinity"),
    D("0"),
    D("-1"),
]


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_rounding_rejects_invalid_values(value: Any) -> None:
    s = spec()
    with pytest.raises(DomainValidationError, match=r"^price must be"):
        round_price(value, s, DOWN)
    with pytest.raises(DomainValidationError, match=r"^qty must be"):
        round_qty(value, s, UP)


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_checks_reject_invalid_values(value: Any) -> None:
    s = spec()
    calls: list[Callable[[], bool]] = [
        lambda: is_price_aligned(value, s),
        lambda: is_qty_aligned(value, s),
        lambda: meets_min_qty(value, s),
        lambda: meets_min_notional(value, D("1"), s),
        lambda: meets_min_notional(D("1"), value, s),
    ]
    for call in calls:
        with pytest.raises(DomainValidationError):
            call()


@pytest.mark.parametrize("direction", ["down", "up", True, None, 0])
def test_direction_must_be_enum(direction: Any) -> None:
    with pytest.raises(DomainValidationError, match="direction must be a RoundingDirection"):
        round_price(D("1"), spec(), direction)
