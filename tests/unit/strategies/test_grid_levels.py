"""Grid geometry: deterministic price levels between two bounds."""

from __future__ import annotations

import itertools
from decimal import Decimal, getcontext, localcontext
from typing import Any

import pytest

from app.domain.enums import GridSpacing
from app.domain.errors import DomainValidationError
from app.strategies.grid.levels import generate_grid_levels

D = Decimal
ARITH = GridSpacing.ARITHMETIC
GEOM = GridSpacing.GEOMETRIC


def grid(lower: str, upper: str, levels: int, spacing: GridSpacing) -> tuple[Decimal, ...]:
    return generate_grid_levels(
        lower_price=D(lower), upper_price=D(upper), levels=levels, spacing=spacing
    )


def assert_invariants(
    result: tuple[Decimal, ...], lower: Decimal, upper: Decimal, levels: int
) -> None:
    assert isinstance(result, tuple)
    assert len(result) == levels
    assert result[0] == lower
    assert result[-1] == upper
    for value in result:
        assert isinstance(value, Decimal)
        assert value.is_finite()
        assert value > 0
    for left, right in itertools.pairwise(result):
        assert left < right


# --- Semantics of `levels` ---------------------------------------------------------------


@pytest.mark.parametrize("spacing", [ARITH, GEOM])
def test_levels_counts_both_bounds(spacing: GridSpacing) -> None:
    result = grid("100", "200", 3, spacing)
    assert len(result) == 3
    assert (result[0], result[-1]) == (D("100"), D("200"))


@pytest.mark.parametrize("spacing", [ARITH, GEOM])
def test_two_levels_are_the_bounds(spacing: GridSpacing) -> None:
    assert grid("100", "200", 2, spacing) == (D("100"), D("200"))


@pytest.mark.parametrize("spacing", [ARITH, GEOM])
def test_endpoints_are_the_input_values(spacing: GridSpacing) -> None:
    lower, upper = D("100.50"), D("200.250")
    result = generate_grid_levels(lower_price=lower, upper_price=upper, levels=5, spacing=spacing)
    assert result[0] is lower
    assert result[-1] is upper


# --- Arithmetic --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "levels", "expected"),
    [
        ("100", "200", 3, ["100", "150", "200"]),
        ("100", "200", 5, ["100", "125", "150", "175", "200"]),
        ("100.5", "200.5", 3, ["100.5", "150.5", "200.5"]),
        ("0.1", "0.4", 4, ["0.1", "0.2", "0.3", "0.4"]),
    ],
)
def test_arithmetic_exact_cases(lower: str, upper: str, levels: int, expected: list[str]) -> None:
    result = grid(lower, upper, levels, ARITH)
    assert result == tuple(D(v) for v in expected)
    assert [str(v) for v in result] == expected  # no trailing-zero noise


def test_arithmetic_uneven_division() -> None:
    result = grid("100", "101", 4, ARITH)
    assert_invariants(result, D("100"), D("101"), 4)
    with localcontext() as ctx:
        ctx.prec = 60
        third = D(1) / 3
        tolerance = D("1E-35")
        assert abs(result[1] - (100 + third)) < tolerance
        assert abs(result[2] - (100 + 2 * third)) < tolerance
    # 40 significant digits: 3 before the decimal point, 37 after.
    assert result[1] == D("100." + "3" * 37)


# --- Geometric ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "levels", "expected"),
    [
        ("100", "400", 3, ["100", "200", "400"]),
        ("100", "800", 4, ["100", "200", "400", "800"]),
        ("1", "16", 5, ["1", "2", "4", "8", "16"]),
        ("0.5", "4.5", 3, ["0.5", "1.5", "4.5"]),
    ],
)
def test_geometric_perfect_ratios_are_exact(
    lower: str, upper: str, levels: int, expected: list[str]
) -> None:
    result = grid(lower, upper, levels, GEOM)
    assert result == tuple(D(v) for v in expected)
    assert [str(v) for v in result] == expected


def test_geometric_non_perfect_ratio() -> None:
    result = grid("100", "200", 4, GEOM)
    assert_invariants(result, D("100"), D("200"), 4)
    # Each step multiplies by 2 ** (1/3); check the cube instead of a decimal string.
    with localcontext() as ctx:
        ctx.prec = 60
        ratios = [right / left for left, right in itertools.pairwise(result)]
        for ratio in ratios:
            assert abs(ratio**3 - 2) < D("1E-35")
        assert abs(result[1] - D("125.9921049894873164767210607278228350570")) < D("1E-34")


# --- Property-style checks ---------------------------------------------------------------

BOUNDS = [
    ("1", "2"),
    ("100", "200"),
    ("0.0001", "0.0002"),
    ("25000.5", "71234.75"),
    ("0.00000123", "0.5"),
    ("3", "3.0001"),
]
LEVELS = [2, 3, 7, 10, 51]


@pytest.mark.parametrize(
    ("bounds", "levels", "spacing"), list(itertools.product(BOUNDS, LEVELS, [ARITH, GEOM]))
)
def test_properties(bounds: tuple[str, str], levels: int, spacing: GridSpacing) -> None:
    lower, upper = D(bounds[0]), D(bounds[1])
    result = generate_grid_levels(
        lower_price=lower, upper_price=upper, levels=levels, spacing=spacing
    )
    assert_invariants(result, lower, upper, levels)
    assert (
        generate_grid_levels(lower_price=lower, upper_price=upper, levels=levels, spacing=spacing)
        == result
    )  # deterministic

    with localcontext() as ctx:
        ctx.prec = 60
        if spacing is ARITH:
            expected = (upper - lower) / (levels - 1)
            for left, right in itertools.pairwise(result):
                assert abs((right - left) - expected) <= expected * D("1E-30")
        else:
            ratios = [right / left for left, right in itertools.pairwise(result)]
            for ratio in ratios:
                assert abs(ratio - ratios[0]) <= ratios[0] * D("1E-30")


# --- Extremes / precision ----------------------------------------------------------------


@pytest.mark.parametrize("spacing", [ARITH, GEOM])
@pytest.mark.parametrize(
    ("lower", "upper", "levels"),
    [
        ("0.00000001", "0.00000002", 10),  # very small prices
        ("1E-12", "1", 100),  # wide geometric range of tiny prices
        ("1000000000", "1000000001", 10),  # large prices, narrow range
        ("99999999999999.5", "100000000000000", 5),
        ("65000", "65000.0001", 11),  # narrow range
        ("100", "200", 1000),  # many levels
    ],
)
def test_extreme_inputs_keep_invariants(
    lower: str, upper: str, levels: int, spacing: GridSpacing
) -> None:
    assert_invariants(grid(lower, upper, levels, spacing), D(lower), D(upper), levels)


@pytest.mark.parametrize("spacing", [ARITH, GEOM])
def test_precision_collapse_fails_closed(spacing: GridSpacing) -> None:
    # Adjacent levels would be equal at the output precision: no invalid grid is returned.
    with pytest.raises(DomainValidationError, match="not strictly increasing"):
        grid("1", "1.0000000000000000000000000000000000000000000001", 3, spacing)


def test_global_decimal_context_untouched() -> None:
    before = getcontext().copy()
    with localcontext() as ctx:
        ctx.prec = 3  # a hostile caller context must not change the result
        result = grid("100", "200", 4, GEOM)
        arith = grid("100", "101", 4, ARITH)
    assert result == grid("100", "200", 4, GEOM)
    assert arith == grid("100", "101", 4, ARITH)
    after = getcontext()
    assert (after.prec, after.rounding, after.flags, after.traps) == (
        before.prec,
        before.rounding,
        before.flags,
        before.traps,
    )


# --- Validation --------------------------------------------------------------------------

BAD_PRICES: list[Any] = [100.0, 100, True, "100", None, D("NaN"), D("sNaN"), D("Infinity")]


@pytest.mark.parametrize("value", [*BAD_PRICES, D("0"), D("-1")])
def test_lower_price_validated(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^lower_price must be"):
        generate_grid_levels(lower_price=value, upper_price=D("200"), levels=3, spacing=ARITH)


@pytest.mark.parametrize("value", BAD_PRICES)
def test_upper_price_validated(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^upper_price must be"):
        generate_grid_levels(lower_price=D("100"), upper_price=value, levels=3, spacing=ARITH)


@pytest.mark.parametrize("upper", ["100", "99.99", "100.000"])
def test_upper_must_exceed_lower(upper: str) -> None:
    with pytest.raises(DomainValidationError, match=r"^upper_price .* must be > lower_price"):
        grid("100", upper, 3, ARITH)


@pytest.mark.parametrize("levels", [1, 0, -3, True, False, 3.0, "3", D("3"), None])
def test_levels_validated(levels: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^levels must be"):
        generate_grid_levels(
            lower_price=D("100"), upper_price=D("200"), levels=levels, spacing=ARITH
        )


@pytest.mark.parametrize("spacing", ["arithmetic", "geometric", None, 1])
def test_spacing_must_be_enum(spacing: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^spacing must be a GridSpacing"):
        generate_grid_levels(lower_price=D("100"), upper_price=D("200"), levels=3, spacing=spacing)
