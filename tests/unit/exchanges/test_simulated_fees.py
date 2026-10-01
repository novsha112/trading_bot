"""Deterministic simulation trading fees: explicit schedule and liquidity role."""

from __future__ import annotations

import ast
import dataclasses
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.errors import DomainValidationError
from app.exchanges import simulated_fees
from app.exchanges.simulated_fees import (
    LiquidityRole,
    SimulatedFeePolicy,
    TradingFeeSchedule,
)

D = Decimal


def schedule(**overrides: Any) -> TradingFeeSchedule:
    values: dict[str, Any] = {
        "maker_rate": D("0.0002"),
        "taker_rate": D("0.00055"),
        "fee_asset": "USDT",
    }
    return TradingFeeSchedule(**{**values, **overrides})


# --- schedule -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "expected"),
    [(LiquidityRole.MAKER, D("0.02")), (LiquidityRole.TAKER, D("0.055"))],
)
def test_role_selects_rate(role: LiquidityRole, expected: Decimal) -> None:
    assert schedule().calculate(price=D("100"), qty=D("1"), liquidity_role=role) == expected


def test_formula_uses_given_price_and_qty() -> None:
    fee = schedule(taker_rate=D("0.001")).calculate(
        price=D("110"), qty=D("3"), liquidity_role=LiquidityRole.TAKER
    )

    assert fee == D("0.33")


def test_zero_rate_is_a_known_zero() -> None:
    fee = schedule(maker_rate=D("0")).calculate(
        price=D("100"), qty=D("5"), liquidity_role=LiquidityRole.MAKER
    )

    assert fee == 0
    assert fee is not None


def test_negative_maker_rate_is_a_rebate() -> None:
    fee = schedule(maker_rate=D("-0.0001")).calculate(
        price=D("100"), qty=D("10"), liquidity_role=LiquidityRole.MAKER
    )

    assert fee == D("-0.1")


def test_fee_is_exact_without_currency_rounding() -> None:
    fee = schedule(taker_rate=D("0.00055")).calculate(
        price=D("65000.5"), qty=D("0.003"), liquidity_role=LiquidityRole.TAKER
    )

    assert fee == D("0.1072508250")  # exact: no settlement rounding modeled


def test_fee_ignores_the_global_decimal_context() -> None:
    def compute() -> Decimal:
        return schedule(taker_rate=D("0.000555")).calculate(
            price=D("12345.6789"), qty=D("0.123"), liquidity_role=LiquidityRole.TAKER
        )

    baseline = compute()
    with localcontext() as context:
        context.prec = 2
        context.rounding = "ROUND_UP"
        low = compute()

    assert low == baseline == D("0.8427777701085")
    assert str(low) == str(baseline)


@pytest.mark.parametrize("field", ["maker_rate", "taker_rate"])
@pytest.mark.parametrize(
    "value", [D("NaN"), D("sNaN"), D("Infinity"), D("-Infinity"), 0.001, 0, "0.001", True, None]
)
def test_invalid_rates_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        schedule(**{field: value})


def test_rate_decimal_subclass_rejected() -> None:
    class Sub(Decimal):
        pass

    with pytest.raises(ValueError, match="maker_rate"):
        schedule(maker_rate=Sub("0.001"))


@pytest.mark.parametrize("value", [D("0.5"), D("2"), D("-0.3")])
def test_no_artificial_rate_range(value: Decimal) -> None:
    assert schedule(maker_rate=value, taker_rate=value).maker_rate == value


@pytest.mark.parametrize("asset", ["", " USDT", "USDT ", None, 1])
def test_invalid_fee_asset_rejected(asset: object) -> None:
    with pytest.raises(DomainValidationError, match="fee_asset"):
        schedule(fee_asset=asset)


@pytest.mark.parametrize("role", ["maker", "MAKER", None, True])
def test_calculate_requires_a_liquidity_role(role: object) -> None:
    with pytest.raises(ValueError, match="liquidity_role"):
        schedule().calculate(price=D("1"), qty=D("1"), liquidity_role=role)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("price", "qty"), [(D("0"), D("1")), (D("1"), D("-1")), (D("NaN"), D("1")), (1, D("1"))]
)
def test_calculate_requires_positive_decimals(price: object, qty: object) -> None:
    with pytest.raises(ValueError, match=r"price|qty"):
        schedule().calculate(
            price=price,  # type: ignore[arg-type]
            qty=qty,  # type: ignore[arg-type]
            liquidity_role=LiquidityRole.TAKER,
        )


def test_schedule_and_policy_are_immutable() -> None:
    s = schedule()
    policy = SimulatedFeePolicy(schedule=s, liquidity_role=LiquidityRole.TAKER)

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.maker_rate = D("1")  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.liquidity_role = LiquidityRole.MAKER  # type: ignore[misc]


# --- policy -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "fee", "is_maker"),
    [(LiquidityRole.MAKER, D("0.02"), True), (LiquidityRole.TAKER, D("0.055"), False)],
)
def test_policy_produces_fill_fee_fields(role: LiquidityRole, fee: Decimal, is_maker: bool) -> None:
    policy = SimulatedFeePolicy(schedule=schedule(), liquidity_role=role)

    assert policy.fill_fee(price=D("100"), qty=D("1")) == (fee, "USDT", is_maker)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"schedule": None, "liquidity_role": LiquidityRole.TAKER}, TypeError),
        ({"schedule": "x", "liquidity_role": LiquidityRole.TAKER}, TypeError),
        ({"liquidity_role": None}, ValueError),
        ({"liquidity_role": "taker"}, ValueError),
    ],
)
def test_policy_must_be_fully_configured(kwargs: dict[str, Any], error: type[Exception]) -> None:
    values: dict[str, Any] = {"schedule": schedule(), **kwargs}
    with pytest.raises(error):
        SimulatedFeePolicy(**values)


def test_liquidity_role_is_simulation_local() -> None:
    assert {r.value for r in LiquidityRole} == {"maker", "taker"}
    assert LiquidityRole.__module__ == "app.exchanges.simulated_fees"


def test_module_has_no_forbidden_dependencies() -> None:
    source = Path(simulated_fees.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith("app.domain"), name
    for banned in ("float(", "getcontext", "setcontext", "localcontext", "balance"):
        assert banned not in source, banned
