"""Exposure decomposition and worst-case pending exposure for Risk V1."""

from __future__ import annotations

import ast
import itertools
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.risk import exposure as exposure_module
from app.risk.exposure import ExposureCalculationError, calculate_exposure
from app.risk.models import ExposureChange, OpenOrderExposure

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def intent(side: Side, qty: str, *, reduce_only: bool = False) -> PlaceOrderIntent:
    return PlaceOrderIntent(
        intent_id="i-1",
        strategy_id="grid",
        symbol="BTCUSDT",
        side=side,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=reduce_only,
        tag=None,
        created_at=T0,
    )


def pending(
    side: Side,
    qty: str,
    *,
    reduce_only: bool = False,
    status: OrderStatus = OrderStatus.OPEN,
) -> OpenOrderExposure:
    return OpenOrderExposure(
        side=side, remaining_qty=D(qty), price=D("100"), reduce_only=reduce_only, status=status
    )


def calc(
    position: str,
    new: PlaceOrderIntent,
    *orders: OpenOrderExposure,
    valid_reduce_only: bool = False,
) -> ExposureChange:
    return calculate_exposure(
        position_qty=D(position),
        intent=new,
        open_orders=tuple(orders),
        valid_reduce_only=valid_reduce_only,
    )


def split(e: ExposureChange) -> tuple[Decimal, Decimal]:
    return e.reducing_qty, e.increasing_qty


def worst(e: ExposureChange) -> tuple[Decimal, Decimal]:
    return e.worst_long_qty, e.worst_short_qty


@contextmanager
def low_precision() -> Iterator[None]:
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        yield


# --- decomposition ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("position", "side", "qty", "reducing", "increasing"),
    [
        ("0", Side.BUY, "5", "0", "5"),  # flat + buy
        ("0", Side.SELL, "5", "0", "5"),  # flat + sell
        ("5", Side.BUY, "2", "0", "2"),  # long increase
        ("5", Side.SELL, "2", "2", "0"),  # long partial close
        ("5", Side.SELL, "5", "5", "0"),  # long exact close
        ("5", Side.SELL, "8", "5", "3"),  # long reversal
        ("-5", Side.SELL, "2", "0", "2"),  # short increase
        ("-5", Side.BUY, "2", "2", "0"),  # short partial close
        ("-5", Side.BUY, "5", "5", "0"),  # short exact close
        ("-5", Side.BUY, "8", "5", "3"),  # short reversal
        ("3.333", Side.SELL, "7.777", "3.333", "4.444"),  # awkward reversal
        ("-7.777", Side.BUY, "3.333", "3.333", "0"),  # awkward partial
        ("7.777", Side.BUY, "3.333", "0", "3.333"),  # awkward increase
    ],
)
def test_decomposition(position: str, side: Side, qty: str, reducing: str, increasing: str) -> None:
    e = calc(position, intent(side, qty))

    assert split(e) == (D(reducing), D(increasing))


def test_open_orders_do_not_change_the_decomposition() -> None:
    orders = (
        pending(Side.SELL, "4"),
        pending(Side.BUY, "9"),
        pending(Side.SELL, "2", reduce_only=True),
    )

    with_orders = calc("5", intent(Side.SELL, "8"), *orders)
    without = calc("5", intent(Side.SELL, "8"))

    assert split(with_orders) == split(without) == (D("5"), D("3"))


# --- valid reduce-only -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("position", "side", "qty", "reducing"),
    [
        ("5", Side.SELL, "2", "2"),  # smaller
        ("5", Side.SELL, "5", "5"),  # exact
        ("5", Side.SELL, "100", "5"),  # oversized: capped at the position
        ("-5", Side.BUY, "2", "2"),
        ("-5", Side.BUY, "5", "5"),
        ("-5", Side.BUY, "100", "5"),
        ("3.333", Side.SELL, "7.777", "3.333"),
    ],
)
def test_valid_reduce_only_never_increases(
    position: str, side: Side, qty: str, reducing: str
) -> None:
    e = calc(position, intent(side, qty, reduce_only=True), valid_reduce_only=True)

    assert split(e) == (D(reducing), D("0"))


def test_valid_reduce_only_flag_requires_a_reduce_only_intent() -> None:
    with pytest.raises(DomainValidationError, match="valid_reduce_only"):
        calc("5", intent(Side.SELL, "2", reduce_only=False), valid_reduce_only=True)


def test_reduce_only_intent_not_marked_valid_is_treated_as_ordinary() -> None:
    # The evaluator decides validity; without its confirmation the intent is not
    # given the reduce-only exemption.
    e = calc("5", intent(Side.SELL, "8", reduce_only=True), valid_reduce_only=False)

    assert split(e) == (D("5"), D("3"))
    assert worst(e) == (D("5"), D("3"))


# --- worst case ---------------------------------------------------------------------------


def test_worst_case_from_flat_does_not_net_opposite_orders() -> None:
    e = calc("0", intent(Side.BUY, "5"), pending(Side.BUY, "3"), pending(Side.SELL, "4"))

    assert worst(e) == (D("8"), D("4"))


def test_worst_case_from_long() -> None:
    e = calc("5", intent(Side.SELL, "8"), pending(Side.BUY, "3"), pending(Side.SELL, "10"))

    # Long path ignores sells: 5 + 3. Short path: 5 - 10 - 8 = -13.
    assert worst(e) == (D("8"), D("13"))


def test_worst_case_from_short() -> None:
    e = calc("-5", intent(Side.BUY, "8"), pending(Side.SELL, "3"), pending(Side.BUY, "10"))

    # Short path ignores buys: -5 - 3. Long path: -5 + 10 + 8 = 13.
    assert worst(e) == (D("13"), D("8"))


def test_worst_case_never_negative() -> None:
    e = calc("5", intent(Side.BUY, "1"))

    assert worst(e) == (D("6"), D("0"))  # a long position has no short worst case


def test_full_non_reduce_only_intent_enters_its_side() -> None:
    # Only 3 of SELL 8 increase risk, but the whole order may fill: short path -3.
    e = calc("5", intent(Side.SELL, "8"))

    assert split(e) == (D("5"), D("3"))
    assert worst(e) == (D("5"), D("3"))


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_pending_reduce_only_orders_are_ignored(side: Side) -> None:
    e = calc(
        "5",
        intent(Side.BUY, "1"),
        pending(side, "100", reduce_only=True),
        pending(Side.SELL, "50", reduce_only=True),
    )

    assert worst(e) == (D("6"), D("0"))


@pytest.mark.parametrize("status", [OrderStatus.UNKNOWN, OrderStatus.SUBMITTING, OrderStatus.NEW])
def test_uncertain_pending_orders_count_in_full(status: OrderStatus) -> None:
    e = calc("0", intent(Side.BUY, "1"), pending(Side.SELL, "7.777", status=status))

    assert worst(e) == (D("1"), D("7.777"))


def test_new_valid_reduce_only_does_not_enter_the_worst_case() -> None:
    orders = (pending(Side.BUY, "2"), pending(Side.SELL, "4"))

    e = calc("5", intent(Side.SELL, "100", reduce_only=True), *orders, valid_reduce_only=True)

    assert worst(e) == (D("7"), D("0"))  # 5 + 2; short path 5 - 4 = 1 > 0


def test_multiple_pending_orders_are_summed_exactly() -> None:
    orders = (
        pending(Side.BUY, "3.333"),
        pending(Side.BUY, "7.777"),
        pending(Side.SELL, "5.555"),
        pending(Side.SELL, "1.111"),
        pending(Side.BUY, "9.999", reduce_only=True),
    )

    e = calc("1.234", intent(Side.BUY, "0.001"), *orders)

    assert worst(e) == (D("12.345"), D("5.432"))


def test_order_of_pending_orders_does_not_matter() -> None:
    orders = [
        pending(Side.BUY, "3.333"),
        pending(Side.SELL, "7.777"),
        pending(Side.BUY, "5.555", status=OrderStatus.UNKNOWN),
        pending(Side.SELL, "1.111", reduce_only=True),
    ]
    results = {
        calc("-2.222", intent(Side.SELL, "4.444"), *permutation)
        for permutation in itertools.permutations(orders)
    }

    assert len(results) == 1


# --- exactness and the global decimal context --------------------------------------------


def test_low_precision_context_keeps_exact_values() -> None:
    def run() -> ExposureChange:
        return calc(
            "3.333",
            intent(Side.BUY, "5.555"),
            pending(Side.BUY, "7.777"),
            pending(Side.SELL, "7.777"),
        )

    baseline = run()
    with low_precision():
        low = run()

    assert low == baseline
    assert worst(low) == (D("16.665"), D("4.444"))  # not 17 / 16.7 / 4.5
    assert str(low.worst_long_qty) == "16.665"


def test_low_precision_decomposition_and_reduce_only() -> None:
    def run() -> tuple[ExposureChange, ExposureChange]:
        return (
            calc("3.333", intent(Side.SELL, "7.777")),
            calc("-7.777", intent(Side.BUY, "9.999", reduce_only=True), valid_reduce_only=True),
        )

    baseline = run()
    with low_precision():
        low = run()

    assert low == baseline
    assert split(low[0]) == (D("3.333"), D("4.444"))
    assert split(low[1]) == (D("7.777"), D("0"))


def test_calculation_does_not_modify_the_global_context() -> None:
    def snapshot() -> tuple[object, ...]:
        context = getcontext()
        return (
            context.prec,
            context.rounding,
            dict(context.traps),
            dict(context.flags),
            context.Emin,
            context.Emax,
        )

    getcontext().clear_flags()
    before = snapshot()
    calc("3.333", intent(Side.SELL, "7.777"), pending(Side.BUY, "5.555"))
    calc("-1", intent(Side.BUY, "2", reduce_only=True), valid_reduce_only=True)

    assert snapshot() == before


def test_unrepresentable_result_raises_calculation_error() -> None:
    # 10^200 + 10^-200 needs more digits than any exact context allows.
    with pytest.raises(ExposureCalculationError):
        calculate_exposure(
            position_qty=D("1E+200"),
            intent=intent(Side.BUY, "1E-200"),
            open_orders=(),
            valid_reduce_only=False,
        )


def test_calculation_error_is_not_a_validation_error() -> None:
    assert issubclass(ExposureCalculationError, ArithmeticError)
    assert not issubclass(ExposureCalculationError, DomainValidationError)


# --- input contract ----------------------------------------------------------------------


class SubDecimal(Decimal):
    pass


@pytest.mark.parametrize(
    "value", [None, 1, 1.5, True, "1", D("NaN"), D("Infinity"), SubDecimal("1")]
)
def test_position_must_be_a_known_exact_decimal(value: object) -> None:
    with pytest.raises(DomainValidationError, match="position_qty"):
        calculate_exposure(
            position_qty=value,  # type: ignore[arg-type]
            intent=intent(Side.BUY, "1"),
            open_orders=(),
            valid_reduce_only=False,
        )


@pytest.mark.parametrize("orders", [None, [pending(Side.BUY, "1")], (pending(Side.BUY, "1"), "x")])
def test_open_orders_must_be_a_known_tuple(orders: object) -> None:
    with pytest.raises(DomainValidationError, match="open_orders"):
        calculate_exposure(
            position_qty=D("0"),
            intent=intent(Side.BUY, "1"),
            open_orders=orders,  # type: ignore[arg-type]
            valid_reduce_only=False,
        )


def test_intent_must_be_a_place_order_intent() -> None:
    with pytest.raises(DomainValidationError, match="intent"):
        calculate_exposure(
            position_qty=D("0"),
            intent="buy 1",  # type: ignore[arg-type]
            open_orders=(),
            valid_reduce_only=False,
        )


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_valid_reduce_only_must_be_bool(value: object) -> None:
    with pytest.raises(DomainValidationError, match="valid_reduce_only"):
        calculate_exposure(
            position_qty=D("0"),
            intent=intent(Side.BUY, "1"),
            open_orders=(),
            valid_reduce_only=value,  # type: ignore[arg-type]
        )


# --- purity --------------------------------------------------------------------------------


def test_inputs_are_not_mutated_and_results_repeat() -> None:
    position = D("3.333")
    new = intent(Side.SELL, "7.777")
    orders = (pending(Side.BUY, "5.555"), pending(Side.SELL, "1.111", reduce_only=True))
    before: tuple[Any, ...] = (position, new, orders, [o.remaining_qty for o in orders])

    first = calculate_exposure(
        position_qty=position, intent=new, open_orders=orders, valid_reduce_only=False
    )
    second = calculate_exposure(
        position_qty=position, intent=new, open_orders=orders, valid_reduce_only=False
    )

    assert first == second
    assert (position, new, orders, [o.remaining_qty for o in orders]) == before


def test_exposure_module_dependencies() -> None:
    source = Path(exposure_module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith(("app.domain", "app.risk")), name
    for banned in ("float(", "getcontext", "setcontext", "localcontext", "RiskReason"):
        assert banned not in source, banned
    assert not re.search(r"(?<!copy_)\babs\(", source)  # built-in abs() rounds to the context
