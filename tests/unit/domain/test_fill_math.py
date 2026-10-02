"""Shared exact execution arithmetic (simulator and local account state)."""

from __future__ import annotations

import ast
from decimal import ROUND_UP, Decimal, DecimalException, getcontext, localcontext
from pathlib import Path

import pytest

from app.domain import fill_math
from app.domain.enums import Side
from app.domain.fill_math import (
    AVERAGE_PRICE_PRECISION,
    ExecutionTotals,
    accumulate_execution,
    next_position_qty,
    signed_quantity,
)

D = Decimal


@pytest.mark.parametrize(("side", "expected"), [(Side.BUY, "1.25"), (Side.SELL, "-1.25")])
def test_signed_quantity(side: Side, expected: str) -> None:
    assert signed_quantity(side, D("1.25")) == D(expected)


@pytest.mark.parametrize(
    ("position", "side", "qty", "expected"),
    [("0", Side.BUY, "2", "2"), ("3", Side.SELL, "5", "-2"), ("-3", Side.BUY, "3", "0")],
)
def test_next_position_qty(position: str, side: Side, qty: str, expected: str) -> None:
    assert next_position_qty(D(position), side=side, qty=D(qty)) == D(expected)


def test_first_execution_keeps_its_price() -> None:
    price = D("1." + "3" * 60)  # more digits than the published average precision

    totals = accumulate_execution(
        filled_qty=D("0"), filled_notional=D("0"), price=price, qty=D("2")
    )

    assert totals == ExecutionTotals(
        filled_qty=D("2"), filled_notional=D("2." + "6" * 60), avg_fill_price=price
    )


def test_average_comes_from_the_exact_notional() -> None:
    first = accumulate_execution(
        filled_qty=D("0"), filled_notional=D("0"), price=D("100"), qty=D("1")
    )
    second = accumulate_execution(
        filled_qty=first.filled_qty,
        filled_notional=first.filled_notional,
        price=D("101"),
        qty=D("1"),
    )
    third = accumulate_execution(
        filled_qty=second.filled_qty,
        filled_notional=second.filled_notional,
        price=D("103"),
        qty=D("1"),
    )

    assert (second.avg_fill_price, third.filled_notional) == (D("100.5"), D("304"))
    assert third.avg_fill_price == D("101.3333333333333333333333333333333333333")
    assert len(third.avg_fill_price.as_tuple().digits) == AVERAGE_PRICE_PRECISION


def test_results_do_not_depend_on_the_global_context() -> None:
    def run() -> tuple[object, ...]:
        totals = accumulate_execution(
            filled_qty=D("1.111111111"),
            filled_notional=D("111.4444444333"),
            price=D("99.7"),
            qty=D("0.222222222"),
        )
        return totals, next_position_qty(D("-0.067"), side=Side.SELL, qty=D("3.333"))

    expected = run()
    getcontext().clear_flags()
    context = getcontext()
    before = (context.prec, context.rounding, dict(context.flags))
    with localcontext() as low:
        low.prec = 2
        low.rounding = ROUND_UP
        result = run()

    assert result == expected
    assert (context.prec, context.rounding, dict(context.flags)) == before


def test_unrepresentable_values_raise_instead_of_rounding() -> None:
    with pytest.raises(DecimalException):
        next_position_qty(D("1E+100"), side=Side.BUY, qty=D("1E-100"))
    with pytest.raises(DecimalException):
        accumulate_execution(
            filled_qty=D("1E+100"), filled_notional=D("1"), price=D("1"), qty=D("1E-100")
        )


def test_module_is_stdlib_and_domain_only() -> None:
    source = Path(fill_math.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert all(not name.startswith("app.") or name.startswith("app.domain") for name in imports)
    for banned in ("getcontext", "localcontext", "float("):
        assert banned not in source, banned
