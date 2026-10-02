"""Signed position after a confirmed fill (quantity only)."""

from __future__ import annotations

import ast
import re
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path

import pytest

from app.domain.enums import Side
from app.domain.errors import DomainValidationError
from app.portfolio import positions as positions_module
from app.portfolio.positions import PositionStateError, position_after_fill

D = Decimal
BUY, SELL = Side.BUY, Side.SELL


@pytest.mark.parametrize(
    ("position", "side", "qty", "expected"),
    [
        ("0", BUY, "2", "2"),  # buy from flat
        ("0", SELL, "2", "-2"),  # sell from flat
        ("3", BUY, "2", "5"),  # add long
        ("3", SELL, "2", "1"),  # reduce long
        ("3", SELL, "3", "0"),  # exact close long
        ("3", SELL, "5", "-2"),  # ordinary reversal long -> short
        ("-3", SELL, "2", "-5"),  # add short
        ("-3", BUY, "2", "-1"),  # reduce short
        ("-3", BUY, "3", "0"),  # exact close short
        ("-3", BUY, "5", "2"),  # ordinary reversal short -> long
    ],
)
def test_ordinary_fill(position: str, side: Side, qty: str, expected: str) -> None:
    result = position_after_fill(D(position), side=side, qty=D(qty), reduce_only=False)

    assert result == D(expected)
    assert type(result) is Decimal


@pytest.mark.parametrize(
    ("position", "side", "qty", "expected"),
    [
        ("3", SELL, "1", "2"),
        ("3", SELL, "3", "0"),  # reduce-only may close exactly
        ("-3", BUY, "1.5", "-1.5"),
        ("-3", BUY, "3", "0"),
    ],
)
def test_valid_reduce_only_fill(position: str, side: Side, qty: str, expected: str) -> None:
    assert position_after_fill(D(position), side=side, qty=D(qty), reduce_only=True) == D(expected)


@pytest.mark.parametrize(
    ("position", "side", "qty", "match"),
    [
        ("3", SELL, "3.000001", "reverse"),  # oversized: long would turn short
        ("-3", BUY, "4", "reverse"),
        ("3", BUY, "1", "same side"),  # wrong side
        ("-3", SELL, "1", "same side"),
        ("0", SELL, "1", "without an open position"),
        ("0", BUY, "1", "without an open position"),
    ],
)
def test_impossible_reduce_only_fill_is_rejected(
    position: str, side: Side, qty: str, match: str
) -> None:
    with pytest.raises(PositionStateError, match=match):
        position_after_fill(D(position), side=side, qty=D(qty), reduce_only=True)


@pytest.mark.parametrize("reduce_only", [False, True])
def test_unknown_position_stays_unknown(reduce_only: bool) -> None:
    assert position_after_fill(None, side=SELL, qty=D("1"), reduce_only=reduce_only) is None


def test_unrepresentable_position_is_rejected() -> None:
    with pytest.raises(PositionStateError, match="exactly"):
        position_after_fill(D("1E+100"), side=BUY, qty=D("1E-100"), reduce_only=False)


@pytest.mark.parametrize(
    ("position", "side", "qty", "reduce_only", "match"),
    [
        (1, BUY, D("1"), False, "position_qty"),
        (D("NaN"), BUY, D("1"), False, "position_qty"),
        (D("0"), "buy", D("1"), False, "side"),
        (D("0"), BUY, D("0"), False, "qty"),
        (D("0"), BUY, 1, False, "qty"),
        (D("0"), BUY, D("1"), 1, "reduce_only"),
    ],
)
def test_invalid_inputs_are_rejected(
    position: object, side: object, qty: object, reduce_only: object, match: str
) -> None:
    with pytest.raises(DomainValidationError, match=match):
        position_after_fill(
            position,  # type: ignore[arg-type]
            side=side,  # type: ignore[arg-type]
            qty=qty,  # type: ignore[arg-type]
            reduce_only=reduce_only,  # type: ignore[arg-type]
        )


def test_exact_under_a_low_precision_context() -> None:
    getcontext().clear_flags()
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        long_side = position_after_fill(
            D("3.333333333"), side=SELL, qty=D("0.000000001"), reduce_only=True
        )
        short_side = position_after_fill(D("-0.067"), side=SELL, qty=D("3.333"), reduce_only=False)

    assert long_side == D("3.333333332")
    assert short_side == D("-3.400")


def test_global_context_is_unchanged() -> None:
    getcontext().clear_flags()
    context = getcontext()
    before = (context.prec, context.rounding, dict(context.traps), dict(context.flags))

    position_after_fill(D("3.333"), side=SELL, qty=D("7.777"), reduce_only=False)

    assert (context.prec, context.rounding, dict(context.traps), dict(context.flags)) == before


def test_module_is_pure() -> None:
    source = Path(positions_module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert all(not name.startswith("app.") or name.startswith("app.domain") for name in imports)
    for banned in ("getcontext", "localcontext", "asyncio", "float("):
        assert banned not in source, banned
    assert not re.search(r"(?<!copy_)\babs\(", source)
