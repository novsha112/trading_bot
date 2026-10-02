"""Exact execution arithmetic shared by every fill consumer.

One rule set for the simulated exchange and the local account state, so the two
never diverge:

* ``signed_quantity``: BUY ``+qty``, SELL ``-qty`` (``copy_negate``, exact and
  context-free; unary minus would round to the global decimal context);
* ``next_position_qty``: signed position after one execution;
* ``accumulate_execution``: cumulative filled quantity, exact notional
  (``sum(price * qty)``) and the published average price. The first execution's
  average is its price as is; later ones are ``notional / filled`` rounded to
  ``AVERAGE_PRICE_PRECISION`` significant digits, ROUND_HALF_EVEN. The exact
  notional is the source of every average, so one rounding never carries into
  the next.

Sums and products run in an explicit exact context (``Inexact`` trapped); a value
needing more than ``EXACT_FILL_PRECISION`` digits raises the ``decimal``
exception (a ``DecimalException``) instead of being rounded. The global decimal
context is neither read nor modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
)
from typing import Final

from app.domain.enums import Side

EXACT_FILL_PRECISION: Final = 80
AVERAGE_PRICE_PRECISION: Final = 40


def _exact() -> Context:
    return Context(
        prec=EXACT_FILL_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
    )


def _average() -> Context:
    return Context(
        prec=AVERAGE_PRICE_PRECISION,
        rounding=ROUND_HALF_EVEN,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionTotals:
    """Cumulative execution state of one order after an execution."""

    filled_qty: Decimal
    filled_notional: Decimal
    """Exact ``sum(price * qty)`` of all executions."""
    avg_fill_price: Decimal


def signed_quantity(side: Side, qty: Decimal) -> Decimal:
    """``+qty`` for BUY, ``-qty`` for SELL (exact)."""
    return qty if side is Side.BUY else qty.copy_negate()


def next_position_qty(position_qty: Decimal, *, side: Side, qty: Decimal) -> Decimal:
    """Signed net position after executing ``qty`` on ``side`` (exact)."""
    return _exact().add(position_qty, signed_quantity(side, qty))


def accumulate_execution(
    *, filled_qty: Decimal, filled_notional: Decimal, price: Decimal, qty: Decimal
) -> ExecutionTotals:
    """Totals after one more execution of ``qty`` at ``price``."""
    exact = _exact()
    filled = exact.add(filled_qty, qty)
    notional = exact.add(filled_notional, exact.multiply(price, qty))
    average = price if filled_qty == 0 else _average().divide(notional, filled)
    return ExecutionTotals(filled_qty=filled, filled_notional=notional, avg_fill_price=average)
