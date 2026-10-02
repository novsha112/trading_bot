"""Exposure of a new intent for Risk V1: decomposition and worst-case pending size.

Pure arithmetic on known inputs only. Unknown position / open orders, reduce-only
validity and every limit or rejection reason are decided by the evaluator
(docs/ARCHITECTURE.md 9.0) before calling ``calculate_exposure``.

Decomposition (relative to the current position ``q`` only; open orders do not
change it), with ``d`` = +qty for BUY, -qty for SELL:
* same direction or flat: reducing 0, increasing ``|d|``;
* opposite: reducing ``min(|q|, |d|)``, increasing ``|d| - reducing`` (a
  reversal has both parts);
* valid reduce-only (confirmed by the caller): reducing ``min(|q|, |d|)``,
  increasing 0 — execution caps the fill, so an oversized order reduces at most
  the position.

Worst case (positive magnitudes; opposite pending orders never net):
* ``worst_long_qty = max(q + all non-reduce-only BUY remaining (+ the intent if it
  is a BUY and not a valid reduce-only), 0)``, assuming no SELL fills;
* ``worst_short_qty = max(-(q - all non-reduce-only SELL remaining (- the intent if
  it is a SELL and not a valid reduce-only)), 0)``, assuming no BUY fills.
Every active status counts in full, UNKNOWN included; reduce-only orders never
increase either side. A non-reduce-only intent enters its side with its whole
qty: LONG 5 + SELL 8 may end SHORT 3.

Exactness: sums run in an explicit exact context (``Inexact`` trapped), signs and
magnitudes use ``copy_negate`` / ``copy_abs``; the global decimal context is
neither read nor modified. A result needing more digits than the exact context
raises ``ExposureCalculationError`` instead of being rounded.
"""

from __future__ import annotations

from decimal import (
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
)
from typing import Final

from app.domain.enums import Side
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.validation import require_bool
from app.risk.models import ExposureChange, OpenOrderExposure

_EXACT_PRECISION: Final = 100
_ZERO: Final = Decimal(0)


class ExposureCalculationError(ArithmeticError):
    """The exposure cannot be computed exactly (more digits than the exact
    context); the evaluator rejects such an intent, it is never rounded."""


def _exact() -> Context:
    return Context(
        prec=_EXACT_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
    )


def _require_inputs(
    position_qty: object, intent: object, open_orders: object, valid_reduce_only: object
) -> None:
    if type(position_qty) is not Decimal or not position_qty.is_finite():
        raise DomainValidationError("position_qty must be a known, finite, exact Decimal")
    if not isinstance(intent, PlaceOrderIntent):
        raise DomainValidationError("intent must be a PlaceOrderIntent")
    if type(open_orders) is not tuple or not all(
        isinstance(order, OpenOrderExposure) for order in open_orders
    ):
        raise DomainValidationError("open_orders must be a known tuple of OpenOrderExposure")
    if require_bool(valid_reduce_only, "valid_reduce_only") and not intent.reduce_only:
        raise DomainValidationError("valid_reduce_only requires a reduce_only intent")


def calculate_exposure(
    *,
    position_qty: Decimal,
    intent: PlaceOrderIntent,
    open_orders: tuple[OpenOrderExposure, ...],
    valid_reduce_only: bool,
) -> ExposureChange:
    """Decompose ``intent`` against the known signed ``position_qty`` and compute
    the worst-case long / short sizes with the known active ``open_orders``."""
    _require_inputs(position_qty, intent, open_orders, valid_reduce_only)
    context = _exact()
    try:
        order_qty = intent.qty
        position_size = position_qty.copy_abs()
        opposite = (position_qty > 0 and intent.side is Side.SELL) or (
            position_qty < 0 and intent.side is Side.BUY
        )
        reducing = min(position_size, order_qty) if opposite else _ZERO
        increasing = _ZERO if valid_reduce_only else context.subtract(order_qty, reducing)

        long_path = position_qty
        short_path = position_qty
        for order in open_orders:
            if order.reduce_only:
                continue  # can never increase either side
            if order.side is Side.BUY:
                long_path = context.add(long_path, order.remaining_qty)
            else:
                short_path = context.subtract(short_path, order.remaining_qty)
        if not valid_reduce_only:
            if intent.side is Side.BUY:
                long_path = context.add(long_path, order_qty)
            else:
                short_path = context.subtract(short_path, order_qty)
    except DecimalException:
        raise ExposureCalculationError("exposure cannot be computed exactly") from None

    return ExposureChange(
        reducing_qty=reducing,
        increasing_qty=increasing,
        worst_long_qty=max(long_path, _ZERO),
        worst_short_qty=max(short_path.copy_negate(), _ZERO),
    )
