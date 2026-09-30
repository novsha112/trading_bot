"""Rounding to tick size / quantity step and exchange minimum checks.

Rounding works on multiples of the step, never on a number of decimal places:
with ``tick_size = 0.25`` the price 100.37 becomes 100.25 (DOWN) or 100.50 (UP).

    remainder = value % step            # exact, 0 <= remainder < step for value > 0
    DOWN      = value - remainder
    UP        = value - remainder + step  (value itself if remainder == 0)

All arithmetic runs in a private decimal context that traps inexact results, so
the answer is either exact or a ``DomainValidationError``. The global decimal
context is neither read nor modified.
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

from app.domain.enums import RoundingDirection
from app.domain.errors import DomainValidationError
from app.domain.instrument import InstrumentSpec
from app.domain.validation import require_positive

# Enough for prices/quantities of any real instrument; anything that would need
# more digits fails loudly instead of being rounded silently.
_EXACT_PRECISION = 40


def _exact_context() -> Context:
    return Context(
        prec=_EXACT_PRECISION,
        traps=[InvalidOperation, DivisionByZero, Overflow, Inexact],
    )


def _remainder(value: Decimal, step: Decimal, field: str) -> Decimal:
    try:
        return _exact_context().remainder(value, step)
    except DecimalException:
        raise DomainValidationError(
            f"{field} {value} with step {step} cannot be computed exactly"
        ) from None


def _round_to_step(value: object, step: Decimal, direction: object, field: str) -> Decimal:
    value = require_positive(value, field)
    if not isinstance(direction, RoundingDirection):
        raise DomainValidationError(
            f"direction must be a RoundingDirection, got {type(direction).__name__}"
        )
    remainder = _remainder(value, step, field)
    if remainder == 0:
        return value
    context = _exact_context()
    try:
        floor = context.subtract(value, remainder)
        result = floor if direction is RoundingDirection.DOWN else context.add(floor, step)
    except DecimalException:
        raise DomainValidationError(
            f"{field} {value} with step {step} cannot be computed exactly"
        ) from None
    if result == 0:
        # A zero price or quantity is never valid; the caller must decide what to do.
        raise DomainValidationError(f"{field} {value} rounds down to zero with step {step}")
    return result


def round_price(price: Decimal, spec: InstrumentSpec, direction: RoundingDirection) -> Decimal:
    """Round a positive price to a multiple of ``spec.tick_size`` in the given direction."""
    return _round_to_step(price, spec.tick_size, direction, "price")


def round_qty(qty: Decimal, spec: InstrumentSpec, direction: RoundingDirection) -> Decimal:
    """Round a positive quantity to a multiple of ``spec.qty_step`` in the given direction."""
    return _round_to_step(qty, spec.qty_step, direction, "qty")


def is_price_aligned(price: Decimal, spec: InstrumentSpec) -> bool:
    """True if the price is an exact multiple of ``spec.tick_size``."""
    price = require_positive(price, "price")
    return _remainder(price, spec.tick_size, "price") == 0


def is_qty_aligned(qty: Decimal, spec: InstrumentSpec) -> bool:
    """True if the quantity is an exact multiple of ``spec.qty_step``."""
    qty = require_positive(qty, "qty")
    return _remainder(qty, spec.qty_step, "qty") == 0


def meets_min_qty(qty: Decimal, spec: InstrumentSpec) -> bool:
    """True if ``qty >= spec.min_qty`` (the boundary itself passes)."""
    return require_positive(qty, "qty") >= spec.min_qty


def meets_min_notional(price: Decimal, qty: Decimal, spec: InstrumentSpec) -> bool:
    """True if the exact ``price * qty >= spec.min_notional`` (the boundary itself passes).

    Only the order value is checked: fees, leverage and margin are not part of it.
    """
    price = require_positive(price, "price")
    qty = require_positive(qty, "qty")
    try:
        notional = _exact_context().multiply(price, qty)
    except DecimalException:
        raise DomainValidationError(
            f"notional of price {price} and qty {qty} cannot be computed exactly"
        ) from None
    return notional >= spec.min_notional
