"""Signed net position after a confirmed fill (one-way mode, quantity only).

The local account state needs only the signed quantity per symbol for Risk:
``None`` = unknown / not reconciled, ``0`` = known flat, ``> 0`` long, ``< 0``
short. No entry price, PnL, fees or marks: those are not known honestly here.

``position_after_fill`` applies the shared exact rule (``app.domain.fill_math``)
to an actual execution; it does not re-decide Risk. An unknown position stays
unknown (a delta on an unknown base is still unknown). With a known position, a
reduce-only fill is checked as defense in depth: it must be on the opposite side
of an open position and may reduce it to zero but never reverse it; anything
else is a state corruption / exchange mismatch and raises
``PositionStateError`` (nothing is computed, the caller changes nothing). A
result needing more digits than the exact context also raises it.
"""

from __future__ import annotations

from decimal import Decimal, DecimalException

from app.domain.enums import Side
from app.domain.errors import DomainValidationError
from app.domain.fill_math import next_position_qty
from app.domain.validation import require_bool, require_enum, require_positive


class PositionStateError(RuntimeError):
    """A fill cannot be applied to the known position (corrupted or mismatched
    state, or an unrepresentable result)."""


def position_after_fill(
    position_qty: Decimal | None, *, side: Side, qty: Decimal, reduce_only: bool
) -> Decimal | None:
    """Signed position after executing ``qty`` on ``side``; None stays None."""
    require_enum(side, Side, "side")
    require_positive(qty, "qty")
    require_bool(reduce_only, "reduce_only")
    if position_qty is None:
        return None
    if type(position_qty) is not Decimal or not position_qty.is_finite():
        raise DomainValidationError("position_qty must be a finite, exact Decimal or None")
    if reduce_only:
        if position_qty == 0:
            raise PositionStateError(f"reduce-only {side.value} fill without an open position")
        if (position_qty > 0) == (side is Side.BUY):
            raise PositionStateError(
                f"reduce-only {side.value} fill on the same side as position {position_qty}"
            )
    try:
        result = next_position_qty(position_qty, side=side, qty=qty)
    except DecimalException:
        raise PositionStateError("position quantity cannot be computed exactly") from None
    if reduce_only and result != 0 and (result > 0) != (position_qty > 0):
        raise PositionStateError(
            f"reduce-only {side.value} fill of {qty} would reverse position {position_qty}"
        )
    return result
