"""Grid geometry: deterministic price levels between two bounds.

``levels`` is the total number of price levels including both bounds:
lower=100, upper=200, levels=3 gives (100, 150, 200).

* Arithmetic: ``lower + (upper - lower) * i / (levels - 1)``.
* Geometric: ``lower * exp(ln(upper / lower) * i / (levels - 1))``.

Every inner level is computed directly from the bounds (no accumulated step),
with Decimal only: ``Decimal.ln`` / ``Decimal.exp`` are correctly rounded, so no
float is involved anywhere. Inner levels are then rounded to a fixed number of
significant digits, which makes perfect ratios exact (100 -> 400: 200, not
199.999...). The first and last level are the input values themselves.

Levels are mathematical target prices: they are NOT rounded to the instrument
tick size here (that needs exchange metadata and is a separate step).

Precision is an internal detail: computations run in a private decimal context
and never read or modify the caller's context.
"""

from __future__ import annotations

import itertools
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Final

from app.domain.enums import GridSpacing
from app.domain.errors import DomainValidationError
from app.domain.validation import require_enum, require_positive

# Internal working precision; the guard digits absorb ln/exp and division error.
_WORKING_PRECISION: Final = 60
# Significant digits of inner levels (far below any exchange tick size).
_OUTPUT_PRECISION: Final = 40


def _working_context() -> Context:
    return Context(
        prec=_WORKING_PRECISION,
        rounding=ROUND_HALF_EVEN,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def _to_output(value: Decimal) -> Decimal:
    """Round to the output precision and drop trailing zeros (value unchanged).

    Every operation uses the explicit output context: Decimal methods without a
    context argument would silently use the caller's (global) precision.
    """
    context = Context(prec=_OUTPUT_PRECISION, rounding=ROUND_HALF_EVEN)
    normalized = context.normalize(context.plus(value))
    # normalize() turns 200 into 2E+2; keep integers in plain notation.
    if normalized.as_tuple().exponent > 0:  # type: ignore[operator]
        return context.quantize(normalized, Decimal(1))
    return normalized


def generate_grid_levels(
    *,
    lower_price: Decimal,
    upper_price: Decimal,
    levels: int,
    spacing: GridSpacing,
) -> tuple[Decimal, ...]:
    """Return ``levels`` strictly increasing prices from ``lower_price`` to ``upper_price``.

    Pure and deterministic: no I/O, config, environment or clock.

    Raises:
        DomainValidationError: invalid input, or the levels cannot be represented as a
            strictly increasing sequence at the output precision (range too narrow for
            the number of levels).
    """
    lower = require_positive(lower_price, "lower_price")
    upper = require_positive(upper_price, "upper_price")
    if upper <= lower:
        raise DomainValidationError(f"upper_price ({upper}) must be > lower_price ({lower})")
    if type(levels) is not int or levels < 2:
        raise DomainValidationError(
            f"levels must be an int >= 2, got {type(levels).__name__}"
            + (f" {levels}" if type(levels) is int else "")
        )
    spacing = require_enum(spacing, GridSpacing, "spacing")

    intervals = levels - 1
    with localcontext(_working_context()):
        if spacing is GridSpacing.ARITHMETIC:
            width = upper - lower
            inner = [lower + width * i / intervals for i in range(1, intervals)]
        else:
            log_ratio = (upper / lower).ln()
            inner = [lower * (log_ratio * i / intervals).exp() for i in range(1, intervals)]

    result = (lower, *(_to_output(value) for value in inner), upper)

    for index, (left, right) in enumerate(itertools.pairwise(result)):
        if not left < right:
            raise DomainValidationError(
                f"grid levels are not strictly increasing at index {index + 1}: "
                f"range too narrow for {levels} levels at the supported precision"
            )
    return result
