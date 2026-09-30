"""Instrument specification: exchange trading constraints of one instrument.

Only exchange-defined characteristics belong here (tick size, quantity step,
limits). User risk policy (leverage, allocation, position limits) does not.
Values are validated but never rounded or normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.domain.errors import DomainValidationError
from app.domain.validation import require_non_negative, require_positive, require_text


@dataclass(frozen=True, slots=True, kw_only=True)
class InstrumentSpec:
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    """Price increment: every valid price is an exact multiple of it."""
    qty_step: Decimal
    """Quantity increment: every valid quantity is an exact multiple of it."""
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    """Minimum order value (price * qty) in the quote asset; 0 means no minimum."""

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        require_text(self.base_asset, "base_asset")
        require_text(self.quote_asset, "quote_asset")
        require_positive(self.tick_size, "tick_size")
        require_positive(self.qty_step, "qty_step")
        require_positive(self.min_qty, "min_qty")
        require_positive(self.max_qty, "max_qty")
        require_non_negative(self.min_notional, "min_notional")
        if self.max_qty < self.min_qty:
            raise DomainValidationError(
                f"max_qty ({self.max_qty}) must be >= min_qty ({self.min_qty})"
            )
