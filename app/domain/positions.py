"""Position snapshot (one-way position mode: one net position per symbol).

``unrealized_pnl`` may be unknown (``None``) for an open position whose source has
no valuation (e.g. no mark price yet); ``None`` never means zero. A flat position
has a known unrealized PnL of exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.domain.enums import PositionSide
from app.domain.errors import DomainValidationError
from app.domain.validation import require_decimal, require_positive, require_text, require_utc


@dataclass(frozen=True, slots=True, kw_only=True)
class Position:
    symbol: str
    qty: Decimal
    """Signed: > 0 long, < 0 short, == 0 flat. The single source of truth for side."""
    entry_price: Decimal | None
    """Required for an open position, None for a flat one."""
    mark_price: Decimal | None
    unrealized_pnl: Decimal | None
    """None: not computed / not known (open positions only); never read as zero.
    Independent of ``mark_price``. A flat position requires exactly 0."""
    realized_pnl: Decimal
    """Gross realized trading PnL from execution-price differences, before fees,
    funding, interest and rebates (those are accounted separately)."""
    updated_at: datetime

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        require_decimal(self.qty, "qty")
        if self.unrealized_pnl is not None:
            require_decimal(self.unrealized_pnl, "unrealized_pnl")
        if self.qty == 0:
            if self.entry_price is not None:
                raise DomainValidationError(
                    f"entry_price must be None for a flat position, got {self.entry_price}"
                )
            if self.unrealized_pnl is None or self.unrealized_pnl != 0:
                raise DomainValidationError(
                    f"unrealized_pnl must be 0 for a flat position, got {self.unrealized_pnl}"
                )
        else:
            require_positive(self.entry_price, "entry_price")
        if self.mark_price is not None:
            require_positive(self.mark_price, "mark_price")
        require_decimal(self.realized_pnl, "realized_pnl")
        require_utc(self.updated_at, "updated_at")

    @property
    def side(self) -> PositionSide:
        if self.qty > 0:
            return PositionSide.LONG
        if self.qty < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT
