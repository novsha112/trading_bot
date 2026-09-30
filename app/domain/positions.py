"""Position snapshot (one-way position mode: one net position per symbol)."""

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
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    updated_at: datetime

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        require_decimal(self.qty, "qty")
        if self.qty == 0:
            if self.entry_price is not None:
                raise DomainValidationError(
                    f"entry_price must be None for a flat position, got {self.entry_price}"
                )
        else:
            require_positive(self.entry_price, "entry_price")
        if self.mark_price is not None:
            require_positive(self.mark_price, "mark_price")
        require_decimal(self.unrealized_pnl, "unrealized_pnl")
        require_decimal(self.realized_pnl, "realized_pnl")
        require_utc(self.updated_at, "updated_at")

    @property
    def side(self) -> PositionSide:
        if self.qty > 0:
            return PositionSide.LONG
        if self.qty < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT
