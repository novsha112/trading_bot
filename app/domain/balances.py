"""Balance snapshot of one asset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.domain.validation import require_decimal, require_text, require_utc


@dataclass(frozen=True, slots=True, kw_only=True)
class Balance:
    """Values are finite but not sign-constrained: the exact semantics of each field
    depend on the exchange account model and are confirmed in the adapter phase."""

    asset: str
    wallet_balance: Decimal
    equity: Decimal
    available: Decimal
    updated_at: datetime

    def __post_init__(self) -> None:
        require_text(self.asset, "asset")
        require_decimal(self.wallet_balance, "wallet_balance")
        require_decimal(self.equity, "equity")
        require_decimal(self.available, "available")
        require_utc(self.updated_at, "updated_at")
