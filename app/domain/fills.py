"""Fill: a confirmed execution reported by the exchange."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.domain.enums import Side
from app.domain.validation import (
    require_bool,
    require_decimal,
    require_enum,
    require_positive,
    require_text,
    require_utc,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill:
    exec_id: str
    """Unique execution id; the deduplication key for fills."""
    exchange_order_id: str
    client_order_id: str | None
    """None for orders not placed by the bot (e.g. created manually)."""
    symbol: str
    side: Side
    price: Decimal
    qty: Decimal
    fee: Decimal
    """Any sign: a negative fee is a rebate."""
    fee_asset: str
    is_maker: bool
    exchange_ts: datetime

    def __post_init__(self) -> None:
        require_text(self.exec_id, "exec_id")
        require_text(self.exchange_order_id, "exchange_order_id")
        if self.client_order_id is not None:
            require_text(self.client_order_id, "client_order_id")
        require_text(self.symbol, "symbol")
        require_enum(self.side, Side, "side")
        require_positive(self.price, "price")
        require_positive(self.qty, "qty")
        require_decimal(self.fee, "fee")
        require_text(self.fee_asset, "fee_asset")
        require_bool(self.is_maker, "is_maker")
        require_utc(self.exchange_ts, "exchange_ts")
