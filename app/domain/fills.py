"""Fill: a confirmed execution reported by the exchange.

The execution itself (ids, side, price, qty, time) is always known. Fee data and
the liquidity role may be unknown (``None``) when the source does not provide them
(e.g. a simulator without a fee or order-book model); ``None`` never means zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.domain.enums import Side
from app.domain.errors import DomainValidationError
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
    fee: Decimal | None
    """Any sign: a negative fee is a rebate; zero is a known zero fee.
    None: the fee is unknown (never interpreted as zero)."""
    fee_asset: str | None
    """None exactly when ``fee`` is None."""
    is_maker: bool | None
    """None: the liquidity role is unknown. Independent of the fee data."""
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
        if self.fee is not None:
            require_decimal(self.fee, "fee")
        if self.fee_asset is not None:
            require_text(self.fee_asset, "fee_asset")
        if (self.fee is None) != (self.fee_asset is None):
            raise DomainValidationError(
                "fee and fee_asset must be both known or both None, "
                f"got fee={self.fee}, fee_asset={self.fee_asset!r}"
            )
        if self.is_maker is not None:
            require_bool(self.is_maker, "is_maker")
        require_utc(self.exchange_ts, "exchange_ts")
