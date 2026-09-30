"""Strategy output: what the strategy wants done, before risk approval.

Context-dependent checks (tick/step alignment, minimums, balance, position,
risk limits) are not part of these models; later layers perform them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TypeAlias

from app.domain.enums import OrderType, Side, TimeInForce
from app.domain.validation import (
    require_bool,
    require_enum,
    require_order_terms,
    require_text,
    require_utc,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaceOrderIntent:
    intent_id: str
    strategy_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: Decimal | None
    """Required for LIMIT, None for MARKET."""
    qty: Decimal
    time_in_force: TimeInForce
    reduce_only: bool
    tag: str | None
    """Free-form strategy label, e.g. "grid:L07:buy"."""
    created_at: datetime

    def __post_init__(self) -> None:
        require_text(self.intent_id, "intent_id")
        require_text(self.strategy_id, "strategy_id")
        require_text(self.symbol, "symbol")
        require_enum(self.side, Side, "side")
        require_order_terms(
            order_type=self.order_type,
            price=self.price,
            qty=self.qty,
            time_in_force=self.time_in_force,
        )
        require_bool(self.reduce_only, "reduce_only")
        if self.tag is not None:
            require_text(self.tag, "tag")
        require_utc(self.created_at, "created_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelOrderIntent:
    intent_id: str
    strategy_id: str
    symbol: str
    client_order_id: str
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_text(self.intent_id, "intent_id")
        require_text(self.strategy_id, "strategy_id")
        require_text(self.symbol, "symbol")
        require_text(self.client_order_id, "client_order_id")
        require_text(self.reason, "reason")
        require_utc(self.created_at, "created_at")


Intent: TypeAlias = PlaceOrderIntent | CancelOrderIntent
