"""Exchange-neutral DTOs used only at the exchange boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.domain.validation import require_text, require_utc


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderAck:
    """The exchange accepted a placement request and assigned its order id.

    Not an order status: acceptance says nothing about whether the order is open,
    filled or already canceled. The order stays SUBMITTING until an OrderUpdate
    (stream or REST) confirms its state. Recording ``exchange_order_id`` on the
    Order is a metadata update, not a state transition.
    """

    client_order_id: str
    exchange_order_id: str
    exchange_ts: datetime | None
    """Exchange time of the acknowledgement, if the exchange reports one."""

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        require_text(self.exchange_order_id, "exchange_order_id")
        if self.exchange_ts is not None:
            require_utc(self.exchange_ts, "exchange_ts")
