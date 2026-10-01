"""Exchange-neutral DTOs used only at the exchange boundary.

They carry exactly what a transport request needs. Lifecycle state (status,
fills, version, timestamps) stays in the domain ``Order``, owned by the
execution layer, and never reaches an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.domain.enums import OrderType, Side, TimeInForce
from app.domain.validation import (
    require_bool,
    require_enum,
    require_order_terms,
    require_text,
    require_utc,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderRequest:
    """Order placement request, built by the execution layer from a persisted Order.

    ``client_order_id`` exists before any network call: it is the idempotency key
    the exchange stores with the order, used to resolve ambiguous outcomes.
    Adapters never generate their own.
    """

    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: Decimal | None
    """Required for LIMIT, None for MARKET."""
    qty: Decimal
    time_in_force: TimeInForce
    reduce_only: bool

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        require_text(self.symbol, "symbol")
        require_enum(self.side, Side, "side")
        require_order_terms(
            order_type=self.order_type,
            price=self.price,
            qty=self.qty,
            time_in_force=self.time_in_force,
        )
        require_bool(self.reduce_only, "reduce_only")


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderRef:
    """Identifies an existing (or possibly existing) order on the exchange.

    ``client_order_id`` is always known: it is our stable identifier for
    idempotency and reconciliation. ``exchange_order_id`` is known only after an
    acknowledgement; after an ambiguous placement it is not, and lookups must
    still work by ``client_order_id``.
    """

    symbol: str
    client_order_id: str
    exchange_order_id: str | None = None

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        require_text(self.client_order_id, "client_order_id")
        if self.exchange_order_id is not None:
            require_text(self.exchange_order_id, "exchange_order_id")


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
