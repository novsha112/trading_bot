"""Order and normalized exchange order report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.validation import (
    require_bool,
    require_enum,
    require_non_negative,
    require_order_terms,
    require_positive,
    require_text,
    require_utc,
)

# Statuses in which nothing can have been executed.
_ZERO_FILL_STATUSES: Final = frozenset(
    {
        OrderStatus.NEW,
        OrderStatus.SUBMITTING,
        OrderStatus.OPEN,
        OrderStatus.REJECTED,
        OrderStatus.FAILED,
    }
)
# Not fully executed: once the cumulative fill reaches qty the order is FILLED,
# including when a cancel request was pending (CANCELING -> FILLED).
_NOT_FULLY_FILLED_STATUSES: Final = frozenset(
    {OrderStatus.CANCELING, OrderStatus.CANCELED, OrderStatus.EXPIRED}
)


def _require_avg_fill_price(filled_qty: Decimal, avg_fill_price: object) -> None:
    if filled_qty == 0:
        if avg_fill_price is not None:
            raise DomainValidationError(
                f"avg_fill_price must be None without fills, got {avg_fill_price}"
            )
    else:
        require_positive(avg_fill_price, "avg_fill_price")


def _require_status_fill(status: OrderStatus, filled_qty: Decimal, qty: Decimal) -> None:
    rule: str | None = None
    if status in _ZERO_FILL_STATUSES and filled_qty != 0:
        rule = "must be 0"
    elif status is OrderStatus.PARTIALLY_FILLED and not 0 < filled_qty < qty:
        rule = "must be > 0 and < qty"
    elif status is OrderStatus.FILLED and filled_qty != qty:
        rule = "must be == qty"
    elif status in _NOT_FULLY_FILLED_STATUSES and filled_qty >= qty:
        rule = "must be < qty"
    # UNKNOWN allows any fill from 0 through qty: its purpose is uncertainty.
    if rule is not None:
        raise DomainValidationError(
            f"filled_qty ({filled_qty}) is not allowed for status {status.value}: {rule}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Order:
    """Local view of an order. Immutable: state changes go through
    ``app.domain.order_state.transition``, which returns a new instance."""

    client_order_id: str
    exchange_order_id: str | None
    strategy_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: Decimal | None
    qty: Decimal
    time_in_force: TimeInForce
    reduce_only: bool
    status: OrderStatus
    filled_qty: Decimal
    """Cumulative executed quantity."""
    avg_fill_price: Decimal | None
    """Cumulative average execution price; None without fills."""
    created_at: datetime
    updated_at: datetime
    last_exchange_update_ts: datetime | None
    version: int
    """Incremented by every transition (optimistic concurrency for persistence)."""

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        if self.exchange_order_id is not None:
            require_text(self.exchange_order_id, "exchange_order_id")
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
        status = require_enum(self.status, OrderStatus, "status")
        filled_qty = require_non_negative(self.filled_qty, "filled_qty")
        if filled_qty > self.qty:
            raise DomainValidationError(f"filled_qty ({filled_qty}) must be <= qty ({self.qty})")
        _require_avg_fill_price(filled_qty, self.avg_fill_price)
        _require_status_fill(status, filled_qty, self.qty)
        require_utc(self.created_at, "created_at")
        require_utc(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise DomainValidationError(
                f"updated_at ({self.updated_at.isoformat()}) must be >= "
                f"created_at ({self.created_at.isoformat()})"
            )
        if self.last_exchange_update_ts is not None:
            require_utc(self.last_exchange_update_ts, "last_exchange_update_ts")
        if type(self.version) is not int or self.version < 0:
            raise DomainValidationError(
                f"version must be an int >= 0, got {self.version!r} ({type(self.version).__name__})"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdate:
    """Normalized exchange report about one order (not exchange-specific).

    It does not know the order quantity, so consistency with a concrete Order
    (cum_filled_qty <= qty, status vs quantity) is checked when it is applied.
    """

    client_order_id: str
    exchange_order_id: str | None
    status: OrderStatus
    cum_filled_qty: Decimal
    avg_fill_price: Decimal | None
    reject_reason: str | None
    exchange_ts: datetime

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        if self.exchange_order_id is not None:
            require_text(self.exchange_order_id, "exchange_order_id")
        require_enum(self.status, OrderStatus, "status")
        cum_filled_qty = require_non_negative(self.cum_filled_qty, "cum_filled_qty")
        _require_avg_fill_price(cum_filled_qty, self.avg_fill_price)
        if self.reject_reason is not None:
            require_text(self.reject_reason, "reject_reason")
        require_utc(self.exchange_ts, "exchange_ts")
