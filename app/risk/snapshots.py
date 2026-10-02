"""Pure mapping of local domain orders into a Risk V1 snapshot.

The boundary between the local order state and the evaluator: the caller (the
future placement coordinator) has already read its registry under the account
lock, chosen the ``snapshot_id`` and decided the effective ``TradingState`` and
the position. Nothing here reads an exchange or a registry, takes a lock, uses a
clock or derives a position from fills.

* ``orders``: the LOCAL UNIFIED active orders of ``symbol`` only (in flight,
  acknowledged and foreign ones alike); ``None`` = unknown, a tuple = a known view
  (``()`` = known none). A foreign symbol, a terminal order or the same
  ``client_order_id`` twice is a programmer error (``DomainValidationError``).
* ``account_open_order_count`` is passed through as given and never derived from
  ``len(orders)``: other symbols may have active orders too.
* The input order is preserved; the final validation is ``RiskSnapshot``'s.

Arithmetic is exact (``calculate_remaining_qty``); the global decimal context is
neither read nor modified. A remainder that cannot be computed exactly raises
``ExposureCalculationError``.
"""

from __future__ import annotations

from decimal import Decimal

from app.domain.errors import DomainValidationError
from app.domain.orders import Order
from app.risk.exposure import calculate_remaining_qty
from app.risk.models import (
    ACTIVE_ORDER_STATUSES,
    OpenOrderExposure,
    RiskSnapshot,
    TradingState,
)


def open_order_exposure(order: Order) -> OpenOrderExposure:
    """The exposure of one active local order: its whole unexecuted remainder.

    Only an exact ``Order`` with an active status (NEW, SUBMITTING, OPEN,
    PARTIALLY_FILLED, CANCELING, UNKNOWN) is accepted; a terminal order raises
    ``DomainValidationError``. The domain guarantees a remainder > 0 for every
    active order; ``OpenOrderExposure`` re-checks it defensively. Exchange metadata
    is not used.
    """
    if type(order) is not Order:
        raise DomainValidationError("order must be a domain Order")
    if order.status not in ACTIVE_ORDER_STATUSES:
        raise DomainValidationError(
            f"order {order.client_order_id}: status {order.status.value} "
            "is not an active order status"
        )
    return OpenOrderExposure(
        side=order.side,
        remaining_qty=calculate_remaining_qty(qty=order.qty, filled_qty=order.filled_qty),
        price=order.price,
        reduce_only=order.reduce_only,
        status=order.status,
    )


def _map_orders(symbol: str, orders: object) -> tuple[OpenOrderExposure, ...] | None:
    if orders is None:
        return None
    if type(orders) is not tuple:
        raise DomainValidationError("orders must be a tuple of domain Order or None")
    seen: set[str] = set()
    exposures: list[OpenOrderExposure] = []
    for order in orders:
        if type(order) is not Order:
            raise DomainValidationError("orders must contain only domain Order")
        if order.symbol != symbol:
            raise DomainValidationError(
                f"order {order.client_order_id} of {order.symbol} is not an order of {symbol}"
            )
        if order.client_order_id in seen:
            raise DomainValidationError(f"order {order.client_order_id} appears more than once")
        seen.add(order.client_order_id)
        exposures.append(open_order_exposure(order))
    return tuple(exposures)


def build_risk_snapshot(
    *,
    snapshot_id: str,
    symbol: str,
    trading_state: TradingState,
    position_qty: Decimal | None,
    orders: tuple[Order, ...] | None,
    account_open_order_count: int | None,
) -> RiskSnapshot:
    """A ``RiskSnapshot`` of ``symbol`` from already collected local state."""
    return RiskSnapshot(
        snapshot_id=snapshot_id,
        symbol=symbol,
        trading_state=trading_state,
        position_qty=position_qty,
        open_orders=_map_orders(symbol, orders),
        account_open_order_count=account_open_order_count,
    )
