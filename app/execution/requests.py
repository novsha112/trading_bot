"""Pure mapping of a local order into an exchange-neutral placement request.

The ``Order`` reservation is the only source: every request field is copied, none
is generated or re-derived from the intent. No exchange-specific knowledge.
"""

from __future__ import annotations

from app.domain.errors import DomainValidationError
from app.domain.orders import Order
from app.exchanges.models import OrderRequest


def order_request_from_order(order: Order) -> OrderRequest:
    """The placement request for ``order`` (field by field, nothing invented)."""
    if type(order) is not Order:
        raise DomainValidationError("order must be a domain Order")
    return OrderRequest(
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        price=order.price,
        qty=order.qty,
        time_in_force=order.time_in_force,
        reduce_only=order.reduce_only,
    )
