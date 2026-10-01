"""Order state machine (docs/ARCHITECTURE.md, section 8).

The transition table is explicit. Two self-transitions exist, and only for new
execution data: PARTIALLY_FILLED -> PARTIALLY_FILLED and CANCELING -> CANCELING,
both requiring a strictly larger cumulative filled_qty. A partial fill that
arrives while a cancel is pending keeps the order in CANCELING; a fill that
completes the order moves it to FILLED (CANCELING requires filled_qty < qty).

An allowed arc does not make every data combination valid: the resulting Order
still has to satisfy its status/fill invariants (e.g. FAILED requires no fill).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Final

from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError, InvalidOrderTransition
from app.domain.orders import Order
from app.domain.validation import require_enum, require_non_negative, require_text, require_utc

_S = OrderStatus

ALLOWED_TRANSITIONS: Final[Mapping[OrderStatus, frozenset[OrderStatus]]] = MappingProxyType(
    {
        _S.NEW: frozenset({_S.SUBMITTING, _S.FAILED}),
        _S.SUBMITTING: frozenset(
            {
                _S.OPEN,
                _S.PARTIALLY_FILLED,
                _S.FILLED,
                _S.CANCELED,
                _S.REJECTED,
                _S.FAILED,
                _S.UNKNOWN,
            }
        ),
        _S.OPEN: frozenset(
            {
                _S.PARTIALLY_FILLED,
                _S.FILLED,
                _S.CANCELING,
                _S.CANCELED,
                _S.EXPIRED,
                _S.UNKNOWN,
            }
        ),
        _S.PARTIALLY_FILLED: frozenset(
            {
                _S.PARTIALLY_FILLED,
                _S.FILLED,
                _S.CANCELING,
                _S.CANCELED,
                _S.EXPIRED,
                _S.UNKNOWN,
            }
        ),
        _S.CANCELING: frozenset(
            {
                _S.CANCELING,
                _S.FILLED,
                _S.CANCELED,
                _S.EXPIRED,
                _S.UNKNOWN,
            }
        ),
        # Resolution of an uncertain outcome (resolver: Phase 5).
        _S.UNKNOWN: frozenset(
            {
                _S.OPEN,
                _S.PARTIALLY_FILLED,
                _S.FILLED,
                _S.CANCELED,
                _S.REJECTED,
                _S.EXPIRED,
                _S.FAILED,
            }
        ),
        _S.FILLED: frozenset(),
        _S.CANCELED: frozenset(),
        _S.REJECTED: frozenset(),
        _S.EXPIRED: frozenset(),
        _S.FAILED: frozenset(),
    }
)

TERMINAL_STATUSES: Final = frozenset({_S.FILLED, _S.CANCELED, _S.REJECTED, _S.EXPIRED, _S.FAILED})

# Self-transitions that carry new execution data and must increase filled_qty.
_FILL_PROGRESS_SELF_TRANSITIONS: Final = frozenset({_S.PARTIALLY_FILLED, _S.CANCELING})


def transition(
    order: Order,
    new_status: OrderStatus,
    *,
    at: datetime,
    filled_qty: Decimal | None = None,
    avg_fill_price: Decimal | None = None,
    exchange_order_id: str | None = None,
    last_exchange_update_ts: datetime | None = None,
) -> Order:
    """Return a new Order in ``new_status``; the given order is not modified.

    Args:
        order: current order.
        new_status: target status; must be allowed by ALLOWED_TRANSITIONS.
        at: time of the change (UTC), not earlier than ``order.updated_at``.
        filled_qty: new cumulative filled quantity; None keeps the current one.
            It never decreases. Must be given together with ``avg_fill_price``
            when positive.
        avg_fill_price: new cumulative average fill price (not computed here).
        exchange_order_id: set when first known; None keeps the current one.
            An already known id cannot change.
        last_exchange_update_ts: exchange time of the report behind this change;
            None keeps the current one.

    Raises:
        InvalidOrderTransition: arc not allowed, time moving backwards, filled_qty
            decreasing or not increasing in a fill-progress self-transition,
            exchange_order_id changing.
        DomainValidationError: invalid argument or the resulting order violates
            an Order invariant.
    """
    new_status = require_enum(new_status, OrderStatus, "new_status")
    source = order.status
    if new_status not in ALLOWED_TRANSITIONS[source]:
        raise InvalidOrderTransition(
            f"{source.value} -> {new_status.value} is not allowed "
            f"(client_order_id={order.client_order_id})"
        )

    at = require_utc(at, "at")
    if at < order.updated_at:
        raise InvalidOrderTransition(
            f"time cannot move backwards: at={at.isoformat()} < "
            f"updated_at={order.updated_at.isoformat()}"
        )

    if filled_qty is None:
        if avg_fill_price is not None:
            raise DomainValidationError("avg_fill_price requires filled_qty")
        new_filled_qty = order.filled_qty
        new_avg_fill_price = order.avg_fill_price
    else:
        new_filled_qty = require_non_negative(filled_qty, "filled_qty")
        new_avg_fill_price = avg_fill_price
        if new_filled_qty < order.filled_qty:
            raise InvalidOrderTransition(
                f"filled_qty cannot decrease: {order.filled_qty} -> {new_filled_qty}"
            )

    is_fill_progress = source is new_status and source in _FILL_PROGRESS_SELF_TRANSITIONS
    if is_fill_progress and new_filled_qty <= order.filled_qty:
        raise InvalidOrderTransition(
            f"{source.value} -> {new_status.value} must increase filled_qty "
            f"(current {order.filled_qty}, given {new_filled_qty})"
        )

    new_exchange_order_id = order.exchange_order_id
    if exchange_order_id is not None:
        require_text(exchange_order_id, "exchange_order_id")
        if order.exchange_order_id not in (None, exchange_order_id):
            raise InvalidOrderTransition(
                f"exchange_order_id cannot change: {order.exchange_order_id} -> {exchange_order_id}"
            )
        new_exchange_order_id = exchange_order_id

    new_last_exchange_update_ts = order.last_exchange_update_ts
    if last_exchange_update_ts is not None:
        new_last_exchange_update_ts = require_utc(
            last_exchange_update_ts, "last_exchange_update_ts"
        )

    # dataclasses.replace re-runs Order.__post_init__: all invariants are checked.
    return dataclasses.replace(
        order,
        status=new_status,
        filled_qty=new_filled_qty,
        avg_fill_price=new_avg_fill_price,
        exchange_order_id=new_exchange_order_id,
        last_exchange_update_ts=new_last_exchange_update_ts,
        updated_at=at,
        version=order.version + 1,
    )
