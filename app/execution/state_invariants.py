"""Reference invariants of a durable account state, owned by the execution layer.

Pure checks of one complete account (placements, orders, fills, exact notionals)
shared by every reader and writer of durable account state: a store adapter
validates the state a commit would produce, startup hydration validates the
snapshot it loaded (docs/ARCHITECTURE.md 11.0 / 12). Defining them here, next to
the persistence port, keeps them out of any concrete adapter.

* an approved placement's ``client_order_id`` belongs to one intent only, its
  order exists and carries the intent's terms (identical payloads);
* an ``exchange_order_id`` belongs to one order only;
* every order has exactly one exact notional (>= 0), zero exactly when nothing
  is filled; every notional belongs to an order;
* every fill has a ``client_order_id`` of an existing order with the same symbol
  and side and a compatible exchange order id;
* a safety block belongs to an existing FAILED order without an exchange id
  (it was never sent).

Errors are the port's: ``StoreConflictError`` for an identity owned twice,
``StoreValidationError`` for a dangling or inconsistent reference. Nothing is
computed, replayed or repaired.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Final

from app.domain.enums import OrderStatus
from app.domain.fills import Fill
from app.domain.orders import Order
from app.execution.models import PlacementRecord, SafetyBlockRecord
from app.execution.persistence import StoreConflictError, StoreValidationError

_INTENT_TERMS: Final = (
    "strategy_id",
    "symbol",
    "side",
    "order_type",
    "price",
    "qty",
    "time_in_force",
    "reduce_only",
)


def identical(left: object, right: object) -> bool:
    """Equal AND equally represented (exact Decimal digits / exponent / sign):
    ``Decimal("4")`` and ``Decimal("4.0")`` are different payloads."""
    return left == right and repr(left) == repr(right)


def check_account_references(
    *,
    placements: Mapping[str, PlacementRecord],
    orders: Mapping[str, Order],
    fills: Mapping[str, Fill],
    notionals: Mapping[str, Decimal],
    safety_blocks: Mapping[str, SafetyBlockRecord],
) -> None:
    """Validate the references of one complete account state (see module doc)."""
    placed: dict[str, str] = {}
    for record in placements.values():
        cid = record.client_order_id
        if cid is None:
            continue
        if cid in placed:
            raise StoreConflictError(
                f"client_order_id {cid} belongs to intents {placed[cid]} and {record.intent_id}"
            )
        placed[cid] = record.intent_id
        order = orders.get(cid)
        if order is None:
            raise StoreValidationError(f"approved placement {record.intent_id}: no order {cid}")
        for term in _INTENT_TERMS:
            if not identical(getattr(order, term), getattr(record.intent, term)):
                raise StoreValidationError(
                    f"order {cid} {term} does not match its placement {record.intent_id}"
                )
    exchange_ids: dict[str, str] = {}
    for order in orders.values():
        if order.exchange_order_id is not None:
            other = exchange_ids.setdefault(order.exchange_order_id, order.client_order_id)
            if other != order.client_order_id:
                raise StoreConflictError(
                    f"exchange_order_id {order.exchange_order_id} belongs to orders "
                    f"{other} and {order.client_order_id}"
                )
        notional = notionals.get(order.client_order_id)
        if notional is None:
            raise StoreValidationError(f"order {order.client_order_id} has no filled notional")
        if type(notional) is not Decimal or not notional.is_finite() or notional < 0:
            raise StoreValidationError(
                f"order {order.client_order_id}: filled notional must be an exact, finite "
                f"Decimal >= 0, got {notional!r}"
            )
        if (notional == 0) != (order.filled_qty == 0):
            raise StoreValidationError(
                f"order {order.client_order_id}: filled notional {notional} is inconsistent "
                f"with filled_qty {order.filled_qty}"
            )
    for cid in notionals:
        if cid not in orders:
            raise StoreValidationError(f"filled notional for unknown order {cid}")
    for cid in safety_blocks:
        order = orders.get(cid)
        if order is None:
            raise StoreValidationError(f"safety block for unknown order {cid}")
        if order.status is not OrderStatus.FAILED or order.exchange_order_id is not None:
            raise StoreValidationError(
                f"safety-blocked order {cid} must be FAILED without an exchange id, "
                f"got {order.status.value}"
            )
    for fill in fills.values():
        if fill.client_order_id is None:
            raise StoreValidationError(f"fill {fill.exec_id} has no client_order_id")
        order = orders.get(fill.client_order_id)
        if order is None:
            raise StoreValidationError(f"fill {fill.exec_id}: no order {fill.client_order_id}")
        if order.symbol != fill.symbol or order.side is not fill.side:
            raise StoreValidationError(
                f"fill {fill.exec_id} does not match order {order.client_order_id}"
            )
        if order.exchange_order_id not in (None, fill.exchange_order_id):
            raise StoreValidationError(
                f"fill {fill.exec_id} exchange_order_id {fill.exchange_order_id} differs "
                f"from order {order.client_order_id}"
            )
