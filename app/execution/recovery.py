"""Pure steps of startup hydration after a crash (docs/ARCHITECTURE.md 12).

``InMemoryAccountState.hydrate`` runs them in this order::

    store.load -> validate_persisted_account_state -> (only if needed) one
    recovery_time(clock) -> classify_orders_after_crash -> ONE durable
    recovery_change commit -> a fresh account state

Only LOCALLY PROVABLE recovery happens here; nothing asks the exchange:

* ``NEW -> FAILED``: the write-ahead ``NEW -> SUBMITTING`` is durably committed
  before any placement request may be sent, so an order still durably NEW was
  never sent. A NEW order with an exchange order id or an applied fill
  contradicts that proof and makes the snapshot corrupt.
* ``SUBMITTING -> UNKNOWN``: the request may or may not have reached the
  exchange; the order stays active (exposure counted) and is never re-sent.
  A known exchange order id is kept.
* every other status is kept exactly as stored (UNKNOWN, OPEN,
  PARTIALLY_FILLED and CANCELING stay active; terminal statuses stay final).

Transitions go through the domain state machine (``transition``): version + 1
and ``updated_at`` = the recovery time, which is read ONCE per batch and floored
per order to its own ``updated_at`` (the ``app.execution.timing`` convention:
a recorded time never moves an order backwards). The clock must return an aware
UTC ``datetime``; anything else aborts the hydration before any commit (unlike
``timing.change_time``, nothing is known yet that must be recorded anyway).

Hydrated is not recovered, and recovered is not safe to trade: positions,
missing fills and UNKNOWN orders still need exchange reconciliation (not here).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Final

from app.domain.clock import Clock
from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError, InvalidOrderTransition
from app.domain.order_state import transition
from app.domain.orders import Order
from app.domain.validation import require_utc
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    StoreConflictError,
    StoreValidationError,
)
from app.execution.state_invariants import check_account_references

# The only locally provable crash outcomes; every other status is kept.
CRASH_TRANSITIONS: Final = {
    OrderStatus.NEW: OrderStatus.FAILED,
    OrderStatus.SUBMITTING: OrderStatus.UNKNOWN,
}


class AccountHydrationError(Exception):
    """The durable snapshot cannot be hydrated (corrupt or foreign snapshot,
    unprovable crash classification, invalid recovery time). Nothing was
    committed and no account state was created; a later hydration re-reads
    the store."""


def validate_persisted_account_state(
    loaded: object, *, account_scope_id: str
) -> PersistedAccountState:
    """``loaded`` as a validated snapshot of ``account_scope_id``.

    Checks the snapshot type and scope, the execution-owned reference invariants
    (``app.execution.state_invariants``) and the preconditions of the crash
    proof: a NEW order has no exchange order id and no applied fill.

    Raises:
        AccountHydrationError: on any violation (an invariant error as its cause).
    """
    if type(loaded) is not PersistedAccountState:
        raise AccountHydrationError(
            f"store returned {type(loaded).__name__}, not a PersistedAccountState"
        )
    if loaded.account_scope_id != account_scope_id:
        raise AccountHydrationError(
            f"store returned account {loaded.account_scope_id!r} for {account_scope_id!r}"
        )
    if type(loaded.revision) is not int or loaded.revision < 0:
        raise AccountHydrationError(f"invalid revision {loaded.revision!r}")
    try:
        check_account_references(
            placements=loaded.placements,
            orders=loaded.orders,
            fills=loaded.fills,
            notionals=loaded.notionals,
            safety_blocks=loaded.safety_blocks,
        )
    except (StoreValidationError, StoreConflictError) as error:
        raise AccountHydrationError(
            f"account {account_scope_id} revision {loaded.revision}: corrupt snapshot"
        ) from error
    filled_orders = {fill.client_order_id for fill in loaded.fills.values()}
    for order in loaded.orders.values():
        if order.status is not OrderStatus.NEW:
            continue
        if order.exchange_order_id is not None:
            raise AccountHydrationError(
                f"NEW order {order.client_order_id} has exchange_order_id "
                f"{order.exchange_order_id}: it cannot be proven unsent"
            )
        if order.client_order_id in filled_orders:
            raise AccountHydrationError(
                f"NEW order {order.client_order_id} has an applied fill: it cannot be proven unsent"
            )
    return loaded


def needs_crash_classification(orders: Mapping[str, Order]) -> bool:
    """True if any order is NEW or SUBMITTING (the clock is read only then)."""
    return any(order.status in CRASH_TRANSITIONS for order in orders.values())


def recovery_time(clock: Clock) -> datetime:
    """One ``clock.now()``; an exception propagates unchanged.

    Raises:
        AccountHydrationError: the clock returned no aware UTC ``datetime``.
    """
    now = clock.now()
    try:
        return require_utc(now, "recovery time")
    except DomainValidationError as error:
        raise AccountHydrationError(f"invalid recovery time {now!r}") from error


def classify_orders_after_crash(orders: Mapping[str, Order], *, at: datetime) -> tuple[Order, ...]:
    """The transitioned orders only (NEW -> FAILED, SUBMITTING -> UNKNOWN), in
    the order of ``orders``; untouched orders are not returned. All or nothing.

    Raises:
        AccountHydrationError: a resulting order violates a domain rule.
    """
    at = require_utc(at, "at")
    changed: list[Order] = []
    for order in orders.values():
        target = CRASH_TRANSITIONS.get(order.status)
        if target is None:
            continue
        # Never move an order backwards in time (timing convention: floor).
        order_time = order.updated_at if at < order.updated_at else at
        try:
            changed.append(transition(order, target, at=order_time))
        except (DomainValidationError, InvalidOrderTransition) as error:
            raise AccountHydrationError(
                f"order {order.client_order_id} cannot move "
                f"{order.status.value} -> {target.value} after a crash"
            ) from error
    return tuple(changed)


def recovery_change(
    snapshot: PersistedAccountState, changed: tuple[Order, ...]
) -> AccountStateChange | None:
    """ONE change writing exactly the transitioned orders at revision + 1, or
    None when nothing changed (no commit, the revision stays)."""
    if not changed:
        return None
    return AccountStateChange(
        account_scope_id=snapshot.account_scope_id,
        expected_revision=snapshot.revision,
        new_revision=snapshot.revision + 1,
        order_writes=changed,
    )
