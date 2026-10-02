"""In-memory order reservation registry: one account, one writer (V1).

The local unified view of the account's orders and the serialization boundary
for the placement coordinator (app/services/placement.py, docs/ARCHITECTURE.md 7.0)::

    async with registry.placement_lock() as locked:
        revision / views          # read the state the decision is based on
        build snapshot, evaluate  # pure, outside this module
        locked.register_approved(...) / locked.register_rejected(...)

Lock design: one non-reentrant ``asyncio.Lock`` per account. Mutations exist
only on the ``LockedOrderRegistry`` handle yielded by ``placement_lock()``; its
methods are synchronous and never take the lock, so registering inside the held
lock cannot deadlock. The registry's own async reads take the lock; calling them
(or ``placement_lock()``) again from the task that holds it raises
``RegistryLockError`` instead of waiting forever. A handle is unusable after the
lock is released. Tasks spawned and awaited inside the lock are not detected.

Intent idempotency: one record per ``intent_id``. Registering an intent equal
(field by field) to the recorded one returns the recorded ``PlacementRecord``
unchanged: no new Order, client id or revision, whatever the new decision. A
different intent under the same ``intent_id``, or an approved placement whose
``client_order_id`` is already taken, raises ``PlacementConflictError``.

Revision: starts at 0 and counts changes of the risk-relevant order state. An
approved reservation (record + ``Order(NEW)``) adds 1; a rejected record and a
replay add nothing. Every new registration must name the current revision
(``expected_revision``) or raises ``StaleRevisionError``, so a decision made on
an older state can never reserve. Client order ids are supplied by the caller
and only validated here; nothing is generated. No exchange, network or storage.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import Order
from app.domain.validation import require_text, require_utc
from app.execution.models import PlacementRecord
from app.risk.models import ACTIVE_ORDER_STATUSES, RiskDecision

_INITIAL_VERSION: Final = 0
_NO_FILL: Final = Decimal(0)


class OrderRegistryError(Exception):
    """Base class of registry errors that are not invalid arguments."""


class PlacementConflictError(OrderRegistryError):
    """An ``intent_id`` or ``client_order_id`` is already registered for different data."""


class StaleRevisionError(OrderRegistryError):
    """The registration is based on an older revision than the current one."""


class RegistryLockError(OrderRegistryError):
    """The placement lock was re-entered or a released handle was used."""


def _require_revision(value: object) -> int:
    if type(value) is not int or value < 0:
        raise DomainValidationError(f"expected_revision must be an int >= 0, got {value!r}")
    return value


class _State:
    """The mutable account state, owned by the registry."""

    __slots__ = ("orders", "placements", "revision")

    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}
        self.placements: dict[str, PlacementRecord] = {}
        self.revision = 0


class LockedOrderRegistry:
    """Access to the registry while its placement lock is held (synchronous)."""

    __slots__ = ("_open", "_state")

    def __init__(self, state: _State) -> None:
        self._state = state
        self._open = True

    def _close(self) -> None:
        self._open = False

    def _live(self) -> _State:
        if not self._open:
            raise RegistryLockError("handle used after the placement lock was released")
        return self._state

    @property
    def revision(self) -> int:
        return self._live().revision

    def active_orders(self, symbol: str) -> tuple[Order, ...]:
        """Active orders of ``symbol`` in reservation order (immutable)."""
        state = self._live()
        require_text(symbol, "symbol")
        return tuple(
            order
            for order in state.orders.values()
            if order.symbol == symbol and order.status in ACTIVE_ORDER_STATUSES
        )

    def account_active_order_count(self) -> int:
        """Active orders of the whole account, all symbols."""
        return sum(
            1 for order in self._live().orders.values() if order.status in ACTIVE_ORDER_STATUSES
        )

    def placement(self, intent_id: str) -> PlacementRecord | None:
        state = self._live()
        return state.placements.get(require_text(intent_id, "intent_id"))

    def order(self, client_order_id: str) -> Order | None:
        state = self._live()
        return state.orders.get(require_text(client_order_id, "client_order_id"))

    def replay_of(self, intent: PlaceOrderIntent) -> PlacementRecord | None:
        """The recorded result of an equal intent, or None for a new ``intent_id``.

        Raises ``PlacementConflictError`` when the ``intent_id`` is recorded with
        different data. Lets the caller detect a replay before any evaluation.
        """
        state = self._live()
        if type(intent) is not PlaceOrderIntent:
            raise DomainValidationError("intent must be a PlaceOrderIntent")
        return self._replay(state, intent)

    def register_approved(
        self,
        *,
        intent: PlaceOrderIntent,
        decision: RiskDecision,
        client_order_id: str,
        expected_revision: int,
        at: datetime,
    ) -> PlacementRecord:
        """Reserve: record the approved intent and add its ``Order(NEW)``.

        ``at`` is the reservation time (the caller's clock, not before the intent
        was created). Returns the recorded result on a replay.
        """
        state = self._live()
        self._require_inputs(intent, decision, expected_revision, approved=True)
        require_text(client_order_id, "client_order_id")
        require_utc(at, "at")
        replay = self._replay(state, intent)
        if replay is not None:
            return replay
        self._require_current(state, expected_revision)
        if client_order_id in state.orders:
            raise PlacementConflictError(
                f"client_order_id {client_order_id} is already registered for another intent"
            )
        if at < intent.created_at:
            raise DomainValidationError(
                f"reservation time {at.isoformat()} is before the intent creation "
                f"{intent.created_at.isoformat()}"
            )
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=None,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            price=intent.price,
            qty=intent.qty,
            time_in_force=intent.time_in_force,
            reduce_only=intent.reduce_only,
            status=OrderStatus.NEW,
            filled_qty=_NO_FILL,
            avg_fill_price=None,
            created_at=at,
            updated_at=at,
            last_exchange_update_ts=None,
            version=_INITIAL_VERSION,
        )
        record = PlacementRecord(intent=intent, decision=decision, client_order_id=client_order_id)
        # Everything above may raise; nothing below can, so the change is atomic.
        state.orders[client_order_id] = order
        state.placements[intent.intent_id] = record
        state.revision += 1
        return record

    def register_rejected(
        self, *, intent: PlaceOrderIntent, decision: RiskDecision, expected_revision: int
    ) -> PlacementRecord:
        """Record a rejected intent: no Order, no revision change."""
        state = self._live()
        self._require_inputs(intent, decision, expected_revision, approved=False)
        replay = self._replay(state, intent)
        if replay is not None:
            return replay
        self._require_current(state, expected_revision)
        record = PlacementRecord(intent=intent, decision=decision, client_order_id=None)
        state.placements[intent.intent_id] = record
        return record

    @staticmethod
    def _require_inputs(
        intent: object, decision: object, expected_revision: object, *, approved: bool
    ) -> None:
        if type(intent) is not PlaceOrderIntent:
            raise DomainValidationError("intent must be a PlaceOrderIntent")
        if type(decision) is not RiskDecision:
            raise DomainValidationError("decision must be a RiskDecision")
        if decision.intent_id != intent.intent_id:
            raise DomainValidationError(
                f"decision intent_id {decision.intent_id} differs from intent_id {intent.intent_id}"
            )
        if decision.approved is not approved:
            expected = "approved" if approved else "rejected"
            raise DomainValidationError(f"decision for {intent.intent_id} is not {expected}")
        _require_revision(expected_revision)

    @staticmethod
    def _replay(state: _State, intent: PlaceOrderIntent) -> PlacementRecord | None:
        recorded = state.placements.get(intent.intent_id)
        if recorded is None:
            return None
        if recorded.intent != intent:
            raise PlacementConflictError(
                f"intent_id {intent.intent_id} is already registered with different data"
            )
        return recorded

    @staticmethod
    def _require_current(state: _State, expected_revision: int) -> None:
        if expected_revision != state.revision:
            raise StaleRevisionError(
                f"registration based on revision {expected_revision}, "
                f"current revision is {state.revision}"
            )


class InMemoryOrderRegistry:
    """Local unified order view of one account, guarded by one placement lock."""

    __slots__ = ("_holder", "_lock", "_state")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._holder: asyncio.Task[Any] | None = None
        self._state = _State()

    @asynccontextmanager
    async def placement_lock(self) -> AsyncIterator[LockedOrderRegistry]:
        """Hold the account lock; the yielded handle reads and registers."""
        task = asyncio.current_task()
        if task is not None and task is self._holder:
            raise RegistryLockError(
                "placement lock is not reentrant: use the LockedOrderRegistry already held"
            )
        async with self._lock:
            self._holder = task
            handle = LockedOrderRegistry(self._state)
            try:
                yield handle
            finally:
                handle._close()
                self._holder = None

    async def revision(self) -> int:
        async with self.placement_lock() as locked:
            return locked.revision

    async def active_orders(self, symbol: str) -> tuple[Order, ...]:
        async with self.placement_lock() as locked:
            return locked.active_orders(symbol)

    async def account_active_order_count(self) -> int:
        async with self.placement_lock() as locked:
            return locked.account_active_order_count()

    async def placement(self, intent_id: str) -> PlacementRecord | None:
        async with self.placement_lock() as locked:
            return locked.placement(intent_id)

    async def order(self, client_order_id: str) -> Order | None:
        async with self.placement_lock() as locked:
            return locked.order(client_order_id)
