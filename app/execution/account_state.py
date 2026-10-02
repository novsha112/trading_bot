"""In-memory local account state: one account, one writer, one lock (V1).

The single owner of the local risk-relevant account state (docs/ARCHITECTURE.md
7.0): orders by ``client_order_id``, the ``intent_id -> PlacementRecord`` index,
signed net positions per symbol, applied fills by ``exec_id`` and ONE monotonic
``revision``. There is no separate order registry or position book with its own
lock, so orders and positions can only be read and changed together::

    async with account.account_lock() as locked:
        revision / orders / position   # one consistent state (sync reads)
        build snapshot, evaluate       # pure, outside this module
        await locked.register_approved(...) / await locked.register_rejected(...)
        await locked.mark_submitting(...) / await locked.record_ack(...)
        await locked.record_submission_outcome(...) / await locked.apply_fill(...)
        await locked.apply_exchange_state(...) / await locked.set_position_qty(...)

Persistence (docs/ARCHITECTURE.md 11.0): every mutation is ``prepare -> durable
commit -> publish``. Under the account lock it reads the published snapshot,
prepares a complete new snapshot (copy-on-write; nothing is modified in place)
and the minimal ``AccountStateChange``, awaits ``AccountStateStore.commit`` and
only after it returned publishes the new snapshot. Until then no reader sees the
change: readers need the lock, which the mutation holds through the commit.
The durable commit is the ONLY I/O awaited under the account lock; network
calls are never awaited under it. A store error propagates and leaves the
published state unchanged. ``StoreUncertainError`` (the durable state may now be
ahead of RAM, e.g. SUBMITTING durable, NEW in RAM) additionally POISONS the
account, atomically under the lock: the original error propagates, and every
later mutation fails with ``AccountStatePoisonedError`` before any change is
prepared or the store is called; workflows check ``ensure_mutations_allowed``
right after taking the lock. Reads keep working on the last confirmed RAM
snapshot, for diagnostics only. Poison is runtime-only (not durable, not the
revision, not a TradingState) and has no reset API: the only way back is a new
account state reloaded from durable storage (startup hydration does not exist
yet; an account state starts empty). Definite failures, conflicts and
validation errors do not poison. A no-op (replay, identical
fill, a known exchange id, a confirming report, an ambiguous outcome superseded
by fills, the current position) does not commit. Store validation errors are
invariant bugs and propagate as such.

Lock design: one non-reentrant ``asyncio.Lock``. Mutations exist only on the
``LockedAccountState`` handle yielded by ``account_lock()``; they never take the
lock, so mutating inside the held lock cannot deadlock. The async reads take the
lock; calling them (or ``account_lock()``) again from the task that holds it
raises ``AccountLockError`` instead of waiting forever. A handle is unusable
after the lock is released. Tasks spawned and awaited inside the lock are not
detected.

Intent idempotency: one record per ``intent_id``. Registering an intent equal
(field by field) to the recorded one returns the recorded ``PlacementRecord``
unchanged: no new Order, client id or revision, whatever the new decision. A
different intent under the same ``intent_id``, or an approved placement whose
``client_order_id`` is already taken, raises ``PlacementConflictError``.

Positions: ``None`` (no entry) = unknown / not reconciled; ``0`` = known flat;
``> 0`` long, ``< 0`` short. A missing entry is never read as flat. Until
reconciliation exists, ``set_position_qty`` seeds or clears a position.

Fills (``apply_fill``): the fill is found by its ``client_order_id``; symbol,
side and a known ``exchange_order_id`` must match the order. Only SUBMITTING,
OPEN, PARTIALLY_FILLED, CANCELING and UNKNOWN orders accept fills (NEW was never
sent; terminal orders are final). Cumulative quantity, exact notional and the
average price follow ``app.domain.fill_math``; the new status is FILLED when the
order is complete, otherwise CANCELING stays CANCELING and every other source
becomes PARTIALLY_FILLED (the existing state machine). The position follows
``app.portfolio.positions`` (unknown stays unknown; a reduce-only fill can never
reverse a known position). Everything is prepared first; the order, the exact
notional, the position, the applied fill and the revision are then committed
together, or nothing changes. An identical ``exec_id`` replay changes nothing;
the same ``exec_id`` with different data raises ``FillConflictError``.

Submission (driven by ``OrderSubmitter``): ``mark_submitting`` is the write-ahead
``NEW -> SUBMITTING`` before any request may be sent; ``record_ack`` stores the
exchange order id as metadata on any sent order (never a status change, never a
rollback); ``record_submission_outcome`` decides only a SUBMITTING order (not
sent -> FAILED, rejected -> REJECTED, ambiguous -> UNKNOWN). Confirmed exchange
progress beats a transport outcome: an ambiguous outcome for an order already
advanced by fills changes nothing; a definite one raises
``SubmissionOutcomeConflictError`` and keeps the state.

Exchange reports (``apply_exchange_state``, an ``ExchangeOrderState``): identity,
executed quantity and average price must agree with the local order; a report
showing more execution than the applied fills raises ``MissingFillsError`` (fills
are never synthesized, so Order and position stay consistent), less or a status
the order cannot reach is an ``ExchangeStateMismatchError``. Stronger local
progress is never regressed; nothing changes on any error.

Revision: starts at 0 and versions the whole local risk-relevant state. +1 for
an approved reservation, an order transition (``mark_submitting``, a recorded
outcome, an applied exchange report), a newly recorded exchange order id, an
applied fill and a changed position; nothing for a rejected record, a replay, a
repeated ack or report, an outcome that the observed state already supersedes
or setting a position to its current value. Every new placement must name the
current revision (``expected_revision``) or raises ``StaleRevisionError``.
Client order ids are supplied by the caller and only validated here; nothing is
generated. No exchange, network or storage.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, DecimalException
from typing import Any, Final, TypeVar

from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError, InvalidOrderTransition
from app.domain.fill_math import accumulate_execution
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.order_state import record_exchange_order_id, transition
from app.domain.orders import Order
from app.domain.validation import require_text, require_utc
from app.execution.models import ExchangeOrderState, PlacementRecord, SubmissionOutcome
from app.execution.persistence import (
    AccountStateChange,
    AccountStateStore,
    PersistedOrderNotional,
    PersistedPosition,
    StoreUncertainError,
)
from app.portfolio.positions import PositionStateError, position_after_fill
from app.risk.models import ACTIVE_ORDER_STATUSES, RiskDecision

_V = TypeVar("_V")
_INITIAL_VERSION: Final = 0
_NO_FILL: Final = Decimal(0)
# Statuses an outcome of the placement request may still decide: only SUBMITTING.
_OUTCOME_TARGET: Final = {
    SubmissionOutcome.NOT_SENT: OrderStatus.FAILED,
    SubmissionOutcome.REJECTED: OrderStatus.REJECTED,
    SubmissionOutcome.AMBIGUOUS: OrderStatus.UNKNOWN,
}
# An acknowledgement contradicts these: never sent, or definitely not accepted.
_ACK_CONTRADICTING_STATUSES: Final = frozenset(
    {OrderStatus.NEW, OrderStatus.FAILED, OrderStatus.REJECTED}
)
# Local statuses in which an exchange execution can arrive (docs/ARCHITECTURE.md 8):
# NEW was never sent, terminal orders are final.
FILLABLE_STATUSES: Final = frozenset(
    {
        OrderStatus.SUBMITTING,
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCELING,
        OrderStatus.UNKNOWN,
    }
)


class AccountStateError(Exception):
    """Base class of account-state errors that are not invalid arguments."""


class PlacementConflictError(AccountStateError):
    """An ``intent_id`` or ``client_order_id`` is already registered for different data."""


class StaleRevisionError(AccountStateError):
    """The registration is based on an older revision than the current one."""


class AccountLockError(AccountStateError):
    """The account lock was re-entered or a released handle was used."""


class FillApplicationError(AccountStateError):
    """A fill does not match the local state (unknown or foreign order, wrong
    status, side or ids, overfill, impossible position change); nothing changed."""


class FillConflictError(FillApplicationError):
    """An ``exec_id`` was already applied with different data."""


class OrderAckMismatchError(AccountStateError):
    """An acknowledgement does not belong to the local order (other client or
    exchange id, or an order that was never sent / definitely not accepted)."""


class ExchangeStateMismatchError(AccountStateError):
    """A confirmed exchange report contradicts the local order (identity, a
    smaller executed quantity, a different average price, a status the local
    order can no longer move to, or a status / fill combination the domain
    forbids); nothing was changed."""


class MissingFillsError(AccountStateError):
    """The exchange reports more execution than the fills applied locally. Fills
    are never synthesized from an order report: the missing fills must be
    applied first (Order and position stay consistent); nothing was changed."""


class AccountStatePoisonedError(AccountStateError):
    """A durable commit of this account had an unknown outcome earlier: the store
    may be ahead of RAM, so no further mutation (or workflow meant to mutate) is
    allowed until the account is reloaded from durable state."""


class SubmissionOutcomeConflictError(AccountStateError):
    """A definite transport outcome (not sent / rejected) contradicts exchange
    progress already observed for the order; the stronger state was kept."""


def _require_revision(value: object) -> int:
    if type(value) is not int or value < 0:
        raise DomainValidationError(f"expected_revision must be an int >= 0, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _State:
    """One published snapshot of the account state. Never modified after it is
    published: every mutation prepares a new snapshot (copy-on-write of the
    collections; the domain objects themselves are immutable)."""

    orders: dict[str, Order] = dataclasses.field(default_factory=dict)
    notionals: dict[str, Decimal] = dataclasses.field(default_factory=dict)
    """Exact executed notional per order: the source of its average price."""
    placements: dict[str, PlacementRecord] = dataclasses.field(default_factory=dict)
    positions: dict[str, Decimal] = dataclasses.field(default_factory=dict)
    """Known signed positions only; a missing symbol is unknown."""
    fills: dict[str, Fill] = dataclasses.field(default_factory=dict)
    revision: int = 0


def _with(items: dict[str, _V], key: str, value: _V) -> dict[str, _V]:
    """A copy of ``items`` with ``key`` set (the original is not modified)."""
    copied = dict(items)
    copied[key] = value
    return copied


class LockedAccountState:
    """Access to the account state while its lock is held. Reads are synchronous;
    every mutation is ``async``: it prepares the next snapshot, awaits the durable
    commit (the only I/O awaited under the account lock) and publishes the
    snapshot only after the commit succeeded."""

    __slots__ = ("_open", "_owner")

    def __init__(self, owner: InMemoryAccountState) -> None:
        self._owner = owner
        self._open = True

    def _close(self) -> None:
        self._open = False

    def _live(self) -> _State:
        if not self._open:
            raise AccountLockError("handle used after the account lock was released")
        return self._owner._state

    @property
    def is_poisoned(self) -> bool:
        """True after an uncertain durable commit: reads still work (the last
        confirmed RAM snapshot, for diagnostics only) but mutations fail."""
        self._live()
        return self._owner._poisoned

    def ensure_mutations_allowed(self) -> None:
        """Raise ``AccountStatePoisonedError`` if this account is poisoned. Workflows
        that would mutate the account (placement, submission, reconciliation) call
        it right after taking the lock, before any evaluation, id, clock or
        network use."""
        self._live()
        if self._owner._poisoned:
            raise AccountStatePoisonedError(
                f"account {self._owner.account_scope_id} is poisoned by an uncertain "
                "durable commit: reload it from durable state before any mutation"
            )

    def _mutable(self) -> _State:
        """The single gate of every mutation: live handle and not poisoned."""
        self.ensure_mutations_allowed()
        return self._live()

    def _change(self, state: _State, *, new_revision: int, **writes: Any) -> AccountStateChange:
        return AccountStateChange(
            account_scope_id=self._owner.account_scope_id,
            expected_revision=state.revision,
            new_revision=new_revision,
            **writes,
        )

    async def _commit_and_publish(
        self, current: _State, prepared: _State, change: AccountStateChange
    ) -> None:
        """Durable commit, then publication. A store error propagates and the
        published state stays ``current``; after ``StoreUncertainError`` the
        durable state may be ahead of RAM: the caller must stop mutating and
        reload (no runtime poison flag yet)."""
        self._mutable()
        try:
            await self._owner._store.commit(change)
        except StoreUncertainError:
            # Still under the account lock: no other task can act on RAM that may
            # now be behind the durable state. The prepared state is discarded.
            self._owner._poisoned = True
            raise
        if self._owner._state is not current:  # pragma: no cover - lock invariant
            raise AccountStateError("account state changed during a commit")
        self._owner._state = prepared

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

    def position_qty(self, symbol: str) -> Decimal | None:
        """Signed position of ``symbol``; None when unknown (never read as flat)."""
        state = self._live()
        return state.positions.get(require_text(symbol, "symbol"))

    def fill(self, exec_id: str) -> Fill | None:
        state = self._live()
        return state.fills.get(require_text(exec_id, "exec_id"))

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

    async def register_approved(
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
        state = self._mutable()
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
        prepared = dataclasses.replace(
            state,
            orders=_with(state.orders, client_order_id, order),
            notionals=_with(state.notionals, client_order_id, _NO_FILL),
            placements=_with(state.placements, intent.intent_id, record),
            revision=state.revision + 1,
        )
        change = self._change(
            state,
            new_revision=prepared.revision,
            placement_writes=(record,),
            order_writes=(order,),
            notional_writes=(
                PersistedOrderNotional(client_order_id=client_order_id, filled_notional=_NO_FILL),
            ),
        )
        await self._commit_and_publish(state, prepared, change)
        return record

    async def register_rejected(
        self, *, intent: PlaceOrderIntent, decision: RiskDecision, expected_revision: int
    ) -> PlacementRecord:
        """Record a rejected intent: no Order, no revision change."""
        state = self._mutable()
        self._require_inputs(intent, decision, expected_revision, approved=False)
        replay = self._replay(state, intent)
        if replay is not None:
            return replay
        self._require_current(state, expected_revision)
        record = PlacementRecord(intent=intent, decision=decision, client_order_id=None)
        prepared = dataclasses.replace(
            state, placements=_with(state.placements, intent.intent_id, record)
        )
        change = self._change(state, new_revision=state.revision, placement_writes=(record,))
        await self._commit_and_publish(state, prepared, change)
        return record

    async def set_position_qty(self, symbol: str, qty: Decimal | None) -> None:
        """Seed or clear the known position of ``symbol`` (None = unknown).

        +1 revision when the value changes; setting the current value is a no-op.
        """
        state = self._mutable()
        require_text(symbol, "symbol")
        if qty is not None and (type(qty) is not Decimal or not qty.is_finite()):
            raise DomainValidationError("position qty must be a finite, exact Decimal or None")
        current = state.positions.get(symbol)
        if (current is None and qty is None) or (
            current is not None and qty is not None and current == qty
        ):
            return
        positions = dict(state.positions)
        if qty is None:
            del positions[symbol]
        else:
            positions[symbol] = qty
        prepared = dataclasses.replace(state, positions=positions, revision=state.revision + 1)
        change = self._change(
            state,
            new_revision=prepared.revision,
            position_writes=(PersistedPosition(symbol=symbol, known=qty is not None, qty=qty),),
        )
        await self._commit_and_publish(state, prepared, change)

    async def _publish_order(self, state: _State, order: Order) -> Order:
        """Commit and publish one changed order (+1 revision)."""
        prepared = dataclasses.replace(
            state,
            orders=_with(state.orders, order.client_order_id, order),
            revision=state.revision + 1,
        )
        change = self._change(state, new_revision=prepared.revision, order_writes=(order,))
        await self._commit_and_publish(state, prepared, change)
        return order

    async def mark_submitting(self, client_order_id: str, *, at: datetime) -> Order:
        """Write-ahead ``NEW -> SUBMITTING`` before a placement request may be sent
        (docs/ARCHITECTURE.md 7.2); +1 revision. Nothing is sent here."""
        state = self._mutable()
        order = self._existing(state, client_order_id)
        updated = transition(order, OrderStatus.SUBMITTING, at=at)
        return await self._publish_order(state, updated)

    async def record_ack(
        self, client_order_id: str, *, exchange_order_id: str, at: datetime
    ) -> Order:
        """Record an acknowledgement's ``exchange_order_id``: metadata enrichment,
        never a status change (an ack does not prove OPEN).

        Applies to any sent order (SUBMITTING and every later state, including a
        state already advanced by fills), so a late ack never rolls an order back.
        +1 revision only when the id is new. Raises ``OrderAckMismatchError``
        (nothing changed) for a different known exchange id or an order that was
        never sent / definitely not accepted (NEW, FAILED, REJECTED). The caller
        matches the ack's ``client_order_id`` to this order.
        """
        state = self._mutable()
        order = self._existing(state, client_order_id)
        require_text(exchange_order_id, "exchange_order_id")
        if order.status in _ACK_CONTRADICTING_STATUSES:
            raise OrderAckMismatchError(
                f"ack for order {order.client_order_id} in status {order.status.value}"
            )
        if order.exchange_order_id not in (None, exchange_order_id):
            raise OrderAckMismatchError(
                f"ack exchange_order_id {exchange_order_id} differs from "
                f"{order.exchange_order_id} of order {order.client_order_id}"
            )
        updated = record_exchange_order_id(order, exchange_order_id, at=at)
        if updated is order:
            return order  # the id is already recorded: no commit
        return await self._publish_order(state, updated)

    async def record_submission_outcome(
        self, client_order_id: str, outcome: SubmissionOutcome, *, at: datetime
    ) -> Order:
        """Apply the transport outcome of the placement request.

        Only a SUBMITTING order is decided by it: NOT_SENT -> FAILED (releases the
        reservation), REJECTED -> REJECTED (releases), AMBIGUOUS -> UNKNOWN (stays
        active); +1 revision. Confirmed exchange progress is stronger than any
        transport outcome: for an order already past SUBMITTING, AMBIGUOUS changes
        nothing, while NOT_SENT / REJECTED contradict the observed facts and raise
        ``SubmissionOutcomeConflictError``; the state is never rolled back.
        """
        state = self._mutable()
        order = self._existing(state, client_order_id)
        if type(outcome) is not SubmissionOutcome:
            raise DomainValidationError("outcome must be a SubmissionOutcome")
        if order.status is OrderStatus.SUBMITTING:
            updated = transition(order, _OUTCOME_TARGET[outcome], at=at)
            return await self._publish_order(state, updated)
        if order.status is OrderStatus.NEW:
            raise AccountStateError(f"order {order.client_order_id} was never marked SUBMITTING")
        if outcome is SubmissionOutcome.AMBIGUOUS:
            return order  # the observed exchange state already says more
        raise SubmissionOutcomeConflictError(
            f"{outcome.value} for order {order.client_order_id} contradicts its observed "
            f"status {order.status.value} (filled {order.filled_qty}); state kept"
        )

    async def apply_exchange_state(self, report: ExchangeOrderState, *, at: datetime) -> Order:
        """Apply a confirmed exchange report to the local order, or change nothing.

        Checked before any change: the order exists and was sent (not NEW or
        FAILED); a known ``exchange_order_id`` matches; the reported execution
        equals the locally applied one (less -> stale ``ExchangeStateMismatchError``,
        more -> ``MissingFillsError``: fills are never synthesized); with fills,
        a reported average equals the local one. Then, for a different status,
        the existing state machine and Order invariants decide the transition
        (a status the order cannot move to, e.g. PARTIALLY_FILLED -> OPEN or any
        change of a terminal order, is a mismatch). The position is never
        changed here. A report confirming the current status only records a new
        ``exchange_order_id``; an identical report is a no-op. +1 revision only
        for a real change.
        """
        state = self._mutable()
        if type(report) is not ExchangeOrderState:
            raise DomainValidationError("report must be an ExchangeOrderState")
        order = self._existing(state, report.client_order_id)
        if order.status in (OrderStatus.NEW, OrderStatus.FAILED):
            raise ExchangeStateMismatchError(
                f"exchange reports order {order.client_order_id} as {report.status.value}, "
                f"but it is locally {order.status.value} (never accepted)"
            )
        exchange_order_id = report.exchange_order_id
        if exchange_order_id is not None and order.exchange_order_id not in (
            None,
            exchange_order_id,
        ):
            raise ExchangeStateMismatchError(
                f"exchange_order_id {exchange_order_id} differs from "
                f"{order.exchange_order_id} of order {order.client_order_id}"
            )
        if report.filled_qty < order.filled_qty:
            raise ExchangeStateMismatchError(
                f"exchange reports {report.filled_qty} executed for order "
                f"{order.client_order_id}, below the {order.filled_qty} applied locally"
            )
        if report.filled_qty > order.filled_qty:
            raise MissingFillsError(
                f"exchange reports {report.filled_qty} executed for order "
                f"{order.client_order_id}, {order.filled_qty} applied locally: "
                "apply the missing fills first"
            )
        if report.avg_fill_price is not None and report.avg_fill_price != order.avg_fill_price:
            raise ExchangeStateMismatchError(
                f"exchange average price {report.avg_fill_price} differs from "
                f"{order.avg_fill_price} of order {order.client_order_id}"
            )
        if report.status is order.status:
            if exchange_order_id is None:
                return order
            updated = record_exchange_order_id(order, exchange_order_id, at=at)
        else:
            try:
                updated = transition(
                    order,
                    report.status,
                    at=at,
                    exchange_order_id=exchange_order_id,
                    last_exchange_update_ts=report.exchange_ts,
                )
            except (InvalidOrderTransition, DomainValidationError) as error:
                raise ExchangeStateMismatchError(
                    f"order {order.client_order_id} cannot move from {order.status.value} "
                    f"to reported {report.status.value}: {error}"
                ) from error
        if updated is order:
            return order  # a confirming report: no commit
        return await self._publish_order(state, updated)

    @staticmethod
    def _existing(state: _State, client_order_id: str) -> Order:
        order = state.orders.get(require_text(client_order_id, "client_order_id"))
        if order is None:
            raise AccountStateError(f"no local order {client_order_id}")
        return order

    async def apply_fill(self, fill: Fill, *, at: datetime) -> Order:
        """Apply one confirmed execution to its order and the position, atomically.

        ``at`` is the local time of the change (the caller's clock); the fill's
        exchange time becomes the order's ``last_exchange_update_ts``. Returns the
        order after the fill (the current order for an identical replay).
        """
        state = self._mutable()
        if type(fill) is not Fill:
            raise DomainValidationError("fill must be a domain Fill")
        require_utc(at, "at")
        known = state.fills.get(fill.exec_id)
        if known is not None:
            if known != fill:
                raise FillConflictError(
                    f"exec_id {fill.exec_id} was already applied with different data"
                )
            return state.orders[known.client_order_id or ""]
        order = self._fill_target(state, fill)
        try:
            totals = accumulate_execution(
                filled_qty=order.filled_qty,
                filled_notional=state.notionals[order.client_order_id],
                price=fill.price,
                qty=fill.qty,
            )
        except DecimalException:
            raise FillApplicationError(
                f"fill {fill.exec_id} cannot be accumulated exactly"
            ) from None
        if totals.filled_qty > order.qty:
            raise FillApplicationError(
                f"fill {fill.exec_id} overfills order {order.client_order_id}: "
                f"{totals.filled_qty} > {order.qty}"
            )
        if totals.filled_qty == order.qty:
            status = OrderStatus.FILLED
        elif order.status is OrderStatus.CANCELING:
            status = OrderStatus.CANCELING
        else:
            status = OrderStatus.PARTIALLY_FILLED
        updated = transition(
            order,
            status,
            at=at,
            filled_qty=totals.filled_qty,
            avg_fill_price=totals.avg_fill_price,
            exchange_order_id=fill.exchange_order_id,
            last_exchange_update_ts=fill.exchange_ts,
        )
        try:
            position = position_after_fill(
                state.positions.get(order.symbol),
                side=fill.side,
                qty=fill.qty,
                reduce_only=order.reduce_only,
            )
        except PositionStateError as error:
            raise FillApplicationError(
                f"fill {fill.exec_id} of order {order.client_order_id}: {error}"
            ) from error
        # One durable change: fill, order, notional, position and revision together.
        prepared = dataclasses.replace(
            state,
            orders=_with(state.orders, order.client_order_id, updated),
            notionals=_with(state.notionals, order.client_order_id, totals.filled_notional),
            positions=(
                state.positions
                if position is None
                else _with(state.positions, order.symbol, position)
            ),
            fills=_with(state.fills, fill.exec_id, fill),
            revision=state.revision + 1,
        )
        change = self._change(
            state,
            new_revision=prepared.revision,
            fill_writes=(fill,),
            order_writes=(updated,),
            notional_writes=(
                PersistedOrderNotional(
                    client_order_id=order.client_order_id,
                    filled_notional=totals.filled_notional,
                ),
            ),
            position_writes=(
                ()
                if position is None
                else (PersistedPosition(symbol=order.symbol, known=True, qty=position),)
            ),
        )
        await self._commit_and_publish(state, prepared, change)
        return updated

    @staticmethod
    def _fill_target(state: _State, fill: Fill) -> Order:
        if fill.client_order_id is None:
            raise FillApplicationError(
                f"fill {fill.exec_id} has no client_order_id (foreign order): "
                "not supported before reconciliation"
            )
        order = state.orders.get(fill.client_order_id)
        if order is None:
            raise FillApplicationError(
                f"fill {fill.exec_id}: no local order {fill.client_order_id}"
            )
        if order.symbol != fill.symbol or order.side is not fill.side:
            raise FillApplicationError(
                f"fill {fill.exec_id} ({fill.symbol} {fill.side.value}) does not match order "
                f"{order.client_order_id} ({order.symbol} {order.side.value})"
            )
        if order.exchange_order_id not in (None, fill.exchange_order_id):
            raise FillApplicationError(
                f"fill {fill.exec_id}: exchange_order_id {fill.exchange_order_id} differs from "
                f"{order.exchange_order_id} of order {order.client_order_id}"
            )
        if order.status not in FILLABLE_STATUSES:
            raise FillApplicationError(
                f"fill {fill.exec_id}: order {order.client_order_id} is {order.status.value} "
                "and cannot receive a fill"
            )
        return order

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


class InMemoryAccountState:
    """Local risk-relevant state of one account (orders, placements, positions,
    fills, revision), guarded by its single account lock."""

    __slots__ = ("_account_scope_id", "_holder", "_lock", "_poisoned", "_state", "_store")

    def __init__(self, *, account_scope_id: str, store: AccountStateStore) -> None:
        self._account_scope_id = require_text(account_scope_id, "account_scope_id")
        if not callable(getattr(store, "commit", None)) or not callable(
            getattr(store, "load", None)
        ):
            raise DomainValidationError("store must provide load() and commit()")
        self._store = store
        self._lock = asyncio.Lock()
        self._holder: asyncio.Task[Any] | None = None
        self._state = _State()
        # Runtime-only (not durable, not revision, not TradingState): set when a
        # commit outcome is unknown; there is deliberately no API to clear it.
        self._poisoned = False

    @property
    def account_scope_id(self) -> str:
        """The account scope of every durable change (the single source of truth)."""
        return self._account_scope_id

    @property
    def is_poisoned(self) -> bool:
        """True after an uncertain durable commit; the only way back is a reload of
        a new account state from durable storage (not available yet)."""
        return self._poisoned

    @asynccontextmanager
    async def account_lock(self) -> AsyncIterator[LockedAccountState]:
        """Hold the account lock; the yielded handle reads and mutates."""
        task = asyncio.current_task()
        if task is not None and task is self._holder:
            raise AccountLockError(
                "account lock is not reentrant: use the LockedAccountState already held"
            )
        async with self._lock:
            self._holder = task
            handle = LockedAccountState(self)
            try:
                yield handle
            finally:
                handle._close()
                self._holder = None

    async def revision(self) -> int:
        async with self.account_lock() as locked:
            return locked.revision

    async def active_orders(self, symbol: str) -> tuple[Order, ...]:
        async with self.account_lock() as locked:
            return locked.active_orders(symbol)

    async def account_active_order_count(self) -> int:
        async with self.account_lock() as locked:
            return locked.account_active_order_count()

    async def position_qty(self, symbol: str) -> Decimal | None:
        async with self.account_lock() as locked:
            return locked.position_qty(symbol)

    async def fill(self, exec_id: str) -> Fill | None:
        async with self.account_lock() as locked:
            return locked.fill(exec_id)

    async def placement(self, intent_id: str) -> PlacementRecord | None:
        async with self.account_lock() as locked:
            return locked.placement(intent_id)

    async def order(self, client_order_id: str) -> Order | None:
        async with self.account_lock() as locked:
            return locked.order(client_order_id)
