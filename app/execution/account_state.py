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
account state reloaded from durable storage (``InMemoryAccountState.hydrate``;
the constructor itself starts empty). Definite failures, conflicts and
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

Startup (``InMemoryAccountState.hydrate``, docs/ARCHITECTURE.md 12): a new
account state from the durable state; the locally provable crash classification
(``app.execution.recovery``: NEW -> FAILED, SUBMITTING -> UNKNOWN) is committed
in ONE change before the account state exists. Every position starts unknown.
Hydrated is not recovered, recovered is not safe to trade.

Positions, two separate views per symbol:
* RUNTIME position (``position_qty``): ``None`` (no entry) = unknown / not
  reconciled; ``0`` = known flat; ``> 0`` long, ``< 0`` short. A missing entry is
  never read as flat. Unknown after ``hydrate`` until exchange reconciliation.
* DURABLE position projection (``durable_position``): the committed
  ``PersistedPosition`` (no row / ``known=False`` / ``known=True`` with qty), the
  bot's own durable evidence, never the exchange's current position. Invariant:
  a durable ``known=True`` qty covers every fill committed for that symbol by
  this account state; each such fill advances it in the same change (also while
  the runtime position is unknown, e.g. a fill recovered after ``hydrate``). A
  durable unknown never becomes known through a fill. ``hydrate`` restores the
  durable projection, not the runtime position.
``set_position_qty`` (a trusted value, or None to clear) sets both.
``publish_exchange_positions`` (position reconciliation) sets RUNTIME values only
from an authoritative exchange snapshot: no commit, no revision change. It makes
two recovery states legitimate: runtime known with an unknown durable projection
(an unexplained exchange position; a fill there moves only the runtime value and
never makes the durable projection known) and runtime known different from a
known durable projection (a mismatch; a fill there fails closed with
``PositionProjectionMismatchError`` until it is resolved).
``commit_position_baseline`` (the privileged primitive of an EXPLICIT baseline
acceptance, ``app.execution.position_baseline``) makes the current runtime
quantity the known durable projection, together with its append-only
``PositionBaselineRecord``, in ONE change (+1 revision); the runtime position is
unchanged. Never called automatically.

Exchange order id completion (``complete_exchange_order_ids``, driven by
open-order discovery): records authoritative exchange ids of local orders
that do not know theirs yet, all in ONE change (+1 revision), all or nothing.
Only ``None -> id``; an id equal to the recorded one is a no-op, a different
recorded id is never replaced. It is identity metadata, not an order event:
status, execution, terms and timestamps are kept (``updated_at`` too; only the
order ``version`` advances, as every persisted order write requires). No clock.

Fills (``apply_fill``): the fill is found by its ``client_order_id``; symbol,
side and a known ``exchange_order_id`` must match the order. Only SUBMITTING,
OPEN, PARTIALLY_FILLED, CANCELING and UNKNOWN orders accept fills (NEW was never
sent; terminal orders are final). Cumulative quantity, exact notional and the
average price follow ``app.domain.fill_math``; the new status is FILLED when the
order is complete, otherwise CANCELING stays CANCELING and every other source
becomes PARTIALLY_FILLED (the existing state machine). The position follows
``app.portfolio.positions`` from the runtime position, or from the known durable
projection when the runtime one is unknown (unknown stays unknown; a reduce-only
fill can never reverse a known position, else ``FillApplicationError``).
Everything is prepared first; the order, the exact notional, the position (runtime
and durable), the applied fill and the revision are then committed
together, or nothing changes. An identical ``exec_id`` replay changes nothing;
the same ``exec_id`` with different data raises ``FillConflictError``.

Safety blocks (``record_safety_block``, driven by ``OrderSubmitter``): when the
effective trading state refuses a send, a NEW or SUBMITTING order (never sent)
becomes FAILED together with a durable ``SafetyBlockRecord`` (the reason survives
a restart); the reservation is released, the placement record stays.

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
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, DecimalException
from types import MappingProxyType
from typing import Any, Final, TypeVar

from app.domain.clock import Clock
from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError, InvalidOrderTransition
from app.domain.fill_math import accumulate_execution
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.order_state import record_exchange_order_id, transition
from app.domain.orders import Order
from app.domain.validation import require_text, require_utc
from app.execution.models import (
    ExchangeOrderState,
    PlacementRecord,
    PositionBaselineRecord,
    SafetyBlockRecord,
    SubmissionBlockStage,
    SubmissionOutcome,
)
from app.execution.persistence import (
    AccountStateChange,
    AccountStateStore,
    PersistedAccountState,
    PersistedOrderNotional,
    PersistedPosition,
    StoreUncertainError,
)
from app.execution.recovery import (
    classify_orders_after_crash,
    needs_crash_classification,
    recovery_change,
    recovery_time,
    validate_persisted_account_state,
)
from app.portfolio.positions import PositionStateError, position_after_fill
from app.risk.models import ACTIVE_ORDER_STATUSES, RiskDecision, TradingState

_V = TypeVar("_V")
_INITIAL_VERSION: Final = 0
_NO_FILL: Final = Decimal(0)
# Statuses an outcome of the placement request may still decide: only SUBMITTING.
_OUTCOME_TARGET: Final = {
    SubmissionOutcome.NOT_SENT: OrderStatus.FAILED,
    SubmissionOutcome.REJECTED: OrderStatus.REJECTED,
    SubmissionOutcome.AMBIGUOUS: OrderStatus.UNKNOWN,
}
# Where a safety block can stop an order: never sent in both cases.
_BLOCK_STAGE: Final = {
    OrderStatus.NEW: SubmissionBlockStage.BEFORE_WRITE_AHEAD,
    OrderStatus.SUBMITTING: SubmissionBlockStage.BEFORE_SEND,
}
# An acknowledgement contradicts these: never sent, or definitely not accepted.
_ACK_CONTRADICTING_STATUSES: Final = frozenset(
    {OrderStatus.NEW, OrderStatus.FAILED, OrderStatus.REJECTED}
)
# Local orders whose exchange id may be completed: sent and still relevant for
# the exchange (the open-order recovery statuses; terminal orders are final).
EXCHANGE_ID_COMPLETION_STATUSES: Final = frozenset(
    {
        OrderStatus.UNKNOWN,
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCELING,
    }
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


class PositionProjectionMismatchError(FillApplicationError):
    """The known runtime position and the known durable projection of a symbol
    differ: a fill is never applied to one of them arbitrarily."""


class PositionBaselineError(AccountStateError):
    """A position baseline cannot be accepted; nothing was changed."""


class BaselineRuntimeUnknownError(PositionBaselineError):
    """There is no authoritative runtime (exchange-published) position to accept."""


class BaselineQtyMismatchError(PositionBaselineError):
    """The accepted quantity is not the current runtime position (a stale
    acceptance: the exchange-published position changed since it was seen)."""


class BaselineIdConflictError(PositionBaselineError):
    """The ``baseline_id`` is already used by a recorded baseline (append-only:
    a record is never overwritten)."""


class ExchangeIdCompletionError(AccountStateError):
    """An exchange order id cannot be completed (unknown or non-relevant order,
    a different id already recorded, an id claimed by another order, a
    duplicate in the batch); nothing was changed."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExchangeOrderIdCompletion:
    """Record ``exchange_order_id`` for the local order ``client_order_id``."""

    client_order_id: str
    exchange_order_id: str

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        require_text(self.exchange_order_id, "exchange_order_id")


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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PositionEvidence:
    """Immutable local position evidence read under the account lock."""

    revision: int
    runtime_positions: Mapping[str, Decimal]
    """Known runtime positions (an absent symbol is unknown)."""
    durable_positions: Mapping[str, PersistedPosition]
    """Committed durable projections (an absent symbol has no durable row)."""
    fill_symbols: frozenset[str]
    """Symbols with at least one committed fill."""


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
    """RUNTIME known signed positions only; a missing symbol is unknown."""
    durable_positions: dict[str, PersistedPosition] = dataclasses.field(default_factory=dict)
    """The committed durable position projection per symbol (evidence, not exchange
    authority): exactly what the store holds; a missing symbol has no durable row."""
    fills: dict[str, Fill] = dataclasses.field(default_factory=dict)
    safety_blocks: dict[str, SafetyBlockRecord] = dataclasses.field(default_factory=dict)
    """Why an order was FAILED by the safety gate (definitely not sent)."""
    position_baselines: dict[str, PositionBaselineRecord] = dataclasses.field(default_factory=dict)
    """Append-only audit records of accepted position baselines, by baseline_id."""
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

    def filled_notional(self, client_order_id: str) -> Decimal | None:
        """Exact accumulated fill notional of an order (None: no such order)."""
        state = self._live()
        return state.notionals.get(require_text(client_order_id, "client_order_id"))

    def durable_position(self, symbol: str) -> PersistedPosition | None:
        """The committed durable position projection of ``symbol`` (None: no
        durable row; ``known=False``: a durable unknown). Local evidence only,
        never the exchange's current position."""
        state = self._live()
        return state.durable_positions.get(require_text(symbol, "symbol"))

    def position_evidence(self) -> PositionEvidence:
        """One immutable read of every position-relevant local fact (runtime
        positions, durable projections, symbols with committed fills, revision)."""
        state = self._live()
        return PositionEvidence(
            revision=state.revision,
            runtime_positions=MappingProxyType(dict(state.positions)),
            durable_positions=MappingProxyType(dict(state.durable_positions)),
            fill_symbols=frozenset(fill.symbol for fill in state.fills.values()),
        )

    def publish_exchange_positions(
        self, positions: Mapping[str, Decimal], *, expected_revision: int
    ) -> None:
        """Publish exchange-authoritative RUNTIME positions, all at once.

        A runtime cache publication, not a durable mutation: the durable
        projections, the revision and the store are untouched (nothing is
        committed; a restart forgets it). Refused for a poisoned account and when
        the local state moved past ``expected_revision`` (stale evidence). A zero
        quantity is published as known flat, never as an absent entry."""
        state = self._mutable()
        if _require_revision(expected_revision) != state.revision:
            raise StaleRevisionError(
                f"positions reconciled at revision {expected_revision}, "
                f"current revision is {state.revision}"
            )
        published = dict(state.positions)
        for symbol, qty in positions.items():
            require_text(symbol, "symbol")
            if type(qty) is not Decimal or not qty.is_finite():
                raise DomainValidationError(
                    f"position of {symbol} must be a finite, exact Decimal, got {qty!r}"
                )
            published[symbol] = qty
        # Synchronous (no await): no reader ever sees a partially published map.
        self._owner._state = dataclasses.replace(state, positions=published)

    def position_baselines(self, symbol: str | None = None) -> tuple[PositionBaselineRecord, ...]:
        """Accepted baseline records (of ``symbol``, or all), ordered by
        ``(account_revision, baseline_id)``; an immutable tuple."""
        state = self._live()
        if symbol is not None:
            require_text(symbol, "symbol")
        return tuple(
            sorted(
                (
                    record
                    for record in state.position_baselines.values()
                    if symbol is None or record.symbol == symbol
                ),
                key=lambda record: (record.account_revision, record.baseline_id),
            )
        )

    async def commit_position_baseline(self, record: PositionBaselineRecord) -> None:
        """PRIVILEGED: durably accept the current runtime position as the known
        durable projection of ``record.symbol``, with ``record`` as its audit
        record, in one change (+1 revision). The runtime position is unchanged.

        Only for an explicit acceptance (``app.execution.position_baseline``),
        never automatic. ``record.qty`` must be exactly the current runtime
        quantity and ``record.account_revision`` the next revision.

        Raises:
            AccountStatePoisonedError: the account is poisoned.
            BaselineIdConflictError: ``record.baseline_id`` is already recorded.
            StaleRevisionError: ``record.account_revision`` is not revision + 1.
            BaselineRuntimeUnknownError: no runtime position of the symbol.
            BaselineQtyMismatchError: ``record.qty`` is not the runtime position.
        """
        state = self._mutable()
        if type(record) is not PositionBaselineRecord:
            raise DomainValidationError("record must be a PositionBaselineRecord")
        if record.baseline_id in state.position_baselines:
            raise BaselineIdConflictError(f"baseline_id {record.baseline_id} is already recorded")
        if record.account_revision != state.revision + 1:
            raise StaleRevisionError(
                f"baseline for revision {record.account_revision}, "
                f"next revision is {state.revision + 1}"
            )
        runtime = state.positions.get(record.symbol)
        if runtime is None:
            raise BaselineRuntimeUnknownError(
                f"no runtime position of {record.symbol} to accept as baseline"
            )
        if record.qty != runtime or repr(record.qty) != repr(runtime):
            raise BaselineQtyMismatchError(
                f"baseline {record.qty} of {record.symbol} is not the runtime position {runtime}"
            )
        row = PersistedPosition(symbol=record.symbol, known=True, qty=runtime)
        prepared = dataclasses.replace(
            state,
            durable_positions=_with(state.durable_positions, record.symbol, row),
            position_baselines=_with(state.position_baselines, record.baseline_id, record),
            revision=record.account_revision,
        )
        change = self._change(
            state,
            new_revision=prepared.revision,
            position_writes=(row,),
            position_baseline_writes=(record,),
        )
        await self._commit_and_publish(state, prepared, change)

    def placement(self, intent_id: str) -> PlacementRecord | None:
        state = self._live()
        return state.placements.get(require_text(intent_id, "intent_id"))

    def order(self, client_order_id: str) -> Order | None:
        state = self._live()
        return state.orders.get(require_text(client_order_id, "client_order_id"))

    def orders(self) -> tuple[Order, ...]:
        """Every local order (all statuses), in reservation order (immutable)."""
        return tuple(self._live().orders.values())

    async def complete_exchange_order_ids(
        self, completions: tuple[ExchangeOrderIdCompletion, ...], *, expected_revision: int
    ) -> tuple[Order, ...]:
        """Record authoritative exchange order ids, all in ONE change, or nothing.

        Every completion is checked before anything is prepared: the order
        exists and is in ``EXCHANGE_ID_COMPLETION_STATUSES``; its recorded id is
        None or exactly the given one; no other local order (any status) has the
        id; no client or exchange id appears twice in the batch. Completions
        whose id is already recorded are no-ops; when all are, nothing is
        committed and ``()`` is returned. Otherwise the changed orders (sorted by
        ``(client_order_id, exchange_order_id)``) are committed with +1 revision
        and returned. Status, execution, terms and ``updated_at`` are kept; only
        ``version`` advances.

        Raises:
            AccountStatePoisonedError: the account is poisoned.
            StaleRevisionError: ``expected_revision`` is not the current revision.
            ExchangeIdCompletionError: any completion is not provable (none applied).
        """
        state = self._mutable()
        if type(completions) is not tuple or not all(
            type(item) is ExchangeOrderIdCompletion for item in completions
        ):
            raise DomainValidationError("completions must be a tuple of ExchangeOrderIdCompletion")
        self._require_current(state, _require_revision(expected_revision))
        ordered = sorted(completions, key=lambda c: (c.client_order_id, c.exchange_order_id))
        for attribute in ("client_order_id", "exchange_order_id"):
            values = [getattr(item, attribute) for item in ordered]
            if len(set(values)) != len(values):
                raise ExchangeIdCompletionError(f"a {attribute} appears twice in the batch")
        owners = {
            order.exchange_order_id: order.client_order_id
            for order in state.orders.values()
            if order.exchange_order_id is not None
        }
        updated: list[Order] = []
        for item in ordered:
            order = state.orders.get(item.client_order_id)
            if order is None:
                raise ExchangeIdCompletionError(f"no local order {item.client_order_id}")
            if order.status not in EXCHANGE_ID_COMPLETION_STATUSES:
                raise ExchangeIdCompletionError(
                    f"order {order.client_order_id} is {order.status.value}: its exchange id "
                    "is not completed"
                )
            owner = owners.get(item.exchange_order_id)
            if owner is not None and owner != order.client_order_id:
                raise ExchangeIdCompletionError(
                    f"exchange_order_id {item.exchange_order_id} belongs to order {owner}"
                )
            if order.exchange_order_id == item.exchange_order_id:
                continue  # already recorded: benign
            if order.exchange_order_id is not None:
                raise ExchangeIdCompletionError(
                    f"order {order.client_order_id} has exchange_order_id "
                    f"{order.exchange_order_id}, never replaced by {item.exchange_order_id}"
                )
            # Identity metadata, not an event: the order's own time is kept.
            updated.append(
                record_exchange_order_id(order, item.exchange_order_id, at=order.updated_at)
            )
        if not updated:
            return ()
        orders = dict(state.orders)
        for order in updated:
            orders[order.client_order_id] = order
        prepared = dataclasses.replace(state, orders=orders, revision=state.revision + 1)
        change = self._change(state, new_revision=prepared.revision, order_writes=tuple(updated))
        await self._commit_and_publish(state, prepared, change)
        return tuple(updated)

    def safety_block(self, client_order_id: str) -> SafetyBlockRecord | None:
        """The durable reason if the safety gate FAILED this order, else None."""
        state = self._live()
        return state.safety_blocks.get(require_text(client_order_id, "client_order_id"))

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
        row = PersistedPosition(symbol=symbol, known=qty is not None, qty=qty)
        current = state.positions.get(symbol)
        durable = state.durable_positions.get(symbol)
        durable_unknown = durable is None or not durable.known
        if current == qty and (durable == row or (qty is None and durable_unknown)):
            return  # runtime and durable already hold this value (absent = unknown)
        positions = dict(state.positions)
        if qty is None:
            positions.pop(symbol, None)
        else:
            positions[symbol] = qty
        prepared = dataclasses.replace(
            state,
            positions=positions,
            durable_positions=_with(state.durable_positions, symbol, row),
            revision=state.revision + 1,
        )
        change = self._change(state, new_revision=prepared.revision, position_writes=(row,))
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

    async def record_safety_block(
        self, client_order_id: str, *, effective_state: TradingState, at: datetime
    ) -> Order:
        """The safety gate refused to send the order: ``NEW -> FAILED`` (before the
        write-ahead marker) or ``SUBMITTING -> FAILED`` (the definite NOT_SENT
        outcome, before the request was sent), together with its durable
        ``SafetyBlockRecord``; +1 revision. FAILED releases the reservation. Only
        the caller knows that nothing was sent; any other status raises."""
        state = self._mutable()
        order = self._existing(state, client_order_id)
        stage = _BLOCK_STAGE.get(order.status)
        if stage is None:
            raise AccountStateError(
                f"order {client_order_id} is {order.status.value}: only a NEW or "
                "SUBMITTING order can be blocked before sending"
            )
        updated = transition(order, OrderStatus.FAILED, at=at)
        block = SafetyBlockRecord(
            client_order_id=client_order_id,
            effective_state=effective_state,
            stage=stage,
            blocked_at=updated.updated_at,
        )
        prepared = dataclasses.replace(
            state,
            orders=_with(state.orders, client_order_id, updated),
            safety_blocks=_with(state.safety_blocks, client_order_id, block),
            revision=state.revision + 1,
        )
        change = self._change(
            state,
            new_revision=prepared.revision,
            order_writes=(updated,),
            safety_block_writes=(block,),
        )
        await self._commit_and_publish(state, prepared, change)
        return updated

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
        runtime, durable = self._positions_after_fill(state, order, fill)
        # One durable change: fill, order, notional, position and revision together.
        prepared = dataclasses.replace(
            state,
            orders=_with(state.orders, order.client_order_id, updated),
            notionals=_with(state.notionals, order.client_order_id, totals.filled_notional),
            positions=(
                state.positions
                if runtime is None
                else _with(state.positions, order.symbol, runtime)
            ),
            durable_positions=(
                state.durable_positions
                if durable is None
                else _with(state.durable_positions, order.symbol, durable)
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
            position_writes=() if durable is None else (durable,),
        )
        await self._commit_and_publish(state, prepared, change)
        return updated

    @staticmethod
    def _positions_after_fill(
        state: _State, order: Order, fill: Fill
    ) -> tuple[Decimal | None, PersistedPosition | None]:
        """(runtime position after the fill or None if it stays unknown, the new
        durable row or None if the durable projection is not known).

        One shared computation (``position_after_fill``) from the position the
        fill applies to: the runtime one when known, else the durable known
        projection. A known durable projection always advances with the fill (in
        the same change), so ``known=True`` keeps covering every committed fill;
        an unknown one never becomes known here. Runtime and durable known with
        different values fail closed."""
        symbol = order.symbol
        runtime = state.positions.get(symbol)
        row = state.durable_positions.get(symbol)
        durable = row.qty if row is not None and row.known else None
        if runtime is not None and durable is not None and runtime != durable:
            raise PositionProjectionMismatchError(
                f"fill {fill.exec_id}: runtime position {runtime} of {symbol} differs from "
                f"its durable projection {durable}"
            )
        base = runtime if runtime is not None else durable
        try:
            after = position_after_fill(
                base, side=fill.side, qty=fill.qty, reduce_only=order.reduce_only
            )
        except PositionStateError as error:
            raise FillApplicationError(
                f"fill {fill.exec_id} of order {order.client_order_id}: {error}"
            ) from error
        runtime_after = after if runtime is not None else None
        if durable is None or after is None:
            return runtime_after, None
        return runtime_after, PersistedPosition(symbol=symbol, known=True, qty=after)

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


def _require_store(store: AccountStateStore) -> AccountStateStore:
    if not callable(getattr(store, "commit", None)) or not callable(getattr(store, "load", None)):
        raise DomainValidationError("store must provide load() and commit()")
    return store


def _runtime_state(snapshot: PersistedAccountState, change: AccountStateChange | None) -> _State:
    """The runtime state of a validated snapshot after its recovery ``change``.

    Orders, placements, fills and exact notionals are taken as stored (nothing
    is recomputed), the classified orders replace theirs in place (the
    reservation order is kept). Positions are deliberately left out: every
    position reads as unknown until exchange reconciliation.
    """
    orders = dict(snapshot.orders)
    revision = snapshot.revision
    if change is not None:
        orders.update((order.client_order_id, order) for order in change.order_writes)
        revision = change.new_revision
    return _State(
        orders=orders,
        notionals=dict(snapshot.notionals),
        placements=dict(snapshot.placements),
        positions={},
        durable_positions=dict(snapshot.positions),
        fills=dict(snapshot.fills),
        safety_blocks=dict(snapshot.safety_blocks),
        position_baselines=dict(snapshot.position_baselines),
        revision=revision,
    )


class InMemoryAccountState:
    """Local risk-relevant state of one account (orders, placements, positions,
    fills, revision), guarded by its single account lock."""

    __slots__ = ("_account_scope_id", "_holder", "_lock", "_poisoned", "_state", "_store")

    def __init__(self, *, account_scope_id: str, store: AccountStateStore) -> None:
        """An EMPTY account state (revision 0, nothing known). Use ``hydrate`` to
        start from the durable state of an account."""
        self._account_scope_id = require_text(account_scope_id, "account_scope_id")
        self._store = _require_store(store)
        self._lock = asyncio.Lock()
        self._holder: asyncio.Task[Any] | None = None
        self._state = _State()
        # Runtime-only (not durable, not revision, not TradingState): set when a
        # commit outcome is unknown; there is deliberately no API to clear it.
        self._poisoned = False

    @classmethod
    async def hydrate(
        cls, *, account_scope_id: str, store: AccountStateStore, clock: Clock
    ) -> InMemoryAccountState:
        """A NEW account state built from the durable state of ``account_scope_id``.

        ``store.load`` -> validation -> local crash classification
        (``app.execution.recovery``: NEW -> FAILED, SUBMITTING -> UNKNOWN, all
        other statuses kept) -> if anything changed, ONE durable commit at
        revision + 1 with exactly the transitioned orders -> construction. The
        clock is read once, and only when an order must be classified. No
        account (``load`` returns None) gives an empty state, without a commit.

        Every position is UNKNOWN in the runtime state, also one persisted as
        known (the durable record is not rewritten): after a restart no
        position is known until exchange position reconciliation.

        Nothing is retried. A load, clock or store error (commit, conflict,
        uncertain) propagates unchanged, a corrupt or foreign snapshot raises
        ``AccountHydrationError``; in every failure case no account state is
        created (none poisoned either: a later hydration re-reads the store).
        Existing account states, poisoned or not, are never touched.

        Hydrated is not recovered, and recovered is not safe to trade: UNKNOWN
        orders, missing fills and positions still need exchange reconciliation;
        there is no readiness flag. One writer per account scope is the
        caller's invariant (no global recovery lock); the revision CAS rejects
        a concurrent hydration's commit.
        """
        require_text(account_scope_id, "account_scope_id")
        _require_store(store)
        if not callable(getattr(clock, "now", None)):
            raise DomainValidationError("clock must provide now()")
        loaded = await store.load(account_scope_id=account_scope_id)
        if loaded is None:
            return cls(account_scope_id=account_scope_id, store=store)
        snapshot = validate_persisted_account_state(loaded, account_scope_id=account_scope_id)
        changed: tuple[Order, ...] = ()
        if needs_crash_classification(snapshot.orders):
            changed = classify_orders_after_crash(snapshot.orders, at=recovery_time(clock))
        change = recovery_change(snapshot, changed)
        if change is not None:
            await store.commit(change)
        # Constructed and published only after the recovery commit succeeded.
        account = cls(account_scope_id=account_scope_id, store=store)
        account._state = _runtime_state(snapshot, change)
        return account

    @property
    def account_scope_id(self) -> str:
        """The account scope of every durable change (the single source of truth)."""
        return self._account_scope_id

    @property
    def is_poisoned(self) -> bool:
        """True after an uncertain durable commit; the only way back is a NEW
        account state from ``hydrate`` (this one stays poisoned)."""
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

    async def filled_notional(self, client_order_id: str) -> Decimal | None:
        async with self.account_lock() as locked:
            return locked.filled_notional(client_order_id)

    async def durable_position(self, symbol: str) -> PersistedPosition | None:
        async with self.account_lock() as locked:
            return locked.durable_position(symbol)

    async def position_baselines(
        self, symbol: str | None = None
    ) -> tuple[PositionBaselineRecord, ...]:
        async with self.account_lock() as locked:
            return locked.position_baselines(symbol)

    async def placement(self, intent_id: str) -> PlacementRecord | None:
        async with self.account_lock() as locked:
            return locked.placement(intent_id)

    async def order(self, client_order_id: str) -> Order | None:
        async with self.account_lock() as locked:
            return locked.order(client_order_id)

    async def orders(self) -> tuple[Order, ...]:
        async with self.account_lock() as locked:
            return locked.orders()

    async def safety_block(self, client_order_id: str) -> SafetyBlockRecord | None:
        async with self.account_lock() as locked:
            return locked.safety_block(client_order_id)
