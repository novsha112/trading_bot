"""Startup hydration: durable snapshot -> validation -> local crash classification
-> one recovery commit -> a fresh ``InMemoryAccountState``.

Two kinds of durable state are used: hand-built snapshots served by
``SnapshotStore`` (exact revisions, corrupt data) and the reference store filled
by a real account state that is then dropped (a "crash").
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from decimal import ROUND_UP, Context, Decimal, Inexact, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import Order
from app.execution import account_state as account_state_module
from app.execution import recovery as recovery_module
from app.execution.account_state import (
    AccountStatePoisonedError,
    InMemoryAccountState,
    LockedAccountState,
)
from app.execution.models import ExchangeOrderState, PlacementRecord, SubmissionOutcome
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    PersistedPosition,
    StoreCommitError,
    StoreConflictError,
    StoreUncertainError,
    StoreValidationError,
)
from app.execution.recovery import (
    AccountHydrationError,
    classify_orders_after_crash,
    recovery_change,
    validate_persisted_account_state,
)
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import (
    ACTIVE_ORDER_STATUSES,
    ExposureChange,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    SymbolRiskLimits,
    TradingState,
)
from app.risk.snapshots import build_risk_snapshot
from app.services.placement import PlacementCoordinator

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
T9 = T0 + timedelta(minutes=9)
SCOPE = "acct-1"
APP_ROOT = Path(__file__).resolve().parents[3] / "app"


# --- doubles --------------------------------------------------------------------------------


class Clock:
    def __init__(self, at: object = T9) -> None:
        self.at = at
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        return self.at  # type: ignore[return-value]


class RaisingClock:
    def __init__(self) -> None:
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        raise RuntimeError("clock unavailable")


class SnapshotStore:
    """Serves a fixed snapshot; records commits; optionally fails them."""

    def __init__(
        self, snapshot: PersistedAccountState | None, *, fail: Exception | None = None
    ) -> None:
        self.snapshot = snapshot
        self.fail = fail
        self.loads: list[str] = []
        self.changes: list[AccountStateChange] = []

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        self.loads.append(account_scope_id)
        return self.snapshot

    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        if self.fail is not None:
            raise self.fail


class RecordingStore:
    """The reference store, recording loads and commits."""

    def __init__(self, inner: InMemoryAccountStateStore | None = None) -> None:
        self.inner = InMemoryAccountStateStore() if inner is None else inner
        self.loads = 0
        self.changes: list[AccountStateChange] = []

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        self.loads += 1
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        await self.inner.commit(change)


# --- builders -------------------------------------------------------------------------------


def intent(intent_id: str = "i-1", **overrides: Any) -> PlaceOrderIntent:
    values: dict[str, Any] = {
        "intent_id": intent_id,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "tag": None,
        "created_at": T0,
    }
    return PlaceOrderIntent(**{**values, **overrides})


def decision(source: PlaceOrderIntent, *, approved: bool = True) -> RiskDecision:
    return RiskDecision(
        intent_id=source.intent_id,
        snapshot_id="s",
        policy_id="p",
        approved=approved,
        reasons=() if approved else (RiskReason.MAX_OPEN_ORDERS,),
        exposure=(
            ExposureChange(
                reducing_qty=D("0"),
                increasing_qty=D("1"),
                worst_long_qty=D("1"),
                worst_short_qty=D("0"),
            )
            if approved
            else None
        ),
    )


def order(cid: str, status: OrderStatus = S.OPEN, **overrides: Any) -> Order:
    values: dict[str, Any] = {
        "client_order_id": cid,
        "exchange_order_id": None if status in (S.NEW, S.SUBMITTING) else f"ex-{cid}",
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "status": status,
        "filled_qty": D("0"),
        "avg_fill_price": None,
        "created_at": T0,
        "updated_at": T1,
        "last_exchange_update_ts": None,
        "version": 3,
    }
    if status in (S.PARTIALLY_FILLED, S.CANCELING, S.UNKNOWN, S.EXPIRED, S.CANCELED):
        values |= {"filled_qty": D("4"), "avg_fill_price": D("100")}
    if status is S.FILLED:
        values |= {"filled_qty": D("10"), "avg_fill_price": D("100")}
    return Order(**{**values, **overrides})


def notional_of(item: Order) -> Decimal:
    return D(0) if item.filled_qty == 0 else item.filled_qty * D("100")


def snapshot(
    *orders: Order,
    revision: int = 17,
    scope: str = SCOPE,
    placements: Mapping[str, PlacementRecord] | None = None,
    fills: Mapping[str, Fill] | None = None,
    positions: Mapping[str, PersistedPosition] | None = None,
    notionals: Mapping[str, Decimal] | None = None,
) -> PersistedAccountState:
    return PersistedAccountState(
        account_scope_id=scope,
        revision=revision,
        placements={} if placements is None else placements,
        orders={item.client_order_id: item for item in orders},
        fills={} if fills is None else fills,
        positions={} if positions is None else positions,
        notionals=(
            {item.client_order_id: notional_of(item) for item in orders}
            if notionals is None
            else notionals
        ),
    )


def fill(exec_id: str = "e-1", **overrides: Any) -> Fill:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": "ex-c-1",
        "client_order_id": "c-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D("100"),
        "qty": D("4"),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "exchange_ts": T1,
    }
    return Fill(**{**values, **overrides})


async def hydrate(store: Any, clock: Any = None) -> InMemoryAccountState:
    return await InMemoryAccountState.hydrate(
        account_scope_id=SCOPE, store=store, clock=Clock() if clock is None else clock
    )


def ram(account: InMemoryAccountState) -> tuple[Any, ...]:
    state = account._state
    return (
        state.revision,
        dict(state.orders),
        dict(state.notionals),
        dict(state.placements),
        dict(state.fills),
        dict(state.positions),
    )


def reprs(items: Mapping[str, object]) -> dict[str, str]:
    return {key: repr(value) for key, value in items.items()}


async def reserve(locked: LockedAccountState, intent_id: str, cid: str, **terms: Any) -> None:
    source = intent(intent_id, **terms)
    await locked.register_approved(
        intent=source,
        decision=decision(source),
        client_order_id=cid,
        expected_revision=locked.revision,
        at=T0,
    )


# --- entry point ----------------------------------------------------------------------------


def test_hydrate_is_a_separate_async_factory_and_the_constructor_stays_empty() -> None:
    assert inspect.iscoroutinefunction(InMemoryAccountState.hydrate)
    params = inspect.signature(InMemoryAccountState.hydrate).parameters
    assert list(params) == ["account_scope_id", "store", "clock"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())

    store = RecordingStore()
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    assert account._state.revision == 0
    assert store.loads == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"account_scope_id": ""}, "account_scope_id"),
        ({"store": object()}, "store"),
        ({"clock": object()}, "clock"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_arguments_fail_before_any_load(kwargs: dict[str, Any], match: str) -> None:
    store = SnapshotStore(None)
    values: dict[str, Any] = {"account_scope_id": SCOPE, "store": store, "clock": Clock()}
    with pytest.raises(DomainValidationError, match=match):
        await InMemoryAccountState.hydrate(**{**values, **kwargs})
    assert store.loads == []


# --- missing account / scope ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_account_gives_an_empty_account_without_commit_or_clock() -> None:
    store = SnapshotStore(None)
    clock = RaisingClock()

    account = await hydrate(store, clock)

    assert store.loads == [SCOPE]
    assert store.changes == []
    assert clock.calls == 0
    assert ram(account) == (0, {}, {}, {}, {}, {})
    assert await account.position_qty("BTCUSDT") is None
    assert not account.is_poisoned
    assert account.account_scope_id == SCOPE


@pytest.mark.asyncio
async def test_foreign_scope_is_rejected_without_store_mutation() -> None:
    store = SnapshotStore(snapshot(order("c-1", S.NEW), scope="acct-2"))
    clock = Clock()

    with pytest.raises(AccountHydrationError, match="acct-2"):
        await hydrate(store, clock)
    assert store.changes == []
    assert clock.calls == 0


@pytest.mark.parametrize("loaded", [object(), {"revision": 1}, 0])
@pytest.mark.asyncio
async def test_a_load_result_that_is_not_a_snapshot_is_rejected(loaded: object) -> None:
    store = SnapshotStore(loaded)  # type: ignore[arg-type]
    with pytest.raises(AccountHydrationError, match="not a PersistedAccountState"):
        await hydrate(store)
    assert store.changes == []


@pytest.mark.asyncio
async def test_load_errors_propagate_unchanged() -> None:
    class FailingLoad(SnapshotStore):
        async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
            raise ConnectionError("database down")

    store = FailingLoad(None)
    clock = Clock()
    with pytest.raises(ConnectionError, match="database down"):
        await hydrate(store, clock)
    assert store.changes == []
    assert clock.calls == 0


# --- validation -----------------------------------------------------------------------------


def _approved(source: PlaceOrderIntent, cid: str) -> PlacementRecord:
    return PlacementRecord(intent=source, decision=decision(source), client_order_id=cid)


CORRUPT: dict[str, tuple[Any, type[Exception] | None]] = {
    "placement without order": (
        lambda: snapshot(placements={"i-1": _approved(intent("i-1"), "c-1")}),
        StoreValidationError,
    ),
    "placement terms differ": (
        lambda: snapshot(
            order("c-1", price=D("100.0")),
            placements={"i-1": _approved(intent("i-1"), "c-1")},
        ),
        StoreValidationError,
    ),
    "client id of two intents": (
        lambda: snapshot(
            order("c-1"),
            placements={
                "i-1": _approved(intent("i-1"), "c-1"),
                "i-2": _approved(intent("i-2"), "c-1"),
            },
        ),
        StoreConflictError,
    ),
    "duplicate exchange id": (
        lambda: snapshot(order("c-1"), order("c-2", exchange_order_id="ex-c-1")),
        StoreConflictError,
    ),
    "order without notional": (lambda: snapshot(order("c-1"), notionals={}), StoreValidationError),
    "notional of unknown order": (
        lambda: snapshot(order("c-1"), notionals={"c-1": D(0), "c-9": D(0)}),
        StoreValidationError,
    ),
    "notional without fill": (
        lambda: snapshot(order("c-1"), notionals={"c-1": D("1")}),
        StoreValidationError,
    ),
    "fill without notional": (
        lambda: snapshot(order("c-1", S.PARTIALLY_FILLED), notionals={"c-1": D(0)}),
        StoreValidationError,
    ),
    "negative notional": (
        lambda: snapshot(order("c-1", S.PARTIALLY_FILLED), notionals={"c-1": D("-400")}),
        StoreValidationError,
    ),
    "fill of unknown order": (lambda: snapshot(order("c-2"), fills={"e-1": fill()}), None),
    "fill of other side": (
        lambda: snapshot(order("c-1", S.PARTIALLY_FILLED), fills={"e-1": fill(side=Side.SELL)}),
        StoreValidationError,
    ),
    "fill of other exchange id": (
        lambda: snapshot(
            order("c-1", S.PARTIALLY_FILLED), fills={"e-1": fill(exchange_order_id="ex-9")}
        ),
        StoreValidationError,
    ),
    "NEW with exchange id": (lambda: snapshot(order("c-1", S.NEW, exchange_order_id="ex-1")), None),
    "NEW with an applied fill": (
        lambda: snapshot(order("c-1", S.NEW), fills={"e-1": fill()}),
        None,
    ),
}


@pytest.mark.parametrize("case", sorted(CORRUPT))
@pytest.mark.asyncio
async def test_corrupt_snapshots_are_rejected_before_clock_or_commit(case: str) -> None:
    build, cause = CORRUPT[case]
    store = SnapshotStore(build())
    clock = Clock()

    with pytest.raises(AccountHydrationError) as raised:
        await hydrate(store, clock)

    if cause is not None:
        assert type(raised.value.__cause__) is cause
    assert store.changes == []
    assert clock.calls == 0


def test_new_with_exchange_id_is_a_valid_domain_order_so_hydrate_must_reject_it() -> None:
    # The domain model alone does not forbid it; the crash proof does.
    assert order("c-1", S.NEW, exchange_order_id="ex-1").exchange_order_id == "ex-1"
    with pytest.raises(AccountHydrationError, match="cannot be proven unsent"):
        validate_persisted_account_state(
            snapshot(order("c-1", S.NEW, exchange_order_id="ex-1")), account_scope_id=SCOPE
        )


@pytest.mark.parametrize("value", [D("NaN"), D("Infinity"), 1.5, "1"])
def test_non_exact_notionals_cannot_even_form_a_snapshot(value: object) -> None:
    with pytest.raises(StoreValidationError):
        snapshot(order("c-1"), notionals={"c-1": value})  # type: ignore[dict-item]


def test_keys_must_match_identities_in_a_snapshot() -> None:
    with pytest.raises(StoreValidationError, match="does not match its key"):
        PersistedAccountState(
            account_scope_id=SCOPE,
            revision=1,
            placements={},
            orders={"c-2": order("c-1")},
            fills={},
            positions={},
            notionals={"c-2": D(0)},
        )


@pytest.mark.asyncio
async def test_the_reference_store_shares_the_same_invariants() -> None:
    store = InMemoryAccountStateStore()
    with pytest.raises(StoreValidationError, match="no filled notional"):
        await store.commit(
            AccountStateChange(
                account_scope_id=SCOPE,
                expected_revision=0,
                new_revision=1,
                order_writes=(order("c-1"),),
            )
        )


# --- classification -------------------------------------------------------------------------

UNCHANGED = [status for status in S if status not in (S.NEW, S.SUBMITTING)]


@pytest.mark.parametrize("status", UNCHANGED, ids=lambda s: s.value)
@pytest.mark.asyncio
async def test_statuses_other_than_new_and_submitting_are_kept(status: OrderStatus) -> None:
    kept = order("c-1", status)
    store = SnapshotStore(snapshot(kept))
    clock = Clock()

    account = await hydrate(store, clock)

    assert store.changes == []
    assert clock.calls == 0
    assert await account.revision() == 17
    assert await account.order("c-1") is kept


@pytest.mark.parametrize(
    ("source", "target"), [(S.NEW, S.FAILED), (S.SUBMITTING, S.UNKNOWN)], ids=["new", "submitting"]
)
@pytest.mark.asyncio
async def test_new_fails_and_submitting_becomes_unknown(
    source: OrderStatus, target: OrderStatus
) -> None:
    stored = order("c-1", source)
    store = SnapshotStore(snapshot(stored))
    clock = Clock()

    account = await hydrate(store, clock)

    restored = await account.order("c-1")
    assert restored is not None
    assert restored.status is target
    assert restored.version == stored.version + 1
    assert restored.updated_at == T9
    assert restored.created_at == stored.created_at
    assert dataclasses.replace(restored, status=source, version=3, updated_at=T1) == stored
    assert clock.calls == 1
    assert [change.order_writes for change in store.changes] == [(restored,)]


@pytest.mark.asyncio
async def test_mixed_batch_is_one_commit_with_changed_orders_only() -> None:
    new_1, sub_1, opened = order("c-1", S.NEW), order("c-2", S.SUBMITTING), order("c-3")
    new_2 = order("c-4", S.NEW, symbol="ETHUSDT")
    sub_2 = order("c-5", S.SUBMITTING, exchange_order_id="ex-c-5")
    terminal = order("c-6", S.FILLED)
    store = SnapshotStore(snapshot(new_1, sub_1, opened, new_2, sub_2, terminal))
    clock = Clock()

    account = await hydrate(store, clock)

    assert clock.calls == 1
    assert len(store.changes) == 1
    change = store.changes[0]
    assert (change.account_scope_id, change.expected_revision, change.new_revision) == (
        SCOPE,
        17,
        18,
    )
    assert [(o.client_order_id, o.status) for o in change.order_writes] == [
        ("c-1", S.FAILED),
        ("c-2", S.UNKNOWN),
        ("c-4", S.FAILED),
        ("c-5", S.UNKNOWN),
    ]
    assert change.placement_writes == ()
    assert change.fill_writes == ()
    assert change.position_writes == ()
    assert change.notional_writes == ()
    assert await account.revision() == 18
    # Untouched orders are the stored objects: same version and timestamps.
    assert await account.order("c-3") is opened
    assert await account.order("c-6") is terminal
    # The SUBMITTING order keeps its known exchange id.
    c5 = await account.order("c-5")
    assert c5 is not None
    assert c5.exchange_order_id == "ex-c-5"
    # Reservation order is kept.
    assert list(account._state.orders) == ["c-1", "c-2", "c-3", "c-4", "c-5", "c-6"]


@pytest.mark.asyncio
async def test_revision_17_with_only_open_orders_stays_17_without_commit() -> None:
    store = SnapshotStore(snapshot(order("c-1"), order("c-2", S.PARTIALLY_FILLED)))
    clock = Clock()

    account = await hydrate(store, clock)

    assert await account.revision() == 17
    assert store.changes == []
    assert clock.calls == 0


@pytest.mark.asyncio
async def test_clock_before_an_order_floors_that_order_to_its_own_update_time() -> None:
    late = order("c-1", S.SUBMITTING, updated_at=T9)
    early = order("c-2", S.NEW, updated_at=T0)
    store = SnapshotStore(snapshot(late, early))
    clock = Clock(T2)

    account = await hydrate(store, clock)

    assert clock.calls == 1
    c1, c2 = await account.order("c-1"), await account.order("c-2")
    assert c1 is not None
    assert c2 is not None
    assert (c1.status, c1.updated_at) == (S.UNKNOWN, T9)
    assert (c2.status, c2.updated_at) == (S.FAILED, T2)


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 1, 15, 12, 30),  # noqa: DTZ001 - naive on purpose
        datetime(2026, 1, 15, 12, 30, tzinfo=timezone(timedelta(hours=2))),
        "2026-01-15T12:30:00+00:00",
        None,
    ],
)
@pytest.mark.asyncio
async def test_invalid_clock_values_abort_before_commit(now: object) -> None:
    store = SnapshotStore(snapshot(order("c-1", S.NEW)))
    with pytest.raises(AccountHydrationError, match="recovery time"):
        await hydrate(store, Clock(now))
    assert store.changes == []


@pytest.mark.asyncio
async def test_a_raising_clock_aborts_before_commit() -> None:
    store = SnapshotStore(snapshot(order("c-1", S.SUBMITTING)))
    clock = RaisingClock()
    with pytest.raises(RuntimeError, match="clock unavailable"):
        await hydrate(store, clock)
    assert clock.calls == 1
    assert store.changes == []


@pytest.mark.asyncio
async def test_one_invalid_resulting_order_commits_nothing() -> None:
    corrupt = order("c-2", S.SUBMITTING)
    # Bypass the Order invariants: a SUBMITTING order with a fill but no price.
    object.__setattr__(corrupt, "filled_qty", D("4"))
    store = SnapshotStore(
        snapshot(order("c-1", S.NEW), corrupt, notionals={"c-1": D(0), "c-2": D("400")})
    )

    with pytest.raises(AccountHydrationError, match="c-2") as raised:
        await hydrate(store)
    assert isinstance(raised.value.__cause__, DomainValidationError)
    assert store.changes == []


def test_pure_helpers_return_changed_orders_only_and_no_change_for_none() -> None:
    kept, new = order("c-1"), order("c-2", S.NEW)
    changed = classify_orders_after_crash({"c-1": kept, "c-2": new}, at=T9)
    assert [(o.client_order_id, o.status) for o in changed] == [("c-2", S.FAILED)]
    assert recovery_change(snapshot(kept), ()) is None
    change = recovery_change(snapshot(kept, new), changed)
    assert change is not None
    assert (change.expected_revision, change.new_revision) == (17, 18)


# --- recovery commit failures ---------------------------------------------------------------


async def crashed_store(inner: InMemoryAccountStateStore | None = None) -> RecordingStore:
    """Reference store after a crash: c-1 NEW, c-2 SUBMITTING, c-3 OPEN, positions known."""
    store = RecordingStore(inner)
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
        await locked.set_position_qty("ETHUSDT", D("-2.50"))
        await reserve(locked, "i-1", "c-1")
        await reserve(locked, "i-2", "c-2", price=D("99"))
        await locked.mark_submitting("c-2", at=T1)
        await reserve(locked, "i-3", "c-3", price=D("98"))
        await locked.mark_submitting("c-3", at=T1)
        await locked.record_ack("c-3", exchange_order_id="ex-3", at=T1)
        await locked.apply_exchange_state(
            exchange_state("c-3", "ex-3", S.OPEN),
            at=T1,
        )
    store.changes.clear()
    return store


def exchange_state(cid: str, ex_id: str, status: OrderStatus) -> ExchangeOrderState:
    return ExchangeOrderState(
        client_order_id=cid,
        exchange_order_id=ex_id,
        status=status,
        filled_qty=D("0"),
        avg_fill_price=None,
        exchange_ts=T1,
    )


@pytest.mark.asyncio
async def test_crash_recovery_against_the_reference_store() -> None:
    store = await crashed_store()
    before = await store.inner.load(account_scope_id=SCOPE)
    assert before is not None

    account = await hydrate(store)

    after = await store.inner.load(account_scope_id=SCOPE)
    assert after is not None
    assert after.revision == before.revision + 1 == await account.revision()
    assert {cid: o.status for cid, o in after.orders.items()} == {
        "c-1": S.FAILED,
        "c-2": S.UNKNOWN,
        "c-3": S.OPEN,
    }
    assert dict(after.orders) == account._state.orders
    # The durable positions were not rewritten.
    assert dict(after.positions) == dict(before.positions)


@pytest.mark.asyncio
async def test_definite_commit_failure_creates_no_account_and_changes_nothing() -> None:
    store = await crashed_store()
    before = await store.inner.load(account_scope_id=SCOPE)
    store.inner.inject_commit_failure(CommitFailure.DEFINITE)

    with pytest.raises(StoreCommitError):
        await hydrate(store)

    assert await store.inner.load(account_scope_id=SCOPE) == before
    retried = await hydrate(store)  # a new hydration starts from the store again
    assert await retried.revision() == before.revision + 1  # type: ignore[union-attr]
    assert len(store.changes) == 2


@pytest.mark.asyncio
async def test_uncertain_commit_creates_no_account_and_the_next_hydrate_rereads() -> None:
    store = await crashed_store()
    before = await store.inner.load(account_scope_id=SCOPE)
    assert before is not None
    store.inner.inject_commit_failure(CommitFailure.UNCERTAIN)
    clock = Clock()

    with pytest.raises(StoreUncertainError):
        await hydrate(store, clock)
    assert len(store.changes) == 1

    # The uncertain commit was in fact applied: the next hydration finds nothing
    # left to classify, commits nothing and does not read the clock.
    second_clock = RaisingClock()
    account = await hydrate(store, second_clock)
    assert store.changes[1:] == []
    assert second_clock.calls == 0
    assert await account.revision() == before.revision + 1
    assert not account.is_poisoned


@pytest.mark.parametrize(
    "error",
    [
        StoreCommitError("x"),
        StoreUncertainError("x"),
        StoreConflictError("x"),
        StoreValidationError("x"),
    ],
)
@pytest.mark.asyncio
async def test_store_errors_propagate_unchanged_once(error: Exception) -> None:
    store = SnapshotStore(snapshot(order("c-1", S.NEW)), fail=error)

    with pytest.raises(type(error)) as raised:
        await hydrate(store)

    assert raised.value is error
    assert len(store.changes) == 1  # never retried


@pytest.mark.asyncio
async def test_concurrent_hydrations_are_decided_by_the_revision_cas() -> None:
    inner = (await crashed_store()).inner

    class BarrierStore(RecordingStore):
        def __init__(self) -> None:
            super().__init__(inner)
            self.both_loaded = asyncio.Event()

        async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
            loaded = await super().load(account_scope_id=account_scope_id)
            if self.loads == 2:
                self.both_loaded.set()
            await self.both_loaded.wait()
            return loaded

    store = BarrierStore()
    results = await asyncio.gather(hydrate(store), hydrate(store), return_exceptions=True)

    winners = [r for r in results if isinstance(r, InMemoryAccountState)]
    losers = [r for r in results if isinstance(r, Exception)]
    assert len(winners) == 1
    assert [type(e) for e in losers] == [StoreConflictError]
    assert len(store.changes) == 2


# --- positions ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persisted_known_positions_are_unknown_at_runtime_and_not_rewritten() -> None:
    positions = {
        "BTCUSDT": PersistedPosition(symbol="BTCUSDT", known=True, qty=D("5")),
        "ETHUSDT": PersistedPosition(symbol="ETHUSDT", known=True, qty=D("0")),
        "SOLUSDT": PersistedPosition(symbol="SOLUSDT", known=False, qty=None),
    }
    store = SnapshotStore(snapshot(order("c-1"), positions=positions))

    account = await hydrate(store)

    for symbol in positions:
        assert await account.position_qty(symbol) is None
    assert account._state.positions == {}
    assert store.changes == []


@pytest.mark.asyncio
async def test_every_referenced_symbol_reads_unknown_after_recovery() -> None:
    store = await crashed_store()
    writer = await hydrate(store)
    async with writer.account_lock() as locked:
        rejected = intent("i-sol", symbol="SOLUSDT")
        await locked.register_rejected(
            intent=rejected,
            decision=decision(rejected, approved=False),
            expected_revision=locked.revision,
        )
        await locked.apply_fill(
            fill("e-3", client_order_id="c-3", exchange_order_id="ex-3", price=D("98"), qty=D("1")),
            at=T2,
        )
    loaded = await store.inner.load(account_scope_id=SCOPE)
    assert loaded is not None
    store.changes.clear()

    account = await hydrate(store)

    symbols = {o.symbol for o in loaded.orders.values()} | set(loaded.positions)
    symbols |= {r.intent.symbol for r in loaded.placements.values()}
    symbols |= {f.symbol for f in loaded.fills.values()}
    assert symbols == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    for symbol in sorted(symbols):
        assert await account.position_qty(symbol) is None
    assert store.changes == []


@pytest.mark.asyncio
async def test_risk_sees_unknown_position_after_hydrate() -> None:
    store = await crashed_store()
    account = await hydrate(store)
    coordinator = PlacementCoordinator(
        account_state=account,
        policy=RiskPolicy(
            policy_id="p",
            max_open_orders=None,
            symbols={
                "BTCUSDT": SymbolRiskLimits(
                    max_order_qty=None, max_order_notional=None, max_position_qty=D("100")
                )
            },
        ),
        clock=Clock(),
        client_order_id_generator=_NoIds(),
    )

    record = await coordinator.place(intent=intent("i-new"), trading_state=TradingState.RUNNING)

    assert not record.approved
    assert RiskReason.UNKNOWN_POSITION in record.decision.reasons


class _NoIds:
    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        raise AssertionError("no id may be generated")


# --- restored state -------------------------------------------------------------------------


async def traded_store() -> tuple[RecordingStore, InMemoryAccountState]:
    """A real account with orders in every reachable status, then dropped."""
    store = RecordingStore()
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
        statuses = ["new", "sub", "unknown", "open", "pf", "canceling", "filled", "failed"]
        for n, name in enumerate(statuses):
            await reserve(locked, f"i-{name}", f"c-{name}", price=D(100 - n))
        rejected = intent("i-rej")
        await locked.register_rejected(
            intent=rejected,
            decision=decision(rejected, approved=False),
            expected_revision=locked.revision,
        )
        for name in statuses[1:]:
            await locked.mark_submitting(f"c-{name}", at=T1)
        await locked.record_submission_outcome("c-unknown", SubmissionOutcome.AMBIGUOUS, at=T1)
        await locked.record_submission_outcome("c-failed", SubmissionOutcome.NOT_SENT, at=T1)
        for name in ("open", "pf", "canceling", "filled"):
            await locked.record_ack(f"c-{name}", exchange_order_id=f"ex-{name}", at=T1)
            await locked.apply_exchange_state(
                exchange_state(f"c-{name}", f"ex-{name}", S.OPEN), at=T1
            )
        await locked.apply_fill(
            fill(
                "e-pf",
                client_order_id="c-pf",
                exchange_order_id="ex-pf",
                price=D("97"),
                qty=D("4.000"),
            ),
            at=T2,
        )
        await locked.apply_fill(
            fill(
                "e-cx",
                client_order_id="c-canceling",
                exchange_order_id="ex-canceling",
                price=D("96"),
                qty=D("1"),
            ),
            at=T2,
        )
        await locked.apply_fill(
            fill(
                "e-f",
                client_order_id="c-filled",
                exchange_order_id="ex-filled",
                price=D("95"),
                qty=D("10"),
            ),
            at=T2,
        )
    return store, account


@pytest.mark.asyncio
async def test_orders_placements_fills_and_notionals_are_restored_exactly() -> None:
    store, crashed = await traded_store()
    durable = await store.inner.load(account_scope_id=SCOPE)
    assert durable is not None

    account = await hydrate(store)

    state = account._state
    assert reprs(state.placements) == reprs(durable.placements)
    assert reprs(state.fills) == reprs(durable.fills)
    assert reprs(state.notionals) == reprs(durable.notionals)
    for cid, stored in durable.orders.items():
        if stored.status in (S.NEW, S.SUBMITTING):
            continue
        assert repr(state.orders[cid]) == repr(stored)
    assert state.orders["c-new"].status is S.FAILED
    assert state.orders["c-sub"].status is S.UNKNOWN
    assert state.revision == durable.revision + 1
    # The crashed in-memory instance is untouched by the hydration.
    assert crashed._state.orders["c-new"].status is S.NEW


@pytest.mark.asyncio
async def test_active_statuses_after_hydrate_feed_the_risk_snapshot() -> None:
    store, _ = await traded_store()
    account = await hydrate(store)

    async with account.account_lock() as locked:
        active = locked.active_orders("BTCUSDT")
        count = locked.account_active_order_count()
        risk = build_risk_snapshot(
            snapshot_id=f"{SCOPE}:{locked.revision}",
            symbol="BTCUSDT",
            trading_state=TradingState.RUNNING,
            position_qty=locked.position_qty("BTCUSDT"),
            orders=active,
            account_open_order_count=count,
        )

    assert {o.client_order_id for o in active} == {
        "c-sub",  # SUBMITTING -> UNKNOWN: still counted
        "c-unknown",
        "c-open",
        "c-pf",
        "c-canceling",
    }
    assert {o.status for o in active} <= ACTIVE_ORDER_STATUSES
    assert count == 5
    assert risk.open_orders is not None
    assert len(risk.open_orders) == 5
    assert risk.position_qty is None


@pytest.mark.asyncio
async def test_canceling_survives_hydrate_as_active() -> None:
    store = SnapshotStore(snapshot(order("c-1", S.CANCELING)))
    account = await hydrate(store)
    assert [o.status for o in await account.active_orders("BTCUSDT")] == [S.CANCELING]


@pytest.mark.asyncio
async def test_exact_values_survive_a_hostile_decimal_context() -> None:
    hundred = D("1." + "1" * 99)
    stored = order(
        "c-1",
        S.PARTIALLY_FILLED,
        price=hundred,
        filled_qty=D("4.000"),
        avg_fill_price=hundred,
        exchange_order_id="ex-weird-1",
    )
    zero_e = order("c-2", exchange_order_id="ex-2")
    tiny = order("c-3", S.UNKNOWN, filled_qty=D("1E-200"), avg_fill_price=D("1E-200"))
    fills = {
        "e-1": fill(
            client_order_id="c-1",
            exchange_order_id="ex-weird-1",
            price=hundred,
            qty=D("4.000"),
            fee=D("-0"),
            fee_asset="USDT",
        )
    }
    notionals = {"c-1": hundred * 4, "c-2": D("-0"), "c-3": D("1E-400")}
    loaded = snapshot(stored, zero_e, tiny, fills=fills, notionals=notionals)
    store = SnapshotStore(loaded)

    hostile = Context(prec=1, rounding=ROUND_UP, traps=[Inexact])
    with localcontext(hostile):
        account = await hydrate(store)

    assert reprs(account._state.notionals) == reprs(loaded.notionals)
    assert reprs(account._state.fills) == reprs(loaded.fills)
    assert reprs(account._state.orders) == reprs(loaded.orders)
    assert repr(account._state.notionals["c-2"]) == "Decimal('-0')"
    assert account._state.orders["c-1"].exchange_order_id == "ex-weird-1"
    assert account._state.orders["c-1"].updated_at == T1
    assert store.changes == []


# --- replay ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_replay_after_hydrate_needs_no_risk_id_clock_or_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RecordingStore()
    first = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with first.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
    risk_policy = RiskPolicy(
        policy_id="p",
        max_open_orders=1,
        symbols={
            "BTCUSDT": SymbolRiskLimits(
                max_order_qty=None, max_order_notional=None, max_position_qty=D("100")
            )
        },
    )

    class Ids:
        def next_id(self, *, intent: PlaceOrderIntent) -> str:
            return f"c-{intent.intent_id}"

    before = PlacementCoordinator(
        account_state=first, policy=risk_policy, clock=Clock(T0), client_order_id_generator=Ids()
    )
    approved = await before.place(intent=intent("i-1"), trading_state=TradingState.RUNNING)
    rejected = await before.place(intent=intent("i-2"), trading_state=TradingState.RUNNING)
    assert approved.approved
    assert not rejected.approved

    account = await hydrate(store)  # crash: c-i-1 NEW -> FAILED
    store.changes.clear()
    clock = RaisingClock()
    calls: list[str] = []

    def spy(**_: Any) -> Any:
        calls.append("evaluate")
        raise AssertionError("Risk must not run for a replay")

    monkeypatch.setattr("app.services.placement.evaluate", spy)
    after = PlacementCoordinator(
        account_state=account, policy=risk_policy, clock=clock, client_order_id_generator=_NoIds()
    )

    assert await after.place(intent=intent("i-1"), trading_state=TradingState.RUNNING) == approved
    assert await after.place(intent=intent("i-2"), trading_state=TradingState.RUNNING) == rejected
    assert calls == []
    assert clock.calls == 0
    assert store.changes == []
    order_after = await account.order("c-i-1")
    assert order_after is not None
    assert order_after.status is S.FAILED


# --- poison / readiness ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poisoned_account_stays_poisoned_and_hydrate_builds_a_clean_one() -> None:
    inner = InMemoryAccountStateStore()
    store = RecordingStore(inner)
    old = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with old.account_lock() as locked:
        await reserve(locked, "i-1", "c-1")
        inner.inject_commit_failure(CommitFailure.UNCERTAIN)
        with pytest.raises(StoreUncertainError):
            await locked.mark_submitting("c-1", at=T1)
    assert old.is_poisoned
    old_state = old._state
    assert old_state.orders["c-1"].status is S.NEW  # RAM behind the durable SUBMITTING

    fresh = await hydrate(store)

    assert not fresh.is_poisoned
    assert fresh is not old
    restored = await fresh.order("c-1")
    assert restored is not None
    assert restored.status is S.UNKNOWN  # durable SUBMITTING, never re-sent
    assert old.is_poisoned
    assert old._state is old_state
    async with old.account_lock() as locked:
        with pytest.raises(AccountStatePoisonedError):
            locked.ensure_mutations_allowed()
    # The fresh account keeps working on top of the recovery revision.
    async with fresh.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
        # reserve 1, durable write-ahead 2, recovery 3, position 4.
        assert locked.revision == 4


def test_there_is_no_readiness_flag() -> None:
    names = set(dir(InMemoryAccountState)) | set(dir(LockedAccountState))
    assert not {n for n in names if "ready" in n.lower() or "recovered" in n.lower()}
    assert "hydrated" not in names


# --- no network -----------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module", [recovery_module, account_state_module])
def test_hydration_path_never_imports_exchange_or_network_code(module: Any) -> None:
    imports = _imports(Path(module.__file__))
    assert not {name for name in imports if name.startswith(("app.exchanges", "app.persistence"))}
    source = inspect.getsource(module)
    assert "TradingClient" not in source


def test_hydrate_accepts_no_trading_client() -> None:
    source = inspect.getsource(InMemoryAccountState.hydrate)
    assert "TradingClient" not in source
    assert "get_order" not in source
    assert set(inspect.signature(InMemoryAccountState.hydrate).parameters) == {
        "account_scope_id",
        "store",
        "clock",
    }
