"""Fail-closed account state after an uncertain durable commit ("poisoned")."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import OrderUpdate
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.execution.account_state import (
    AccountStatePoisonedError,
    ExchangeStateMismatchError,
    FillApplicationError,
    InMemoryAccountState,
    LockedAccountState,
    MissingFillsError,
)
from app.execution.models import ExchangeOrderState, SubmissionOutcome
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    StoreCommitError,
    StoreConflictError,
    StoreUncertainError,
    StoreValidationError,
)
from app.execution.reconciliation import UnknownOrderReconciler
from app.execution.submitter import OrderSubmitter
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.manager import evaluate
from app.risk.models import (
    ExposureChange,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    SymbolRiskLimits,
    TradingState,
)
from app.services import placement as placement_module
from app.services.placement import PlacementCoordinator

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
SCOPE = "acct-1"
FLAT = D("0")


class CountingStore:
    """Reference store wrapper: counts commit attempts and can inject a failure
    into a chosen (1-based) attempt, or block a commit until released."""

    def __init__(self) -> None:
        self.inner = InMemoryAccountStateStore()
        self.attempts = 0
        self.fail_on: dict[int, CommitFailure] = {}
        self.validation_on: set[int] = set()
        self.block_on: int | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.attempts += 1
        if self.attempts in self.validation_on:
            raise StoreValidationError("rejected by the test store")
        if self.attempts == self.block_on:
            self.entered.set()
            await self.release.wait()
        failure = self.fail_on.get(self.attempts)
        if failure is not None:
            self.inner.inject_commit_failure(failure)
        await self.inner.commit(change)

    def fail_next(self, failure: CommitFailure = CommitFailure.UNCERTAIN) -> None:
        self.fail_on[self.attempts + 1] = failure


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


def fill(exec_id: str = "e-1", qty: str = "4") -> Fill:
    return Fill(
        exec_id=exec_id,
        exchange_order_id="ex-1",
        client_order_id="c-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        price=D("100"),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=T1,
    )


def report(status: OrderStatus, filled: str = "0") -> ExchangeOrderState:
    return ExchangeOrderState(
        client_order_id="c-1",
        exchange_order_id="ex-1",
        status=status,
        filled_qty=D(filled),
        avg_fill_price=D("100") if D(filled) > 0 else None,
        exchange_ts=T1,
    )


async def reserve(locked: LockedAccountState, intent_id: str = "i-1", cid: str = "c-1") -> None:
    source = intent(intent_id)
    await locked.register_approved(
        intent=source,
        decision=decision(source),
        client_order_id=cid,
        expected_revision=locked.revision,
        at=T0,
    )


async def account_with_order(
    store: CountingStore, *, submitted: bool = True
) -> InMemoryAccountState:
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", FLAT)
        await reserve(locked)
        if submitted:
            await locked.mark_submitting("c-1", at=T0)
    return account


def snapshot(account: InMemoryAccountState) -> tuple[Any, ...]:
    state = account._state
    return (
        state.revision,
        dict(state.orders),
        dict(state.notionals),
        dict(state.positions),
        dict(state.fills),
        dict(state.placements),
    )


async def poison(account: InMemoryAccountState, store: CountingStore) -> None:
    """Poison through an uncertain position commit (the most neutral mutation)."""
    store.fail_next()
    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty("ETHUSDT", D("1"))


# --- construction / trigger -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_account_is_not_poisoned_and_has_no_poison_parameter() -> None:
    account = InMemoryAccountState(account_scope_id=SCOPE, store=CountingStore())

    assert account.is_poisoned is False
    async with account.account_lock() as locked:
        assert locked.is_poisoned is False
        locked.ensure_mutations_allowed()
    assert set(inspect.signature(InMemoryAccountState).parameters) == {
        "account_scope_id",
        "store",
    }


def test_there_is_no_api_to_clear_the_poison() -> None:
    names = {name.lower() for name in dir(InMemoryAccountState) + dir(LockedAccountState)}

    for forbidden in ("clear_poison", "reset_poison", "resume", "unpoison", "reset"):
        assert forbidden not in names


@pytest.mark.asyncio
async def test_uncertain_commit_poisons_and_raises_the_original_error() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    before = snapshot(account)
    store.fail_next()

    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError) as caught:
            await locked.set_position_qty("BTCUSDT", D("3"))
        assert not isinstance(caught.value, AccountStatePoisonedError)
        assert locked.is_poisoned is True

    assert account.is_poisoned is True
    assert snapshot(account) == before  # the prepared state was never published
    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    assert durable.revision == before[0] + 1  # the store may be ahead of RAM
    assert durable.positions["BTCUSDT"].qty == D("3")


@pytest.mark.parametrize(
    ("label", "operation"),
    [
        ("approved", lambda locked: reserve(locked, "i-2", "c-2")),
        (
            "rejected",
            lambda locked: locked.register_rejected(
                intent=intent("i-2"),
                decision=decision(intent("i-2"), approved=False),
                expected_revision=locked.revision,
            ),
        ),
        ("submitting", lambda locked: locked.mark_submitting("c-1", at=T1)),
        ("fill", lambda locked: locked.apply_fill(fill(), at=T2)),
        ("ack", lambda locked: locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)),
        (
            "outcome",
            lambda locked: locked.record_submission_outcome(
                "c-1", SubmissionOutcome.AMBIGUOUS, at=T2
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_every_uncertain_mutation_poisons(
    label: str, operation: Callable[[LockedAccountState], Awaitable[Any]]
) -> None:
    store = CountingStore()
    account = await account_with_order(store, submitted=label != "submitting")
    before = snapshot(account)
    store.fail_next()

    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await operation(locked)

    assert account.is_poisoned is True
    assert snapshot(account) == before
    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    assert durable.revision >= before[0]  # rejected placements keep the revision


@pytest.mark.asyncio
async def test_uncertain_rejected_placement_poisons_although_the_revision_stays() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    source = intent("i-9")
    store.fail_next()

    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.register_rejected(
                intent=source,
                decision=decision(source, approved=False),
                expected_revision=locked.revision,
            )
        assert locked.placement("i-9") is None  # RAM does not know it

    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    assert "i-9" in durable.placements  # the store does
    assert account.is_poisoned is True


# --- non-triggering errors ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_definite_failure_does_not_poison_and_trading_continues() -> None:
    store = CountingStore()
    account = await account_with_order(store, submitted=False)
    store.fail_next(CommitFailure.DEFINITE)

    async with account.account_lock() as locked:
        with pytest.raises(StoreCommitError):
            await reserve(locked, "i-2", "c-2")
        assert locked.is_poisoned is False
        await reserve(locked, "i-2", "c-2")  # the next correct mutation succeeds
        await locked.set_position_qty("BTCUSDT", D("1"))

    assert account.is_poisoned is False
    assert (await account.order("c-2")) is not None


@pytest.mark.asyncio
async def test_conflict_does_not_poison() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    # Another (unsupported) writer moves the durable revision: the CAS fails.
    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    await store.inner.commit(
        AccountStateChange(
            account_scope_id=SCOPE,
            expected_revision=durable.revision,
            new_revision=durable.revision + 1,
        )
    )

    async with account.account_lock() as locked:
        with pytest.raises(StoreConflictError):
            await locked.set_position_qty("BTCUSDT", D("1"))
        assert locked.is_poisoned is False
        assert locked.position_qty("BTCUSDT") == FLAT


@pytest.mark.asyncio
async def test_store_validation_error_does_not_poison() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    store.validation_on.add(store.attempts + 1)

    async with account.account_lock() as locked:
        with pytest.raises(StoreValidationError):
            await locked.set_position_qty("BTCUSDT", D("1"))

    assert account.is_poisoned is False


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (
            lambda locked: locked.apply_exchange_state(report(S.PARTIALLY_FILLED, "4"), at=T2),
            MissingFillsError,
        ),
        (
            lambda locked: locked.apply_exchange_state(report(S.FILLED, "0"), at=T2),
            ExchangeStateMismatchError,
        ),
        (lambda locked: locked.apply_fill(fill(qty="11"), at=T2), FillApplicationError),
        (lambda locked: locked.mark_submitting("c-9", at=T2), Exception),
        (lambda locked: locked.set_position_qty("BTCUSDT", 1), DomainValidationError),
    ],
)
@pytest.mark.asyncio
async def test_domain_errors_do_not_poison(
    operation: Callable[[LockedAccountState], Awaitable[Any]], error: type[Exception]
) -> None:
    store = CountingStore()
    account = await account_with_order(store)

    async with account.account_lock() as locked:
        with pytest.raises(error):
            await operation(locked)

    assert account.is_poisoned is False


# --- after the poison -----------------------------------------------------------------------


BLOCKED_MUTATIONS: list[tuple[str, Callable[[LockedAccountState], Awaitable[Any]]]] = [
    ("register_approved", lambda locked: reserve(locked, "i-3", "c-3")),
    (
        "register_rejected",
        lambda locked: locked.register_rejected(
            intent=intent("i-3"),
            decision=decision(intent("i-3"), approved=False),
            expected_revision=locked.revision,
        ),
    ),
    ("set_position_qty", lambda locked: locked.set_position_qty("BTCUSDT", D("1"))),
    ("mark_submitting", lambda locked: locked.mark_submitting("c-1", at=T2)),
    ("record_ack", lambda locked: locked.record_ack("c-1", exchange_order_id="ex-9", at=T2)),
    (
        "record_submission_outcome",
        lambda locked: locked.record_submission_outcome("c-1", SubmissionOutcome.NOT_SENT, at=T2),
    ),
    ("apply_fill", lambda locked: locked.apply_fill(fill(), at=T2)),
    ("apply_exchange_state", lambda locked: locked.apply_exchange_state(report(S.OPEN), at=T2)),
    # Even an invalid call fails on the poison first (the check precedes validation).
    ("invalid_position", lambda locked: locked.set_position_qty("BTCUSDT", 1)),  # type: ignore[arg-type]
    ("unknown_order", lambda locked: locked.mark_submitting("c-404", at=T2)),
]


@pytest.mark.parametrize(
    ("label", "operation"), BLOCKED_MUTATIONS, ids=[m[0] for m in BLOCKED_MUTATIONS]
)
@pytest.mark.asyncio
async def test_every_mutation_fails_closed_without_calling_the_store(
    label: str, operation: Callable[[LockedAccountState], Awaitable[Any]]
) -> None:
    store = CountingStore()
    account = await account_with_order(store, submitted=label != "mark_submitting")
    await poison(account, store)
    attempts = store.attempts
    before = snapshot(account)

    async with account.account_lock() as locked:
        with pytest.raises(AccountStatePoisonedError, match="poisoned"):
            await operation(locked)

    assert store.attempts == attempts  # the store is never called again
    assert snapshot(account) == before


@pytest.mark.asyncio
async def test_reads_keep_the_last_confirmed_snapshot() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    async with account.account_lock() as locked:
        await locked.apply_fill(fill(), at=T1)
    before = snapshot(account)
    store.fail_next()
    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.apply_fill(fill("e-2", "6"), at=T2)

        assert locked.is_poisoned is True
        assert locked.revision == before[0]
        order = locked.order("c-1")
        assert order is not None
        assert (order.status, order.filled_qty) == (S.PARTIALLY_FILLED, D("4"))
        assert locked.position_qty("BTCUSDT") == D("4")
        assert locked.fill("e-1") == fill()
        assert locked.fill("e-2") is None
        assert locked.placement("i-1") is not None
        assert locked.replay_of(intent("i-1")) is not None
        assert len(locked.active_orders("BTCUSDT")) == 1
        assert locked.account_active_order_count() == 1

    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    assert durable.orders["c-1"].status is S.FILLED  # the store holds the full fill
    assert await account.position_qty("BTCUSDT") == D("4")  # async reads work too


# --- workflows ------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        return T1


class Ids:
    def __init__(self) -> None:
        self.calls = 0

    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        self.calls += 1
        return f"c-{self.calls + 10}"


class Client:
    def __init__(self) -> None:
        self.placed: list[OrderRequest] = []
        self.reads: list[OrderRef] = []
        self.answer: OrderUpdate | None = None

    async def place_order(self, order: OrderRequest) -> OrderAck:
        self.placed.append(order)
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="ex-1", exchange_ts=T1
        )

    async def cancel_order(self, order: OrderRef) -> None:
        raise AssertionError

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        self.reads.append(order)
        return self.answer

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise AssertionError


@pytest.mark.asyncio
async def test_uncertain_submitting_poisons_and_no_request_is_ever_sent() -> None:
    store = CountingStore()
    account = await account_with_order(store, submitted=False)
    client, clock = Client(), Clock()
    sender = OrderSubmitter(account_state=account, client=client, clock=clock)
    store.fail_next()

    with pytest.raises(StoreUncertainError):
        await sender.submit(client_order_id="c-1")

    assert client.placed == []
    assert account.is_poisoned is True
    assert (await account.order("c-1")).status is S.NEW  # type: ignore[union-attr]
    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    assert durable.orders["c-1"].status is S.SUBMITTING  # a reload would see it
    clock_calls, attempts = clock.calls, store.attempts

    with pytest.raises(AccountStatePoisonedError):
        await sender.submit(client_order_id="c-1")

    assert client.placed == []
    assert (clock.calls, store.attempts) == (clock_calls, attempts)


def policy() -> RiskPolicy:
    return RiskPolicy(
        policy_id="p",
        max_open_orders=None,
        symbols={
            "BTCUSDT": SymbolRiskLimits(
                max_order_qty=None, max_order_notional=None, max_position_qty=None
            )
        },
    )


@pytest.mark.asyncio
async def test_uncertain_reservation_poisons_and_the_coordinator_stops_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluations: list[str] = []

    def spy(**kwargs: Any) -> RiskDecision:
        evaluations.append(kwargs["intent"].intent_id)
        return evaluate(**kwargs)

    monkeypatch.setattr(placement_module, "evaluate", spy)
    store = CountingStore()
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", FLAT)
    ids, clock = Ids(), Clock()
    coord = PlacementCoordinator(
        account_state=account, policy=policy(), clock=clock, client_order_id_generator=ids
    )
    store.fail_next()

    with pytest.raises(StoreUncertainError):
        await coord.place(intent=intent(), trading_state=TradingState.RUNNING)

    assert account.is_poisoned is True
    assert await account.placement("i-1") is None  # RAM: no reservation
    durable = await store.load(account_scope_id=SCOPE)
    assert durable is not None
    assert "i-1" in durable.placements  # the store: placement + NEW
    counts = (len(evaluations), ids.calls, clock.calls, store.attempts)

    for source in (intent(), intent("i-2")):  # the same intent and a new one
        with pytest.raises(AccountStatePoisonedError):
            await coord.place(intent=source, trading_state=TradingState.RUNNING)

    assert (len(evaluations), ids.calls, clock.calls, store.attempts) == counts


@pytest.mark.asyncio
async def test_uncertain_exchange_state_poisons_and_reconcile_stops_before_the_network() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    async with account.account_lock() as locked:
        await locked.record_submission_outcome("c-1", SubmissionOutcome.AMBIGUOUS, at=T1)
    client = Client()
    client.answer = OrderUpdate(
        client_order_id="c-1",
        exchange_order_id="ex-1",
        status=S.OPEN,
        cum_filled_qty=D("0"),
        avg_fill_price=None,
        reject_reason=None,
        exchange_ts=T1,
    )
    reconciler = UnknownOrderReconciler(account_state=account, client=client, clock=Clock())
    store.fail_next()

    with pytest.raises(StoreUncertainError):
        await reconciler.reconcile(client_order_id="c-1")

    assert account.is_poisoned is True
    assert (await account.order("c-1")).status is S.UNKNOWN  # type: ignore[union-attr]
    reads, attempts = len(client.reads), store.attempts

    with pytest.raises(AccountStatePoisonedError):
        await reconciler.reconcile(client_order_id="c-1")

    assert (len(client.reads), store.attempts) == (reads, attempts)  # no get_order


# --- race -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_waiting_mutation_sees_the_poison_set_before_unlock() -> None:
    store = CountingStore()
    account = await account_with_order(store)
    store.block_on = store.attempts + 1
    store.fail_on[store.attempts + 1] = CommitFailure.UNCERTAIN

    async def task_a() -> None:
        async with account.account_lock() as locked:
            await locked.apply_fill(fill(), at=T2)

    async def task_b() -> None:
        async with account.account_lock() as locked:
            await locked.set_position_qty("BTCUSDT", D("9"))

    a = asyncio.create_task(task_a())
    await store.entered.wait()  # A is inside its commit, holding the lock
    b = asyncio.create_task(task_b())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not b.done()  # B waits for the lock
    attempts = store.attempts

    store.release.set()
    results = await asyncio.wait_for(asyncio.gather(a, b, return_exceptions=True), timeout=5)

    assert isinstance(results[0], StoreUncertainError)
    assert isinstance(results[1], AccountStatePoisonedError)
    assert store.attempts == attempts  # B never reached the store
    assert account.is_poisoned is True
