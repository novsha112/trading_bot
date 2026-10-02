"""Account state mutations: prepare -> durable commit -> publish.

Uses the in-memory reference store (and thin instrumented wrappers around it).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import OrderUpdate
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeNotSentError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.execution.account_state import (
    InMemoryAccountState,
    LockedAccountState,
    MissingFillsError,
)
from app.execution.models import ExchangeOrderState, SubmissionOutcome
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    PersistedOrderNotional,
    PersistedPosition,
    StoreCommitError,
    StoreUncertainError,
    StoreValidationError,
)
from app.execution.reconciliation import UnknownOrderReconciler
from app.execution.submitter import OrderSubmitter
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import (
    ExposureChange,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    SymbolRiskLimits,
    TradingState,
)
from app.services.placement import PlacementCoordinator

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
SCOPE = "acct-1"
FLAT = D("0")


class RecordingStore:
    """Delegates to the reference store and records every attempted change, and
    whether the account lock was held during the commit."""

    def __init__(self, account_ref: list[InMemoryAccountState] | None = None) -> None:
        self.inner = InMemoryAccountStateStore()
        self.changes: list[AccountStateChange] = []
        self.lock_held: list[bool] = []
        self.account_ref = account_ref

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        if self.account_ref:
            self.lock_held.append(self.account_ref[0]._lock.locked())
        await self.inner.commit(change)


class BlockingStore(RecordingStore):
    """Holds every commit until ``release`` is set; optionally fails it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = False

    async def commit(self, change: AccountStateChange) -> None:
        self.entered.set()
        await self.release.wait()
        if self.fail:
            raise StoreCommitError("blocked commit failed")
        await super().commit(change)


class FailingOnNthCommitStore(RecordingStore):
    """Injects ``failure`` into the ``n``-th commit (1-based) of the reference store."""

    def __init__(self, n: int, failure: CommitFailure) -> None:
        super().__init__()
        self.n = n
        self.failure = failure

    async def commit(self, change: AccountStateChange) -> None:
        if len(self.changes) + 1 == self.n:
            self.inner.inject_commit_failure(self.failure)
        await super().commit(change)


class ValidationFailingStore(RecordingStore):
    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        raise StoreValidationError("store rejects the change")


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


def fill(exec_id: str = "e-1", qty: str = "4", **overrides: Any) -> Fill:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": "ex-1",
        "client_order_id": "c-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D("100"),
        "qty": D(qty),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "exchange_ts": T1,
    }
    return Fill(**{**values, **overrides})


def store_error(failure: CommitFailure) -> type[Exception]:
    return StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError


def make_account(store: Any) -> InMemoryAccountState:
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    if isinstance(store, RecordingStore) and store.account_ref is not None:
        store.account_ref.append(account)
    return account


async def reserve(locked: LockedAccountState, intent_id: str = "i-1", cid: str = "c-1") -> None:
    source = intent(intent_id)
    await locked.register_approved(
        intent=source,
        decision=decision(source),
        client_order_id=cid,
        expected_revision=locked.revision,
        at=T0,
    )


async def prepared_account(
    store: Any, *, submitted: bool = True, position: Decimal | None = FLAT
) -> InMemoryAccountState:
    """Order c-1 reserved (and SUBMITTING), BTCUSDT position known."""
    account = make_account(store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", position)
        await reserve(locked)
        if submitted:
            await locked.mark_submitting("c-1", at=T0)
    return account


def ram(account: InMemoryAccountState) -> tuple[Any, ...]:
    state = account._state
    return (
        state.revision,
        dict(state.orders),
        dict(state.notionals),
        dict(state.positions),
        dict(state.fills),
        dict(state.placements),
    )


async def durable(store: Any) -> tuple[Any, ...] | None:
    state = await store.load(account_scope_id=SCOPE)
    if state is None:
        return None
    return (
        state.revision,
        dict(state.orders),
        dict(state.notionals),
        {s: p.qty for s, p in state.positions.items() if p.known},
        dict(state.fills),
        dict(state.placements),
    )


async def assert_ram_matches_store(account: InMemoryAccountState, store: Any) -> None:
    assert await durable(store) == ram(account)


# --- construction ---------------------------------------------------------------------------


def test_account_state_requires_an_explicit_store_and_scope() -> None:
    with pytest.raises(TypeError):
        InMemoryAccountState()  # type: ignore[call-arg]
    with pytest.raises(Exception, match="store"):
        InMemoryAccountState(account_scope_id=SCOPE, store=object())  # type: ignore[arg-type]

    account = InMemoryAccountState(account_scope_id=SCOPE, store=InMemoryAccountStateStore())
    assert account.account_scope_id == SCOPE


@pytest.mark.asyncio
async def test_construction_does_not_load_or_commit() -> None:
    store = RecordingStore()
    make_account(store)

    assert store.changes == []
    assert await durable(store) is None


# --- change sets ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_operation_commits_its_minimal_change_and_ram_matches_the_store() -> None:
    store = RecordingStore()
    account = make_account(store)

    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
        await reserve(locked)
        source = intent("i-2")
        await locked.register_rejected(
            intent=source, decision=decision(source, approved=False), expected_revision=2
        )
        await locked.mark_submitting("c-1", at=T1)
        await locked.record_ack("c-1", exchange_order_id="ex-1", at=T1)
        await locked.apply_fill(fill(), at=T2)

    position, approved, rejected, submitting, ack, filled = store.changes
    assert (position.expected_revision, position.new_revision) == (0, 1)
    assert position.position_writes == (
        PersistedPosition(symbol="BTCUSDT", known=True, qty=D("0")),
    )
    assert (approved.expected_revision, approved.new_revision) == (1, 2)
    assert [p.intent_id for p in approved.placement_writes] == ["i-1"]
    assert [o.status for o in approved.order_writes] == [S.NEW]
    assert approved.notional_writes == (
        PersistedOrderNotional(client_order_id="c-1", filled_notional=D("0")),
    )
    assert (approved.fill_writes, approved.position_writes) == ((), ())
    assert (rejected.expected_revision, rejected.new_revision) == (2, 2)
    assert [p.client_order_id for p in rejected.placement_writes] == [None]
    assert (rejected.order_writes, rejected.notional_writes) == ((), ())
    assert (submitting.new_revision, [o.status for o in submitting.order_writes]) == (
        3,
        [S.SUBMITTING],
    )
    assert [o.exchange_order_id for o in ack.order_writes] == ["ex-1"]
    assert ack.new_revision == 4
    # The fill is one change: fill, order, notional and position together.
    assert filled.fill_writes == (fill(),)
    assert [(o.status, o.filled_qty) for o in filled.order_writes] == [(S.PARTIALLY_FILLED, D("4"))]
    assert filled.notional_writes == (
        PersistedOrderNotional(client_order_id="c-1", filled_notional=D("400")),
    )
    assert filled.position_writes == (PersistedPosition(symbol="BTCUSDT", known=True, qty=D("4")),)
    assert (filled.expected_revision, filled.new_revision) == (4, 5)
    assert all(change.account_scope_id == SCOPE for change in store.changes)
    await assert_ram_matches_store(account, store)


@pytest.mark.asyncio
async def test_clearing_a_position_writes_an_explicit_unknown_marker() -> None:
    store = RecordingStore()
    account = make_account(store)

    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("2"))
        await locked.set_position_qty("BTCUSDT", None)

    assert store.changes[-1].position_writes == (
        PersistedPosition(symbol="BTCUSDT", known=False, qty=None),
    )
    loaded = await store.load(account_scope_id=SCOPE)
    assert loaded is not None
    assert loaded.positions["BTCUSDT"] == PersistedPosition(symbol="BTCUSDT", known=False, qty=None)


@pytest.mark.asyncio
async def test_fill_with_unknown_position_writes_no_position() -> None:
    store = RecordingStore()
    account = await prepared_account(store, position=None)

    async with account.account_lock() as locked:
        await locked.apply_fill(fill(), at=T2)

    assert store.changes[-1].position_writes == ()


@pytest.mark.asyncio
async def test_outcome_and_report_transitions_commit_one_order_write() -> None:
    store = RecordingStore()
    account = await prepared_account(store)
    async with account.account_lock() as locked:
        await locked.record_submission_outcome("c-1", SubmissionOutcome.AMBIGUOUS, at=T1)
        await locked.apply_exchange_state(
            ExchangeOrderState(
                client_order_id="c-1",
                exchange_order_id="ex-1",
                status=S.OPEN,
                filled_qty=D("0"),
                avg_fill_price=None,
                exchange_ts=T1,
            ),
            at=T2,
        )

    unknown, opened = store.changes[-2:]
    assert [o.status for o in unknown.order_writes] == [S.UNKNOWN]
    assert [o.status for o in opened.order_writes] == [S.OPEN]
    assert opened.new_revision == unknown.new_revision + 1
    await assert_ram_matches_store(account, store)


# --- no-ops ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_operations_do_not_commit() -> None:
    store = RecordingStore()
    account = await prepared_account(store)
    async with account.account_lock() as locked:
        await locked.apply_fill(fill(), at=T2)
    committed = len(store.changes)
    revision = account._state.revision

    async with account.account_lock() as locked:
        await reserve(locked)  # replay of the same intent
        await locked.apply_fill(fill(), at=T2)  # identical fill
        await locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)  # known id
        await locked.record_submission_outcome(  # superseded by the fill
            "c-1", SubmissionOutcome.AMBIGUOUS, at=T2
        )
        await locked.apply_exchange_state(  # confirming report
            ExchangeOrderState(
                client_order_id="c-1",
                exchange_order_id="ex-1",
                status=S.PARTIALLY_FILLED,
                filled_qty=D("4"),
                avg_fill_price=D("100"),
                exchange_ts=T1,
            ),
            at=T2,
        )
        await locked.set_position_qty("BTCUSDT", D("4"))  # current value
        assert locked.replay_of(intent("i-1")) is not None  # read-only

    assert len(store.changes) == committed
    assert account._state.revision == revision


@pytest.mark.asyncio
async def test_rejected_errors_before_the_commit_do_not_call_the_store() -> None:
    store = RecordingStore()
    account = await prepared_account(store)
    committed = len(store.changes)
    before = ram(account)

    async with account.account_lock() as locked:
        with pytest.raises(MissingFillsError):
            await locked.apply_exchange_state(
                ExchangeOrderState(
                    client_order_id="c-1",
                    exchange_order_id="ex-1",
                    status=S.PARTIALLY_FILLED,
                    filled_qty=D("4"),
                    avg_fill_price=D("100"),
                    exchange_ts=T1,
                ),
                at=T2,
            )
        with pytest.raises(Exception, match="overfills"):
            await locked.apply_fill(fill(qty="11"), at=T2)

    assert len(store.changes) == committed
    assert ram(account) == before


# --- failure injection ----------------------------------------------------------------------


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_reservation_failure_leaves_ram_unchanged(failure: CommitFailure) -> None:
    store = InMemoryAccountStateStore()
    account = make_account(store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
    before_ram, before_store = ram(account), await durable(store)
    store.inject_commit_failure(failure)

    async with account.account_lock() as locked:
        with pytest.raises(store_error(failure)):
            await reserve(locked)

    assert ram(account) == before_ram
    after_store = await durable(store)
    if failure is CommitFailure.DEFINITE:
        assert after_store == before_store
    else:  # durable is ahead of RAM: the reservation exists only in the store
        assert after_store is not None
        assert "c-1" in after_store[1]
        assert after_store[0] == before_ram[0] + 1


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_rejected_placement_is_also_durable_first(failure: CommitFailure) -> None:
    store = InMemoryAccountStateStore()
    account = make_account(store)
    source = intent("i-9")
    store.inject_commit_failure(failure)

    async with account.account_lock() as locked:
        with pytest.raises(
            StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError
        ):
            await locked.register_rejected(
                intent=source, decision=decision(source, approved=False), expected_revision=0
            )
        assert locked.placement("i-9") is None


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_fill_failure_publishes_nothing(failure: CommitFailure) -> None:
    store = InMemoryAccountStateStore()
    account = await prepared_account(store)
    before_ram, before_store = ram(account), await durable(store)
    store.inject_commit_failure(failure)

    async with account.account_lock() as locked:
        with pytest.raises(
            StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError
        ):
            await locked.apply_fill(fill(), at=T2)
        # Nothing of the fill is visible: order, position, fill, notional, revision.
        assert locked.order("c-1").filled_qty == D("0")  # type: ignore[union-attr]
        assert locked.position_qty("BTCUSDT") == D("0")
        assert locked.fill("e-1") is None

    assert ram(account) == before_ram
    after_store = await durable(store)
    if failure is CommitFailure.DEFINITE:
        assert after_store == before_store
    else:  # the store holds the complete fill transaction
        assert after_store is not None
        revision, orders, notionals, positions, fills, _ = after_store
        assert revision == before_ram[0] + 1
        assert orders["c-1"].filled_qty == D("4")
        assert notionals["c-1"] == D("400")
        assert positions["BTCUSDT"] == D("4")
        assert fills["e-1"] == fill()


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_exchange_update_failure_publishes_nothing(failure: CommitFailure) -> None:
    store = InMemoryAccountStateStore()
    account = await prepared_account(store)
    async with account.account_lock() as locked:
        await locked.record_submission_outcome("c-1", SubmissionOutcome.AMBIGUOUS, at=T1)
    before_ram = ram(account)
    store.inject_commit_failure(failure)

    async with account.account_lock() as locked:
        with pytest.raises(
            StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError
        ):
            await locked.apply_exchange_state(
                ExchangeOrderState(
                    client_order_id="c-1",
                    exchange_order_id="ex-1",
                    status=S.CANCELED,
                    filled_qty=D("0"),
                    avg_fill_price=None,
                    exchange_ts=T1,
                ),
                at=T2,
            )

    assert ram(account) == before_ram
    loaded = await store.load(account_scope_id=SCOPE)
    assert loaded is not None
    expected = S.UNKNOWN if failure is CommitFailure.DEFINITE else S.CANCELED
    assert loaded.orders["c-1"].status is expected


@pytest.mark.asyncio
async def test_store_validation_error_propagates_as_an_invariant_bug() -> None:
    store = ValidationFailingStore()
    account = make_account(store)

    async with account.account_lock() as locked:
        with pytest.raises(StoreValidationError):
            await locked.set_position_qty("BTCUSDT", D("1"))
        assert locked.position_qty("BTCUSDT") is None
        assert locked.revision == 0


# --- read visibility ------------------------------------------------------------------------


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.asyncio
async def test_prepared_state_is_invisible_until_the_commit_returns(fail: bool) -> None:
    store = BlockingStore()
    account = make_account(store)
    store.fail = fail
    seen: dict[str, Any] = {}

    async def mutate() -> None:
        async with account.account_lock() as locked:
            await locked.set_position_qty("BTCUSDT", D("5"))

    async def read() -> None:
        async with account.account_lock() as locked:
            seen["position"] = locked.position_qty("BTCUSDT")
            seen["revision"] = locked.revision

    writer = asyncio.create_task(mutate())
    await store.entered.wait()  # the commit is in progress, the lock is held
    reader = asyncio.create_task(read())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not reader.done()  # the reader waits for the lock
    assert account._state.positions == {}  # nothing published yet

    store.release.set()
    results = await asyncio.wait_for(asyncio.gather(writer, reader, return_exceptions=True), 5)

    if fail:
        assert isinstance(results[0], StoreCommitError)
        assert seen == {"position": None, "revision": 0}
    else:
        assert results[0] is None
        assert seen == {"position": D("5"), "revision": 1}


@pytest.mark.asyncio
async def test_store_commit_runs_under_the_account_lock() -> None:
    accounts: list[InMemoryAccountState] = []
    store = RecordingStore(accounts)
    account = await prepared_account(store)
    async with account.account_lock() as locked:
        await locked.apply_fill(fill(), at=T2)

    assert store.lock_held
    assert all(store.lock_held)
    assert account is accounts[0]


# --- submitter ------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.times = iter([T1, T2, T2 + timedelta(seconds=1)])

    def now(self) -> datetime:
        return next(self.times)


class Client:
    """TradingClient double that records the account lock state at each call."""

    def __init__(
        self, account: InMemoryAccountState, *, error: BaseException | None = None
    ) -> None:
        self.account = account
        self.error = error
        self.placed: list[OrderRequest] = []
        self.reads: list[OrderRef] = []
        self.lock_held: list[bool] = []
        self.answer: OrderUpdate | None = None

    async def place_order(self, order: OrderRequest) -> OrderAck:
        self.placed.append(order)
        self.lock_held.append(self.account._lock.locked())
        if self.error is not None:
            raise self.error
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="ex-1", exchange_ts=T1
        )

    async def cancel_order(self, order: OrderRef) -> None:
        raise AssertionError

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        self.reads.append(order)
        self.lock_held.append(self.account._lock.locked())
        return self.answer

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise AssertionError


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_submitting_commit_failure_forbids_the_network(failure: CommitFailure) -> None:
    store = InMemoryAccountStateStore()
    account = await prepared_account(store, submitted=False)
    client = Client(account)
    store.inject_commit_failure(failure)

    with pytest.raises(store_error(failure)):
        await OrderSubmitter(account_state=account, client=client, clock=Clock()).submit(
            client_order_id="c-1"
        )

    assert client.placed == []  # write-ahead: no durable SUBMITTING, no request
    assert (await account.order("c-1")).status is S.NEW  # type: ignore[union-attr]
    loaded = await store.load(account_scope_id=SCOPE)
    assert loaded is not None
    # Uncertain: the store already holds SUBMITTING while RAM still says NEW; a
    # reload / restart would see SUBMITTING (the caller must stop and reload).
    expected = S.NEW if failure is CommitFailure.DEFINITE else S.SUBMITTING
    assert loaded.orders["c-1"].status is expected


@pytest.mark.asyncio
async def test_network_is_outside_the_lock_and_the_store_inside() -> None:
    accounts: list[InMemoryAccountState] = []
    store = RecordingStore(accounts)
    account = await prepared_account(store, submitted=False)
    client = Client(account)

    order = await OrderSubmitter(account_state=account, client=client, clock=Clock()).submit(
        client_order_id="c-1"
    )

    assert (order.status, order.exchange_order_id) == (S.SUBMITTING, "ex-1")
    assert client.lock_held == [False]
    assert all(store.lock_held)
    await assert_ram_matches_store(account, store)


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_ack_persistence_failure_never_resends(failure: CommitFailure) -> None:
    # Commits: position, reservation, SUBMITTING, then the ack (the 4th) fails.
    store = FailingOnNthCommitStore(4, failure)
    account = await prepared_account(store, submitted=False)
    client = Client(account)
    sender = OrderSubmitter(account_state=account, client=client, clock=Clock())

    with pytest.raises(store_error(failure)):
        await sender.submit(client_order_id="c-1")

    assert len(client.placed) == 1
    order = await account.order("c-1")
    assert order is not None
    assert (order.status, order.exchange_order_id) == (S.SUBMITTING, None)  # RAM: previous state
    with pytest.raises(Exception, match="submitting"):
        await sender.submit(client_order_id="c-1")
    assert len(client.placed) == 1  # never sent again


@pytest.mark.parametrize(
    ("transport", "failure"),
    [
        (ExchangeNotSentError("down"), CommitFailure.DEFINITE),
        (ExchangeAmbiguousResultError("timeout"), CommitFailure.DEFINITE),
        (ExchangeAmbiguousResultError("timeout"), CommitFailure.UNCERTAIN),
    ],
)
@pytest.mark.asyncio
async def test_outcome_persistence_failure_wins_and_keeps_the_transport_error(
    transport: BaseException, failure: CommitFailure
) -> None:
    store = FailingOnNthCommitStore(4, failure)  # the outcome commit fails
    account = await prepared_account(store, submitted=False)
    client = Client(account, error=transport)

    with pytest.raises(store_error(failure)) as caught:
        await OrderSubmitter(account_state=account, client=client, clock=Clock()).submit(
            client_order_id="c-1"
        )

    assert caught.value.__context__ is transport
    assert len(client.placed) == 1
    assert (await account.order("c-1")).status is S.SUBMITTING  # type: ignore[union-attr]


# --- reconciler / coordinator ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciler_reads_outside_the_lock_and_commits_inside() -> None:
    accounts: list[InMemoryAccountState] = []
    store = RecordingStore(accounts)
    account = await prepared_account(store)
    async with account.account_lock() as locked:
        await locked.record_submission_outcome("c-1", SubmissionOutcome.AMBIGUOUS, at=T1)
    client = Client(account)
    client.answer = OrderUpdate(
        client_order_id="c-1",
        exchange_order_id="ex-1",
        status=S.OPEN,
        cum_filled_qty=D("0"),
        avg_fill_price=None,
        reject_reason=None,
        exchange_ts=T1,
    )
    store.lock_held.clear()

    order = await UnknownOrderReconciler(
        account_state=account, client=client, clock=Clock()
    ).reconcile(client_order_id="c-1")

    assert order.status is S.OPEN
    assert client.lock_held == [False]
    assert store.lock_held == [True]
    await assert_ram_matches_store(account, store)


class Ids:
    def __init__(self) -> None:
        self.calls = 0

    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        self.calls += 1
        return f"c-{self.calls}"


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_coordinator_store_failure_leaves_no_reservation(failure: CommitFailure) -> None:
    store = InMemoryAccountStateStore()
    account = make_account(store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
    ids = Ids()
    coord = PlacementCoordinator(
        account_state=account,
        policy=RiskPolicy(
            policy_id="p",
            max_open_orders=None,
            symbols={
                "BTCUSDT": SymbolRiskLimits(
                    max_order_qty=None, max_order_notional=None, max_position_qty=None
                )
            },
        ),
        clock=Clock(),
        client_order_id_generator=ids,
    )
    store.inject_commit_failure(failure)

    with pytest.raises(
        StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError
    ):
        await coord.place(intent=intent(), trading_state=TradingState.RUNNING)

    assert ids.calls == 1  # the generated id is simply unused
    assert await account.placement("i-1") is None
    assert await account.account_active_order_count() == 0
    assert await account.revision() == 1
