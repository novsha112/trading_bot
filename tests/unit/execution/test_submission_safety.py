"""OrderSubmitter safety gate: the current effective trading state is checked
before the write-ahead marker and again right before the network send."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import Order, OrderUpdate
from app.exchanges.errors import ExchangeError
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.execution.account_state import (
    AccountStateError,
    AccountStatePoisonedError,
    InMemoryAccountState,
)
from app.execution.models import (
    ExchangeOrderState,
    SafetyBlockRecord,
    SubmissionBlockStage,
    SubmissionOutcome,
)
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    StoreCommitError,
    StoreConflictError,
    StoreUncertainError,
    StoreValidationError,
)
from app.execution.reconciliation import UnknownOrderReconciler
from app.execution.safety import SafetyController, SafetyStateError, submission_allowed
from app.execution.submitter import OrderSubmissionBlockedError, OrderSubmitter
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import (
    ExposureChange,
    RiskDecision,
    RiskPolicy,
    SymbolRiskLimits,
    TradingState,
)
from app.services import placement as placement_module
from app.services.placement import PlacementCoordinator

D = Decimal
S = OrderStatus
RUNNING = TradingState.RUNNING
REDUCE_ONLY = TradingState.REDUCE_ONLY
PAUSED = TradingState.PAUSED
HALTED = TradingState.HALTED
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
T3 = T0 + timedelta(seconds=3)
SCOPE = "acct-1"


# --- doubles --------------------------------------------------------------------------------


class Clock:
    def __init__(self, *times: datetime) -> None:
        self.times = list(times) or [T1, T2, T3]
        self.calls = 0

    def now(self) -> datetime:
        value = self.times[min(self.calls, len(self.times) - 1)]
        self.calls += 1
        return value


class FirstOnlyClock:
    """One valid read, then failures (the fallback path of later reads)."""

    def __init__(self) -> None:
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("clock unavailable")
        return T1


class Client:
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []
        self.seen_effective: list[TradingState] = []
        self.safety: SafetyController | None = None
        self.events: list[str] | None = None

    async def place_order(self, order: OrderRequest) -> OrderAck:
        # Runs synchronously up to here when awaited: records what was true at send.
        if self.events is not None:
            self.events.append("send")
        if self.safety is not None:
            self.seen_effective.append(self.safety.effective_state)
        self.requests.append(order)
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="ex-1", exchange_ts=T1
        )

    async def cancel_order(self, order: OrderRef) -> None:
        raise AssertionError("cancel_order must not be called")

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        raise AssertionError("get_order must not be called")

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise AssertionError("get_open_orders must not be called")


class ScriptedStore:
    """The reference store; the n-th commit (1-based, counted from ``arm``) can
    block until released, and any armed commit can fail."""

    def __init__(self) -> None:
        self.inner = InMemoryAccountStateStore()
        self.commits = 0
        self.block_at: int | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failures: dict[int, Exception | CommitFailure] = {}

    def arm(self) -> None:
        self.commits = 0

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.commits += 1
        number = self.commits
        if number == self.block_at:
            self.entered.set()
            await self.release.wait()
        failure = self.failures.get(number)
        if isinstance(failure, Exception):
            raise failure
        if isinstance(failure, CommitFailure):
            self.inner.inject_commit_failure(failure)
        await self.inner.commit(change)


def intent(intent_id: str = "i-1", **overrides: Any) -> PlaceOrderIntent:
    values: dict[str, Any] = {
        "intent_id": intent_id,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100"),
        "qty": D("1"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "tag": None,
        "created_at": T0,
    }
    return PlaceOrderIntent(**{**values, **overrides})


async def reserved(
    store: Any = None, *, reduce_only: bool = False
) -> tuple[InMemoryAccountState, Any]:
    store = ScriptedStore() if store is None else store
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    source = intent(reduce_only=reduce_only)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
        await locked.register_approved(
            intent=source,
            decision=RiskDecision(
                intent_id=source.intent_id,
                snapshot_id="s",
                policy_id="p",
                approved=True,
                reasons=(),
                exposure=ExposureChange(
                    reducing_qty=D("0"),
                    increasing_qty=D("1"),
                    worst_long_qty=D("1"),
                    worst_short_qty=D("0"),
                ),
            ),
            client_order_id="c-1",
            expected_revision=locked.revision,
            at=T0,
        )
    if isinstance(store, ScriptedStore):
        store.arm()
    return account, store


def safety_for(
    account: InMemoryAccountState, requested: TradingState = RUNNING, *, ready: bool = True
) -> SafetyController:
    safety = SafetyController(account_state=account)
    if ready:
        safety.mark_hydrated()
        safety.mark_exchange_reconciled()
    safety.request_state(requested)
    return safety


def sender(
    account: InMemoryAccountState, safety: SafetyController, client: Client, clock: Any = None
) -> OrderSubmitter:
    return OrderSubmitter(
        account_state=account,
        safety=safety,
        client=client,
        clock=Clock() if clock is None else clock,
    )


async def order_of(account: InMemoryAccountState) -> Order:
    order = await account.order("c-1")
    assert order is not None
    return order


async def durable_order(store: ScriptedStore) -> Order:
    loaded = await store.load(account_scope_id=SCOPE)
    assert loaded is not None
    return loaded.orders["c-1"]


# --- construction ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submitter_requires_the_controller_of_the_same_account() -> None:
    account, _ = await reserved()
    other, _ = await reserved()
    client, clock = Client(), Clock()

    OrderSubmitter(account_state=account, safety=safety_for(account), client=client, clock=clock)
    with pytest.raises(DomainValidationError, match="same account_state"):
        OrderSubmitter(account_state=account, safety=safety_for(other), client=client, clock=clock)
    with pytest.raises(DomainValidationError, match="SafetyController"):
        OrderSubmitter(account_state=account, safety=object(), client=client, clock=clock)  # type: ignore[arg-type]

    class Proxy(SafetyController):
        __slots__ = ()

    with pytest.raises(DomainValidationError, match="SafetyController"):
        OrderSubmitter(
            account_state=account, safety=Proxy(account_state=account), client=client, clock=clock
        )
    with pytest.raises(TypeError):
        OrderSubmitter(account_state=account, client=client, clock=clock)  # type: ignore[call-arg]


# --- permission matrix ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "reduce_only", "allowed"),
    [
        (RUNNING, False, True),
        (RUNNING, True, True),
        (REDUCE_ONLY, False, False),
        (REDUCE_ONLY, True, True),
        (PAUSED, False, False),
        (PAUSED, True, False),
        (HALTED, False, False),
        (HALTED, True, False),
    ],
)
def test_submission_permission_matrix(
    state: TradingState, reduce_only: bool, allowed: bool
) -> None:
    assert submission_allowed(state, reduce_only=reduce_only) is allowed


@pytest.mark.parametrize(("state", "reduce_only"), [("running", False), (RUNNING, 1), (None, True)])
def test_submission_permission_validates_inputs(state: Any, reduce_only: Any) -> None:
    with pytest.raises(DomainValidationError):
        submission_allowed(state, reduce_only=reduce_only)


# --- initial check --------------------------------------------------------------------------


DENIED_FIRST: dict[str, tuple[TradingState, bool, bool, TradingState]] = {
    # name: (requested, gates ready, reduce_only, expected effective)
    "default paused": (PAUSED, False, False, PAUSED),
    "running, recovery incomplete": (RUNNING, False, False, PAUSED),
    "halted": (HALTED, True, False, HALTED),
    "halted, reduce-only": (HALTED, True, True, HALTED),
    "paused, reduce-only": (PAUSED, True, True, PAUSED),
    "reduce-only state, normal order": (REDUCE_ONLY, True, False, REDUCE_ONLY),
}


@pytest.mark.parametrize("case", sorted(DENIED_FIRST))
@pytest.mark.asyncio
async def test_denied_before_write_ahead_fails_the_new_order_durably(case: str) -> None:
    requested, ready, reduce_only, effective = DENIED_FIRST[case]
    account, store = await reserved(reduce_only=reduce_only)
    safety = SafetyController(account_state=account)
    if case != "default paused":
        safety = safety_for(account, requested, ready=ready)
    client, clock = Client(), Clock()
    before = await order_of(account)
    revision = await account.revision()

    with pytest.raises(OrderSubmissionBlockedError) as raised:
        await sender(account, safety, client, clock).submit(client_order_id="c-1")

    error = raised.value
    assert (error.client_order_id, error.effective_state, error.reduce_only, error.stage) == (
        "c-1",
        effective,
        reduce_only,
        SubmissionBlockStage.BEFORE_WRITE_AHEAD,
    )
    assert client.requests == []
    assert clock.calls == 1
    failed = await order_of(account)
    assert failed.status is S.FAILED
    assert failed.version == before.version + 1
    assert failed.updated_at == T1
    assert failed.exchange_order_id is None
    assert await account.revision() == revision + 1
    block = await account.safety_block("c-1")
    assert block == SafetyBlockRecord(
        client_order_id="c-1",
        effective_state=effective,
        stage=SubmissionBlockStage.BEFORE_WRITE_AHEAD,
        blocked_at=T1,
    )
    # Durable: the store has the FAILED order and its reason.
    loaded = await store.load(account_scope_id=SCOPE)
    assert loaded is not None
    assert loaded.orders["c-1"] == failed
    assert loaded.safety_blocks["c-1"] == block
    assert store.commits == 1


@pytest.mark.parametrize(
    ("requested", "reduce_only"),
    [(RUNNING, False), (RUNNING, True), (REDUCE_ONLY, True)],
)
@pytest.mark.asyncio
async def test_allowed_submissions_are_sent(requested: TradingState, reduce_only: bool) -> None:
    account, _ = await reserved(reduce_only=reduce_only)
    client = Client()

    order = await sender(account, safety_for(account, requested), client).submit(
        client_order_id="c-1"
    )

    assert [r.client_order_id for r in client.requests] == ["c-1"]
    assert order.status is S.SUBMITTING
    assert order.exchange_order_id == "ex-1"
    assert await account.safety_block("c-1") is None


@pytest.mark.asyncio
async def test_the_block_survives_a_restart() -> None:
    account, store = await reserved()
    with pytest.raises(OrderSubmissionBlockedError):
        await sender(account, safety_for(account, HALTED), Client()).submit(client_order_id="c-1")

    restarted = await InMemoryAccountState.hydrate(
        account_scope_id=SCOPE, store=store, clock=Clock()
    )

    assert (await order_of(restarted)).status is S.FAILED
    block = await restarted.safety_block("c-1")
    assert block is not None
    assert (block.effective_state, block.stage) == (HALTED, SubmissionBlockStage.BEFORE_WRITE_AHEAD)


@pytest.mark.asyncio
async def test_a_blocked_order_is_never_submitted_again() -> None:
    account, _ = await reserved()
    safety = safety_for(account, PAUSED)
    client = Client()
    with pytest.raises(OrderSubmissionBlockedError):
        await sender(account, safety, client).submit(client_order_id="c-1")

    safety.request_state(RUNNING)
    with pytest.raises(AccountStateError, match="failed: not sent again"):
        await sender(account, safety, client).submit(client_order_id="c-1")
    assert client.requests == []


def test_blocked_error_is_a_local_safety_decision() -> None:
    assert issubclass(OrderSubmissionBlockedError, SafetyStateError)
    assert not issubclass(OrderSubmissionBlockedError, ExchangeError)


# --- race: state changes during the durable SUBMITTING commit ---------------------------------


async def run_with_state_change_during_marker(
    requested_after: TradingState, *, reduce_only: bool = False
) -> tuple[InMemoryAccountState, ScriptedStore, Client, Any]:
    account, store = await reserved(reduce_only=reduce_only)
    store.block_at = 1  # the SUBMITTING marker
    safety = safety_for(account, RUNNING)
    client = Client()
    task = asyncio.create_task(sender(account, safety, client).submit(client_order_id="c-1"))
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    safety.request_state(requested_after)  # the operator acts while the commit waits
    store.release.set()
    try:
        result: Any = await asyncio.wait_for(task, timeout=5)
    except Exception as error:  # inspected by the caller
        result = error
    return account, store, client, result


@pytest.mark.parametrize("after", [PAUSED, HALTED, REDUCE_ONLY])
@pytest.mark.asyncio
async def test_final_check_fails_submitting_without_sending(after: TradingState) -> None:
    account, store, client, result = await run_with_state_change_during_marker(after)

    assert isinstance(result, OrderSubmissionBlockedError)
    assert result.stage is SubmissionBlockStage.BEFORE_SEND
    assert result.effective_state is after
    assert client.requests == []
    failed = await order_of(account)
    assert failed.status is S.FAILED
    assert failed.version == 2  # NEW -> SUBMITTING -> FAILED
    assert await durable_order(store) == failed
    block = await account.safety_block("c-1")
    assert block is not None
    assert (block.stage, block.effective_state) == (SubmissionBlockStage.BEFORE_SEND, after)
    assert store.commits == 2  # marker + FAILED, nothing else


@pytest.mark.asyncio
async def test_final_check_still_sends_a_reduce_only_order_under_reduce_only() -> None:
    account, _, client, result = await run_with_state_change_during_marker(
        REDUCE_ONLY, reduce_only=True
    )

    assert isinstance(result, Order)
    assert [r.client_order_id for r in client.requests] == ["c-1"]
    assert await account.safety_block("c-1") is None


@pytest.mark.asyncio
async def test_final_check_has_no_await_before_the_send(monkeypatch: pytest.MonkeyPatch) -> None:
    account, _ = await reserved()
    safety = safety_for(account, RUNNING)
    client = Client()
    client.safety = safety
    events: list[str] = []
    client.events = events
    real_snapshot = SafetyController.snapshot

    def snapshot(self: SafetyController) -> Any:
        events.append("safety")
        if events.count("safety") == 2:
            # Runs at the first suspension point after the final check: if the
            # submitter awaited anything before place_order, the send would see PAUSED.
            asyncio.get_running_loop().call_soon(self.request_state, PAUSED)
        return real_snapshot(self)

    monkeypatch.setattr(SafetyController, "snapshot", snapshot)

    await sender(account, safety, client).submit(client_order_id="c-1")

    # first check, final check, the send (and the client's own read inside it)
    assert events == ["safety", "safety", "send", "safety"]
    assert client.seen_effective == [RUNNING]  # no await between the check and the send
    assert safety.effective_state is PAUSED  # the callback did run afterwards
    assert len(client.requests) == 1


# --- persistence failures -------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        CommitFailure.DEFINITE,
        StoreConflictError("conflict"),
        StoreValidationError("invalid"),
    ],
    ids=["definite", "conflict", "validation"],
)
@pytest.mark.asyncio
async def test_initial_block_persistence_failure_keeps_new_and_sends_nothing(
    failure: Any,
) -> None:
    account, store = await reserved()
    store.failures[1] = failure
    client = Client()
    expected = StoreCommitError if failure is CommitFailure.DEFINITE else type(failure)

    with pytest.raises(expected):
        await sender(account, safety_for(account, PAUSED), client).submit(client_order_id="c-1")

    assert client.requests == []
    assert (await order_of(account)).status is S.NEW
    assert (await durable_order(store)).status is S.NEW
    assert await account.safety_block("c-1") is None
    assert not account.is_poisoned


@pytest.mark.asyncio
async def test_initial_block_uncertain_commit_poisons_and_sends_nothing() -> None:
    account, store = await reserved()
    store.failures[1] = CommitFailure.UNCERTAIN
    safety = safety_for(account, HALTED)
    client, clock = Client(), Clock()

    with pytest.raises(StoreUncertainError):
        await sender(account, safety, client, clock).submit(client_order_id="c-1")

    assert account.is_poisoned
    assert client.requests == []
    assert (await order_of(account)).status is S.NEW  # RAM: last confirmed state
    calls = clock.calls
    with pytest.raises(AccountStatePoisonedError):
        await sender(account, safety, client, clock).submit(client_order_id="c-1")
    assert clock.calls == calls
    assert client.requests == []


@pytest.mark.parametrize("failure", [CommitFailure.DEFINITE, CommitFailure.UNCERTAIN])
@pytest.mark.asyncio
async def test_final_block_persistence_failure_sends_nothing(failure: CommitFailure) -> None:
    account, store = await reserved()
    store.block_at = 1
    store.failures[2] = failure  # the SUBMITTING -> FAILED commit
    safety = safety_for(account, RUNNING)
    client = Client()
    task = asyncio.create_task(sender(account, safety, client).submit(client_order_id="c-1"))
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    safety.request_state(PAUSED)
    store.release.set()

    expected = StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError
    with pytest.raises(expected):
        await asyncio.wait_for(task, timeout=5)

    assert client.requests == []
    assert (await order_of(account)).status is S.SUBMITTING  # RAM: last confirmed
    assert account.is_poisoned is (failure is CommitFailure.UNCERTAIN)
    if failure is CommitFailure.UNCERTAIN:
        with pytest.raises(AccountStatePoisonedError):
            await sender(account, safety, client).submit(client_order_id="c-1")
    assert client.requests == []


# --- poison ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poisoned_account_fails_before_safety_clock_and_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account, store = await reserved()
    store.failures[1] = CommitFailure.UNCERTAIN
    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty("BTCUSDT", D("1"))
    reads: list[int] = []
    monkeypatch.setattr(SafetyController, "snapshot", lambda self: reads.append(1))
    client, clock = Client(), Clock()

    with pytest.raises(AccountStatePoisonedError):
        await sender(account, SafetyController(account_state=account), client, clock).submit(
            client_order_id="c-1"
        )

    assert (reads, clock.calls, client.requests) == ([], 0, [])


@pytest.mark.asyncio
async def test_poison_between_the_checks_wins_over_the_final_check() -> None:
    account, store = await reserved()
    store.block_at = 1
    store.failures[2] = CommitFailure.UNCERTAIN  # the other writer's commit
    safety = safety_for(account, RUNNING)
    client = Client()
    task = asyncio.create_task(sender(account, safety, client).submit(client_order_id="c-1"))
    await asyncio.wait_for(store.entered.wait(), timeout=5)

    async def other_writer() -> None:
        async with account.account_lock() as locked:
            await locked.set_position_qty("BTCUSDT", D("1"))

    # Queued on the lock: it runs between the marker and the final check (FIFO).
    other = asyncio.create_task(other_writer())
    await asyncio.sleep(0)
    store.release.set()
    with pytest.raises(StoreUncertainError):
        await asyncio.wait_for(other, timeout=5)
    with pytest.raises(AccountStatePoisonedError):
        await asyncio.wait_for(task, timeout=5)

    assert client.requests == []
    assert (await order_of(account)).status is S.SUBMITTING  # not mutated while poisoned
    assert await account.safety_block("c-1") is None


# --- clock ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_block_falls_back_to_the_order_time_when_the_clock_fails() -> None:
    account, store = await reserved()
    store.block_at = 1
    safety = safety_for(account, RUNNING)
    clock = FirstOnlyClock()
    task = asyncio.create_task(
        sender(account, safety, Client(), clock).submit(client_order_id="c-1")
    )
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    safety.request_state(PAUSED)
    store.release.set()

    with pytest.raises(OrderSubmissionBlockedError):
        await asyncio.wait_for(task, timeout=5)

    failed = await order_of(account)
    assert failed.status is S.FAILED
    assert failed.updated_at == T1  # the SUBMITTING time, never earlier
    assert clock.calls == 2


@pytest.mark.asyncio
async def test_initial_block_reads_the_clock_strictly() -> None:
    account, _ = await reserved()

    class NaiveClock:
        def now(self) -> datetime:
            return datetime(2026, 1, 15, 12, 30)  # noqa: DTZ001 - naive on purpose

    with pytest.raises(DomainValidationError):
        await sender(account, safety_for(account, PAUSED), Client(), NaiveClock()).submit(
            client_order_id="c-1"
        )
    assert (await order_of(account)).status is S.NEW


# --- reservation / replay -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_reservation_releases_exposure_and_replay_returns_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ScriptedStore()
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
    safety = safety_for(account, RUNNING)
    policy = RiskPolicy(
        policy_id="p",
        max_open_orders=None,
        symbols={
            "BTCUSDT": SymbolRiskLimits(
                max_order_qty=None, max_order_notional=None, max_position_qty=D("10")
            )
        },
    )

    class Ids:
        def __init__(self) -> None:
            self.calls = 0

        def next_id(self, *, intent: PlaceOrderIntent) -> str:
            self.calls += 1
            return f"c-{intent.intent_id}"

    ids, clock = Ids(), Clock()
    coord = PlacementCoordinator(
        account_state=account,
        safety=safety,
        policy=policy,
        clock=clock,
        client_order_id_generator=ids,
    )
    record = await coord.place(intent=intent("i-1"))
    assert record.approved
    assert await account.account_active_order_count() == 1

    safety.request_state(PAUSED)
    client = Client()
    with pytest.raises(OrderSubmissionBlockedError):
        await sender(account, safety, client).submit(client_order_id="c-i-1")

    assert await account.account_active_order_count() == 0
    assert await account.active_orders("BTCUSDT") == ()
    assert await account.placement("i-1") == record

    evaluations: list[int] = []
    monkeypatch.setattr(placement_module, "evaluate", lambda **_: evaluations.append(1))
    counts = (ids.calls, clock.calls, store.commits)
    replay = await coord.place(intent=intent("i-1"))
    assert replay == record
    assert (await account.order("c-i-1")).status is S.FAILED  # type: ignore[union-attr]
    assert (ids.calls, clock.calls, store.commits) == counts
    assert evaluations == []
    monkeypatch.undo()

    # A new trading decision needs a new intent_id.
    safety.request_state(RUNNING)
    fresh = await coord.place(intent=intent("i-2"))
    assert fresh.approved
    await sender(account, safety, client, Clock(T3)).submit(client_order_id="c-i-2")
    assert [r.client_order_id for r in client.requests] == ["c-i-2"]


# --- account state API ----------------------------------------------------------------------


@pytest.mark.parametrize("status", [S.OPEN, S.UNKNOWN, S.FAILED])
@pytest.mark.asyncio
async def test_only_new_or_submitting_orders_can_be_blocked(status: OrderStatus) -> None:
    account, _ = await reserved()
    async with account.account_lock() as locked:
        await locked.mark_submitting("c-1", at=T1)
        if status is S.OPEN:
            await locked.record_ack("c-1", exchange_order_id="ex-1", at=T1)
            await locked.apply_exchange_state(
                ExchangeOrderState(
                    client_order_id="c-1",
                    exchange_order_id="ex-1",
                    status=S.OPEN,
                    filled_qty=D("0"),
                    avg_fill_price=None,
                    exchange_ts=T1,
                ),
                at=T1,
            )
        elif status is S.UNKNOWN:
            await locked.record_submission_outcome("c-1", SubmissionOutcome.AMBIGUOUS, at=T1)
        else:
            await locked.record_submission_outcome("c-1", SubmissionOutcome.NOT_SENT, at=T1)
        revision = locked.revision
        with pytest.raises(AccountStateError, match="only a NEW or SUBMITTING"):
            await locked.record_safety_block("c-1", effective_state=PAUSED, at=T2)
        assert locked.revision == revision


@pytest.mark.asyncio
async def test_store_rejects_inconsistent_safety_blocks() -> None:
    account, store = await reserved()
    block = SafetyBlockRecord(
        client_order_id="c-1",
        effective_state=PAUSED,
        stage=SubmissionBlockStage.BEFORE_WRITE_AHEAD,
        blocked_at=T1,
    )
    loaded = await store.load(account_scope_id=SCOPE)
    assert loaded is not None
    # A block of an order that is not FAILED.
    with pytest.raises(StoreValidationError, match="must be FAILED"):
        await store.inner.commit(
            AccountStateChange(
                account_scope_id=SCOPE,
                expected_revision=loaded.revision,
                new_revision=loaded.revision + 1,
                safety_block_writes=(block,),
            )
        )
    # A block can never ride on a same-revision change.
    with pytest.raises(StoreValidationError, match="keeps the revision"):
        AccountStateChange(
            account_scope_id=SCOPE,
            expected_revision=loaded.revision,
            new_revision=loaded.revision,
            safety_block_writes=(block,),
        )
    # Once stored, a block is immutable.
    with pytest.raises(OrderSubmissionBlockedError):
        await sender(account, safety_for(account, PAUSED), Client(), Clock(T1)).submit(
            client_order_id="c-1"
        )
    stored = await store.load(account_scope_id=SCOPE)
    assert stored is not None
    with pytest.raises(StoreConflictError, match="safety block"):
        await store.inner.commit(
            AccountStateChange(
                account_scope_id=SCOPE,
                expected_revision=stored.revision,
                new_revision=stored.revision + 1,
                safety_block_writes=(
                    SafetyBlockRecord(
                        client_order_id="c-1",
                        effective_state=HALTED,
                        stage=SubmissionBlockStage.BEFORE_WRITE_AHEAD,
                        blocked_at=T1,
                    ),
                ),
            )
        )


# --- unchanged paths ------------------------------------------------------------------------


def test_reconciler_is_not_gated_by_safety() -> None:
    params = inspect.signature(UnknownOrderReconciler).parameters
    assert "safety" not in params
    assert "safety" in inspect.signature(OrderSubmitter).parameters
