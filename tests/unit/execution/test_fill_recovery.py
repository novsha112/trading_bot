"""Missing-fill recovery of one order from the complete execution history."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.instrument import InstrumentSpec
from app.domain.intents import PlaceOrderIntent
from app.exchanges.errors import ExchangeResponseError
from app.exchanges.models import OrderRequest
from app.exchanges.recovery import (
    ExchangeExecution,
    ExchangeOrder,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
)
from app.exchanges.simulated import SimulatedExchange
from app.execution import fill_recovery as module
from app.execution.account_state import AccountStatePoisonedError, InMemoryAccountState
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.fill_recovery import (
    ExecutionCumulativeMismatchError,
    ExecutionHistoryProtocolError,
    ExecutionIdentityConflictError,
    ExecutionRecoveryOutcome,
    ExecutionRecoveryResult,
    UnsupportedExecutionKindError,
    recover_missing_fills,
)
from app.execution.models import ExchangeOrderState, SubmissionOutcome
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    StoreCommitError,
    StoreUncertainError,
)
from app.execution.recovery_matching import ManagedOrderMatch, classify_open_orders
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NS = ClientOrderNamespace("bot01")
CID = NS.build("aaa")
EID = "X-1"


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


# --- doubles --------------------------------------------------------------------------------


class Store:
    """Reference store counting commits; failures armed per commit number."""

    def __init__(self) -> None:
        self.inner = InMemoryAccountStateStore()
        self.commits = 0
        self.failures: dict[int, CommitFailure] = {}

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.commits += 1
        failure = self.failures.get(self.commits)
        if failure is not None:
            self.inner.inject_commit_failure(failure)
        await self.inner.commit(change)


class Reader:
    """Serves the given executions in the given order, ``page_size`` per page."""

    def __init__(self, executions: list[ExchangeExecution], *, page_size: int = 2) -> None:
        self.executions = executions
        self.page_size = page_size
        self.fail_pages: set[int] = set()
        self.calls = 0
        self.wrong_echo_page: int | None = None
        self.cycle = False
        self.block: asyncio.Event | None = None

    async def list_executions(
        self, query: ExecutionQuery, *, cursor: str | None = None
    ) -> ExecutionPage:
        self.calls += 1
        if self.block is not None:
            await self.block.wait()
        offset = 0 if cursor is None else int(cursor.removeprefix("c"))
        page_number = offset // self.page_size + 1
        if page_number in self.fail_pages:
            raise ExchangeResponseError(f"page {page_number} failed")
        end = offset + self.page_size
        next_cursor = f"c{end}" if end < len(self.executions) else None
        if self.cycle and next_cursor is not None:
            next_cursor = "c0"
        echo = query
        if self.wrong_echo_page == page_number:
            echo = dataclasses.replace(query, end=query.end + timedelta(seconds=1))
        return ExecutionPage(
            query=echo, executions=tuple(self.executions[offset:end]), next_cursor=next_cursor
        )

    async def list_open_orders(self) -> Any:
        raise AssertionError("not used")

    async def get_position_snapshot(self) -> Any:
        raise AssertionError("not used")


def execution(
    exec_id: str, qty: str, ts: int, price: str = "100", **overrides: Any
) -> ExchangeExecution:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": EID,
        "client_order_id": CID,
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D(price),
        "qty": D(qty),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "kind": ExecutionKind.TRADE,
        "exchange_ts": at(ts),
    }
    return ExchangeExecution(**{**values, **overrides})


def fill_of(e: ExchangeExecution) -> Fill:
    return Fill(
        exec_id=e.exec_id,
        exchange_order_id=e.exchange_order_id,
        client_order_id=e.client_order_id,
        symbol=e.symbol,
        side=e.side,
        price=e.price,
        qty=e.qty,
        fee=e.fee,
        fee_asset=e.fee_asset,
        is_maker=e.is_maker,
        exchange_ts=e.exchange_ts,
    )


QUERY = ExecutionQuery(symbol="BTCUSDT", exchange_order_id=EID, start=at(-100), end=at(1000))


async def account_with_order(
    *, status: OrderStatus = S.OPEN, qty: str = "10", applied: tuple[ExchangeExecution, ...] = ()
) -> tuple[InMemoryAccountState, Store]:
    store = Store()
    account = InMemoryAccountState(account_scope_id="acct-1", store=store)
    source = PlaceOrderIntent(
        intent_id="i-1",
        strategy_id="grid-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        tag=None,
        created_at=T0,
    )
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D("0"))
        await locked.register_approved(
            intent=source,
            decision=RiskDecision(
                intent_id="i-1",
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
            client_order_id=CID,
            expected_revision=locked.revision,
            at=T0,
        )
        await locked.mark_submitting(CID, at=T0)
        if status is S.UNKNOWN:
            await locked.record_submission_outcome(CID, SubmissionOutcome.AMBIGUOUS, at=T0)
        else:
            await locked.record_ack(CID, exchange_order_id=EID, at=T0)
            await locked.apply_exchange_state(
                ExchangeOrderState(
                    client_order_id=CID,
                    exchange_order_id=EID,
                    status=S.OPEN,
                    filled_qty=D("0"),
                    avg_fill_price=None,
                    exchange_ts=T0,
                ),
                at=T0,
            )
        for e in applied:
            await locked.apply_fill(fill_of(e), at=at(1))
    store.commits = 0
    return account, store


def exchange_order(
    cum: str, *, qty: str = "10", status: OrderStatus | None = None
) -> ExchangeOrder:
    filled = D(cum)
    if status is None:
        status = S.OPEN if filled == 0 else (S.FILLED if filled == D(qty) else S.PARTIALLY_FILLED)
    return ExchangeOrder(
        exchange_order_id=EID,
        client_order_id=CID,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        status=status,
        cum_filled_qty=filled,
        cum_filled_notional=None,
        avg_fill_price=None if filled == 0 else D("100"),
        created_ts=T0,
        updated_ts=at(500),
    )


async def match_for(account: InMemoryAccountState, cum: str, **kwargs: Any) -> ManagedOrderMatch:
    local = await account.order(CID)
    assert local is not None
    exchange = exchange_order(cum, **kwargs)
    classified = (
        classify_open_orders(local_orders=(local,), exchange_orders=(exchange,), namespace=NS)
        if exchange.status in (S.OPEN, S.PARTIALLY_FILLED)
        else None
    )
    if classified is not None:
        (match,) = classified.managed
        return match
    return ManagedOrderMatch(
        local_order=local,
        exchange_order=exchange,
        exchange_id_completion_required=local.exchange_order_id is None,
    )


async def recover(
    account: InMemoryAccountState,
    reader: Reader,
    match: ManagedOrderMatch,
    clock: Any = None,
) -> ExecutionRecoveryResult:
    return await recover_missing_fills(
        account_state=account,
        reader=reader,
        match=match,
        query=QUERY,
        clock=ManualClock(at(900)) if clock is None else clock,
    )


async def snapshot(account: InMemoryAccountState) -> tuple[Any, ...]:
    order = await account.order(CID)
    return (
        await account.revision(),
        order,
        await account.filled_notional(CID),
        await account.position_qty("BTCUSDT"),
    )


# --- happy paths ----------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [1, 2, 3, 5])
@pytest.mark.asyncio
async def test_missing_fills_are_applied_in_time_then_exec_id_order(page_size: int) -> None:
    history = [
        execution("e-3", "2", 30, price="101"),
        execution("e-1", "1", 10),
        execution("e-2b", "1", 20, price="99.5"),
        execution("e-2a", "1", 20),
        execution("e-4", "0.5", 40, price="100.25"),
    ]
    account, store = await account_with_order()
    reader = Reader(history, page_size=page_size)

    result = await recover(account, reader, await match_for(account, "5.5"))

    assert result.outcome is ExecutionRecoveryOutcome.RECOVERED
    assert (result.fills_applied, result.already_present, result.unique_executions) == (5, 0, 5)
    assert result.pages_read == -(-5 // page_size)
    order = await account.order(CID)
    assert order is not None
    assert order.filled_qty == D("5.5") == result.final_filled_qty
    assert order.status is S.PARTIALLY_FILLED
    expected_notional = D("101") * 2 + 100 + D("99.5") + 100 + D("100.25") * D("0.5")
    assert await account.filled_notional(CID) == expected_notional == result.final_notional
    assert order.last_exchange_update_ts == at(40)  # applied in (ts, exec_id) order
    assert await account.position_qty("BTCUSDT") == D("5.5")
    assert store.commits == 5
    for e in history:
        assert await account.fill(e.exec_id) == fill_of(e)


@pytest.mark.asyncio
async def test_unknown_order_without_exchange_id_is_completed_by_its_fills() -> None:
    account, _ = await account_with_order(status=S.UNKNOWN)
    reader = Reader([execution("e-1", "10", 5)])

    result = await recover(account, reader, await match_for(account, "10"))

    order = await account.order(CID)
    assert order is not None
    assert (order.status, order.exchange_order_id, order.filled_qty) == (S.FILLED, EID, D("10"))
    assert result.fills_applied == 1


@pytest.mark.asyncio
async def test_already_complete_is_a_no_op_without_clock_or_commit() -> None:
    applied = (execution("e-1", "2", 10), execution("e-2", "3", 20))
    account, store = await account_with_order(applied=applied)
    before = await snapshot(account)

    class NoClock:
        def now(self) -> datetime:
            raise AssertionError("the clock must not be read")

    result = await recover(account, Reader(list(applied)), await match_for(account, "5"), NoClock())

    assert result.outcome is ExecutionRecoveryOutcome.ALREADY_COMPLETE
    assert (result.fills_applied, result.already_present) == (0, 2)
    assert store.commits == 0
    assert await snapshot(account) == before


@pytest.mark.asyncio
async def test_zero_fill_order_with_empty_history_is_a_no_op() -> None:
    account, store = await account_with_order()
    before = await snapshot(account)

    result = await recover(account, Reader([]), await match_for(account, "0"))

    assert result.outcome is ExecutionRecoveryOutcome.ALREADY_COMPLETE
    assert (result.pages_read, result.unique_executions, result.final_filled_qty) == (1, 0, D("0"))
    assert store.commits == 0
    assert await snapshot(account) == before


@pytest.mark.asyncio
async def test_partially_applied_history_applies_only_the_rest() -> None:
    first = execution("e-1", "2", 10)
    account, store = await account_with_order(applied=(first,))

    result = await recover(
        account, Reader([execution("e-2", "3", 20), first]), await match_for(account, "5")
    )

    assert (result.already_present, result.fills_applied) == (1, 1)
    assert store.commits == 1


@pytest.mark.asyncio
async def test_duplicate_identical_execution_across_pages_counts_once() -> None:
    e = execution("e-1", "2", 10)
    account, _ = await account_with_order()

    result = await recover(
        account,
        Reader([e, execution("e-2", "1", 20), e], page_size=1),
        await match_for(account, "3"),
    )

    assert (result.executions_seen, result.unique_executions, result.fills_applied) == (3, 2, 2)
    order = await account.order(CID)
    assert order is not None
    assert order.filled_qty == D("3")


@pytest.mark.asyncio
async def test_result_does_not_depend_on_the_response_order() -> None:
    history = [execution(f"e-{n}", "1", 10 * (n % 3), price=str(100 + n)) for n in range(6)]
    outcomes = []
    for ordering in (history, list(reversed(history)), history[3:] + history[:3]):
        account, _ = await account_with_order()
        await recover(account, Reader(list(ordering), page_size=2), await match_for(account, "6"))
        outcomes.append(await snapshot(account))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.asyncio
async def test_hostile_decimal_values_are_exact() -> None:
    history = [
        execution("e-1", "0.000000001", 10, price="12345.678912345"),
        execution("e-2", "2.000", 20, price="99.10"),
    ]
    account, _ = await account_with_order()

    result = await recover(account, Reader(history), await match_for(account, "2.000000001"))

    expected = D("12345.678912345") * D("0.000000001") + D("99.10") * D("2.000")
    assert result.final_notional == expected
    assert await account.filled_notional(CID) == expected


# --- protocol failures: nothing applied ------------------------------------------------------


def five() -> list[ExchangeExecution]:
    return [execution(f"e-{n}", "1", n) for n in range(5)]


@pytest.mark.parametrize("page", [1, 3, 5])
@pytest.mark.asyncio
async def test_a_failing_page_applies_nothing(page: int) -> None:
    account, store = await account_with_order()
    reader = Reader(five(), page_size=1)
    reader.fail_pages.add(page)
    before = await snapshot(account)

    with pytest.raises(ExchangeResponseError):
        await recover(account, reader, await match_for(account, "5"))

    assert store.commits == 0
    assert await snapshot(account) == before


@pytest.mark.parametrize("page", [1, 2])
@pytest.mark.asyncio
async def test_a_wrong_query_echo_applies_nothing(page: int) -> None:
    account, store = await account_with_order()
    reader = Reader(five(), page_size=3)
    reader.wrong_echo_page = page

    with pytest.raises(ExecutionHistoryProtocolError, match="echo"):
        await recover(account, reader, await match_for(account, "5"))
    assert store.commits == 0


@pytest.mark.asyncio
async def test_a_cursor_cycle_applies_nothing() -> None:
    account, store = await account_with_order()
    reader = Reader(five(), page_size=1)
    reader.cycle = True

    with pytest.raises(ExecutionHistoryProtocolError, match="cycle"):
        await recover(account, reader, await match_for(account, "5"))
    assert store.commits == 0
    assert reader.calls == 2


@pytest.mark.asyncio
async def test_a_page_that_is_not_an_execution_page_applies_nothing() -> None:
    account, store = await account_with_order()

    class Bad(Reader):
        async def list_executions(self, query: ExecutionQuery, *, cursor: str | None = None) -> Any:
            return {"executions": []}

    with pytest.raises(ExecutionHistoryProtocolError):
        await recover(account, Bad([]), await match_for(account, "0"))
    assert store.commits == 0


# --- history preflight: nothing applied -----------------------------------------------------


@pytest.mark.asyncio
async def test_conflicting_duplicate_exec_id_applies_nothing() -> None:
    account, store = await account_with_order()
    history = [
        execution("e-1", "1", 10),
        execution("e-2", "1", 20),
        execution("e-1", "1", 10, fee=D("0.1"), fee_asset="USDT"),
    ]

    with pytest.raises(ExecutionIdentityConflictError, match="different data"):
        await recover(account, Reader(history, page_size=1), await match_for(account, "2"))
    assert store.commits == 0


@pytest.mark.parametrize(
    "kind", [ExecutionKind.LIQUIDATION, ExecutionKind.ADL, ExecutionKind.BUST, ExecutionKind.OTHER]
)
@pytest.mark.asyncio
async def test_non_trade_executions_block_recovery(kind: ExecutionKind) -> None:
    account, store = await account_with_order()
    history = [execution("e-1", "1", 10), execution("e-2", "1", 20, kind=kind)]

    with pytest.raises(UnsupportedExecutionKindError, match=kind.value):
        await recover(account, Reader(history), await match_for(account, "1"))
    assert store.commits == 0


def _tamper(page: ExecutionPage, replacement: ExchangeExecution) -> ExecutionPage:
    # Bypass the page's own validation, as a broken reader might.
    object.__setattr__(page, "executions", (replacement,))
    return page


@pytest.mark.parametrize(
    "overrides",
    [
        {"exchange_order_id": "X-2"},
        {"client_order_id": NS.build("bbb")},
        {"client_order_id": None},
        {"symbol": "ETHUSDT"},
        {"side": Side.SELL},
    ],
    ids=["exchange-id", "client-id", "no-client-id", "symbol", "side"],
)
@pytest.mark.asyncio
async def test_executions_of_another_identity_apply_nothing(overrides: dict[str, Any]) -> None:
    account, store = await account_with_order()
    alien = execution("e-1", "1", 10, **overrides)

    class Alien(Reader):
        async def list_executions(
            self, query: ExecutionQuery, *, cursor: str | None = None
        ) -> ExecutionPage:
            page = ExecutionPage(query=query, executions=(), next_cursor=None)
            return _tamper(page, alien)

    with pytest.raises(ExecutionIdentityConflictError, match="does not belong"):
        await recover(account, Alien([]), await match_for(account, "1"))
    assert store.commits == 0


@pytest.mark.parametrize(
    ("cum", "history_qty"),
    [("3", "2"), ("1", "2")],
    ids=["exchange-above-history", "history-above-exchange"],
)
@pytest.mark.asyncio
async def test_exchange_and_history_quantities_must_match(cum: str, history_qty: str) -> None:
    account, store = await account_with_order()

    with pytest.raises(ExecutionCumulativeMismatchError):
        await recover(
            account, Reader([execution("e-1", history_qty, 10)]), await match_for(account, cum)
        )
    assert store.commits == 0


@pytest.mark.asyncio
async def test_exchange_below_local_is_a_conflict() -> None:
    applied = (execution("e-1", "2", 10), execution("e-2", "2", 20))
    account, store = await account_with_order(applied=applied)
    # The exchange now shows less than what was applied locally.

    with pytest.raises(ExecutionCumulativeMismatchError):
        await recover(account, Reader([applied[0]]), await match_for(account, "2"))
    assert store.commits == 0
    order = await account.order(CID)
    assert order is not None
    assert order.filled_qty == D("4")  # nothing removed


@pytest.mark.parametrize(
    "overrides",
    [
        {"qty": D("3")},
        {"price": D("101")},
        {"exchange_ts": at(11)},
        {"fee": D("0.1"), "fee_asset": "USDT"},
        {"is_maker": True},
    ],
    ids=["qty", "price", "time", "fee", "maker"],
)
@pytest.mark.asyncio
async def test_a_locally_applied_exec_id_with_other_data_applies_nothing(
    overrides: dict[str, Any],
) -> None:
    local_version = execution("e-1", "2", 10)
    account, store = await account_with_order(applied=(local_version,))
    exchange_version = dataclasses.replace(local_version, **overrides)
    cum = str(exchange_version.qty + 1)

    with pytest.raises(ExecutionIdentityConflictError, match="applied locally"):
        await recover(
            account,
            Reader([exchange_version, execution("e-2", "1", 20)]),
            await match_for(account, cum),
        )
    assert store.commits == 0


@pytest.mark.asyncio
async def test_a_local_fill_missing_from_the_history_is_a_mismatch() -> None:
    applied = (execution("e-1", "2", 10),)
    account, store = await account_with_order(applied=applied)

    with pytest.raises(ExecutionCumulativeMismatchError, match="applied part"):
        await recover(account, Reader([execution("e-9", "2", 10)]), await match_for(account, "2"))
    assert store.commits == 0


# --- preconditions / stale match ------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_must_name_the_matched_order() -> None:
    account, _ = await account_with_order()
    match = await match_for(account, "0")
    for query in (
        dataclasses.replace(QUERY, exchange_order_id=None),
        dataclasses.replace(QUERY, exchange_order_id="X-2"),
        dataclasses.replace(QUERY, symbol="ETHUSDT"),
    ):
        with pytest.raises(DomainValidationError, match="query"):
            await recover_missing_fills(
                account_state=account,
                reader=Reader([]),
                match=match,
                query=query,
                clock=ManualClock(T0),
            )


@pytest.mark.asyncio
async def test_a_stale_match_with_another_exchange_id_is_refused() -> None:
    account, store = await account_with_order()
    match = await match_for(account, "1")
    stale = dataclasses.replace(
        match, exchange_order=dataclasses.replace(match.exchange_order, exchange_order_id="X-2")
    )
    query = dataclasses.replace(QUERY, exchange_order_id="X-2")

    with pytest.raises(ExecutionIdentityConflictError, match="exchange id"):
        await recover_missing_fills(
            account_state=account,
            reader=Reader([]),
            match=stale,
            query=query,
            clock=ManualClock(T0),
        )
    assert store.commits == 0


@pytest.mark.asyncio
async def test_poisoned_account_reads_nothing() -> None:
    account, store = await account_with_order()
    store.failures[1] = CommitFailure.UNCERTAIN
    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty("BTCUSDT", D("1"))
    reader = Reader([execution("e-1", "1", 10)])

    with pytest.raises(AccountStatePoisonedError):
        await recover(account, reader, await match_for(account, "1"))
    assert reader.calls == 0


# --- mutation-phase failures ----------------------------------------------------------------


@pytest.mark.parametrize("failing", [1, 2])
@pytest.mark.asyncio
async def test_definite_failure_keeps_earlier_fills_and_a_rerun_completes(failing: int) -> None:
    history = [execution("e-1", "1", 10), execution("e-2", "2", 20), execution("e-3", "3", 30)]
    account, store = await account_with_order()
    store.failures[failing] = CommitFailure.DEFINITE

    with pytest.raises(StoreCommitError):
        await recover(account, Reader(history), await match_for(account, "6"))

    order = await account.order(CID)
    assert order is not None
    assert order.filled_qty == sum((e.qty for e in history[: failing - 1]), D(0))
    assert not account.is_poisoned

    result = await recover(account, Reader(history), await match_for(account, "6"))
    assert (result.already_present, result.fills_applied) == (failing - 1, 4 - failing)
    final = await account.order(CID)
    assert final is not None
    assert final.filled_qty == D("6")
    assert await account.position_qty("BTCUSDT") == D("6")


@pytest.mark.asyncio
async def test_uncertain_failure_poisons_and_stops_immediately() -> None:
    history = [execution("e-1", "1", 10), execution("e-2", "2", 20), execution("e-3", "3", 30)]
    account, store = await account_with_order()
    store.failures[2] = CommitFailure.UNCERTAIN

    with pytest.raises(StoreUncertainError):
        await recover(account, Reader(history), await match_for(account, "6"))

    assert account.is_poisoned
    assert store.commits == 2  # the third fill was never attempted
    order = await account.order(CID)
    assert order is not None
    assert order.filled_qty == D("1")  # RAM: the last confirmed state
    with pytest.raises(AccountStatePoisonedError):
        await recover(account, Reader(history), await match_for(account, "6"))


# --- concurrency ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_recoveries_apply_each_fill_once() -> None:
    history = [execution("e-1", "2", 10), execution("e-2", "3", 20)]
    account, store = await account_with_order()
    match = await match_for(account, "5")
    gate = asyncio.Event()
    first, second = Reader(history), Reader(history)
    first.block = second.block = gate

    tasks = [asyncio.create_task(recover(account, reader, match)) for reader in (first, second)]
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(*tasks)

    assert sorted(r.fills_applied for r in results) == [0, 2]
    assert store.commits == 2
    order = await account.order(CID)
    assert order is not None
    assert order.filled_qty == D("5")
    assert await account.position_qty("BTCUSDT") == D("5")
    assert await account.filled_notional(CID) == D("500")


# --- simulator ------------------------------------------------------------------------------


def _spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=D("0.1"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        max_qty=D("1000000"),
        min_notional=D("0"),
    )


@pytest.mark.parametrize("page_size", [1, 2, 7])
@pytest.mark.asyncio
async def test_simulator_history_matches_its_fills_and_notional(page_size: int) -> None:
    clock = ManualClock(T0)
    sim = SimulatedExchange(clock=clock, instruments=(_spec(),), execution_page_size=page_size)
    await sim.place_order(
        OrderRequest(
            client_order_id=CID,
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            price=D("100"),
            qty=D("10"),
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
        )
    )
    fills: list[Fill] = []
    for n, price in enumerate(("99.9", "99.5", "98.7")):
        clock.advance(timedelta(seconds=1))
        fills += await sim.fill_crossed_limit_orders(
            symbol="BTCUSDT", execution_price=D(price), available_qty=D(str(n + 1))
        )
    (open_order,) = (await sim.list_open_orders()).orders
    query = ExecutionQuery(
        symbol="BTCUSDT",
        exchange_order_id=open_order.exchange_order_id,
        start=T0,
        end=at(10),
    )

    # Simulator guarantee (not a generic reader guarantee): exact cumulative notional.
    exact = sum((f.price * f.qty for f in fills), D(0))
    assert open_order.cum_filled_notional == exact

    # Local state that has seen none of the fills (e.g. an UNKNOWN order).
    account, _ = await account_with_order(status=S.UNKNOWN)
    local = await account.order(CID)
    assert local is not None
    (match,) = classify_open_orders(
        local_orders=(local,), exchange_orders=(open_order,), namespace=NS
    ).managed

    result = await recover_missing_fills(
        account_state=account, reader=sim, match=match, query=query, clock=clock
    )

    assert result.fills_applied == 3
    assert result.final_filled_qty == open_order.cum_filled_qty == D("6")
    assert result.final_notional == exact == await account.filled_notional(CID)
    for fill in fills:
        assert await account.fill(fill.exec_id) == fill


# --- boundaries -----------------------------------------------------------------------------


def test_component_is_read_only_towards_the_exchange() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    for word in (
        "place_order",
        "cancel_order",
        "cum_filled_notional ==",
        ".commit(",
        "datetime.now",
        "simulated",
        "bybit",
    ):
        assert word not in source, word
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if name.startswith(("app.persistence", "app.services"))}
    assert inspect.iscoroutinefunction(recover_missing_fills)
