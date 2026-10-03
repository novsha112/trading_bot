"""Automatic position reconciliation of a supplied exchange snapshot: known is
not reconciled; runtime publication only; nothing durable changes."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import itertools
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.exchanges.recovery import (
    ExchangeExecution,
    ExchangeOrder,
    ExchangePosition,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
    PositionSnapshot,
)
from app.execution import position_reconciliation as module
from app.execution.account_state import (
    AccountStatePoisonedError,
    InMemoryAccountState,
    PositionProjectionMismatchError,
    StaleRevisionError,
)
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.fill_recovery import recover_missing_fills
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    PersistedPosition,
    StoreUncertainError,
)
from app.execution.position_reconciliation import (
    PositionExplanation,
    PositionReconciliationResult,
    classify_positions,
    reconcile_positions,
)
from app.execution.recovery_matching import ManagedOrderMatch
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision

D = Decimal
E = PositionExplanation
BTC, ETH, SOL = "BTCUSDT", "ETHUSDT", "SOLUSDT"
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NS = ClientOrderNamespace("bot01")


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


class Store:
    def __init__(self) -> None:
        self.inner = InMemoryAccountStateStore()
        self.changes: list[AccountStateChange] = []
        self.failures: dict[int, CommitFailure] = {}
        self.block: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        if self.block is not None:
            self.entered.set()
            await self.block.wait()
        failure = self.failures.get(len(self.changes))
        if failure is not None:
            self.inner.inject_commit_failure(failure)
        await self.inner.commit(change)


def snap(positions: Mapping[str, str] | None = None, *, complete: bool = True) -> PositionSnapshot:
    return PositionSnapshot(
        positions=tuple(ExchangePosition(symbol=s, qty=D(q)) for s, q in (positions or {}).items()),
        complete=complete,
        server_ts=T0,
    )


def row(symbol: str, qty: str | None) -> PersistedPosition:
    return PersistedPosition(
        symbol=symbol, known=qty is not None, qty=None if qty is None else D(qty)
    )


def cid(n: int) -> str:
    return NS.build(f"o{n}")


async def reserve(
    account: InMemoryAccountState,
    n: int,
    symbol: str,
    side: Side,
    qty: str,
    *,
    reduce_only: bool = False,
) -> None:
    source = PlaceOrderIntent(
        intent_id=f"i-{n}",
        strategy_id="grid-1",
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=reduce_only,
        tag=None,
        created_at=T0,
    )
    async with account.account_lock() as locked:
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
            client_order_id=cid(n),
            expected_revision=locked.revision,
            at=T0,
        )
        await locked.mark_submitting(cid(n), at=T0)


def fill(n: int, exec_id: str, symbol: str, side: Side, qty: str, ts: int = 1) -> Fill:
    return Fill(
        exec_id=exec_id,
        exchange_order_id=f"X-{n}",
        client_order_id=cid(n),
        symbol=symbol,
        side=side,
        price=D("100"),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=at(ts),
    )


async def apply(account: InMemoryAccountState, f: Fill) -> None:
    async with account.account_lock() as locked:
        await locked.apply_fill(f, at=at(10))


async def hydrate(store: Store) -> InMemoryAccountState:
    return await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=store, clock=ManualClock(at(5))
    )


async def restarted(
    *,
    known: dict[str, str] | None = None,
    unknown: tuple[str, ...] = (),
    fills: tuple[tuple[str, str], ...] = (),
) -> tuple[InMemoryAccountState, Store]:
    """Positions seeded durable (known / known=False rows), optional fills of
    symbols with an unknown projection, then a restart (hydrate)."""
    store = Store()
    first = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with first.account_lock() as locked:
        for symbol, qty in (known or {}).items():
            await locked.set_position_qty(symbol, D(qty))
        for symbol in unknown:
            await locked.set_position_qty(symbol, D("1"))
            await locked.set_position_qty(symbol, None)
    for n, (symbol, qty) in enumerate(fills, start=1):
        await reserve(first, n, symbol, Side.BUY, qty)
        await apply(first, fill(n, f"e-{n}", symbol, Side.BUY, qty))
    account = await hydrate(store)
    store.changes.clear()
    return account, store


async def reconcile(
    account: InMemoryAccountState, snapshot: PositionSnapshot
) -> PositionReconciliationResult:
    return await reconcile_positions(account_state=account, snapshot=snapshot)


def by_symbol(result: PositionReconciliationResult) -> dict[str, Any]:
    return {
        p.symbol: (p.explanation, p.known, p.explained, p.exchange_qty, p.runtime_qty)
        for p in result.positions
    }


async def assert_nothing_durable(
    account: InMemoryAccountState, store: Store, revision: int, rows: dict[str, Any]
) -> None:
    assert store.changes == []
    assert await account.revision() == revision
    assert dict(account._state.durable_positions) == rows


# --- durable known projection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("durable", "exchange", "explanation"),
    [
        ("0", "0", E.DURABLE_PROJECTION_MATCH),
        ("5", "5", E.DURABLE_PROJECTION_MATCH),
        ("-5", "-5", E.DURABLE_PROJECTION_MATCH),
        ("5", "5.000", E.DURABLE_PROJECTION_MATCH),
        ("0", "-0", E.DURABLE_PROJECTION_MATCH),
        ("5", "4", E.DURABLE_PROJECTION_MISMATCH),
        ("-5", "-4", E.DURABLE_PROJECTION_MISMATCH),
        ("1E-200", "1E-200", E.DURABLE_PROJECTION_MATCH),
        ("1" + "0" * 40, "1E+40", E.DURABLE_PROJECTION_MATCH),
        ("1" + "0" * 40, "1" + "0" * 39 + "1", E.DURABLE_PROJECTION_MISMATCH),
    ],
)
@pytest.mark.asyncio
async def test_known_projection_against_the_exchange(
    durable: str, exchange: str, explanation: E
) -> None:
    account, store = await restarted(known={BTC: durable})
    revision = await account.revision()
    rows = dict(account._state.durable_positions)

    result = await reconcile(account, snap({BTC: exchange}))

    (position,) = result.positions
    assert position.explanation is explanation
    assert position.known is True
    assert position.explained is (explanation is E.DURABLE_PROJECTION_MATCH)
    assert result.reconciled is position.explained
    assert await account.position_qty(BTC) == D(exchange)
    assert repr(await account.position_qty(BTC)) == repr(D(exchange))
    await assert_nothing_durable(account, store, revision, rows)  # never overwritten


@pytest.mark.parametrize(
    ("durable", "explanation"),
    [("0", E.DURABLE_PROJECTION_MATCH), ("3", E.DURABLE_PROJECTION_MISMATCH)],
)
@pytest.mark.asyncio
async def test_symbol_missing_from_a_complete_snapshot_is_flat(
    durable: str, explanation: E
) -> None:
    account, store = await restarted(known={BTC: durable})

    result = await reconcile(account, snap())

    assert by_symbol(result)[BTC][:4] == (
        explanation,
        True,
        explanation is E.DURABLE_PROJECTION_MATCH,
        D("0"),
    )
    assert await account.position_qty(BTC) == D("0")  # known flat, not None
    assert store.changes == []


# --- no known durable projection ------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "exchange", "explanation"),
    [
        ({}, "5", E.UNEXPLAINED_NONZERO),
        ({}, "-5", E.UNEXPLAINED_NONZERO),
        ({"unknown": (BTC,)}, "5", E.UNEXPLAINED_NONZERO),
        ({"unknown": (BTC,)}, "0", E.UNKNOWN_FLAT_WITH_NO_EVIDENCE),
        ({}, "0", E.UNKNOWN_FLAT_WITH_NO_EVIDENCE),
        ({"unknown": (BTC,), "fills": ((BTC, "2"),)}, "0", E.UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE),
        ({"fills": ((BTC, "2"),)}, "0", E.UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE),
        ({"fills": ((BTC, "2"),)}, "2", E.UNEXPLAINED_NONZERO),
    ],
    ids=[
        "absent+5",
        "absent-5",
        "unknown+5",
        "unknown-flat",
        "absent-flat",
        "unknown-flat-fills",
        "absent-flat-fills",
        "absent-nonzero-fills",
    ],
)
@pytest.mark.asyncio
async def test_without_a_known_projection(
    setup: dict[str, Any], exchange: str, explanation: E
) -> None:
    account, store = await restarted(**setup)
    revision = await account.revision()
    rows = dict(account._state.durable_positions)

    result = await reconcile(account, snap({BTC: exchange}))

    position = next(p for p in result.positions if p.symbol == BTC)
    assert position.explanation is explanation
    assert position.known is True
    assert position.explained is (explanation is E.UNKNOWN_FLAT_WITH_NO_EVIDENCE)
    assert await account.position_qty(BTC) == D(exchange)
    await assert_nothing_durable(account, store, revision, rows)


@pytest.mark.asyncio
async def test_fill_symbols_and_runtime_only_symbols_join_the_universe() -> None:
    account, store = await restarted(fills=((ETH, "1"),))
    account._state = dataclasses.replace(account._state, positions={SOL: D("4")})  # runtime-only

    result = await reconcile(account, snap())

    assert by_symbol(result) == {
        ETH: (E.UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE, True, False, D("0"), D("0")),
        SOL: (E.UNKNOWN_FLAT_WITH_NO_EVIDENCE, True, True, D("0"), D("0")),
    }
    assert await account.position_qty(SOL) == D("0")  # the stale runtime value is replaced
    assert store.changes == []


# --- aggregate ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_complete_snapshot_and_empty_account_is_reconciled() -> None:
    account, store = await restarted()
    result = await reconcile(account, snap())
    assert result == PositionReconciliationResult(
        snapshot_complete=True, positions=(), all_known=True, all_explained=True, reconciled=True
    )
    assert store.changes == []


@pytest.mark.asyncio
async def test_one_unexplained_symbol_blocks_the_account() -> None:
    account, _ = await restarted(known={BTC: "2", ETH: "0"})
    result = await reconcile(account, snap({BTC: "2", SOL: "1"}))
    assert result.all_known is True
    assert result.all_explained is False
    assert result.reconciled is False
    assert [p.symbol for p in result.positions if not p.explained] == [SOL]


# --- partial snapshot -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "positions",
    [{}, {BTC: "5"}, {BTC: "0"}, {ETH: "1"}],
    ids=["empty", "explicit-nonzero", "explicit-zero", "local-symbols-missing"],
)
@pytest.mark.asyncio
async def test_partial_snapshot_infers_and_publishes_nothing(positions: dict[str, str]) -> None:
    account, store = await restarted(known={BTC: "5"}, unknown=(SOL,))
    state = account._state
    revision = await account.revision()

    result = await reconcile(account, snap(positions, complete=False))

    assert result.snapshot_complete is False
    assert result.reconciled is False
    assert result.all_known is False
    assert result.all_explained is False
    assert {p.explanation for p in result.positions} == {E.INCOMPLETE_SNAPSHOT}
    assert all(p.known is False and p.runtime_qty is None for p in result.positions)
    btc = next(p for p in result.positions if p.symbol == BTC)
    assert btc.exchange_qty == (D(positions[BTC]) if BTC in positions else None)
    assert account._state is state  # nothing published
    await assert_nothing_durable(account, store, revision, dict(state.durable_positions))


@pytest.mark.asyncio
async def test_partial_empty_snapshot_on_an_empty_account_is_not_reconciled() -> None:
    account, _ = await restarted()
    result = await reconcile(account, snap(complete=False))
    assert (result.positions, result.reconciled) == ((), False)


# --- recovered fills: no second application -------------------------------------------------


class Reader:
    def __init__(self, executions: list[ExchangeExecution]) -> None:
        self.executions = executions

    async def list_executions(
        self, query: ExecutionQuery, *, cursor: str | None = None
    ) -> ExecutionPage:
        return ExecutionPage(query=query, executions=tuple(self.executions), next_cursor=None)

    async def list_open_orders(self) -> Any:
        raise AssertionError("not used")

    async def get_position_snapshot(self) -> Any:
        raise AssertionError("not used")


def trade(exec_id: str, qty: str, ts: int) -> ExchangeExecution:
    return ExchangeExecution(
        exec_id=exec_id,
        exchange_order_id="X-1",
        client_order_id=cid(1),
        symbol=BTC,
        side=Side.BUY,
        price=D("100"),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        kind=ExecutionKind.TRADE,
        exchange_ts=at(ts),
    )


@pytest.mark.asyncio
async def test_reconciliation_after_fill_recovery_does_not_add_fills_again() -> None:
    store = Store()
    first = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with first.account_lock() as locked:
        await locked.set_position_qty(BTC, D("0"))
    await reserve(first, 1, BTC, Side.BUY, "10")
    await apply(first, fill(1, "e-1", BTC, Side.BUY, "2", ts=1))
    account = await hydrate(store)
    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (
        None,
        row(BTC, "2"),
    )

    local = await account.order(cid(1))
    assert local is not None
    exchange = ExchangeOrder(
        exchange_order_id="X-1",
        client_order_id=cid(1),
        symbol=BTC,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D("10"),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        status=OrderStatus.PARTIALLY_FILLED,
        cum_filled_qty=D("5"),
        cum_filled_notional=None,
        avg_fill_price=D("100"),
        created_ts=T0,
        updated_ts=at(3),
    )
    await recover_missing_fills(
        account_state=account,
        reader=Reader([trade("e-1", "2", 1), trade("e-2", "3", 2)]),
        match=ManagedOrderMatch(
            local_order=local, exchange_order=exchange, exchange_id_completion_required=False
        ),
        query=ExecutionQuery(symbol=BTC, exchange_order_id="X-1", start=T0, end=at(100)),
        clock=ManualClock(at(6)),
    )
    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (
        None,
        row(BTC, "5"),
    )
    store.changes.clear()
    revision = await account.revision()

    result = await reconcile(account, snap({BTC: "5"}))

    assert by_symbol(result)[BTC] == (E.DURABLE_PROJECTION_MATCH, True, True, D("5"), D("5"))
    assert result.reconciled is True
    assert await account.position_qty(BTC) == D("5")
    assert await account.durable_position(BTC) == row(BTC, "5")
    assert store.changes == []
    assert await account.revision() == revision


# --- runtime vs durable after publication ---------------------------------------------------


@pytest.mark.asyncio
async def test_mismatch_keeps_both_views_and_later_fills_fail_closed() -> None:
    account, store = await restarted(known={BTC: "2"})
    await reserve(account, 1, BTC, Side.BUY, "1")
    store.changes.clear()
    revision = await account.revision()

    result = await reconcile(account, snap({BTC: "5"}))

    assert result.reconciled is False
    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (
        D("5"),
        row(BTC, "2"),
    )
    assert store.changes == []
    assert await account.revision() == revision
    with pytest.raises(PositionProjectionMismatchError):
        await apply(account, fill(1, "e-1", BTC, Side.BUY, "1"))
    assert store.changes == []


@pytest.mark.asyncio
async def test_fills_after_an_unexplained_publication_move_only_the_runtime_value() -> None:
    account, store = await restarted()
    await reserve(account, 1, BTC, Side.BUY, "2")
    await reserve(account, 2, BTC, Side.SELL, "2", reduce_only=True)
    await reconcile(account, snap({BTC: "5"}))

    await apply(account, fill(1, "e-1", BTC, Side.BUY, "2"))
    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (D("7"), None)

    await apply(account, fill(2, "e-2", BTC, Side.SELL, "2"))  # reduce-only against runtime 7
    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (D("5"), None)
    assert all(change.position_writes == () for change in store.changes)


@pytest.mark.asyncio
async def test_after_a_match_fills_move_both_views_as_before() -> None:
    account, _ = await restarted(known={BTC: "2"})
    await reserve(account, 1, BTC, Side.BUY, "1")
    await reconcile(account, snap({BTC: "2.0"}))

    await apply(account, fill(1, "e-1", BTC, Side.BUY, "1"))

    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (
        D("3"),
        row(BTC, "3"),
    )


# --- restart --------------------------------------------------------------------------------


@pytest.mark.parametrize(("known", "exchange"), [({}, "5"), ({BTC: "2"}, "5"), ({BTC: "5"}, "5")])
@pytest.mark.asyncio
async def test_publication_is_forgotten_by_a_restart(known: dict[str, str], exchange: str) -> None:
    account, store = await restarted(known=known)
    await reconcile(account, snap({BTC: exchange}))
    durable_before = await account.durable_position(BTC)

    again = await hydrate(store)

    assert await again.position_qty(BTC) is None
    assert await again.durable_position(BTC) == durable_before


# --- poison / atomicity / concurrency -------------------------------------------------------


@pytest.mark.asyncio
async def test_poisoned_account_publishes_nothing() -> None:
    account, store = await restarted(known={BTC: "2"})
    store.failures[1] = CommitFailure.UNCERTAIN
    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty(ETH, D("1"))
    state = account._state
    store.changes.clear()

    with pytest.raises(AccountStatePoisonedError):
        await reconcile(account, snap({BTC: "2"}))

    assert account._state is state
    assert store.changes == []


@pytest.mark.asyncio
async def test_a_failure_before_publication_leaves_the_runtime_map_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account, _ = await restarted(known={BTC: "2", ETH: "1"})
    state = account._state

    def broken(*_: Any) -> Any:
        raise RuntimeError("classification failed")

    monkeypatch.setattr(module, "classify_positions", broken)
    with pytest.raises(RuntimeError):
        await reconcile(account, snap({BTC: "2", ETH: "1"}))
    assert account._state is state
    assert account._state.positions == {}


@pytest.mark.asyncio
async def test_publication_refuses_stale_evidence() -> None:
    account, _ = await restarted(known={BTC: "2"})
    async with account.account_lock() as locked:
        with pytest.raises(StaleRevisionError):
            locked.publish_exchange_positions({BTC: D("2")}, expected_revision=locked.revision - 1)
        with pytest.raises(DomainValidationError):
            locked.publish_exchange_positions({BTC: 2}, expected_revision=locked.revision)  # type: ignore[dict-item]
        assert locked.position_qty(BTC) is None


@pytest.mark.asyncio
async def test_a_fill_in_flight_is_either_fully_before_or_after_reconciliation() -> None:
    account, store = await restarted(known={BTC: "2"})
    await reserve(account, 1, BTC, Side.BUY, "3")
    store.changes.clear()
    store.block = asyncio.Event()

    fill_task = asyncio.create_task(apply(account, fill(1, "e-1", BTC, Side.BUY, "3")))
    await asyncio.wait_for(store.entered.wait(), timeout=5)  # the fill holds the lock
    reconcile_task = asyncio.create_task(reconcile(account, snap({BTC: "5"})))
    await asyncio.sleep(0)
    assert not reconcile_task.done()  # waits for the account lock
    store.block.set()
    await asyncio.wait_for(fill_task, timeout=5)
    result = await asyncio.wait_for(reconcile_task, timeout=5)

    # The reconciliation saw the committed fill (durable 5), never the torn state.
    assert by_symbol(result)[BTC] == (E.DURABLE_PROJECTION_MATCH, True, True, D("5"), D("5"))
    assert (await account.position_qty(BTC), await account.durable_position(BTC)) == (
        D("5"),
        row(BTC, "5"),
    )


# --- determinism / boundaries ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_does_not_depend_on_the_snapshot_order() -> None:
    account, _ = await restarted(known={BTC: "2", ETH: "-1"}, unknown=(SOL,))
    rows = [
        ExchangePosition(symbol=s, qty=D(q))
        for s, q in ((BTC, "2"), (ETH, "-3"), (SOL, "0"), ("XRPUSDT", "7"))
    ]
    async with account.account_lock() as locked:
        local = locked.position_evidence()
    results = {
        classify_positions(PositionSnapshot(positions=tuple(p), complete=True, server_ts=T0), local)
        for p in itertools.permutations(rows)
    }
    assert len(results) == 1
    (result,) = results
    assert [p.symbol for p in result.positions] == sorted([BTC, ETH, SOL, "XRPUSDT"])


def test_module_is_pure_of_readers_adapters_and_safety() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert {name for name in imported if name.startswith("app.")} <= {
        "app.domain.errors",
        "app.exchanges.recovery",
        "app.execution.account_state",
    }
    for word in (
        "ExchangeStateReader",
        "SafetyController",
        "simulated",
        "bybit",
        ".commit(",
        "set_position_qty",
        "Clock",
    ):
        assert word not in source, word
