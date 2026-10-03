"""Durable position projection: a known durable position always covers every
committed fill of its symbol, also while the runtime position is unknown."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.exchanges.recovery import (
    ExchangeExecution,
    ExchangeOrder,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
)
from app.execution.account_state import (
    AccountStatePoisonedError,
    FillApplicationError,
    FillConflictError,
    InMemoryAccountState,
    PositionProjectionMismatchError,
)
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.fill_recovery import recover_missing_fills
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    PersistedPosition,
    StoreCommitError,
    StoreUncertainError,
)
from app.execution.recovery_matching import ManagedOrderMatch
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision

D = Decimal
S = OrderStatus
SYMBOL = "BTCUSDT"
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NS = ClientOrderNamespace("bot01")


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


class Store:
    """Reference store recording changes; failures armed per commit number."""

    def __init__(self, inner: InMemoryAccountStateStore | None = None) -> None:
        self.inner = InMemoryAccountStateStore() if inner is None else inner
        self.changes: list[AccountStateChange] = []
        self.failures: dict[int, CommitFailure] = {}

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        failure = self.failures.get(len(self.changes))
        if failure is not None:
            self.inner.inject_commit_failure(failure)
        await self.inner.commit(change)

    def arm(self) -> None:
        self.changes.clear()
        self.failures.clear()


async def durable(store: Store) -> tuple[PersistedPosition | None, int, int]:
    loaded = await store.load(account_scope_id="acct-1")
    assert loaded is not None
    return loaded.positions.get(SYMBOL), loaded.revision, len(loaded.fills)


def row(qty: str | None) -> PersistedPosition:
    return PersistedPosition(
        symbol=SYMBOL, known=qty is not None, qty=None if qty is None else D(qty)
    )


def cid(n: int) -> str:
    return NS.build(f"o{n}")


async def reserve(
    account: InMemoryAccountState, n: int, side: Side, qty: str, *, reduce_only: bool = False
) -> None:
    source = PlaceOrderIntent(
        intent_id=f"i-{n}",
        strategy_id="grid-1",
        symbol=SYMBOL,
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


def fill(n: int, exec_id: str, side: Side, qty: str, ts: int = 1, **overrides: Any) -> Fill:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": f"X-{n}",
        "client_order_id": cid(n),
        "symbol": SYMBOL,
        "side": side,
        "price": D("100"),
        "qty": D(qty),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "exchange_ts": at(ts),
    }
    return Fill(**{**values, **overrides})


async def apply(account: InMemoryAccountState, f: Fill) -> None:
    async with account.account_lock() as locked:
        await locked.apply_fill(f, at=at(10))


async def hydrate(store: Store) -> InMemoryAccountState:
    return await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=store, clock=ManualClock(at(5))
    )


async def crashed_with(
    position: str | None,
    orders: tuple[tuple[int, Side, str, bool], ...],
    *,
    unknown_row: bool = False,
) -> tuple[InMemoryAccountState, Store]:
    """A fresh account (position seeded or not), orders reserved and SUBMITTING,
    then a restart: the returned account is the hydrated one."""
    store = Store()
    first = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with first.account_lock() as locked:
        if position is not None:
            await locked.set_position_qty(SYMBOL, D(position))
        elif unknown_row:
            await locked.set_position_qty(SYMBOL, D("1"))
            await locked.set_position_qty(SYMBOL, None)
    for n, side, qty, reduce_only in orders:
        await reserve(first, n, side, qty, reduce_only=reduce_only)
    account = await hydrate(store)
    store.arm()
    return account, store


# --- hydrate / evidence API -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_hydrate_restores_the_durable_projection_but_not_the_runtime_position() -> None:
    account, _ = await crashed_with("2", ())

    assert await account.position_qty(SYMBOL) is None
    assert await account.durable_position(SYMBOL) == row("2")
    assert await account.durable_position("ETHUSDT") is None


@pytest.mark.asyncio
async def test_evidence_api_distinguishes_absent_unknown_flat_and_open() -> None:
    absent, _ = await crashed_with(None, ())
    unknown, _ = await crashed_with(None, (), unknown_row=True)
    flat, _ = await crashed_with("0", ())
    long_, _ = await crashed_with("3", ())

    assert await absent.durable_position(SYMBOL) is None
    assert await unknown.durable_position(SYMBOL) == row(None)
    assert await flat.durable_position(SYMBOL) == row("0")
    assert await long_.durable_position(SYMBOL) == row("3")
    for account in (absent, unknown, flat, long_):
        assert await account.position_qty(SYMBOL) is None


@pytest.mark.asyncio
async def test_empty_store_has_no_durable_rows() -> None:
    account = await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=Store(), clock=ManualClock(T0)
    )
    assert await account.durable_position(SYMBOL) is None
    assert account._state.durable_positions == {}


# --- set_position_qty -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_position_sets_runtime_and_durable_together() -> None:
    account, store = await crashed_with("2", ())
    async with account.account_lock() as locked:
        await locked.set_position_qty(SYMBOL, D("4"))
        assert (locked.position_qty(SYMBOL), locked.durable_position(SYMBOL)) == (D("4"), row("4"))
        await locked.set_position_qty(SYMBOL, D("4"))  # no-op
        await locked.set_position_qty(SYMBOL, None)
        assert (locked.position_qty(SYMBOL), locked.durable_position(SYMBOL)) == (None, row(None))
        await locked.set_position_qty(SYMBOL, None)  # no-op
    assert [c.position_writes for c in store.changes] == [(row("4"),), (row(None),)]


@pytest.mark.asyncio
async def test_clearing_a_hydrated_known_projection_writes_unknown() -> None:
    account, store = await crashed_with("2", ())
    async with account.account_lock() as locked:
        await locked.set_position_qty(SYMBOL, None)
    assert [c.position_writes for c in store.changes] == [(row(None),)]
    assert (await durable(store))[0] == row(None)


# --- the audited regression -----------------------------------------------------------------


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
        symbol=SYMBOL,
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
async def test_recovered_fill_after_hydrate_advances_the_known_projection() -> None:
    # Before the crash: known 2 = the projection of the one applied fill.
    store = Store()
    first = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with first.account_lock() as locked:
        await locked.set_position_qty(SYMBOL, D("0"))
    await reserve(first, 1, Side.BUY, "10")
    e1 = trade("e-1", "2", 1)
    async with first.account_lock() as locked:
        await locked.apply_fill(fill(1, "e-1", Side.BUY, "2"), at=at(1))
    assert await durable(store) == (row("2"), 4, 1)

    account = await hydrate(store)
    assert await account.position_qty(SYMBOL) is None
    assert await account.durable_position(SYMBOL) == row("2")
    store.arm()

    local = await account.order(cid(1))
    assert local is not None
    exchange = ExchangeOrder(
        exchange_order_id="X-1",
        client_order_id=cid(1),
        symbol=SYMBOL,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D("10"),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        status=S.PARTIALLY_FILLED,
        cum_filled_qty=D("5"),
        cum_filled_notional=None,
        avg_fill_price=D("100"),
        created_ts=T0,
        updated_ts=at(3),
    )
    result = await recover_missing_fills(
        account_state=account,
        reader=Reader([e1, trade("e-2", "3", 2)]),
        match=ManagedOrderMatch(
            local_order=local, exchange_order=exchange, exchange_id_completion_required=False
        ),
        query=ExecutionQuery(symbol=SYMBOL, exchange_order_id="X-1", start=T0, end=at(100)),
        clock=ManualClock(at(6)),
    )

    assert result.fills_applied == 1
    assert await account.position_qty(SYMBOL) is None  # runtime: still unknown
    assert await account.durable_position(SYMBOL) == row("5")
    assert await durable(store) == (row("5"), 5, 2)
    (change,) = store.changes
    assert (change.expected_revision, change.new_revision) == (4, 5)
    assert [f.exec_id for f in change.fill_writes] == ["e-2"]
    assert change.position_writes == (row("5"),)
    assert len(change.order_writes) == len(change.notional_writes) == 1

    # A second restart keeps the evidence.
    again = await hydrate(store)
    assert await again.position_qty(SYMBOL) is None
    assert await again.durable_position(SYMBOL) == row("5")


# --- signed cases ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "side", "qty", "expected"),
    [
        ("5", Side.SELL, "2", "3"),
        ("2", Side.SELL, "3", "-1"),  # crossing zero (not reduce-only)
        ("-2", Side.SELL, "3", "-5"),
        ("-2", Side.BUY, "1", "-1"),
        ("-2", Side.BUY, "2", "0"),
        ("0", Side.BUY, "0.000000001", "0.000000001"),
        ("1" + "0" * 30, Side.SELL, "1", "9" * 30),
        ("2.000", Side.BUY, "3.0", "5.000"),
    ],
)
@pytest.mark.asyncio
async def test_known_projection_follows_signed_fills_while_runtime_unknown(
    start: str, side: Side, qty: str, expected: str
) -> None:
    account, store = await crashed_with(start, ((1, side, qty, False),))

    await apply(account, fill(1, "e-1", side, qty))

    assert await account.position_qty(SYMBOL) is None
    assert await account.durable_position(SYMBOL) == row(expected)
    assert store.changes[0].position_writes == (row(expected),)


@pytest.mark.parametrize("unknown_row", [False, True], ids=["absent", "known-false"])
@pytest.mark.asyncio
async def test_unknown_projection_never_becomes_known(unknown_row: bool) -> None:
    account, store = await crashed_with(None, ((1, Side.BUY, "3", False),), unknown_row=unknown_row)
    before = await account.durable_position(SYMBOL)

    await apply(account, fill(1, "e-1", Side.BUY, "3"))

    assert await account.position_qty(SYMBOL) is None
    assert await account.durable_position(SYMBOL) == before
    (change,) = store.changes
    assert change.position_writes == ()
    assert [f.exec_id for f in change.fill_writes] == ["e-1"]
    order = await account.order(cid(1))
    assert order is not None
    assert order.filled_qty == D("3")


@pytest.mark.asyncio
async def test_runtime_known_path_moves_both_views_in_one_change() -> None:
    store = Store()
    account = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty(SYMBOL, D("1"))
    await reserve(account, 1, Side.BUY, "4")
    store.arm()

    await apply(account, fill(1, "e-1", Side.BUY, "4"))

    assert await account.position_qty(SYMBOL) == D("5")
    assert await account.durable_position(SYMBOL) == row("5")
    assert store.changes[0].position_writes == (row("5"),)


# --- reduce-only / disagreement -------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "side", "qty"),
    [("2", Side.BUY, "1"), ("2", Side.SELL, "3"), ("0", Side.SELL, "1"), ("-2", Side.BUY, "3")],
    ids=["increase", "reverse-long", "from-flat", "reverse-short"],
)
@pytest.mark.asyncio
async def test_reduce_only_is_checked_against_the_known_projection(
    start: str, side: Side, qty: str
) -> None:
    account, store = await crashed_with(start, ((1, side, qty, True),))

    with pytest.raises(FillApplicationError, match="reduce-only"):
        await apply(account, fill(1, "e-1", side, qty))

    assert store.changes == []
    assert await account.durable_position(SYMBOL) == row(start)


@pytest.mark.asyncio
async def test_reduce_only_within_the_known_projection_applies() -> None:
    account, _ = await crashed_with("2", ((1, Side.SELL, "2", True),))
    await apply(account, fill(1, "e-1", Side.SELL, "2"))
    assert await account.durable_position(SYMBOL) == row("0")


@pytest.mark.asyncio
async def test_reduce_only_with_unknown_projection_keeps_the_existing_contract() -> None:
    account, _ = await crashed_with(None, ((1, Side.SELL, "2", True),))
    await apply(account, fill(1, "e-1", Side.SELL, "2"))  # unknown stays unknown
    assert await account.durable_position(SYMBOL) is None


@pytest.mark.asyncio
async def test_runtime_and_durable_disagreement_fails_closed() -> None:
    account, store = await crashed_with("2", ((1, Side.BUY, "1", False),))
    account._state = dataclasses.replace(account._state, positions={SYMBOL: D("3")})  # white-box
    before = account._state

    with pytest.raises(PositionProjectionMismatchError, match="differs"):
        await apply(account, fill(1, "e-1", Side.BUY, "1"))

    assert store.changes == []
    assert account._state is before


@pytest.mark.asyncio
async def test_runtime_known_with_unknown_projection_moves_runtime_only() -> None:
    account, store = await crashed_with(None, ((1, Side.BUY, "1", False),))
    account._state = dataclasses.replace(account._state, positions={SYMBOL: D("3")})  # white-box

    await apply(account, fill(1, "e-1", Side.BUY, "1"))

    assert await account.position_qty(SYMBOL) == D("4")
    assert await account.durable_position(SYMBOL) is None
    assert store.changes[0].position_writes == ()


# --- idempotency / conflicts ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_replay_changes_nothing() -> None:
    account, store = await crashed_with("2", ((1, Side.BUY, "5", False),))
    f = fill(1, "e-1", Side.BUY, "3")
    await apply(account, f)
    revision = await account.revision()

    await apply(account, f)

    assert len(store.changes) == 1
    assert await account.revision() == revision
    assert await account.durable_position(SYMBOL) == row("5")


@pytest.mark.asyncio
async def test_conflicting_replay_leaves_the_projection() -> None:
    account, store = await crashed_with("2", ((1, Side.BUY, "5", False),))
    await apply(account, fill(1, "e-1", Side.BUY, "3"))

    with pytest.raises(FillConflictError):
        await apply(account, fill(1, "e-1", Side.BUY, "2"))

    assert len(store.changes) == 1
    assert await account.durable_position(SYMBOL) == row("5")


# --- store failures / copy-on-write ---------------------------------------------------------


@pytest.mark.asyncio
async def test_definite_failure_publishes_nothing_and_keeps_the_old_dict() -> None:
    account, store = await crashed_with("2", ((1, Side.BUY, "5", False),))
    store.failures[1] = CommitFailure.DEFINITE
    state = account._state
    old_rows = state.durable_positions
    snapshot = dict(old_rows)

    with pytest.raises(StoreCommitError):
        await apply(account, fill(1, "e-1", Side.BUY, "3"))

    assert account._state is state
    assert old_rows == snapshot
    assert await account.durable_position(SYMBOL) == row("2")
    assert await account.fill("e-1") is None
    assert await durable(store) == (row("2"), state.revision, 0)


@pytest.mark.asyncio
async def test_uncertain_failure_poisons_and_publishes_nothing() -> None:
    account, store = await crashed_with("2", ((1, Side.BUY, "5", False),))
    store.failures[1] = CommitFailure.UNCERTAIN
    state = account._state

    with pytest.raises(StoreUncertainError):
        await apply(account, fill(1, "e-1", Side.BUY, "3"))

    assert account.is_poisoned
    assert account._state is state
    assert await account.durable_position(SYMBOL) == row("2")  # RAM: last confirmed
    with pytest.raises(AccountStatePoisonedError):
        await apply(account, fill(1, "e-2", Side.BUY, "1"))


@pytest.mark.asyncio
async def test_success_replaces_the_dict_without_mutating_the_published_one() -> None:
    account, _ = await crashed_with("2", ((1, Side.BUY, "5", False),))
    old_rows = account._state.durable_positions

    await apply(account, fill(1, "e-1", Side.BUY, "3"))

    assert old_rows == {SYMBOL: row("2")}
    assert account._state.durable_positions is not old_rows


# --- concurrency ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_fills_of_one_symbol_both_advance_the_projection() -> None:
    account, store = await crashed_with("0", ((1, Side.BUY, "2", False), (2, Side.BUY, "3", False)))
    revision = await account.revision()

    await asyncio.gather(
        apply(account, fill(1, "e-1", Side.BUY, "2")),
        apply(account, fill(2, "e-2", Side.BUY, "3")),
    )

    assert await account.durable_position(SYMBOL) == row("5")
    assert await account.revision() == revision + 2
    assert (await durable(store))[0] == row("5")


@pytest.mark.asyncio
async def test_concurrent_replays_of_one_exec_id_apply_once() -> None:
    account, store = await crashed_with("0", ((1, Side.BUY, "2", False),))
    f = fill(1, "e-1", Side.BUY, "2")

    await asyncio.gather(apply(account, f), apply(account, f))

    assert len(store.changes) == 1
    assert await account.durable_position(SYMBOL) == row("2")
