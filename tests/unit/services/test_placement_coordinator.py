"""PlacementCoordinator: atomic snapshot -> evaluate -> reserve under the account lock."""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import Clock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import Order
from app.execution.account_state import (
    AccountStatePoisonedError,
    InMemoryAccountState,
    PlacementConflictError,
)
from app.execution.models import PlacementRecord
from app.execution.persistence import StoreUncertainError
from app.execution.safety import SafetyController
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.exposure import ExposureCalculationError
from app.risk.manager import evaluate
from app.risk.models import (
    ExposureChange,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    SymbolRiskLimits,
    TradingState,
)
from app.risk.snapshots import build_risk_snapshot
from app.services import placement as placement_module
from app.services.placement import (
    ClientOrderIdGenerator,
    PlacementCoordinator,
    PlacementPreparationError,
)

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
SCOPE = "acct-1"
RUNNING = TradingState.RUNNING
FLAT = D("0")


def new_account(account_scope_id: str = "acct-1") -> InMemoryAccountState:
    """An account state on a fresh in-memory reference store."""
    return InMemoryAccountState(
        account_scope_id=account_scope_id, store=InMemoryAccountStateStore()
    )


class SequenceIds:
    """Deterministic generator: c-1, c-2, ... (counts every call)."""

    def __init__(self) -> None:
        self.calls = 0

    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        self.calls += 1
        return f"c-{self.calls}"


class FixedIds:
    """Returns the same value on every call (invalid or duplicate ids)."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        self.calls += 1
        return self.value  # type: ignore[no-any-return]


class FailingIds:
    def __init__(self) -> None:
        self.calls = 0

    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        self.calls += 1
        raise RuntimeError("id source unavailable")


class CountingClock:
    def __init__(self, at: datetime = T1) -> None:
        self.at = at
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        return self.at


class FailingClock:
    def __init__(self) -> None:
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        raise RuntimeError("clock unavailable")


def limits(max_position_qty: str | None = "10") -> SymbolRiskLimits:
    return SymbolRiskLimits(
        max_order_qty=None,
        max_order_notional=None,
        max_position_qty=None if max_position_qty is None else D(max_position_qty),
    )


def policy(
    *, max_open_orders: int | None = None, max_position_qty: str | None = "10"
) -> RiskPolicy:
    return RiskPolicy(
        policy_id="policy-1",
        max_open_orders=max_open_orders,
        symbols={
            "BTCUSDT": limits(max_position_qty),
            "ETHUSDT": limits(max_position_qty),
        },
    )


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


def ready_safety(
    account: InMemoryAccountState, requested: TradingState = RUNNING
) -> SafetyController:
    """A controller with every recovery gate confirmed: effective == requested."""
    safety = SafetyController(account_state=account)
    safety.mark_hydrated()
    safety.mark_exchange_reconciled()
    safety.request_state(requested)
    return safety


def coordinator(
    *,
    account: InMemoryAccountState | None = None,
    safety: SafetyController | None = None,
    risk_policy: RiskPolicy | None = None,
    clock: Clock | None = None,
    ids: ClientOrderIdGenerator | None = None,
) -> PlacementCoordinator:
    account = new_account(SCOPE) if account is None else account
    return PlacementCoordinator(
        account_state=account,
        safety=ready_safety(account) if safety is None else safety,
        policy=policy() if risk_policy is None else risk_policy,
        clock=CountingClock() if clock is None else clock,
        client_order_id_generator=SequenceIds() if ids is None else ids,
    )


# Known flat BTCUSDT and ETHUSDT are seeded first: two position changes.
SEEDED = 2


async def flat_account(**positions: Decimal | None) -> InMemoryAccountState:
    """Account state with known positions (flat unless overridden; None = unknown)."""
    account = new_account()
    seeds: dict[str, Decimal | None] = {"BTCUSDT": FLAT, "ETHUSDT": FLAT, **positions}
    async with account.account_lock() as locked:
        for symbol, qty in seeds.items():
            await locked.set_position_qty(symbol, qty)
    return account


async def set_position(account: InMemoryAccountState, symbol: str, qty: Decimal | None) -> None:
    async with account.account_lock() as locked:
        await locked.set_position_qty(symbol, qty)


def exchange_fill(
    exec_id: str, client_order_id: str, *, qty: str, price: str = "100", side: Side = Side.BUY
) -> Fill:
    return Fill(
        exec_id=exec_id,
        exchange_order_id=f"ex-{client_order_id}",
        client_order_id=client_order_id,
        symbol="BTCUSDT",
        side=side,
        price=D(price),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=T1,
    )


async def place(
    coord: PlacementCoordinator,
    source: PlaceOrderIntent,
    *,
    trading_state: TradingState | None = None,
) -> PlacementRecord:
    """Place ``source``; ``trading_state`` is first requested on the (fully
    recovered) controller, so it is also the effective state."""
    if trading_state is not None:
        coord.safety.request_state(trading_state)
    return await coord.place(intent=source)


async def registry_state(account: InMemoryAccountState) -> tuple[Any, ...]:
    async with account.account_lock() as locked:
        return (
            locked.revision,
            locked.active_orders("BTCUSDT"),
            locked.active_orders("ETHUSDT"),
            locked.account_active_order_count(),
        )


@pytest.fixture
def evaluate_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Counts the evaluator calls made by the coordinator (real evaluate inside)."""
    calls: list[str] = []
    real: Callable[..., RiskDecision] = evaluate

    def spy(**kwargs: Any) -> RiskDecision:
        calls.append(kwargs["intent"].intent_id)
        return real(**kwargs)

    monkeypatch.setattr(placement_module, "evaluate", spy)
    return calls


# --- construction ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", " acct-1", "acct-1 ", 5, None])
def test_account_scope_id_is_validated_by_the_account_state(value: object) -> None:
    with pytest.raises(DomainValidationError, match="account_scope_id"):
        new_account(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["acct:1", ":", "a:"])
def test_snapshot_delimiter_in_the_account_scope_is_rejected(value: str) -> None:
    account = new_account(value)
    with pytest.raises(DomainValidationError, match="must not contain"):
        PlacementCoordinator(
            account_state=account,
            safety=SafetyController(account_state=account),
            policy=policy(),
            clock=CountingClock(),
            client_order_id_generator=SequenceIds(),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("account_state", object(), "account_state"),
        ("safety", object(), "safety"),
        ("policy", object(), "policy"),
        ("clock", object(), "clock"),
        ("client_order_id_generator", object(), "client_order_id_generator"),
    ],
)
def test_dependencies_are_validated(field: str, value: object, match: str) -> None:
    account = new_account()
    arguments: dict[str, Any] = {
        "account_state": account,
        "safety": SafetyController(account_state=account),
        "policy": policy(),
        "clock": CountingClock(),
        "client_order_id_generator": SequenceIds(),
    }

    with pytest.raises(DomainValidationError, match=match):
        PlacementCoordinator(**{**arguments, field: value})


@pytest.mark.parametrize("value", ["acct-1", "desk/a", "ACCT_7.main"])
def test_valid_account_scope_ids(value: str) -> None:
    account = new_account(value)
    coord = PlacementCoordinator(
        account_state=account,
        safety=SafetyController(account_state=account),
        policy=policy(),
        clock=CountingClock(),
        client_order_id_generator=SequenceIds(),
    )

    assert coord.account_scope_id == value


@pytest.mark.asyncio
async def test_scope_with_a_slash_keeps_the_snapshot_format() -> None:
    account = new_account("desk/a")
    await set_position(account, "BTCUSDT", FLAT)
    await set_position(account, "ETHUSDT", FLAT)
    coord = PlacementCoordinator(
        account_state=account,
        safety=ready_safety(account),
        policy=policy(),
        clock=CountingClock(),
        client_order_id_generator=SequenceIds(),
    )

    record = await place(coord, intent())

    assert record.decision.snapshot_id == f"desk/a:{SEEDED}"


def test_account_scope_id_comes_from_the_account_state() -> None:
    account = new_account("desk/b")

    assert coordinator(account=account).account_scope_id == "desk/b"
    assert account.account_scope_id == "desk/b"


def test_sequence_generator_satisfies_the_protocol() -> None:
    generator: ClientOrderIdGenerator = SequenceIds()

    assert generator.next_id(intent=intent()) == "c-1"
    assert generator.next_id(intent=intent()) == "c-2"


@pytest.mark.asyncio
async def test_non_intent_is_rejected_before_the_lock() -> None:
    account = await flat_account()

    with pytest.raises(DomainValidationError, match="PlaceOrderIntent"):
        await coordinator(account=account).place(
            intent="i-1",  # type: ignore[arg-type]
        )

    assert await account.revision() == SEEDED


# --- approved / rejected --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_placement_reserves_new_order() -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)
    source = intent(qty=D("4"))

    record = await place(coord, source)

    assert record.approved is True
    assert record.client_order_id == "c-1"
    assert record.decision.snapshot_id == f"acct-1:{SEEDED}"
    assert record.decision.policy_id == "policy-1"
    assert record.decision.exposure == ExposureChange(
        reducing_qty=D("0"), increasing_qty=D("4"), worst_long_qty=D("4"), worst_short_qty=D("0")
    )
    order = await account.order("c-1")
    assert order is not None
    assert (order.status, order.qty, order.created_at) == (OrderStatus.NEW, D("4"), T1)
    assert await account.placement("i-1") is record
    assert await account.revision() == SEEDED + 1
    assert await account.account_active_order_count() == 1
    assert (ids.calls, clock.calls) == (1, 1)


@pytest.mark.asyncio
async def test_snapshot_ids_follow_the_revision_read_under_the_lock() -> None:
    coord = coordinator(account=await flat_account())

    first = await place(coord, intent("i-1"))
    rejected = await place(coord, intent("i-2"), trading_state=TradingState.PAUSED)
    second = await place(coord, intent("i-3"))

    assert [r.decision.snapshot_id for r in (first, rejected, second)] == [
        f"acct-1:{SEEDED}",
        f"acct-1:{SEEDED + 1}",
        f"acct-1:{SEEDED + 1}",
    ]


@pytest.mark.asyncio
async def test_second_placement_sees_the_first_reservation() -> None:
    account = await flat_account()
    coord = coordinator(account=account)

    await place(coord, intent("i-1", qty=D("6")))
    second = await place(coord, intent("i-2", qty=D("4")))
    third = await place(coord, intent("i-3", qty=D("0.1")))

    assert second.approved is True
    assert second.decision.exposure is not None
    assert second.decision.exposure.worst_long_qty == D("10")
    assert third.decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert await account.account_active_order_count() == 2


@pytest.mark.parametrize(
    ("trading_state", "reason"),
    [
        (TradingState.HALTED, RiskReason.KILL_SWITCH_ACTIVE),
        (TradingState.PAUSED, RiskReason.TRADING_PAUSED),
        (TradingState.REDUCE_ONLY, RiskReason.REDUCE_ONLY_STATE),
    ],
)
@pytest.mark.asyncio
async def test_rejected_placement_is_recorded_without_reservation(
    trading_state: TradingState, reason: RiskReason
) -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    record = await place(coord, intent(), trading_state=trading_state)

    assert record.approved is False
    assert record.client_order_id is None
    assert record.decision.reasons == (reason,)
    assert record.decision.snapshot_id == f"acct-1:{SEEDED}"
    assert await account.placement("i-1") is record
    assert await registry_state(account) == (SEEDED, (), (), 0)
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_place_has_no_position_argument() -> None:
    parameters = inspect.signature(PlacementCoordinator.place).parameters

    assert set(parameters) == {"self", "intent"}
    coord = coordinator(account=await flat_account())
    with pytest.raises(TypeError):
        await coord.place(intent=intent(), position_qty=FLAT)  # type: ignore[call-arg]
    # The caller can no longer supply a trading state either.
    with pytest.raises(TypeError):
        await coord.place(intent=intent(), trading_state=RUNNING)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_missing_position_is_unknown_not_flat() -> None:
    # Nothing seeded for BTCUSDT: the account never read it as zero.
    account = new_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    record = await place(coord, intent())

    assert record.decision.reasons == (RiskReason.UNKNOWN_POSITION,)
    assert record.decision.snapshot_id == "acct-1:0"
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_cleared_position_is_unknown_again() -> None:
    account = await flat_account()
    await set_position(account, "BTCUSDT", None)
    coord = coordinator(account=account)

    record = await place(coord, intent())

    assert record.decision.reasons == (RiskReason.UNKNOWN_POSITION,)
    assert record.decision.snapshot_id == f"acct-1:{SEEDED + 1}"


@pytest.mark.parametrize(
    ("position", "qty", "approved"),
    [("0", "10", True), ("8", "2", True), ("8", "3", False), ("-9", "10", True)],
)
@pytest.mark.asyncio
async def test_account_position_is_read_under_the_lock(
    position: str, qty: str, approved: bool
) -> None:
    account = await flat_account(BTCUSDT=D(position))
    coord = coordinator(account=account)

    record = await place(coord, intent(qty=D(qty)))

    assert record.approved is approved
    if not approved:
        assert record.decision.reasons == (RiskReason.MAX_POSITION_QTY,)


# --- concurrency ----------------------------------------------------------------------------


async def run_contended(
    account: InMemoryAccountState, coord: PlacementCoordinator, *sources: PlaceOrderIntent
) -> list[PlacementRecord]:
    """Start every placement while the lock is held, so all of them contend for it."""
    async with account.account_lock():
        tasks = [asyncio.create_task(place(coord, source)) for source in sources]
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
    return list(await asyncio.wait_for(asyncio.gather(*tasks), timeout=5))


@pytest.mark.asyncio
async def test_concurrent_intents_cannot_both_pass_max_position() -> None:
    account = await flat_account()
    coord = coordinator(account=account, risk_policy=policy(max_position_qty="10"))
    a, b = intent("i-a", qty=D("6")), intent("i-b", qty=D("6"))

    records = await run_contended(account, coord, a, b)

    winners = [r for r in records if r.approved]
    losers = [r for r in records if not r.approved]
    assert len(winners) == 1
    assert len(losers) == 1
    winner, loser = winners[0], losers[0]
    assert winner.decision.snapshot_id == f"acct-1:{SEEDED}"
    assert loser.decision.snapshot_id == f"acct-1:{SEEDED + 1}"
    assert loser.decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert loser.decision.exposure is not None
    assert loser.decision.exposure.worst_long_qty == D("12")  # pending 6 + candidate 6
    orders = await account.active_orders("BTCUSDT")
    assert [(o.client_order_id, o.status) for o in orders] == [
        (winner.client_order_id, OrderStatus.NEW)
    ]
    assert await account.account_active_order_count() == 1
    assert await account.revision() == SEEDED + 1
    assert await account.placement(loser.intent_id) is loser


@pytest.mark.asyncio
async def test_concurrent_intents_on_different_symbols_share_max_open_orders() -> None:
    account = await flat_account()
    coord = coordinator(account=account, risk_policy=policy(max_open_orders=1))
    btc, eth = intent("i-btc"), intent("i-eth", symbol="ETHUSDT")

    records = await run_contended(account, coord, btc, eth)

    winners = [r for r in records if r.approved]
    losers = [r for r in records if not r.approved]
    assert len(winners) == 1
    assert len(losers) == 1
    loser = losers[0]
    assert loser.decision.reasons == (RiskReason.MAX_OPEN_ORDERS,)
    assert loser.decision.snapshot_id == f"acct-1:{SEEDED + 1}"
    # The loser's own symbol had no orders: only the account-wide view rejects it.
    loser_symbol = loser.intent.symbol
    assert await account.active_orders(loser_symbol) == ()
    assert await account.account_active_order_count() == 1
    assert await account.revision() == SEEDED + 1


@pytest.mark.asyncio
async def test_many_concurrent_intents_never_exceed_the_limit() -> None:
    account = await flat_account()
    coord = coordinator(account=account, risk_policy=policy(max_position_qty="10"))
    sources = [intent(f"i-{n}", qty=D("3")) for n in range(8)]

    records = await run_contended(account, coord, *sources)

    approved = [r for r in records if r.approved]
    assert len(approved) == 3  # 3 * 3 = 9 <= 10, a fourth would reach 12
    assert sum(o.qty for o in await account.active_orders("BTCUSDT")) == D("9")
    assert sorted(r.decision.snapshot_id for r in approved) == [
        f"acct-1:{SEEDED}",
        f"acct-1:{SEEDED + 1}",
        f"acct-1:{SEEDED + 2}",
    ]
    assert {r.decision.snapshot_id for r in records if not r.approved} == {f"acct-1:{SEEDED + 3}"}


# --- orders and position read together -----------------------------------------------------


async def account_with_sent_buy_6() -> tuple[InMemoryAccountState, PlacementCoordinator]:
    """Known flat BTCUSDT and one sent (SUBMITTING) BUY 6, max position 10."""
    account = await flat_account()
    coord = coordinator(account=account, risk_policy=policy(max_position_qty="10"))
    first = await place(coord, intent("i-0", qty=D("6")))
    assert first.client_order_id == "c-1"
    async with account.account_lock() as locked:
        await locked.mark_submitting("c-1", at=T1)
    return account, coord


async def apply_fill_4(account: InMemoryAccountState) -> None:
    async with account.account_lock() as locked:
        await locked.apply_fill(exchange_fill("e-1", "c-1", qty="4"), at=T1)


@pytest.mark.asyncio
async def test_fill_moves_position_and_remainder_atomically_for_risk() -> None:
    account, coord = await account_with_sent_buy_6()

    await apply_fill_4(account)
    record = await place(coord, intent("i-new", qty=D("5")))

    assert await account.position_qty("BTCUSDT") == D("4")
    order = await account.order("c-1")
    assert order is not None
    assert order.qty - order.filled_qty == D("2")
    # 4 (position) + 2 (remaining) + 5 (candidate) = 11 > 10
    assert record.decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert record.decision.exposure is not None
    assert record.decision.exposure.worst_long_qty == D("11")
    assert record.decision.snapshot_id == f"acct-1:{SEEDED + 3}"


def test_the_torn_state_would_have_approved_the_same_intent() -> None:
    # Proof that the scenario matters: position BEFORE the fill with the remainder
    # AFTER it (0 + 2 + 5 = 7) would wrongly pass the limit of 10.
    order = Order(
        client_order_id="c-1",
        exchange_order_id="ex-c-1",
        strategy_id="grid-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D("6"),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_qty=D("4"),
        avg_fill_price=D("100"),
        created_at=T0,
        updated_at=T1,
        last_exchange_update_ts=T1,
        version=2,
    )
    torn = build_risk_snapshot(
        snapshot_id="torn",
        symbol="BTCUSDT",
        trading_state=RUNNING,
        position_qty=FLAT,
        orders=(order,),
        account_open_order_count=1,
    )

    decision = evaluate(intent=intent("i-new", qty=D("5")), snapshot=torn, policy=policy())

    assert decision.approved is True


@pytest.mark.parametrize("fill_first", [True, False])
@pytest.mark.asyncio
async def test_placement_sees_the_state_before_or_after_a_fill_never_a_mix(
    fill_first: bool,
) -> None:
    account, coord = await account_with_sent_buy_6()
    base = await account.revision()

    async with account.account_lock():
        # Both wait on the held lock; asyncio.Lock wakes waiters in FIFO order.
        if fill_first:
            fill_task = asyncio.create_task(apply_fill_4(account))
            place_task = asyncio.create_task(place(coord, intent("i-new", qty=D("5"))))
        else:
            place_task = asyncio.create_task(place(coord, intent("i-new", qty=D("5"))))
            fill_task = asyncio.create_task(apply_fill_4(account))
        await asyncio.sleep(0)
        assert not fill_task.done()
        assert not place_task.done()
    await asyncio.wait_for(asyncio.gather(fill_task, place_task), timeout=5)
    record = place_task.result()

    # Before the fill: 0 + 6 + 5; after it: 4 + 2 + 5. Both are 11, and the
    # snapshot names exactly the revision it was read at.
    expected_revision = base + 1 if fill_first else base
    assert record.decision.snapshot_id == f"acct-1:{expected_revision}"
    assert record.decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert record.decision.exposure is not None
    assert record.decision.exposure.worst_long_qty == D("11")
    assert await account.position_qty("BTCUSDT") == D("4")
    assert await account.revision() == base + 1


@pytest.mark.asyncio
async def test_reduce_only_fill_and_placement_share_one_state() -> None:
    account = await flat_account(BTCUSDT=D("3"))
    coord = coordinator(account=account)
    close = await place(coord, intent("i-close", side=Side.SELL, qty=D("3"), reduce_only=True))
    assert close.approved is True
    async with account.account_lock() as locked:
        await locked.mark_submitting("c-1", at=T1)
        await locked.apply_fill(exchange_fill("e-1", "c-1", qty="3", side=Side.SELL), at=T1)

    record = await place(coord, intent("i-again", side=Side.SELL, qty=D("1"), reduce_only=True))

    assert await account.position_qty("BTCUSDT") == FLAT
    assert record.decision.reasons == (RiskReason.REDUCE_ONLY_WITHOUT_POSITION,)


# --- replay ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_replay_returns_the_record_without_reevaluation(
    evaluate_calls: list[str],
) -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    first = await place(coord, intent(), trading_state=TradingState.PAUSED)
    # Inputs that would now be approved: the recorded result still stands.
    second = await place(coord, intent(), trading_state=RUNNING)

    assert second is first
    assert second.decision.reasons == (RiskReason.TRADING_PAUSED,)
    assert evaluate_calls == ["i-1"]
    assert (ids.calls, clock.calls) == (0, 0)
    assert await registry_state(account) == (SEEDED, (), (), 0)


@pytest.mark.asyncio
async def test_approved_replay_returns_the_same_reservation(evaluate_calls: list[str]) -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    first = await place(coord, intent())
    await set_position(account, "BTCUSDT", None)  # would now be UNKNOWN_POSITION
    second = await place(coord, intent(), trading_state=TradingState.HALTED)

    assert second is first
    assert second.client_order_id == "c-1"
    assert evaluate_calls == ["i-1"]
    assert (ids.calls, clock.calls) == (1, 1)
    assert await account.revision() == SEEDED + 2  # reservation + cleared position
    assert await account.account_active_order_count() == 1


@pytest.mark.asyncio
async def test_conflicting_replay_raises_without_evaluation(evaluate_calls: list[str]) -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)
    await place(coord, intent())

    with pytest.raises(PlacementConflictError, match="i-1"):
        await place(coord, intent(qty=D("2")))

    assert evaluate_calls == ["i-1"]
    assert (ids.calls, clock.calls) == (1, 1)
    assert await account.revision() == SEEDED + 1
    record = await account.placement("i-1")
    assert record is not None
    assert record.intent.qty == D("1")


# --- preparation failure --------------------------------------------------------------------

# A valid domain quantity with more significant digits than Risk's exact context.
HUGE_QTY = D("1." + "1" * 100)


async def seed_unrepresentable_order(account: InMemoryAccountState) -> None:
    """A legitimate NEW reservation whose remainder Risk cannot compute exactly."""
    seed = intent("seed", qty=HUGE_QTY)
    async with account.account_lock() as locked:
        await locked.register_approved(
            intent=seed,
            decision=RiskDecision(
                intent_id="seed",
                snapshot_id="setup",
                policy_id="setup",
                approved=True,
                reasons=(),
                exposure=ExposureChange(
                    reducing_qty=D("0"),
                    increasing_qty=D("1"),
                    worst_long_qty=D("1"),
                    worst_short_qty=D("0"),
                ),
            ),
            client_order_id="seed-1",
            expected_revision=locked.revision,
            at=T0,
        )


@pytest.mark.asyncio
async def test_unrepresentable_snapshot_is_a_preparation_error(
    evaluate_calls: list[str],
) -> None:
    account = await flat_account()
    await seed_unrepresentable_order(account)
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)
    before = await registry_state(account)

    with pytest.raises(PlacementPreparationError, match="BTCUSDT") as caught:
        await place(coord, intent())

    assert isinstance(caught.value.__cause__, ExposureCalculationError)
    assert not isinstance(caught.value, ExposureCalculationError | DomainValidationError)
    assert await account.placement("i-1") is None
    assert await registry_state(account) == before
    assert evaluate_calls == []
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_preparation_error_is_scoped_to_the_symbol_snapshot() -> None:
    account = await flat_account()
    await seed_unrepresentable_order(account)
    coord = coordinator(account=account)

    record = await place(coord, intent("i-eth", symbol="ETHUSDT"))

    assert record.approved is True
    assert record.decision.snapshot_id == f"acct-1:{SEEDED + 1}"


@pytest.mark.parametrize("trading_state", ["running", None, 1, True])
@pytest.mark.asyncio
async def test_invalid_requested_state_is_rejected_and_nothing_is_placed(
    trading_state: Any, evaluate_calls: list[str]
) -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock, safety=ready_safety(account))

    with pytest.raises(DomainValidationError, match="TradingState"):
        coord.safety.request_state(trading_state)
    assert coord.safety.snapshot().requested_state is RUNNING

    assert await account.placement("i-1") is None
    assert await account.revision() == SEEDED
    assert evaluate_calls == []
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_evaluator_programmer_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**kwargs: Any) -> RiskDecision:
        raise DomainValidationError("contract violated")

    monkeypatch.setattr(placement_module, "evaluate", broken)
    account = await flat_account()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    with pytest.raises(DomainValidationError, match="contract violated"):
        await place(coord, intent())

    assert await account.placement("i-1") is None
    assert await account.revision() == SEEDED
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_unrepresentable_intent_is_a_risk_rejection_not_a_preparation_error() -> None:
    account = await flat_account()
    ids = SequenceIds()
    coord = coordinator(account=account, ids=ids)

    record = await place(coord, intent(qty=HUGE_QTY))

    assert record.decision.reasons == (RiskReason.UNREPRESENTABLE_CALCULATION,)
    assert ids.calls == 0
    assert await account.revision() == SEEDED


# --- generator / clock atomicity ------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_failure_leaves_no_reservation() -> None:
    account = await flat_account()
    ids, clock = FailingIds(), CountingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    with pytest.raises(RuntimeError, match="id source unavailable"):
        await place(coord, intent())

    assert ids.calls == 1
    assert clock.calls == 0
    assert await account.placement("i-1") is None
    assert await registry_state(account) == (SEEDED, (), (), 0)


@pytest.mark.parametrize("value", ["", " c-1", None, 7])
@pytest.mark.asyncio
async def test_invalid_generated_id_leaves_no_reservation(value: object) -> None:
    account = await flat_account()
    coord = coordinator(account=account, ids=FixedIds(value))

    with pytest.raises(DomainValidationError, match="client_order_id"):
        await place(coord, intent())

    assert await account.placement("i-1") is None
    assert await registry_state(account) == (SEEDED, (), (), 0)


@pytest.mark.asyncio
async def test_duplicate_generated_id_leaves_state_unchanged() -> None:
    account = await flat_account()
    ids = FixedIds("dup")
    coord = coordinator(account=account, ids=ids)
    await place(coord, intent("i-1"))
    before = await registry_state(account)

    with pytest.raises(PlacementConflictError, match="dup"):
        await place(coord, intent("i-2"))

    assert ids.calls == 2  # the generator's own side effect happened
    assert await registry_state(account) == before
    assert await account.placement("i-2") is None


@pytest.mark.asyncio
async def test_clock_failure_leaves_no_reservation() -> None:
    account = await flat_account()
    ids, clock = SequenceIds(), FailingClock()
    coord = coordinator(account=account, ids=ids, clock=clock)

    with pytest.raises(RuntimeError, match="clock unavailable"):
        await place(coord, intent())

    assert (ids.calls, clock.calls) == (1, 1)  # id first, then the clock
    assert await account.placement("i-1") is None
    assert await registry_state(account) == (SEEDED, (), (), 0)


@pytest.mark.parametrize(
    ("at", "match"),
    [
        (T0 - timedelta(microseconds=1), "before"),
        (datetime(2026, 1, 15, 12, 0), "at"),  # noqa: DTZ001 - naive on purpose
    ],
)
@pytest.mark.asyncio
async def test_invalid_reservation_time_leaves_no_reservation(at: datetime, match: str) -> None:
    account = await flat_account()
    coord = coordinator(account=account, clock=CountingClock(at))

    with pytest.raises(DomainValidationError, match=match):
        await place(coord, intent())

    assert await account.placement("i-1") is None
    assert await registry_state(account) == (SEEDED, (), (), 0)


@pytest.mark.asyncio
async def test_lock_is_released_after_a_failure() -> None:
    account = await flat_account()

    with pytest.raises(RuntimeError):
        await place(coordinator(account=account, ids=FailingIds()), intent())

    record = await asyncio.wait_for(place(coordinator(account=account), intent()), timeout=1)
    assert record.approved is True


# --- Decimal context ------------------------------------------------------------------------


async def awkward_scenario() -> tuple[list[PlacementRecord], tuple[Any, ...], Any]:
    account = await flat_account()
    coord = coordinator(account=account, risk_policy=policy(max_position_qty="10"))
    records = [
        await place(coord, intent("i-1", qty=D("3.333333333333333333333333333"))),
        await place(coord, intent("i-2", qty=D("6.666666666666666666666666667"))),
    ]
    # c-1 is sent and partly executed: the position grows, its remainder shrinks.
    async with account.account_lock() as locked:
        await locked.mark_submitting("c-1", at=T1)
        await locked.apply_fill(
            exchange_fill("e-1", "c-1", qty="1.111111111111111111111111111", price="100.3"),
            at=T1,
        )
    records += [
        await place(coord, intent("i-3", qty=D("0.000000000000000000000000001"))),
        await place(coord, intent("i-4", side=Side.SELL, qty=D("1.1"), reduce_only=True)),
    ]
    return records, await registry_state(account), await account.position_qty("BTCUSDT")


def context_state() -> tuple[object, ...]:
    context = getcontext()
    return (
        context.prec,
        context.rounding,
        dict(context.traps),
        dict(context.flags),
        context.Emin,
        context.Emax,
    )


@pytest.mark.asyncio
async def test_full_path_is_exact_under_a_low_precision_context() -> None:
    expected_records, expected_state, expected_position = await awkward_scenario()
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        records, state, position = await awkward_scenario()

    assert [r.decision for r in records] == [r.decision for r in expected_records]
    assert [r.client_order_id for r in records] == [r.client_order_id for r in expected_records]
    assert (state, position) == (expected_state, expected_position)
    assert [r.approved for r in records] == [True, True, False, True]
    assert records[1].decision.exposure is not None
    assert records[1].decision.exposure.worst_long_qty == D("10.000000000000000000000000000")
    assert position == D("1.111111111111111111111111111")
    # position 1.1..1 + remaining 2.2..2 + pending 6.6..7 = 10, plus 1E-27
    assert records[2].decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert records[2].decision.exposure is not None
    assert records[2].decision.exposure.worst_long_qty == D("10.000000000000000000000000001")
    assert records[3].decision.exposure is not None
    assert records[3].decision.exposure.reducing_qty == D("1.1")
    assert state[0] == SEEDED + 5  # 2 reservations, SUBMITTING, fill, 1 reservation


@pytest.mark.asyncio
async def test_global_decimal_context_is_unchanged() -> None:
    getcontext().clear_flags()
    before = context_state()

    await awkward_scenario()

    assert context_state() == before


# --- dependencies ---------------------------------------------------------------------------


def test_coordinator_has_no_network_identity_or_time_dependencies() -> None:
    source = Path(placement_module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith(
            ("app.domain", "app.risk", "app.execution")
        ), name
    banned = {"uuid", "random", "secrets", "time", "httpx", "socket", "sqlite3", "logging"}
    assert not {name.split(".")[0] for name in imports} & banned
    for word in ("place_order", "TradingClient", "exchanges", "datetime.now", "sleep("):
        assert word not in source, word


# --- safety controller ----------------------------------------------------------------------


@pytest.fixture
def risk_states(monkeypatch: pytest.MonkeyPatch) -> list[TradingState]:
    """The trading_state of every RiskSnapshot the coordinator evaluates."""
    seen: list[TradingState] = []
    real = evaluate

    def spy(**kwargs: Any) -> Any:
        seen.append(kwargs["snapshot"].trading_state)
        return real(**kwargs)

    monkeypatch.setattr(placement_module, "evaluate", spy)
    return seen


def test_safety_must_cover_the_same_account() -> None:
    account = new_account()
    with pytest.raises(DomainValidationError, match="same account_state"):
        PlacementCoordinator(
            account_state=account,
            safety=SafetyController(account_state=new_account()),
            policy=policy(),
            clock=CountingClock(),
            client_order_id_generator=SequenceIds(),
        )


@pytest.mark.asyncio
async def test_incomplete_recovery_makes_risk_see_paused(risk_states: list[TradingState]) -> None:
    account = await flat_account()
    safety = SafetyController(account_state=account)
    safety.mark_hydrated()  # hydrated, exchange gates not confirmed yet
    safety.request_state(RUNNING)
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(account=account, safety=safety, ids=ids, clock=clock)

    record = await coord.place(intent=intent())

    assert risk_states == [TradingState.PAUSED]
    assert record.approved is False
    assert record.decision.reasons == (RiskReason.TRADING_PAUSED,)
    assert await registry_state(account) == (SEEDED, (), (), 0)
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_default_controller_makes_risk_see_paused(risk_states: list[TradingState]) -> None:
    account = await flat_account()
    coord = coordinator(account=account, safety=SafetyController(account_state=account))

    record = await coord.place(intent=intent())

    assert risk_states == [TradingState.PAUSED]
    assert record.decision.reasons == (RiskReason.TRADING_PAUSED,)


@pytest.mark.parametrize("requested", [RUNNING, TradingState.REDUCE_ONLY])
@pytest.mark.asyncio
async def test_complete_recovery_passes_the_requested_state_to_risk(
    requested: TradingState, risk_states: list[TradingState]
) -> None:
    account = await flat_account()
    coord = coordinator(account=account, safety=ready_safety(account, requested))

    record = await coord.place(intent=intent())

    assert risk_states == [requested]
    if requested is RUNNING:
        assert record.approved is True
    else:
        assert record.decision.reasons == (RiskReason.REDUCE_ONLY_STATE,)


@pytest.mark.asyncio
async def test_the_atomic_exchange_confirmation_lets_the_next_placement_run(
    risk_states: list[TradingState],
) -> None:
    account = await flat_account()
    safety = SafetyController(account_state=account)
    safety.request_state(RUNNING)
    safety.mark_hydrated()
    coord = coordinator(account=account, safety=safety)

    first = await coord.place(intent=intent("i-1"))
    safety.mark_exchange_reconciled()
    second = await coord.place(intent=intent("i-2"))

    assert risk_states == [TradingState.PAUSED, RUNNING]
    assert first.approved is False
    assert second.approved is True


@pytest.mark.asyncio
async def test_poison_after_recovery_pauses_and_placement_fails_closed_before_risk(
    risk_states: list[TradingState],
) -> None:
    store = InMemoryAccountStateStore()
    account = InMemoryAccountState(account_scope_id=SCOPE, store=store)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", FLAT)
    safety = ready_safety(account)
    coord = coordinator(account=account, safety=safety)
    assert safety.effective_state is RUNNING
    store.inject_commit_failure(CommitFailure.UNCERTAIN)
    async with account.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty("ETHUSDT", FLAT)

    effective = safety.effective_state
    assert effective is TradingState.PAUSED
    # The RiskSnapshot a placement would build from it is rejected as paused...
    async with account.account_lock() as locked:
        snapshot = build_risk_snapshot(
            snapshot_id=f"{SCOPE}:{locked.revision}",
            symbol="BTCUSDT",
            trading_state=effective,
            position_qty=locked.position_qty("BTCUSDT"),
            orders=locked.active_orders("BTCUSDT"),
            account_open_order_count=locked.account_active_order_count(),
        )
    decision = evaluate(intent=intent(), snapshot=snapshot, policy=policy())
    assert decision.reasons == (RiskReason.TRADING_PAUSED,)
    # ...and the coordinator does not even get that far: poison fails first.
    with pytest.raises(AccountStatePoisonedError):
        await coord.place(intent=intent())
    assert risk_states == []


@pytest.mark.asyncio
async def test_replay_is_returned_while_recovery_is_incomplete(
    risk_states: list[TradingState], monkeypatch: pytest.MonkeyPatch
) -> None:
    account = await flat_account()
    safety = ready_safety(account)
    coord = coordinator(account=account, safety=safety)
    approved = await coord.place(intent=intent("i-1"))
    assert approved.approved is True

    # Same account, a new (incomplete) session controller: effective PAUSED.
    fresh = SafetyController(account_state=account)
    fresh.request_state(RUNNING)
    reads: list[int] = []
    real_snapshot = SafetyController.snapshot

    def counting_snapshot(self: SafetyController) -> Any:
        reads.append(1)
        return real_snapshot(self)

    monkeypatch.setattr(SafetyController, "snapshot", counting_snapshot)
    ids, clock = SequenceIds(), CountingClock()
    paused = coordinator(account=account, safety=fresh, ids=ids, clock=clock)

    assert await paused.place(intent=intent("i-1")) is approved
    assert reads == []  # a replay never consults safety
    assert risk_states == [RUNNING]  # only the original evaluation
    assert (ids.calls, clock.calls) == (0, 0)

    record = await paused.place(intent=intent("i-2"))
    assert reads == [1]  # one safety read per new placement
    assert record.decision.reasons == (RiskReason.TRADING_PAUSED,)
