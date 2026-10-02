"""PlacementCoordinator: atomic snapshot -> evaluate -> reserve under the account lock."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import Clock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.execution.models import PlacementRecord
from app.execution.registry import InMemoryOrderRegistry, PlacementConflictError
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


def coordinator(
    *,
    registry: InMemoryOrderRegistry | None = None,
    risk_policy: RiskPolicy | None = None,
    clock: Clock | None = None,
    ids: ClientOrderIdGenerator | None = None,
) -> PlacementCoordinator:
    return PlacementCoordinator(
        account_scope_id=SCOPE,
        registry=InMemoryOrderRegistry() if registry is None else registry,
        policy=policy() if risk_policy is None else risk_policy,
        clock=CountingClock() if clock is None else clock,
        client_order_id_generator=SequenceIds() if ids is None else ids,
    )


async def place(
    coord: PlacementCoordinator,
    source: PlaceOrderIntent,
    *,
    position_qty: Decimal | None = FLAT,
    trading_state: TradingState = RUNNING,
) -> PlacementRecord:
    return await coord.place(intent=source, position_qty=position_qty, trading_state=trading_state)


async def registry_state(registry: InMemoryOrderRegistry) -> tuple[Any, ...]:
    async with registry.placement_lock() as locked:
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
def test_account_scope_id_is_validated_without_normalization(value: object) -> None:
    with pytest.raises(DomainValidationError, match="account_scope_id"):
        PlacementCoordinator(
            account_scope_id=value,  # type: ignore[arg-type]
            registry=InMemoryOrderRegistry(),
            policy=policy(),
            clock=CountingClock(),
            client_order_id_generator=SequenceIds(),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("registry", object(), "registry"),
        ("policy", object(), "policy"),
        ("clock", object(), "clock"),
        ("client_order_id_generator", object(), "client_order_id_generator"),
    ],
)
def test_dependencies_are_validated(field: str, value: object, match: str) -> None:
    arguments: dict[str, Any] = {
        "account_scope_id": SCOPE,
        "registry": InMemoryOrderRegistry(),
        "policy": policy(),
        "clock": CountingClock(),
        "client_order_id_generator": SequenceIds(),
    }

    with pytest.raises(DomainValidationError, match=match):
        PlacementCoordinator(**{**arguments, field: value})


def test_account_scope_id_is_exposed() -> None:
    assert coordinator().account_scope_id == SCOPE


def test_sequence_generator_satisfies_the_protocol() -> None:
    generator: ClientOrderIdGenerator = SequenceIds()

    assert generator.next_id(intent=intent()) == "c-1"
    assert generator.next_id(intent=intent()) == "c-2"


@pytest.mark.asyncio
async def test_non_intent_is_rejected_before_the_lock() -> None:
    registry = InMemoryOrderRegistry()

    with pytest.raises(DomainValidationError, match="PlaceOrderIntent"):
        await coordinator(registry=registry).place(
            intent="i-1",  # type: ignore[arg-type]
            position_qty=FLAT,
            trading_state=RUNNING,
        )

    assert await registry.revision() == 0


# --- approved / rejected --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_placement_reserves_new_order() -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)
    source = intent(qty=D("4"))

    record = await place(coord, source)

    assert record.approved is True
    assert record.client_order_id == "c-1"
    assert record.decision.snapshot_id == "acct-1:0"
    assert record.decision.policy_id == "policy-1"
    assert record.decision.exposure == ExposureChange(
        reducing_qty=D("0"), increasing_qty=D("4"), worst_long_qty=D("4"), worst_short_qty=D("0")
    )
    order = await registry.order("c-1")
    assert order is not None
    assert (order.status, order.qty, order.created_at) == (OrderStatus.NEW, D("4"), T1)
    assert await registry.placement("i-1") is record
    assert await registry.revision() == 1
    assert await registry.account_active_order_count() == 1
    assert (ids.calls, clock.calls) == (1, 1)


@pytest.mark.asyncio
async def test_snapshot_ids_follow_the_revision_read_under_the_lock() -> None:
    coord = coordinator()

    first = await place(coord, intent("i-1"))
    rejected = await place(coord, intent("i-2"), trading_state=TradingState.PAUSED)
    second = await place(coord, intent("i-3"))

    assert [r.decision.snapshot_id for r in (first, rejected, second)] == [
        "acct-1:0",
        "acct-1:1",
        "acct-1:1",
    ]


@pytest.mark.asyncio
async def test_second_placement_sees_the_first_reservation() -> None:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry)

    await place(coord, intent("i-1", qty=D("6")))
    second = await place(coord, intent("i-2", qty=D("4")))
    third = await place(coord, intent("i-3", qty=D("0.1")))

    assert second.approved is True
    assert second.decision.exposure is not None
    assert second.decision.exposure.worst_long_qty == D("10")
    assert third.decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert await registry.account_active_order_count() == 2


@pytest.mark.parametrize(
    ("trading_state", "position_qty", "reason"),
    [
        (TradingState.HALTED, FLAT, RiskReason.KILL_SWITCH_ACTIVE),
        (TradingState.PAUSED, FLAT, RiskReason.TRADING_PAUSED),
        (TradingState.REDUCE_ONLY, FLAT, RiskReason.REDUCE_ONLY_STATE),
        (RUNNING, None, RiskReason.UNKNOWN_POSITION),  # None is unknown, never zero
    ],
)
@pytest.mark.asyncio
async def test_rejected_placement_is_recorded_without_reservation(
    trading_state: TradingState, position_qty: Decimal | None, reason: RiskReason
) -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    record = await place(coord, intent(), position_qty=position_qty, trading_state=trading_state)

    assert record.approved is False
    assert record.client_order_id is None
    assert record.decision.reasons == (reason,)
    assert record.decision.snapshot_id == "acct-1:0"
    assert await registry.placement("i-1") is record
    assert await registry_state(registry) == (0, (), (), 0)
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_caller_position_is_passed_to_risk() -> None:
    coord = coordinator()

    record = await place(coord, intent(qty=D("3")), position_qty=D("8"))

    assert record.decision.reasons == (RiskReason.MAX_POSITION_QTY,)


# --- concurrency ----------------------------------------------------------------------------


async def run_contended(
    registry: InMemoryOrderRegistry, coord: PlacementCoordinator, *sources: PlaceOrderIntent
) -> list[PlacementRecord]:
    """Start every placement while the lock is held, so all of them contend for it."""
    async with registry.placement_lock():
        tasks = [asyncio.create_task(place(coord, source)) for source in sources]
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
    return list(await asyncio.wait_for(asyncio.gather(*tasks), timeout=5))


@pytest.mark.asyncio
async def test_concurrent_intents_cannot_both_pass_max_position() -> None:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry, risk_policy=policy(max_position_qty="10"))
    a, b = intent("i-a", qty=D("6")), intent("i-b", qty=D("6"))

    records = await run_contended(registry, coord, a, b)

    winners = [r for r in records if r.approved]
    losers = [r for r in records if not r.approved]
    assert len(winners) == 1
    assert len(losers) == 1
    winner, loser = winners[0], losers[0]
    assert winner.decision.snapshot_id == "acct-1:0"
    assert loser.decision.snapshot_id == "acct-1:1"
    assert loser.decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert loser.decision.exposure is not None
    assert loser.decision.exposure.worst_long_qty == D("12")  # pending 6 + candidate 6
    orders = await registry.active_orders("BTCUSDT")
    assert [(o.client_order_id, o.status) for o in orders] == [
        (winner.client_order_id, OrderStatus.NEW)
    ]
    assert await registry.account_active_order_count() == 1
    assert await registry.revision() == 1
    assert await registry.placement(loser.intent_id) is loser


@pytest.mark.asyncio
async def test_concurrent_intents_on_different_symbols_share_max_open_orders() -> None:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry, risk_policy=policy(max_open_orders=1))
    btc, eth = intent("i-btc"), intent("i-eth", symbol="ETHUSDT")

    records = await run_contended(registry, coord, btc, eth)

    winners = [r for r in records if r.approved]
    losers = [r for r in records if not r.approved]
    assert len(winners) == 1
    assert len(losers) == 1
    loser = losers[0]
    assert loser.decision.reasons == (RiskReason.MAX_OPEN_ORDERS,)
    assert loser.decision.snapshot_id == "acct-1:1"
    # The loser's own symbol had no orders: only the account-wide view rejects it.
    loser_symbol = loser.intent.symbol
    assert await registry.active_orders(loser_symbol) == ()
    assert await registry.account_active_order_count() == 1
    assert await registry.revision() == 1


@pytest.mark.asyncio
async def test_many_concurrent_intents_never_exceed_the_limit() -> None:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry, risk_policy=policy(max_position_qty="10"))
    sources = [intent(f"i-{n}", qty=D("3")) for n in range(8)]

    records = await run_contended(registry, coord, *sources)

    approved = [r for r in records if r.approved]
    assert len(approved) == 3  # 3 * 3 = 9 <= 10, a fourth would reach 12
    assert sum(o.qty for o in await registry.active_orders("BTCUSDT")) == D("9")
    assert sorted(r.decision.snapshot_id for r in approved) == [
        "acct-1:0",
        "acct-1:1",
        "acct-1:2",
    ]
    assert {r.decision.snapshot_id for r in records if not r.approved} == {"acct-1:3"}


# --- replay ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_replay_returns_the_record_without_reevaluation(
    evaluate_calls: list[str],
) -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    first = await place(coord, intent(), trading_state=TradingState.PAUSED)
    # Inputs that would now be approved: the recorded result still stands.
    second = await place(coord, intent(), trading_state=RUNNING, position_qty=FLAT)

    assert second is first
    assert second.decision.reasons == (RiskReason.TRADING_PAUSED,)
    assert evaluate_calls == ["i-1"]
    assert (ids.calls, clock.calls) == (0, 0)
    assert await registry_state(registry) == (0, (), (), 0)


@pytest.mark.asyncio
async def test_approved_replay_returns_the_same_reservation(evaluate_calls: list[str]) -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    first = await place(coord, intent())
    second = await place(coord, intent(), trading_state=TradingState.HALTED, position_qty=None)

    assert second is first
    assert second.client_order_id == "c-1"
    assert evaluate_calls == ["i-1"]
    assert (ids.calls, clock.calls) == (1, 1)
    assert await registry.revision() == 1
    assert await registry.account_active_order_count() == 1


@pytest.mark.asyncio
async def test_conflicting_replay_raises_without_evaluation(evaluate_calls: list[str]) -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)
    await place(coord, intent())

    with pytest.raises(PlacementConflictError, match="i-1"):
        await place(coord, intent(qty=D("2")))

    assert evaluate_calls == ["i-1"]
    assert (ids.calls, clock.calls) == (1, 1)
    assert await registry.revision() == 1
    record = await registry.placement("i-1")
    assert record is not None
    assert record.intent.qty == D("1")


# --- preparation failure --------------------------------------------------------------------

# A valid domain quantity with more significant digits than Risk's exact context.
HUGE_QTY = D("1." + "1" * 100)


async def seed_unrepresentable_order(registry: InMemoryOrderRegistry) -> None:
    """A legitimate NEW reservation whose remainder Risk cannot compute exactly."""
    seed = intent("seed", qty=HUGE_QTY)
    async with registry.placement_lock() as locked:
        locked.register_approved(
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
            expected_revision=0,
            at=T0,
        )


@pytest.mark.asyncio
async def test_unrepresentable_snapshot_is_a_preparation_error(
    evaluate_calls: list[str],
) -> None:
    registry = InMemoryOrderRegistry()
    await seed_unrepresentable_order(registry)
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)
    before = await registry_state(registry)

    with pytest.raises(PlacementPreparationError, match="BTCUSDT") as caught:
        await place(coord, intent())

    assert isinstance(caught.value.__cause__, ExposureCalculationError)
    assert not isinstance(caught.value, ExposureCalculationError | DomainValidationError)
    assert await registry.placement("i-1") is None
    assert await registry_state(registry) == before
    assert evaluate_calls == []
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_preparation_error_is_scoped_to_the_symbol_snapshot() -> None:
    registry = InMemoryOrderRegistry()
    await seed_unrepresentable_order(registry)
    coord = coordinator(registry=registry)

    record = await place(coord, intent("i-eth", symbol="ETHUSDT"))

    assert record.approved is True
    assert record.decision.snapshot_id == "acct-1:1"


@pytest.mark.parametrize(
    ("trading_state", "position_qty", "match"),
    [
        ("running", FLAT, "trading_state"),
        (RUNNING, 1, "position_qty"),
        (RUNNING, D("NaN"), "position_qty"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_caller_inputs_propagate_as_validation_errors(
    trading_state: Any, position_qty: Any, match: str, evaluate_calls: list[str]
) -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    with pytest.raises(DomainValidationError, match=match):
        await place(coord, intent(), trading_state=trading_state, position_qty=position_qty)

    assert await registry.placement("i-1") is None
    assert await registry.revision() == 0
    assert evaluate_calls == []
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_evaluator_programmer_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**kwargs: Any) -> RiskDecision:
        raise DomainValidationError("contract violated")

    monkeypatch.setattr(placement_module, "evaluate", broken)
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    with pytest.raises(DomainValidationError, match="contract violated"):
        await place(coord, intent())

    assert await registry.placement("i-1") is None
    assert await registry.revision() == 0
    assert (ids.calls, clock.calls) == (0, 0)


@pytest.mark.asyncio
async def test_unrepresentable_intent_is_a_risk_rejection_not_a_preparation_error() -> None:
    registry = InMemoryOrderRegistry()
    ids = SequenceIds()
    coord = coordinator(registry=registry, ids=ids)

    record = await place(coord, intent(qty=HUGE_QTY))

    assert record.decision.reasons == (RiskReason.UNREPRESENTABLE_CALCULATION,)
    assert ids.calls == 0
    assert await registry.revision() == 0


# --- generator / clock atomicity ------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_failure_leaves_no_reservation() -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = FailingIds(), CountingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    with pytest.raises(RuntimeError, match="id source unavailable"):
        await place(coord, intent())

    assert ids.calls == 1
    assert clock.calls == 0
    assert await registry.placement("i-1") is None
    assert await registry_state(registry) == (0, (), (), 0)


@pytest.mark.parametrize("value", ["", " c-1", None, 7])
@pytest.mark.asyncio
async def test_invalid_generated_id_leaves_no_reservation(value: object) -> None:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry, ids=FixedIds(value))

    with pytest.raises(DomainValidationError, match="client_order_id"):
        await place(coord, intent())

    assert await registry.placement("i-1") is None
    assert await registry_state(registry) == (0, (), (), 0)


@pytest.mark.asyncio
async def test_duplicate_generated_id_leaves_state_unchanged() -> None:
    registry = InMemoryOrderRegistry()
    ids = FixedIds("dup")
    coord = coordinator(registry=registry, ids=ids)
    await place(coord, intent("i-1"))
    before = await registry_state(registry)

    with pytest.raises(PlacementConflictError, match="dup"):
        await place(coord, intent("i-2"))

    assert ids.calls == 2  # the generator's own side effect happened
    assert await registry_state(registry) == before
    assert await registry.placement("i-2") is None


@pytest.mark.asyncio
async def test_clock_failure_leaves_no_reservation() -> None:
    registry = InMemoryOrderRegistry()
    ids, clock = SequenceIds(), FailingClock()
    coord = coordinator(registry=registry, ids=ids, clock=clock)

    with pytest.raises(RuntimeError, match="clock unavailable"):
        await place(coord, intent())

    assert (ids.calls, clock.calls) == (1, 1)  # id first, then the clock
    assert await registry.placement("i-1") is None
    assert await registry_state(registry) == (0, (), (), 0)


@pytest.mark.parametrize(
    ("at", "match"),
    [
        (T0 - timedelta(microseconds=1), "before"),
        (datetime(2026, 1, 15, 12, 0), "at"),  # noqa: DTZ001 - naive on purpose
    ],
)
@pytest.mark.asyncio
async def test_invalid_reservation_time_leaves_no_reservation(at: datetime, match: str) -> None:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry, clock=CountingClock(at))

    with pytest.raises(DomainValidationError, match=match):
        await place(coord, intent())

    assert await registry.placement("i-1") is None
    assert await registry_state(registry) == (0, (), (), 0)


@pytest.mark.asyncio
async def test_lock_is_released_after_a_failure() -> None:
    registry = InMemoryOrderRegistry()

    with pytest.raises(RuntimeError):
        await place(coordinator(registry=registry, ids=FailingIds()), intent())

    record = await asyncio.wait_for(place(coordinator(registry=registry), intent()), timeout=1)
    assert record.approved is True


# --- Decimal context ------------------------------------------------------------------------


async def awkward_scenario() -> tuple[list[PlacementRecord], tuple[Any, ...]]:
    registry = InMemoryOrderRegistry()
    coord = coordinator(registry=registry, risk_policy=policy(max_position_qty="10"))
    records = [
        await place(coord, intent("i-1", qty=D("3.333333333333333333333333333"))),
        await place(coord, intent("i-2", qty=D("6.666666666666666666666666667"))),
        await place(coord, intent("i-3", qty=D("0.000000000000000000000000001"))),
        await place(
            coord,
            intent("i-4", side=Side.BUY, qty=D("1.1"), reduce_only=True),
            position_qty=D("-0.067"),
        ),
    ]
    return records, await registry_state(registry)


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
    expected_records, expected_state = await awkward_scenario()
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        records, state = await awkward_scenario()

    assert [r.decision for r in records] == [r.decision for r in expected_records]
    assert [r.client_order_id for r in records] == [r.client_order_id for r in expected_records]
    assert state == expected_state
    assert [r.approved for r in records] == [True, True, False, True]
    assert records[1].decision.exposure is not None
    assert records[1].decision.exposure.worst_long_qty == D("10.000000000000000000000000000")
    assert records[2].decision.reasons == (RiskReason.MAX_POSITION_QTY,)
    assert records[3].decision.exposure is not None
    assert records[3].decision.exposure.reducing_qty == D("0.067")
    assert records[3].decision.exposure.worst_long_qty == D("9.933000000000000000000000000")
    assert state[0] == 3


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
