"""In-memory order reservation registry: idempotency, revision and the account lock."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.orders import Order
from app.execution import models as models_module
from app.execution import registry as registry_module
from app.execution.models import PlacementRecord
from app.execution.registry import (
    InMemoryOrderRegistry,
    LockedOrderRegistry,
    PlacementConflictError,
    RegistryLockError,
    StaleRevisionError,
)
from app.risk.models import ExposureChange, RiskDecision, RiskReason

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
EXPOSURE = ExposureChange(
    reducing_qty=D("0"), increasing_qty=D("1"), worst_long_qty=D("1"), worst_short_qty=D("0")
)


def intent(intent_id: str = "i-1", **overrides: Any) -> PlaceOrderIntent:
    values: dict[str, Any] = {
        "intent_id": intent_id,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100.5"),
        "qty": D("1.25"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "tag": "grid:L01:buy",
        "created_at": T0,
    }
    return PlaceOrderIntent(**{**values, **overrides})


def approved(source: PlaceOrderIntent, snapshot_id: str = "acct:0") -> RiskDecision:
    return RiskDecision(
        intent_id=source.intent_id,
        snapshot_id=snapshot_id,
        policy_id="policy-1",
        approved=True,
        reasons=(),
        exposure=EXPOSURE,
    )


def rejected(source: PlaceOrderIntent, snapshot_id: str = "acct:0") -> RiskDecision:
    return RiskDecision(
        intent_id=source.intent_id,
        snapshot_id=snapshot_id,
        policy_id="policy-1",
        approved=False,
        reasons=(RiskReason.MAX_OPEN_ORDERS,),
        exposure=None,
    )


def reserve(
    locked: LockedOrderRegistry,
    source: PlaceOrderIntent,
    client_order_id: str,
    *,
    expected_revision: int | None = None,
    at: datetime = T1,
) -> PlacementRecord:
    return locked.register_approved(
        intent=source,
        decision=approved(source),
        client_order_id=client_order_id,
        expected_revision=locked.revision if expected_revision is None else expected_revision,
        at=at,
    )


def reject(locked: LockedOrderRegistry, source: PlaceOrderIntent) -> PlacementRecord:
    return locked.register_rejected(
        intent=source, decision=rejected(source), expected_revision=locked.revision
    )


# --- initial state --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_state_is_empty() -> None:
    registry = InMemoryOrderRegistry()

    assert await registry.revision() == 0
    assert await registry.active_orders("BTCUSDT") == ()
    assert await registry.account_active_order_count() == 0
    assert await registry.placement("i-1") is None
    assert await registry.order("c-1") is None


# --- approved reservation -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_reservation_creates_new_order_and_bumps_revision() -> None:
    registry = InMemoryOrderRegistry()
    source = intent()

    async with registry.placement_lock() as locked:
        record = reserve(locked, source, "c-1")
        assert locked.revision == 1

    expected_order = Order(
        client_order_id="c-1",
        exchange_order_id=None,
        strategy_id="grid-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100.5"),
        qty=D("1.25"),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        status=OrderStatus.NEW,
        filled_qty=D("0"),
        avg_fill_price=None,
        created_at=T1,
        updated_at=T1,
        last_exchange_update_ts=None,
        version=0,
    )
    assert record == PlacementRecord(
        intent=source, decision=approved(source), client_order_id="c-1"
    )
    assert record.approved is True
    assert record.intent_id == "i-1"
    assert await registry.order("c-1") == expected_order
    assert await registry.placement("i-1") == record
    assert await registry.active_orders("BTCUSDT") == (expected_order,)
    assert await registry.account_active_order_count() == 1
    assert await registry.revision() == 1


@pytest.mark.asyncio
async def test_market_and_reduce_only_terms_are_copied() -> None:
    registry = InMemoryOrderRegistry()
    source = intent(
        order_type=OrderType.MARKET,
        price=None,
        time_in_force=TimeInForce.IOC,
        side=Side.SELL,
        reduce_only=True,
        tag=None,
    )

    async with registry.placement_lock() as locked:
        reserve(locked, source, "c-1")

    order = await registry.order("c-1")
    assert order is not None
    assert (order.order_type, order.price, order.time_in_force, order.side, order.reduce_only) == (
        OrderType.MARKET,
        None,
        TimeInForce.IOC,
        Side.SELL,
        True,
    )


@pytest.mark.asyncio
async def test_reservation_time_may_equal_intent_time() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        reserve(locked, intent(), "c-1", at=T0)

    order = await registry.order("c-1")
    assert order is not None
    assert order.created_at == T0


@pytest.mark.asyncio
async def test_reservation_before_intent_time_is_rejected() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        with pytest.raises(DomainValidationError, match="before"):
            reserve(locked, intent(), "c-1", at=T0 - timedelta(microseconds=1))
        assert locked.revision == 0


# --- rejected placement ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_placement_is_recorded_without_order_or_revision() -> None:
    registry = InMemoryOrderRegistry()
    source = intent()

    async with registry.placement_lock() as locked:
        record = reject(locked, source)

    assert record == PlacementRecord(intent=source, decision=rejected(source), client_order_id=None)
    assert record.approved is False
    assert await registry.placement("i-1") == record
    assert await registry.revision() == 0
    assert await registry.active_orders("BTCUSDT") == ()
    assert await registry.account_active_order_count() == 0


@pytest.mark.asyncio
async def test_rejected_placement_does_not_change_existing_exposure() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        reserve(locked, intent("i-1"), "c-1")
        reject(locked, intent("i-2"))

    assert await registry.revision() == 1
    assert len(await registry.active_orders("BTCUSDT")) == 1
    assert await registry.account_active_order_count() == 1


# --- intent idempotency ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_replay_returns_the_same_record() -> None:
    registry = InMemoryOrderRegistry()
    source = intent()

    async with registry.placement_lock() as locked:
        first = reserve(locked, source, "c-1")
        # A replay with a new client id and a stale revision is still a replay.
        second = reserve(locked, intent(), "c-2", expected_revision=0)
        assert locked.revision == 1

    assert second is first
    assert await registry.order("c-2") is None
    assert await registry.account_active_order_count() == 1


@pytest.mark.asyncio
async def test_rejected_replay_returns_the_same_record() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        first = reject(locked, intent())
        second = reject(locked, intent())

    assert second is first
    assert await registry.revision() == 0


@pytest.mark.asyncio
async def test_intent_is_single_use_across_outcomes() -> None:
    registry = InMemoryOrderRegistry()
    source = intent()

    async with registry.placement_lock() as locked:
        first = reject(locked, source)
        # A later approval of the same intent does not replace the recorded result.
        second = reserve(locked, source, "c-1")

    assert second is first
    assert await registry.order("c-1") is None
    assert await registry.revision() == 0


@pytest.mark.asyncio
async def test_replay_after_lock_release_returns_the_same_record() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        first = reserve(locked, intent(), "c-1")
    async with registry.placement_lock() as locked:
        second = reserve(locked, intent(), "c-9")

    assert second is first
    assert await registry.revision() == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"strategy_id": "grid-2"},
        {"symbol": "ETHUSDT"},
        {"side": Side.SELL},
        {"price": D("100.6")},
        {"qty": D("1.26")},
        {"time_in_force": TimeInForce.POST_ONLY},
        {"reduce_only": True},
        {"tag": "grid:L02:buy"},
        {"tag": None},
        {"created_at": T1},
        {"order_type": OrderType.MARKET, "price": None},
    ],
)
@pytest.mark.asyncio
async def test_same_intent_id_with_different_data_is_a_conflict(
    overrides: dict[str, Any],
) -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        reserve(locked, intent(), "c-1")
        with pytest.raises(PlacementConflictError, match="i-1"):
            reserve(locked, intent(**overrides), "c-2", at=T1 + timedelta(seconds=1))
        with pytest.raises(PlacementConflictError, match="i-1"):
            reject(locked, intent(**overrides))
        assert locked.revision == 1

    assert await registry.order("c-2") is None


@pytest.mark.asyncio
async def test_numerically_equal_decimals_are_the_same_intent() -> None:
    # Identity is field equality: Decimal compares by value, not representation.
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        first = reserve(locked, intent(price=D("100.5")), "c-1")
        second = reserve(locked, intent(price=D("100.50")), "c-2")

    assert second is first


@pytest.mark.asyncio
async def test_replay_of_reports_new_equal_and_conflicting_intents() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        assert locked.replay_of(intent()) is None
        record = reject(locked, intent())
        assert locked.replay_of(intent()) is record
        with pytest.raises(PlacementConflictError, match="i-1"):
            locked.replay_of(intent(qty=D("2")))
        with pytest.raises(DomainValidationError, match="PlaceOrderIntent"):
            locked.replay_of("i-1")  # type: ignore[arg-type]
        assert locked.revision == 0

    with pytest.raises(RegistryLockError, match="released"):
        locked.replay_of(intent())


# --- client_order_id ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_order_id_of_another_intent_is_a_conflict() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        reserve(locked, intent("i-1"), "c-1")
        with pytest.raises(PlacementConflictError, match="c-1"):
            reserve(locked, intent("i-2"), "c-1")
        assert locked.revision == 1

    assert await registry.placement("i-2") is None
    assert await registry.account_active_order_count() == 1


@pytest.mark.parametrize("client_order_id", ["", " c-1", "c-1 ", 7, None])
@pytest.mark.asyncio
async def test_invalid_client_order_id_is_rejected(client_order_id: object) -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        with pytest.raises(DomainValidationError, match="client_order_id"):
            reserve(locked, intent(), client_order_id)  # type: ignore[arg-type]
        assert locked.revision == 0

    assert await registry.placement("i-1") is None


def test_registry_does_not_generate_identifiers() -> None:
    source = Path(registry_module.__file__).read_text(encoding="utf-8")

    for banned in ("uuid", "random", "secrets", "hash(", "time.", "datetime.now"):
        assert banned not in source, banned


# --- input validation -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_must_belong_to_the_intent() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        with pytest.raises(DomainValidationError, match="intent_id"):
            locked.register_approved(
                intent=intent("i-1"),
                decision=approved(intent("i-2")),
                client_order_id="c-1",
                expected_revision=0,
                at=T1,
            )
        with pytest.raises(DomainValidationError, match="intent_id"):
            locked.register_rejected(
                intent=intent("i-1"), decision=rejected(intent("i-2")), expected_revision=0
            )


@pytest.mark.asyncio
async def test_decision_outcome_must_match_the_operation() -> None:
    registry = InMemoryOrderRegistry()
    source = intent()

    async with registry.placement_lock() as locked:
        with pytest.raises(DomainValidationError, match="approved"):
            locked.register_approved(
                intent=source,
                decision=rejected(source),
                client_order_id="c-1",
                expected_revision=0,
                at=T1,
            )
        with pytest.raises(DomainValidationError, match="rejected"):
            locked.register_rejected(intent=source, decision=approved(source), expected_revision=0)
        assert locked.revision == 0

    assert await registry.placement("i-1") is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("intent", "intent", "PlaceOrderIntent"),
        ("decision", "decision", "RiskDecision"),
        ("expected_revision", True, "expected_revision"),
        ("expected_revision", -1, "expected_revision"),
        ("expected_revision", "0", "expected_revision"),
        ("at", datetime(2026, 1, 15, 12, 0), "at"),  # noqa: DTZ001 - naive on purpose
        ("at", "2026-01-15", "at"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected(field: str, value: object, match: str) -> None:
    registry = InMemoryOrderRegistry()
    source = intent()
    arguments: dict[str, Any] = {
        "intent": source,
        "decision": approved(source),
        "client_order_id": "c-1",
        "expected_revision": 0,
        "at": T1,
    }

    async with registry.placement_lock() as locked:
        with pytest.raises(DomainValidationError, match=match):
            locked.register_approved(**{**arguments, field: value})
        assert locked.revision == 0


# --- revision -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_revision_is_rejected_without_changes() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        reserve(locked, intent("i-1"), "c-1")
        with pytest.raises(StaleRevisionError, match="0"):
            reserve(locked, intent("i-2"), "c-2", expected_revision=0)
        with pytest.raises(StaleRevisionError):
            locked.register_rejected(
                intent=intent("i-3"), decision=rejected(intent("i-3")), expected_revision=0
            )
        assert locked.revision == 1

    assert await registry.placement("i-2") is None
    assert await registry.placement("i-3") is None


@pytest.mark.asyncio
async def test_revision_counts_only_reservations() -> None:
    registry = InMemoryOrderRegistry()
    revisions: list[int] = []

    async with registry.placement_lock() as locked:
        revisions.append(locked.revision)
        reserve(locked, intent("i-1"), "c-1")
        revisions.append(locked.revision)
        reject(locked, intent("i-2"))
        revisions.append(locked.revision)
        reserve(locked, intent("i-1"), "c-1")  # replay
        revisions.append(locked.revision)
        reserve(locked, intent("i-3"), "c-3")
        revisions.append(locked.revision)

    assert revisions == [0, 1, 1, 1, 2]


# --- views ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_symbol_views_and_account_count_across_symbols() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        reserve(locked, intent("i-1"), "c-1")
        reserve(locked, intent("i-2", symbol="ETHUSDT"), "c-2")
        reserve(locked, intent("i-3", side=Side.SELL), "c-3")
        reject(locked, intent("i-4", symbol="SOLUSDT"))

        btc = locked.active_orders("BTCUSDT")
        eth = locked.active_orders("ETHUSDT")
        sol = locked.active_orders("SOLUSDT")
        count = locked.account_active_order_count()

    assert [o.client_order_id for o in btc] == ["c-1", "c-3"]  # reservation order
    assert [o.client_order_id for o in eth] == ["c-2"]
    assert sol == ()
    assert count == 3


@pytest.mark.asyncio
async def test_views_are_immutable_and_defensive() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        record = reserve(locked, intent(), "c-1")
        view = locked.active_orders("BTCUSDT")

    assert type(view) is tuple
    with pytest.raises(dataclasses.FrozenInstanceError):
        view[0].status = OrderStatus.FILLED  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.client_order_id = "c-2"  # type: ignore[misc]

    async with registry.placement_lock() as locked:
        reserve(locked, intent("i-2"), "c-2")

    assert len(view) == 1  # an earlier view does not change
    assert len(await registry.active_orders("BTCUSDT")) == 2
    public = [getattr(registry, name) for name in dir(registry) if not name.startswith("_")]
    assert not any(isinstance(value, dict | list | set) for value in public)


@pytest.mark.parametrize("symbol", ["", " BTCUSDT", 7])
@pytest.mark.asyncio
async def test_invalid_symbol_view_is_rejected(symbol: object) -> None:
    registry = InMemoryOrderRegistry()

    with pytest.raises(DomainValidationError, match="symbol"):
        await registry.active_orders(symbol)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "value"),
    [("placement", ""), ("placement", 1), ("order", " c-1"), ("order", None)],
)
@pytest.mark.asyncio
async def test_invalid_lookup_ids_are_rejected(method: str, value: object) -> None:
    registry = InMemoryOrderRegistry()

    with pytest.raises(DomainValidationError):
        await getattr(registry, method)(value)


# --- lock -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_waiting_task_sees_the_reservation_of_the_lock_holder() -> None:
    registry = InMemoryOrderRegistry()
    a_inside = asyncio.Event()
    b_waiting = asyncio.Event()
    b_inside = asyncio.Event()
    seen: dict[str, Any] = {}

    async def task_a() -> None:
        async with registry.placement_lock() as locked:
            a_inside.set()
            await b_waiting.wait()
            await asyncio.sleep(0)  # B is now blocked on the lock
            assert not b_inside.is_set()
            seen["a_revision"] = locked.revision
            reserve(locked, intent("i-a"), "c-a", expected_revision=seen["a_revision"])

    async def task_b() -> None:
        await a_inside.wait()
        b_waiting.set()
        async with registry.placement_lock() as locked:
            b_inside.set()
            seen["b_revision"] = locked.revision
            seen["b_view"] = locked.active_orders("BTCUSDT")
            seen["b_count"] = locked.account_active_order_count()

    await asyncio.wait_for(asyncio.gather(task_a(), task_b()), timeout=5)

    assert seen["a_revision"] == 0
    assert seen["b_revision"] == 1
    assert [(o.client_order_id, o.status) for o in seen["b_view"]] == [("c-a", OrderStatus.NEW)]
    assert seen["b_count"] == 1


@pytest.mark.asyncio
async def test_concurrent_reservations_do_not_lose_updates() -> None:
    registry = InMemoryOrderRegistry()
    attempts = 20

    async def attempt(index: int) -> int:
        async with registry.placement_lock() as locked:
            revision = locked.revision
            await asyncio.sleep(0)  # yield inside the lock (e.g. a future persistence write)
            reserve(locked, intent(f"i-{index}"), f"c-{index}", expected_revision=revision)
            return revision

    seen = await asyncio.wait_for(asyncio.gather(*(attempt(i) for i in range(attempts))), timeout=5)

    assert sorted(seen) == list(range(attempts))
    assert await registry.revision() == attempts
    assert await registry.account_active_order_count() == attempts


@pytest.mark.asyncio
async def test_concurrent_replays_of_one_intent_create_one_order() -> None:
    registry = InMemoryOrderRegistry()

    async def attempt(index: int) -> PlacementRecord:
        async with registry.placement_lock() as locked:
            await asyncio.sleep(0)
            return reserve(locked, intent("i-1"), f"c-{index}")

    records = await asyncio.wait_for(asyncio.gather(*(attempt(i) for i in range(10))), timeout=5)

    assert all(record is records[0] for record in records)
    assert await registry.revision() == 1
    assert await registry.account_active_order_count() == 1


@pytest.mark.asyncio
async def test_registration_inside_the_held_lock_does_not_deadlock() -> None:
    registry = InMemoryOrderRegistry()

    async def flow() -> PlacementRecord:
        async with registry.placement_lock() as locked:
            _ = locked.revision, locked.active_orders("BTCUSDT")
            return reserve(locked, intent(), "c-1")

    record = await asyncio.wait_for(flow(), timeout=1)

    assert record.client_order_id == "c-1"


@pytest.mark.asyncio
async def test_reentering_the_lock_fails_fast_instead_of_deadlocking() -> None:
    registry = InMemoryOrderRegistry()

    async def flow() -> None:
        async with registry.placement_lock():
            with pytest.raises(RegistryLockError, match="reentrant"):
                await registry.revision()
            with pytest.raises(RegistryLockError, match="reentrant"):
                async with registry.placement_lock():
                    pass  # pragma: no cover - never entered

    await asyncio.wait_for(flow(), timeout=1)
    assert await registry.revision() == 0  # the lock was released


@pytest.mark.asyncio
async def test_handle_cannot_be_used_after_release() -> None:
    registry = InMemoryOrderRegistry()

    async with registry.placement_lock() as locked:
        pass

    with pytest.raises(RegistryLockError, match="released"):
        _ = locked.revision
    with pytest.raises(RegistryLockError, match="released"):
        reserve(locked, intent(), "c-1", expected_revision=0)
    with pytest.raises(RegistryLockError, match="released"):
        locked.active_orders("BTCUSDT")
    assert await registry.placement("i-1") is None


@pytest.mark.asyncio
async def test_lock_is_released_after_an_exception() -> None:
    registry = InMemoryOrderRegistry()

    with pytest.raises(StaleRevisionError):
        async with registry.placement_lock() as locked:
            reserve(locked, intent(), "c-1", expected_revision=5)

    assert await asyncio.wait_for(registry.revision(), timeout=1) == 0


def test_mutation_is_only_available_on_the_locked_handle() -> None:
    public = {name for name in dir(InMemoryOrderRegistry) if not name.startswith("_")}

    assert not any(name.startswith("register") for name in public)
    assert {"register_approved", "register_rejected"} <= set(dir(LockedOrderRegistry))


# --- PlacementRecord ------------------------------------------------------------------------


def test_placement_record_is_frozen() -> None:
    source = intent()
    record = PlacementRecord(intent=source, decision=approved(source), client_order_id="c-1")

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.decision = rejected(source)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"intent": "intent"}, "PlaceOrderIntent"),
        ({"decision": "decision"}, "RiskDecision"),
        ({"decision": approved(intent("i-2"))}, "intent_id"),
        ({"client_order_id": None}, "client_order_id"),
        ({"client_order_id": ""}, "client_order_id"),
        ({"decision": rejected(intent())}, "client_order_id"),
    ],
)
def test_placement_record_validation(kwargs: dict[str, Any], match: str) -> None:
    source = intent()
    values: dict[str, Any] = {
        "intent": source,
        "decision": approved(source),
        "client_order_id": "c-1",
    }

    with pytest.raises(DomainValidationError, match=match):
        PlacementRecord(**{**values, **kwargs})


# --- dependencies ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [models_module, registry_module])
def test_no_network_exchange_or_persistence_dependencies(module: object) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith(
            ("app.domain", "app.risk.models", "app.execution")
        ), name
    banned = {"httpx", "pybit", "ccxt", "sqlite3", "sqlalchemy", "socket", "logging"}
    assert not {name.split(".")[0] for name in imports} & banned
    assert "place_order" not in source
