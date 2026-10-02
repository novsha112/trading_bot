"""SafetyController: requested vs effective TradingState and recovery gates."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.errors import DomainValidationError
from app.execution import safety as safety_module
from app.execution.account_state import InMemoryAccountState
from app.execution.persistence import StoreUncertainError
from app.execution.safety import (
    RecoveryGateOrderError,
    RecoveryReadiness,
    SafetyController,
    SafetySnapshot,
    SafetyStateError,
    effective_trading_state,
)
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import TradingState

RUNNING = TradingState.RUNNING
REDUCE_ONLY = TradingState.REDUCE_ONLY
PAUSED = TradingState.PAUSED
HALTED = TradingState.HALTED
GATES = (
    "hydrated",
    "orders_reconciled",
    "positions_reconciled",
    "open_orders_reconciled",
    "fills_complete",
)
EXCHANGE_GATES = GATES[1:]
OLD_PARTIAL_MARKS = (
    "mark_orders_reconciled",
    "mark_positions_reconciled",
    "mark_open_orders_reconciled",
    "mark_fills_complete",
)
COMPLETE = RecoveryReadiness(**dict.fromkeys(GATES, True))


def account() -> InMemoryAccountState:
    return InMemoryAccountState(account_scope_id="acct-1", store=InMemoryAccountStateStore())


def controller(target: InMemoryAccountState | None = None) -> SafetyController:
    return SafetyController(account_state=account() if target is None else target)


def complete(safety: SafetyController) -> SafetyController:
    safety.mark_hydrated()
    safety.mark_exchange_reconciled()
    return safety


async def poisoned_account() -> InMemoryAccountState:
    store = InMemoryAccountStateStore()
    target = InMemoryAccountState(account_scope_id="acct-1", store=store)
    store.inject_commit_failure(CommitFailure.UNCERTAIN)
    async with target.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty("BTCUSDT", Decimal(0))
    assert target.is_poisoned
    return target


# --- pure evaluator -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "ready", "poisoned", "expected"),
    [
        (RUNNING, False, False, PAUSED),
        (RUNNING, True, False, RUNNING),
        (REDUCE_ONLY, False, False, PAUSED),
        (REDUCE_ONLY, True, False, REDUCE_ONLY),
        (PAUSED, False, False, PAUSED),
        (PAUSED, True, False, PAUSED),
        (HALTED, False, False, HALTED),
        (HALTED, True, False, HALTED),
        (RUNNING, False, True, PAUSED),
        (RUNNING, True, True, PAUSED),
        (REDUCE_ONLY, True, True, PAUSED),
        (PAUSED, True, True, PAUSED),
        (HALTED, False, True, HALTED),
        (HALTED, True, True, HALTED),
    ],
)
def test_effective_state_table(
    requested: TradingState, ready: bool, poisoned: bool, expected: TradingState
) -> None:
    readiness = COMPLETE if ready else RecoveryReadiness()
    assert (
        effective_trading_state(requested=requested, readiness=readiness, account_poisoned=poisoned)
        is expected
    )


@pytest.mark.parametrize("missing", GATES)
@pytest.mark.parametrize("requested", [RUNNING, REDUCE_ONLY])
def test_every_single_missing_gate_keeps_paused(missing: str, requested: TradingState) -> None:
    readiness = dataclasses.replace(COMPLETE, **{missing: False})
    assert not readiness.complete
    assert (
        effective_trading_state(requested=requested, readiness=readiness, account_poisoned=False)
        is PAUSED
    )


@pytest.mark.parametrize("values", list(itertools.product([False, True], repeat=5)))
def test_running_only_with_every_gate(values: tuple[bool, ...]) -> None:
    readiness = RecoveryReadiness(**dict(zip(GATES, values, strict=True)))
    effective = effective_trading_state(
        requested=RUNNING, readiness=readiness, account_poisoned=False
    )
    assert (effective is RUNNING) is all(values)
    assert effective in (RUNNING, PAUSED)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requested": "running"},
        {"requested": 1},
        {"requested": None},
        {"readiness": None},
        {"readiness": {"hydrated": True}},
        {"account_poisoned": 0},
        {"account_poisoned": None},
    ],
)
def test_pure_evaluator_validates_its_inputs(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "requested": RUNNING,
        "readiness": COMPLETE,
        "account_poisoned": False,
    }
    with pytest.raises(DomainValidationError):
        effective_trading_state(**{**values, **kwargs})


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_readiness_flags_must_be_exact_bools(value: object) -> None:
    with pytest.raises(DomainValidationError, match="hydrated"):
        RecoveryReadiness(hydrated=value)  # type: ignore[arg-type]


def test_readiness_defaults_to_nothing_confirmed_and_is_frozen() -> None:
    readiness = RecoveryReadiness()
    assert [getattr(readiness, gate) for gate in GATES] == [False] * 5
    assert not readiness.complete
    assert COMPLETE.complete
    with pytest.raises(dataclasses.FrozenInstanceError):
        readiness.hydrated = True  # type: ignore[misc]


# --- controller -----------------------------------------------------------------------------


def test_new_controller_fails_closed() -> None:
    safety = controller()
    snap = safety.snapshot()

    assert snap == SafetySnapshot(
        requested_state=PAUSED,
        effective_state=PAUSED,
        readiness=RecoveryReadiness(),
        account_poisoned=False,
    )
    assert safety.effective_state is PAUSED


def test_controller_requires_an_account_state() -> None:
    with pytest.raises(DomainValidationError, match="account_state"):
        SafetyController(account_state=object())  # type: ignore[arg-type]
    target = account()
    assert SafetyController(account_state=target).account_state is target


def test_requested_running_is_remembered_while_recovery_is_incomplete() -> None:
    safety = controller()
    safety.request_state(RUNNING)

    assert safety.snapshot().requested_state is RUNNING
    assert safety.effective_state is PAUSED

    safety.mark_hydrated()
    assert safety.effective_state is PAUSED
    complete(safety)
    assert safety.effective_state is RUNNING
    assert safety.snapshot().requested_state is RUNNING


@pytest.mark.parametrize("hydrated", [False, True])
def test_controller_stays_paused_until_the_exchange_gates_are_confirmed(hydrated: bool) -> None:
    safety = controller()
    safety.request_state(RUNNING)
    if hydrated:
        safety.mark_hydrated()
    readiness = safety.snapshot().readiness
    assert readiness.hydrated is hydrated
    assert [getattr(readiness, gate) for gate in EXCHANGE_GATES] == [False] * 4
    assert safety.effective_state is PAUSED


@pytest.mark.parametrize(
    ("requested", "incomplete", "done"),
    [
        (PAUSED, PAUSED, PAUSED),
        (REDUCE_ONLY, PAUSED, REDUCE_ONLY),
        (HALTED, HALTED, HALTED),
        (RUNNING, PAUSED, RUNNING),
    ],
)
def test_requested_states_before_and_after_recovery(
    requested: TradingState, incomplete: TradingState, done: TradingState
) -> None:
    safety = controller()
    safety.request_state(requested)
    assert safety.effective_state is incomplete
    complete(safety)
    assert safety.effective_state is done


def test_halted_is_never_lifted_by_readiness_only_by_an_explicit_request() -> None:
    safety = complete(controller())
    safety.request_state(HALTED)
    assert safety.effective_state is HALTED
    # Re-confirming gates changes nothing.
    complete(safety)
    assert safety.effective_state is HALTED
    safety.request_state(RUNNING)
    assert safety.effective_state is RUNNING


@pytest.mark.parametrize("value", ["running", "RUNNING", 1, True, None, object()])
def test_request_state_accepts_only_an_exact_trading_state(value: object) -> None:
    safety = complete(controller())
    safety.request_state(RUNNING)
    with pytest.raises(DomainValidationError, match="TradingState"):
        safety.request_state(value)  # type: ignore[arg-type]
    assert safety.snapshot().requested_state is RUNNING


# --- poison ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poison_after_readiness_pauses_on_the_next_read() -> None:
    store = InMemoryAccountStateStore()
    target = InMemoryAccountState(account_scope_id="acct-1", store=store)
    safety = complete(SafetyController(account_state=target))
    safety.request_state(RUNNING)
    assert safety.effective_state is RUNNING

    store.inject_commit_failure(CommitFailure.UNCERTAIN)
    async with target.account_lock() as locked:
        with pytest.raises(StoreUncertainError):
            await locked.set_position_qty("BTCUSDT", Decimal(0))

    snap = safety.snapshot()
    assert snap.account_poisoned is True
    assert snap.effective_state is PAUSED
    assert snap.requested_state is RUNNING  # the request is not forgotten
    assert snap.readiness == COMPLETE  # gates are not lowered either
    assert target.is_poisoned  # never cleared by the controller


@pytest.mark.asyncio
async def test_poison_with_incomplete_recovery_is_paused_and_halted_wins() -> None:
    safety = SafetyController(account_state=await poisoned_account())
    safety.request_state(RUNNING)
    assert safety.effective_state is PAUSED
    complete(safety)
    assert safety.effective_state is PAUSED
    safety.request_state(HALTED)
    assert safety.snapshot() == SafetySnapshot(
        requested_state=HALTED,
        effective_state=HALTED,
        readiness=COMPLETE,
        account_poisoned=True,
    )


# --- gates ----------------------------------------------------------------------------------


def test_exchange_confirmation_before_hydrate_is_an_orchestration_error() -> None:
    safety = controller()
    with pytest.raises(RecoveryGateOrderError, match="hydrated"):
        safety.mark_exchange_reconciled()
    assert safety.snapshot().readiness == RecoveryReadiness()
    assert issubclass(RecoveryGateOrderError, SafetyStateError)


def test_exchange_gates_are_confirmed_together_in_one_snapshot() -> None:
    safety = controller()
    safety.request_state(RUNNING)
    safety.mark_hydrated()
    before = safety.snapshot()
    assert before.effective_state is PAUSED

    safety.mark_exchange_reconciled()

    after = safety.snapshot()
    assert after.readiness == COMPLETE
    assert all(getattr(after.readiness, gate) for gate in EXCHANGE_GATES)
    assert after.effective_state is RUNNING
    assert before.readiness.hydrated is True
    assert not any(getattr(before.readiness, gate) for gate in EXCHANGE_GATES)


def test_marking_twice_is_idempotent() -> None:
    safety = complete(controller())
    before = safety.snapshot()
    complete(safety)
    assert safety.snapshot() == before


def test_there_is_no_api_to_lower_a_gate() -> None:
    public = {name for name in dir(SafetyController) if not name.startswith("_")}
    assert public == {
        "account_state",
        "effective_state",
        "mark_exchange_reconciled",
        "mark_hydrated",
        "request_state",
        "snapshot",
    }
    for name in public:
        member = getattr(SafetyController, name)
        if name.startswith("mark_"):
            assert list(inspect.signature(member).parameters) == ["self"]
    # No partial confirmation of the exchange gates is possible.
    assert not any(hasattr(SafetyController, name) for name in OLD_PARTIAL_MARKS)


def test_snapshot_is_immutable_and_detached() -> None:
    safety = controller()
    snap = safety.snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.effective_state = RUNNING  # type: ignore[misc]
    safety.mark_hydrated()
    assert snap.readiness.hydrated is False
    assert safety.snapshot().readiness.hydrated is True


# --- boundaries -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hydrate_does_not_create_or_touch_a_controller() -> None:
    store = InMemoryAccountStateStore()

    class Clock:
        def now(self) -> datetime:
            return datetime(2026, 1, 1, tzinfo=UTC)

    hydrated = await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=store, clock=Clock()
    )
    safety = SafetyController(account_state=hydrated)
    assert safety.snapshot().readiness.hydrated is False  # the bootstrap marks it
    assert "safety" not in inspect.getsource(InMemoryAccountState.hydrate).lower()


def test_controller_is_runtime_only_and_lock_free() -> None:
    source = Path(safety_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= {
        "__future__",
        "dataclasses",
        "app.domain.errors",
        "app.execution.account_state",
        "app.risk.models",
    }
    assert "asyncio" not in source
    assert "store" not in source.lower().replace("restore", "")


def test_controller_methods_are_synchronous() -> None:
    members: list[Callable[..., Any]] = [
        SafetyController.request_state,
        SafetyController.snapshot,
        SafetyController.mark_hydrated,
        SafetyController.mark_exchange_reconciled,
    ]
    assert not any(inspect.iscoroutinefunction(member) for member in members)
