"""Explicit position baseline acceptance: a privileged, per-symbol durable action
that makes the current runtime (exchange-published) position the known durable
projection, atomically with an append-only audit record."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.exchanges.recovery import ExchangePosition, PositionSnapshot
from app.execution import position_baseline as module
from app.execution.account_state import (
    AccountStatePoisonedError,
    BaselineIdConflictError,
    BaselineQtyMismatchError,
    BaselineRuntimeUnknownError,
    FillApplicationError,
    InMemoryAccountState,
    PositionProjectionMismatchError,
    StaleRevisionError,
)
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.models import (
    AUDIT_REASON_MAX_LENGTH,
    PositionBaselineRecord,
    require_audit_reason,
)
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    PersistedPosition,
    StoreCommitError,
    StoreConflictError,
    StoreUncertainError,
    StoreValidationError,
)
from app.execution.position_baseline import (
    BaselineOutcome,
    PositionBaselineAcceptance,
    PositionBaselineAcceptanceResult,
    accept_position_baseline,
    default_baseline_id,
)
from app.execution.position_reconciliation import PositionExplanation, reconcile_positions
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision

D = Decimal
E = PositionExplanation
BTC, ETH = "BTCUSDT", "ETHUSDT"
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NS = ClientOrderNamespace("bot01")
REASON = "operator confirmed exchange position"


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


class Store:
    def __init__(self) -> None:
        self.inner = InMemoryAccountStateStore()
        self.changes: list[AccountStateChange] = []
        self.block: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        return await self.inner.load(account_scope_id=account_scope_id)

    async def commit(self, change: AccountStateChange) -> None:
        self.changes.append(change)
        if self.block is not None:
            self.entered.set()
            await self.block.wait()
        await self.inner.commit(change)


class CountingClock:
    def __init__(self, now: object = None) -> None:
        self.value = at(100) if now is None else now
        self.calls = 0

    def now(self) -> Any:
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class Ids:
    def __init__(self, *ids: str) -> None:
        self.ids = list(ids) or [f"bl_{n:04d}" for n in range(1, 50)]
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.ids.pop(0)


def snap(positions: Mapping[str, str]) -> PositionSnapshot:
    return PositionSnapshot(
        positions=tuple(ExchangePosition(symbol=s, qty=D(q)) for s, q in positions.items()),
        complete=True,
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


def fill(n: int, side: Side, qty: str, symbol: str = BTC) -> Fill:
    return Fill(
        exec_id=f"e-{n}",
        exchange_order_id=f"X-{n}",
        client_order_id=cid(n),
        symbol=symbol,
        side=side,
        price=D("100"),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=at(1),
    )


async def trade(
    account: InMemoryAccountState,
    n: int,
    side: Side,
    qty: str,
    *,
    reduce_only: bool = False,
    symbol: str = BTC,
) -> None:
    await reserve(account, n, symbol, side, qty, reduce_only=reduce_only)
    async with account.account_lock() as locked:
        await locked.apply_fill(fill(n, side, qty, symbol), at=at(10))


async def hydrate(store: Store) -> InMemoryAccountState:
    return await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=store, clock=ManualClock(at(5))
    )


async def unexplained(
    durable: str | None, runtime: str, *, durable_row: bool = True, fills: int = 0
) -> tuple[InMemoryAccountState, Store]:
    """Restarted account with a durable row of BTC (known ``durable`` / unknown
    / none) and ``fills`` committed BTC fills, reconciled to ``runtime``."""
    store = Store()
    first = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with first.account_lock() as locked:
        if durable_row:
            await locked.set_position_qty(BTC, D("1"))
            await locked.set_position_qty(BTC, None if durable is None else D(durable))
    for n in range(1, fills + 1):
        await trade(first, 900 + n, Side.BUY, "1")
    account = await hydrate(store)
    await reconcile_positions(account_state=account, snapshot=snap({BTC: runtime}))
    store.changes.clear()
    return account, store


def command(qty: str, revision: int, *, symbol: str = BTC, reason: str = REASON) -> Any:
    return PositionBaselineAcceptance(
        symbol=symbol, expected_exchange_qty=D(qty), expected_revision=revision, reason=reason
    )


async def accept(
    account: InMemoryAccountState,
    qty: str,
    *,
    revision: int | None = None,
    clock: Any = None,
    ids: Any = None,
    symbol: str = BTC,
    reason: str = REASON,
) -> PositionBaselineAcceptanceResult:
    rev = await account.revision() if revision is None else revision
    return await accept_position_baseline(
        account_state=account,
        command=command(qty, rev, symbol=symbol, reason=reason),
        clock=CountingClock() if clock is None else clock,
        baseline_ids=Ids() if ids is None else ids,
    )


async def views(account: InMemoryAccountState, symbol: str = BTC) -> tuple[Any, Any]:
    return await account.position_qty(symbol), await account.durable_position(symbol)


async def assert_unchanged(
    account: InMemoryAccountState,
    store: Store,
    *,
    revision: int,
    runtime: str | None,
    durable: PersistedPosition | None,
) -> None:
    assert store.changes == []
    assert await account.revision() == revision
    assert await views(account) == (None if runtime is None else D(runtime), durable)
    assert await account.position_baselines() == ()


# --- the command / record value objects ------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "",
        " ",
        "\t\n",
        "\x00",
        "\x07\x1b",
        " operator confirmed",
        "operator confirmed ",
        "operator\nconfirmed",
        "operator​confirmed",
        "x" * (AUDIT_REASON_MAX_LENGTH + 1),
        b"operator",
        42,
    ],
)
def test_invalid_reasons_are_rejected_never_normalized(reason: object) -> None:
    with pytest.raises(DomainValidationError):
        PositionBaselineAcceptance(
            symbol=BTC,
            expected_exchange_qty=D("5"),
            expected_revision=1,
            reason=reason,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "reason", ["operator confirmed position", "x", "x" * AUDIT_REASON_MAX_LENGTH, "ok: 5 BTC"]
)
def test_valid_reasons_are_kept_exactly(reason: str) -> None:
    assert require_audit_reason(reason) is reason
    assert command("5", 1, reason=reason).reason is reason


class HostileDecimal(Decimal):
    pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", ""),
        ("symbol", " BTCUSDT"),
        ("symbol", None),
        ("expected_exchange_qty", 5),
        ("expected_exchange_qty", 5.0),
        ("expected_exchange_qty", "5"),
        ("expected_exchange_qty", D("NaN")),
        ("expected_exchange_qty", D("sNaN")),
        ("expected_exchange_qty", D("Infinity")),
        ("expected_exchange_qty", D("-Infinity")),
        ("expected_exchange_qty", HostileDecimal("5")),
        ("expected_exchange_qty", None),
        ("expected_revision", -1),
        ("expected_revision", True),
        ("expected_revision", 1.0),
        ("expected_revision", None),
    ],
)
def test_invalid_command_fields(field: str, value: object) -> None:
    fields: dict[str, Any] = {
        "symbol": BTC,
        "expected_exchange_qty": D("5"),
        "expected_revision": 1,
        "reason": REASON,
    }
    fields[field] = value
    with pytest.raises(DomainValidationError):
        PositionBaselineAcceptance(**fields)


def test_the_record_is_immutable_and_validated() -> None:
    record = PositionBaselineRecord(
        baseline_id="bl_1",
        symbol=BTC,
        qty=D("5"),
        reason=REASON,
        accepted_at=T0,
        account_revision=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.qty = D("6")  # type: ignore[misc]
    good: dict[str, Any] = dataclasses.asdict(record)
    for field, value in [
        ("baseline_id", ""),
        ("baseline_id", "bl 1"),
        ("baseline_id", "x" * 65),
        ("baseline_id", "bl_é"),
        ("qty", 5),
        ("qty", D("NaN")),
        ("qty", HostileDecimal("5")),
        ("reason", " x"),
        ("accepted_at", T0.replace(tzinfo=None)),
        ("accepted_at", T0.astimezone(timezone(timedelta(hours=2)))),
        ("account_revision", 0),
        ("account_revision", True),
    ]:
        with pytest.raises(DomainValidationError):
            PositionBaselineRecord(**{**good, field: value})


def test_default_baseline_ids_are_unpredictable_and_valid() -> None:
    ids = {default_baseline_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("bl_") and len(i) == 35 for i in ids)


@pytest.mark.asyncio
async def test_wrong_argument_types_are_rejected_before_the_lock() -> None:
    account, store = await unexplained(None, "5")
    good: dict[str, Any] = {
        "account_state": account,
        "command": command("5", await account.revision()),
        "clock": CountingClock(),
        "baseline_ids": Ids(),
    }
    for field, value in [
        ("account_state", object()),
        ("command", dataclasses.asdict(good["command"])),
        ("clock", object()),
        ("baseline_ids", "bl_1"),
    ]:
        with pytest.raises(DomainValidationError):
            await accept_position_baseline(**{**good, field: value})
    assert store.changes == []


# --- acceptance --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("durable", "durable_row", "runtime", "fills", "before"),
    [
        (None, True, "5", 0, E.UNEXPLAINED_NONZERO),
        (None, True, "-5", 0, E.UNEXPLAINED_NONZERO),
        (None, False, "5", 0, E.UNEXPLAINED_NONZERO),
        (None, True, "0", 2, E.UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE),
        ("2", True, "5", 0, E.DURABLE_PROJECTION_MISMATCH),
        ("-2", True, "-5", 0, E.DURABLE_PROJECTION_MISMATCH),
        ("2", True, "0", 0, E.DURABLE_PROJECTION_MISMATCH),
        ("2", True, "1E-200", 0, E.DURABLE_PROJECTION_MISMATCH),
        ("2", True, "-1" + "0" * 40, 0, E.DURABLE_PROJECTION_MISMATCH),
        ("2", True, "5.000", 0, E.DURABLE_PROJECTION_MISMATCH),
    ],
    ids=[
        "unknown+5",
        "unknown-5",
        "absent+5",
        "unknown-flat-fills",
        "mismatch-2-5",
        "mismatch-neg",
        "mismatch-to-flat",
        "tiny",
        "huge-short",
        "trailing-zeros",
    ],
)
@pytest.mark.asyncio
async def test_acceptance_makes_the_runtime_position_the_durable_baseline(
    durable: str | None, durable_row: bool, runtime: str, fills: int, before: E
) -> None:
    account, store = await unexplained(durable, runtime, durable_row=durable_row, fills=fills)
    revision = await account.revision()
    check = await reconcile_positions(account_state=account, snapshot=snap({BTC: runtime}))
    assert check.positions[0].explanation is before
    assert not check.reconciled
    clock, ids = CountingClock(at(100)), Ids("bl_a")

    result = await accept(account, runtime, clock=clock, ids=ids)

    record = PositionBaselineRecord(
        baseline_id="bl_a",
        symbol=BTC,
        qty=D(runtime),
        reason=REASON,
        accepted_at=at(100),
        account_revision=revision + 1,
    )
    assert result == PositionBaselineAcceptanceResult(
        symbol=BTC,
        qty=D(runtime),
        outcome=BaselineOutcome.ACCEPTED,
        baseline_record=record,
        revision=revision + 1,
    )
    assert repr(result.qty) == repr(D(runtime))  # the runtime object, exactly
    assert (clock.calls, ids.calls) == (1, 1)
    # Exactly one change: the known position, the record and the revision CAS.
    assert store.changes == [
        AccountStateChange(
            account_scope_id="acct-1",
            expected_revision=revision,
            new_revision=revision + 1,
            position_writes=(row(BTC, runtime),),
            position_baseline_writes=(record,),
        )
    ]
    assert await views(account) == (D(runtime), row(BTC, runtime))
    assert await account.position_baselines() == (record,)
    assert await account.position_baselines(BTC) == (record,)
    assert await account.position_baselines(ETH) == ()
    stored = await store.inner.load(account_scope_id="acct-1")
    assert stored is not None
    assert dict(stored.position_baselines) == {"bl_a": record}
    assert stored.positions[BTC] == row(BTC, runtime)
    # The same complete snapshot is now explained by the durable projection.
    after = await reconcile_positions(account_state=account, snapshot=snap({BTC: runtime}))
    assert after.positions[0].explanation is E.DURABLE_PROJECTION_MATCH
    assert after.reconciled


@pytest.mark.asyncio
async def test_acceptance_does_not_change_other_symbols() -> None:
    account, store = await unexplained("2", "5")
    await reconcile_positions(account_state=account, snapshot=snap({BTC: "5", ETH: "3"}))
    await accept(account, "5")
    assert await views(account, ETH) == (D("3"), None)
    assert store.changes[0].position_writes == (row(BTC, "5"),)


# --- no-op ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("durable", "runtime"), [("5", "5"), ("0", "0"), ("5", "5.00")])
@pytest.mark.asyncio
async def test_already_accepted_is_a_no_op_without_clock_id_or_commit(
    durable: str, runtime: str
) -> None:
    account, store = await unexplained(durable, runtime)
    revision = await account.revision()
    clock, ids = CountingClock(), Ids()

    result = await accept(account, runtime, clock=clock, ids=ids)

    assert result == PositionBaselineAcceptanceResult(
        symbol=BTC,
        qty=D(runtime),
        outcome=BaselineOutcome.ALREADY_ACCEPTED,
        baseline_record=None,
        revision=revision,
    )
    assert (clock.calls, ids.calls) == (0, 0)
    await assert_unchanged(
        account, store, revision=revision, runtime=runtime, durable=row(BTC, durable)
    )


@pytest.mark.asyncio
async def test_repeating_an_accepted_command_with_the_new_revision_is_a_no_op() -> None:
    account, store = await unexplained("2", "5")
    first = await accept(account, "5")
    store.changes.clear()
    ids = Ids()
    again = await accept(account, "5", revision=first.revision, ids=ids)
    assert again.outcome is BaselineOutcome.ALREADY_ACCEPTED
    assert ids.calls == 0
    assert store.changes == []
    assert len(await account.position_baselines()) == 1


# --- rejected acceptance ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_runtime_position_is_rejected() -> None:
    store = Store()
    first = InMemoryAccountState(account_scope_id="acct-1", store=store)
    async with first.account_lock() as locked:
        await locked.set_position_qty(BTC, D("2"))
    account = await hydrate(store)  # runtime unknown, durable 2
    store.changes.clear()
    clock, ids = CountingClock(), Ids()
    for qty in ("2", "0"):
        with pytest.raises(BaselineRuntimeUnknownError):
            await accept(account, qty, clock=clock, ids=ids)
    assert (clock.calls, ids.calls) == (0, 0)
    await assert_unchanged(account, store, revision=1, runtime=None, durable=row(BTC, "2"))


@pytest.mark.parametrize("expected", ["4", "-5", "0", "5.0000000001"])
@pytest.mark.asyncio
async def test_a_different_expected_qty_is_rejected(expected: str) -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    clock, ids = CountingClock(), Ids()
    with pytest.raises(BaselineQtyMismatchError):
        await accept(account, expected, clock=clock, ids=ids)
    assert (clock.calls, ids.calls) == (0, 0)
    await assert_unchanged(account, store, revision=revision, runtime="5", durable=row(BTC, "2"))


@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.asyncio
async def test_a_stale_or_future_revision_is_rejected(delta: int) -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    clock, ids = CountingClock(), Ids()
    with pytest.raises(StaleRevisionError):
        await accept(account, "5", revision=revision + delta, clock=clock, ids=ids)
    assert (clock.calls, ids.calls) == (0, 0)
    await assert_unchanged(account, store, revision=revision, runtime="5", durable=row(BTC, "2"))


@pytest.mark.asyncio
async def test_poison_is_checked_before_the_clock() -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    account._poisoned = True
    clock, ids = CountingClock(), Ids()
    with pytest.raises(AccountStatePoisonedError):
        await accept(account, "5", clock=clock, ids=ids)
    assert (clock.calls, ids.calls) == (0, 0)
    await assert_unchanged(account, store, revision=revision, runtime="5", durable=row(BTC, "2"))


@pytest.mark.parametrize(
    "now",
    [
        RuntimeError("clock down"),
        T0.replace(tzinfo=None),
        datetime(2026, 1, 15, 14, 0, tzinfo=timezone(timedelta(hours=2))),
        "2026-01-15T12:00:00Z",
        None,
    ],
    ids=["raises", "naive", "non-utc", "string", "none"],
)
@pytest.mark.asyncio
async def test_an_invalid_clock_commits_nothing(now: object) -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    clock = CountingClock(now)
    if now is None:
        clock.value = None
    ids = Ids()
    with pytest.raises((RuntimeError, DomainValidationError)):
        await accept(account, "5", clock=clock, ids=ids)
    assert clock.calls == 1
    assert ids.calls == 0
    await assert_unchanged(account, store, revision=revision, runtime="5", durable=row(BTC, "2"))
    assert not account._poisoned


@pytest.mark.parametrize("bad_id", ["", "bl 1", "x" * 65, None, 7])
@pytest.mark.asyncio
async def test_an_invalid_generated_id_commits_nothing(bad_id: object) -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    with pytest.raises(DomainValidationError):
        await accept(account, "5", ids=lambda: bad_id)
    await assert_unchanged(account, store, revision=revision, runtime="5", durable=row(BTC, "2"))


@pytest.mark.asyncio
async def test_a_colliding_baseline_id_fails_closed_and_never_overwrites() -> None:
    account, store = await unexplained("2", "5")
    first = await accept(account, "5", ids=Ids("bl_same"))
    await reconcile_positions(account_state=account, snapshot=snap({BTC: "7"}))
    store.changes.clear()
    revision = await account.revision()
    with pytest.raises(BaselineIdConflictError):
        await accept(account, "7", ids=Ids("bl_same"))
    assert store.changes == []
    assert await account.revision() == revision
    assert await views(account) == (D("7"), row(BTC, "5"))
    assert await account.position_baselines() == (first.baseline_record,)


# --- store failures ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_definite_store_failure_changes_nothing() -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    store.inner.inject_commit_failure(CommitFailure.DEFINITE)
    with pytest.raises(StoreCommitError):
        await accept(account, "5")
    assert len(store.changes) == 1
    assert await account.revision() == revision
    assert await views(account) == (D("5"), row(BTC, "2"))
    assert await account.position_baselines() == ()
    assert not account._poisoned
    stored = await store.inner.load(account_scope_id="acct-1")
    assert stored is not None
    assert dict(stored.position_baselines) == {}
    # Not poisoned: a retry with the same revision succeeds.
    store.changes.clear()
    retried = await accept(account, "5", revision=revision)
    assert retried.outcome is BaselineOutcome.ACCEPTED


@pytest.mark.asyncio
async def test_an_uncertain_store_failure_poisons_and_publishes_nothing() -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    store.inner.inject_commit_failure(CommitFailure.UNCERTAIN)
    with pytest.raises(StoreUncertainError):
        await accept(account, "5", ids=Ids("bl_u"))
    assert account._poisoned
    assert await account.revision() == revision
    assert await views(account) == (D("5"), row(BTC, "2"))
    assert await account.position_baselines() == ()
    with pytest.raises(AccountStatePoisonedError):
        await accept(account, "5", revision=revision)
    # The store applied it (outcome unknown to RAM): hydration decides.
    restarted = await hydrate(store)
    assert await views(restarted) == (None, row(BTC, "5"))
    assert [r.baseline_id for r in await restarted.position_baselines()] == ["bl_u"]


# --- restart and reconciliation -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_baseline_survives_restart_and_explains_the_next_reconciliation() -> None:
    account, store = await unexplained(None, "5")
    result = await accept(account, "5", reason="operator confirmed: manual buy on web UI")
    assert result.baseline_record is not None

    restarted = await hydrate(store)

    assert await views(restarted) == (None, row(BTC, "5"))  # runtime forgotten
    assert await restarted.position_baselines() == (result.baseline_record,)
    assert (await restarted.position_baselines())[0].reason == (
        "operator confirmed: manual buy on web UI"
    )
    reconciled = await reconcile_positions(account_state=restarted, snapshot=snap({BTC: "5"}))
    assert reconciled.positions[0].explanation is E.DURABLE_PROJECTION_MATCH
    assert reconciled.reconciled
    assert await restarted.position_qty(BTC) == D("5")
    # Not a lasting proof: another exchange quantity is a mismatch again.
    again = await hydrate(store)
    moved = await reconcile_positions(account_state=again, snapshot=snap({BTC: "6"}))
    assert moved.positions[0].explanation is E.DURABLE_PROJECTION_MISMATCH


# --- subsequent fills ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_an_unknown_baseline_fills_move_both_projections() -> None:
    account, _ = await unexplained(None, "5")
    await trade(account, 1, Side.BUY, "2")  # before: runtime only
    assert await views(account) == (D("7"), row(BTC, None))
    await accept(account, "7")
    await trade(account, 2, Side.BUY, "2")
    assert await views(account) == (D("9"), row(BTC, "9"))
    await trade(account, 3, Side.SELL, "4")
    assert await views(account) == (D("5"), row(BTC, "5"))


@pytest.mark.asyncio
async def test_a_mismatch_fails_closed_until_accepted_then_fills_move_both() -> None:
    account, _ = await unexplained("2", "5")
    await reserve(account, 1, BTC, Side.SELL, "1")
    with pytest.raises(PositionProjectionMismatchError):
        async with account.account_lock() as locked:
            await locked.apply_fill(fill(1, Side.SELL, "1"), at=at(10))
    await accept(account, "5")
    async with account.account_lock() as locked:
        await locked.apply_fill(fill(1, Side.SELL, "1"), at=at(10))
    assert await views(account) == (D("4"), row(BTC, "4"))


@pytest.mark.asyncio
async def test_reduce_only_after_acceptance() -> None:
    account, _ = await unexplained("2", "5")
    await accept(account, "5")
    await trade(account, 1, Side.SELL, "2", reduce_only=True)
    assert await views(account) == (D("3"), row(BTC, "3"))
    revision = await account.revision()
    for n, side, qty in ((2, Side.BUY, "1"), (3, Side.SELL, "4")):  # increase / reverse
        await reserve(account, n, BTC, side, qty, reduce_only=True)
        revision = await account.revision()
        with pytest.raises(FillApplicationError):
            async with account.account_lock() as locked:
                await locked.apply_fill(fill(n, side, qty), at=at(10))
        assert await account.revision() == revision
    assert await views(account) == (D("3"), row(BTC, "3"))


# --- audit history ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_is_append_only_and_ordered_by_revision_then_id() -> None:
    account, store = await unexplained("2", "5")
    clock = CountingClock(at(100))
    first = await accept(account, "5", ids=Ids("bl_z"), clock=clock)
    await reconcile_positions(account_state=account, snapshot=snap({BTC: "7", ETH: "1"}))
    clock.value = at(200)
    second = await accept(account, "7", ids=Ids("bl_a"), clock=clock, reason="second look")
    third = await accept(account, "1", ids=Ids("bl_m"), clock=clock, symbol=ETH)
    records = (first.baseline_record, second.baseline_record, third.baseline_record)
    assert [r.account_revision for r in records if r] == sorted(
        r.account_revision for r in records if r
    )
    assert await account.position_baselines() == records
    assert await account.position_baselines(BTC) == records[:2]
    assert await views(account) == (D("7"), row(BTC, "7"))
    # Hydrate restores the whole history in the same deterministic order.
    restarted = await hydrate(store)
    assert await restarted.position_baselines() == records
    returned = await restarted.position_baselines()
    assert isinstance(returned, tuple)


@pytest.mark.asyncio
async def test_accepted_at_never_goes_back_before_an_earlier_baseline() -> None:
    account, _ = await unexplained("2", "5")
    ids = Ids()
    await accept(account, "5", clock=CountingClock(at(300)), ids=ids)
    await reconcile_positions(account_state=account, snapshot=snap({BTC: "7"}))
    later = await accept(account, "7", clock=CountingClock(at(100)), ids=ids)  # clock went back
    assert later.baseline_record is not None
    assert later.baseline_record.accepted_at == at(300)


def test_records_sort_by_revision_then_id_regardless_of_insertion() -> None:
    def rec(bid: str, revision: int) -> PositionBaselineRecord:
        return PositionBaselineRecord(
            baseline_id=bid,
            symbol=BTC,
            qty=D("1"),
            reason=REASON,
            accepted_at=T0,
            account_revision=revision,
        )

    records = [rec("bl_b", 3), rec("bl_a", 3), rec("bl_c", 1)]
    account = InMemoryAccountState(account_scope_id="acct-1", store=Store())
    account._state = dataclasses.replace(
        account._state, position_baselines={r.baseline_id: r for r in records}, revision=3
    )

    async def read() -> tuple[PositionBaselineRecord, ...]:
        return await account.position_baselines()

    assert [r.baseline_id for r in asyncio.run(read())] == ["bl_c", "bl_a", "bl_b"]


# --- concurrency (deterministic, no sleep) -------------------------------------------------------


@pytest.mark.asyncio
async def test_acceptance_waits_for_an_in_flight_fill_and_sees_its_result() -> None:
    account, store = await unexplained(None, "5")
    await reserve(account, 1, BTC, Side.BUY, "2")
    revision = await account.revision()
    store.changes.clear()
    store.block = asyncio.Event()
    fill_task = asyncio.create_task(
        _apply(account, fill(1, Side.BUY, "2"))
    )  # holds the lock through the commit
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    accept_task = asyncio.create_task(accept(account, "5", revision=revision))
    await asyncio.sleep(0)
    assert not accept_task.done()
    store.block.set()
    await asyncio.wait_for(fill_task, timeout=5)
    # The command saw revision R and runtime 5; the fill moved both: stale.
    with pytest.raises(StaleRevisionError):
        await asyncio.wait_for(accept_task, timeout=5)
    assert await views(account) == (D("7"), row(BTC, None))
    assert await account.position_baselines() == ()


@pytest.mark.asyncio
async def test_a_fill_waits_for_an_in_flight_acceptance_then_moves_both() -> None:
    account, store = await unexplained("2", "5")
    await reserve(account, 1, BTC, Side.SELL, "1")
    store.changes.clear()
    store.block = asyncio.Event()
    accept_task = asyncio.create_task(accept(account, "5"))
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    fill_task = asyncio.create_task(_apply(account, fill(1, Side.SELL, "1")))
    await asyncio.sleep(0)
    assert not fill_task.done()
    store.block.set()
    await asyncio.wait_for(accept_task, timeout=5)
    await asyncio.wait_for(fill_task, timeout=5)
    assert await views(account) == (D("4"), row(BTC, "4"))
    assert [c.position_baseline_writes != () for c in store.changes] == [True, False]


@pytest.mark.asyncio
async def test_a_newer_reconciliation_makes_an_older_acceptance_stale() -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()  # the operator saw runtime 5 at R
    store.changes.clear()
    store.block = asyncio.Event()
    # Hold the account lock with an unrelated committed mutation, then queue the
    # reconciliation (runtime 7) BEFORE the operator's acceptance of 5.
    holder = asyncio.create_task(_set_eth(account))
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    reconcile_task = asyncio.create_task(
        reconcile_positions(account_state=account, snapshot=snap({BTC: "7", ETH: "1"}))
    )
    await asyncio.sleep(0)
    accept_task = asyncio.create_task(accept(account, "5", revision=revision + 1))
    await asyncio.sleep(0)
    store.block.set()
    store.block = None
    await asyncio.wait_for(holder, timeout=5)
    await asyncio.wait_for(reconcile_task, timeout=5)
    with pytest.raises(BaselineQtyMismatchError):
        await asyncio.wait_for(accept_task, timeout=5)
    assert await views(account) == (D("7"), row(BTC, "2"))
    assert await account.position_baselines() == ()
    assert [c.position_baseline_writes for c in store.changes] == [()]


@pytest.mark.asyncio
async def test_two_identical_commands_create_one_record() -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()
    ids = Ids()
    results = await asyncio.gather(
        accept(account, "5", revision=revision, ids=ids),
        accept(account, "5", revision=revision, ids=ids),
        return_exceptions=True,
    )
    assert isinstance(results[0], PositionBaselineAcceptanceResult)
    assert results[0].outcome is BaselineOutcome.ACCEPTED
    assert isinstance(results[1], StaleRevisionError)
    assert ids.calls == 1
    assert len(store.changes) == 1
    assert len(await account.position_baselines()) == 1


async def _apply(account: InMemoryAccountState, f: Fill) -> None:
    async with account.account_lock() as locked:
        await locked.apply_fill(f, at=at(10))


async def _set_eth(account: InMemoryAccountState) -> None:
    async with account.account_lock() as locked:
        await locked.set_position_qty(ETH, D("1"))


# --- the privileged primitive and the persistence contract -----------------------------------


@pytest.mark.asyncio
async def test_the_primitive_rechecks_its_preconditions() -> None:
    account, store = await unexplained("2", "5")
    revision = await account.revision()

    def rec(**changes: Any) -> PositionBaselineRecord:
        fields: dict[str, Any] = {
            "baseline_id": "bl_p",
            "symbol": BTC,
            "qty": D("5"),
            "reason": REASON,
            "accepted_at": T0,
            "account_revision": revision + 1,
        }
        return PositionBaselineRecord(**{**fields, **changes})

    cases: list[tuple[Any, type[Exception]]] = [
        (rec(account_revision=revision), StaleRevisionError),
        (rec(account_revision=revision + 2), StaleRevisionError),
        (rec(symbol=ETH), BaselineRuntimeUnknownError),
        (rec(qty=D("4")), BaselineQtyMismatchError),
        (rec(qty=D("5.0")), BaselineQtyMismatchError),  # exactly the runtime object
        (dataclasses.asdict(rec()), DomainValidationError),
    ]
    for record, error in cases:
        with pytest.raises(error):
            async with account.account_lock() as locked:
                await locked.commit_position_baseline(record)
    assert store.changes == []
    assert await views(account) == (D("5"), row(BTC, "2"))


def _record(**changes: Any) -> PositionBaselineRecord:
    fields: dict[str, Any] = {
        "baseline_id": "bl_s",
        "symbol": BTC,
        "qty": D("5"),
        "reason": REASON,
        "accepted_at": T0,
        "account_revision": 1,
    }
    return PositionBaselineRecord(**{**fields, **changes})


@pytest.mark.parametrize(
    "writes",
    [
        {"position_writes": ()},
        {"position_writes": (row(BTC, "4"),)},
        {"position_writes": (row(BTC, "5.0"),)},
        {"position_writes": (row(BTC, None),)},
        {"position_writes": (row(ETH, "5"),)},
    ],
    ids=["no-position", "other-qty", "other-repr", "unknown", "other-symbol"],
)
def test_a_change_never_carries_a_record_without_its_position(writes: dict[str, Any]) -> None:
    with pytest.raises(StoreValidationError):
        AccountStateChange(
            account_scope_id="acct-1",
            expected_revision=0,
            new_revision=1,
            position_baseline_writes=(_record(),),
            **writes,
        )


def test_a_change_record_must_be_at_the_new_revision() -> None:
    for expected, new, revision in ((0, 1, 2), (1, 1, 1), (1, 2, 1)):
        with pytest.raises(StoreValidationError):
            AccountStateChange(
                account_scope_id="acct-1",
                expected_revision=expected,
                new_revision=new,
                position_writes=(row(BTC, "5"),),
                position_baseline_writes=(_record(account_revision=revision),),
            )
    with pytest.raises(StoreValidationError):
        AccountStateChange(
            account_scope_id="acct-1",
            expected_revision=0,
            new_revision=1,
            position_writes=(row(BTC, "5"),),
            position_baseline_writes=[_record()],  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_the_store_keeps_records_append_only() -> None:
    store = InMemoryAccountStateStore()

    def change(expected: int, record: PositionBaselineRecord) -> AccountStateChange:
        return AccountStateChange(
            account_scope_id="acct-1",
            expected_revision=expected,
            new_revision=expected + 1,
            position_writes=(row(record.symbol, str(record.qty)),),
            position_baseline_writes=(record,),
        )

    first = _record()
    await store.commit(change(0, first))
    with pytest.raises(StoreConflictError):  # same id, other payload: never overwritten
        await store.commit(change(1, _record(qty=D("6"), account_revision=2)))
    second = _record(baseline_id="bl_t", qty=D("6"), account_revision=2)
    await store.commit(change(1, second))
    loaded = await store.load(account_scope_id="acct-1")
    assert loaded is not None
    assert dict(loaded.position_baselines) == {"bl_s": first, "bl_t": second}
    with pytest.raises(TypeError):
        loaded.position_baselines["x"] = first  # type: ignore[index]
    with pytest.raises(StoreValidationError):  # duplicates inside one change
        await store.commit(
            AccountStateChange(
                account_scope_id="acct-1",
                expected_revision=2,
                new_revision=3,
                position_writes=(row(BTC, "5"),),
                position_baseline_writes=(
                    _record(baseline_id="bl_u", account_revision=3),
                    _record(baseline_id="bl_u", account_revision=3),
                ),
            )
        )


def test_a_snapshot_rejects_records_from_the_future_or_under_a_wrong_key() -> None:
    base: dict[str, Any] = {
        "account_scope_id": "acct-1",
        "revision": 1,
        "placements": {},
        "orders": {},
        "fills": {},
        "positions": {BTC: row(BTC, "5")},
        "notionals": {},
    }
    PersistedAccountState(**base, position_baselines={"bl_s": _record()})
    for baselines in ({"bl_x": _record()}, {"bl_s": _record(account_revision=2)}):
        with pytest.raises(StoreValidationError):
            PersistedAccountState(**base, position_baselines=baselines)


# --- architecture ------------------------------------------------------------------------------


def test_module_is_explicit_and_never_touches_gates_readers_or_network() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "typing",
        "app.domain.clock",
        "app.domain.errors",
        "app.domain.validation",
        "app.execution.account_state",
        "app.execution.models",
        "app.execution.timing",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in (
        "SafetyController",
        "mark_exchange_reconciled",
        "mark_hydrated",
        "reconcile_positions",
        "ExchangeStateReader",
        "set_position_qty",
        "publish_exchange_positions",
    ):
        assert forbidden not in names, forbidden
    # Exactly one durable mutation path, and it is the privileged primitive.
    assert source.count("commit_position_baseline(") == 1
