"""Authoritative open-order discovery: one complete snapshot, the pure matcher
against the CURRENT local orders, BLOCKED findings as results, and the only
correction (exchange id completion) atomically and only for a clean snapshot."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import itertools
from collections.abc import Awaitable, Callable
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
from app.exchanges.models import OrderRequest
from app.exchanges.protocols import ExchangeStateReader
from app.exchanges.recovery import ExchangeOrder, OpenOrdersSnapshot
from app.exchanges.simulated import SimulatedExchange
from app.execution import open_order_discovery as module
from app.execution.account_state import (
    EXCHANGE_ID_COMPLETION_STATUSES,
    AccountStatePoisonedError,
    ExchangeIdCompletionError,
    ExchangeOrderIdCompletion,
    InMemoryAccountState,
    StaleRevisionError,
)
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.models import ExchangeOrderState, SubmissionOutcome
from app.execution.open_order_discovery import (
    OpenOrderDiscoveryError,
    OpenOrderDiscoveryOutcome,
    OpenOrderDiscoveryResult,
    discover_open_orders,
)
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    StoreCommitError,
    StoreUncertainError,
)
from app.execution.recovery_matching import RELEVANT_LOCAL_STATUSES, IdentityConflictReason
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision

D = Decimal
O = OpenOrderDiscoveryOutcome  # noqa: E741
R = IdentityConflictReason
BTC = "BTCUSDT"
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NS = ClientOrderNamespace("bot01")
OTHER_NS = ClientOrderNamespace("bot02")


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def cid(n: int) -> str:
    return NS.build(f"o{n}")


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


class Reader:
    """Fake ``ExchangeStateReader``: one snapshot; ``during`` runs while the
    read is in flight (no account lock held), to stage races."""

    def __init__(
        self,
        orders: tuple[ExchangeOrder, ...] = (),
        *,
        error: Exception | None = None,
        result: object = None,
        during: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.orders = orders
        self.error = error
        self.result = result
        self.during = during
        self.calls = 0

    async def list_open_orders(self) -> Any:
        self.calls += 1
        if self.during is not None:
            await self.during()
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return OpenOrdersSnapshot(orders=self.orders, server_ts=at(60))

    async def get_position_snapshot(self) -> Any:  # pragma: no cover - not used
        raise AssertionError("not used")

    async def list_executions(self, query: Any, *, cursor: str | None = None) -> Any:
        raise AssertionError("not used")  # pragma: no cover


def ex(
    client_order_id: str | None,
    exchange_order_id: str,
    *,
    qty: str = "1",
    side: Side = Side.BUY,
    price: str = "100",
    symbol: str = BTC,
) -> ExchangeOrder:
    return ExchangeOrder(
        exchange_order_id=exchange_order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        price=D(price),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        status=OrderStatus.OPEN,
        cum_filled_qty=D(0),
        cum_filled_notional=None,
        avg_fill_price=None,
        created_ts=T0,
        updated_ts=T0,
    )


async def place_local(
    account: InMemoryAccountState,
    client_order_id: str,
    *,
    exchange_order_id: str | None = None,
    qty: str = "1",
    side: Side = Side.BUY,
) -> None:
    """A sent order with an ambiguous outcome: UNKNOWN (relevant), with an
    exchange id only if an ack recorded one."""
    intent = PlaceOrderIntent(
        intent_id=f"i-{client_order_id}",
        strategy_id="grid-1",
        symbol=BTC,
        side=side,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        tag=None,
        created_at=T0,
    )
    async with account.account_lock() as locked:
        await locked.register_approved(
            intent=intent,
            decision=RiskDecision(
                intent_id=intent.intent_id,
                snapshot_id="s",
                policy_id="p",
                approved=True,
                reasons=(),
                exposure=ExposureChange(
                    reducing_qty=D("0"),
                    increasing_qty=D(qty),
                    worst_long_qty=D(qty),
                    worst_short_qty=D("0"),
                ),
            ),
            client_order_id=client_order_id,
            expected_revision=locked.revision,
            at=T0,
        )
        await locked.mark_submitting(client_order_id, at=at(1))
        await locked.record_submission_outcome(
            client_order_id, SubmissionOutcome.AMBIGUOUS, at=at(2)
        )
        if exchange_order_id is not None:
            await locked.record_ack(client_order_id, exchange_order_id=exchange_order_id, at=at(3))


async def account_with(
    *orders: tuple[str, str | None], store: Store | None = None
) -> tuple[InMemoryAccountState, Store]:
    store = Store() if store is None else store
    account = InMemoryAccountState(account_scope_id="acct-1", store=store)
    for client_order_id, exchange_order_id in orders:
        await place_local(account, client_order_id, exchange_order_id=exchange_order_id)
    store.changes.clear()
    return account, store


async def discover(
    account: InMemoryAccountState,
    reader: ExchangeStateReader,
    namespace: ClientOrderNamespace = NS,
) -> OpenOrderDiscoveryResult:
    return await discover_open_orders(account_state=account, reader=reader, namespace=namespace)


async def exchange_ids(account: InMemoryAccountState) -> dict[str, str | None]:
    return {o.client_order_id: o.exchange_order_id for o in await account.orders()}


async def assert_blocked_untouched(
    account: InMemoryAccountState,
    store: Store,
    result: OpenOrderDiscoveryResult,
    ids: dict[str, str | None],
) -> None:
    assert result.outcome is O.BLOCKED
    assert not result.clean
    assert result.exchange_ids_completed == ()
    assert result.revision_after == result.revision_before
    assert store.changes == []
    assert await exchange_ids(account) == ids


# --- the reader boundary --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_exchange_and_no_local_order_is_clean() -> None:
    account, store = await account_with()
    reader = Reader()
    result = await discover(account, reader)
    assert result.outcome is O.CLEAN
    assert result.clean
    assert reader.calls == 1
    assert (result.revision_before, result.revision_after) == (0, 0)
    assert result.snapshot.orders == ()
    assert result.classification.is_clean
    assert store.changes == []


@pytest.mark.asyncio
async def test_a_poisoned_account_does_not_read_the_exchange() -> None:
    account, _ = await account_with((cid(1), None))
    account._poisoned = True
    reader = Reader((ex(cid(1), "X-1"),))
    with pytest.raises(AccountStatePoisonedError):
        await discover(account, reader)
    assert reader.calls == 0


@pytest.mark.asyncio
async def test_poison_during_the_read_is_checked_again_under_the_lock() -> None:
    account, store = await account_with((cid(1), None))

    async def poison() -> None:
        account._poisoned = True

    with pytest.raises(AccountStatePoisonedError):
        await discover(account, Reader((ex(cid(1), "X-1"),), during=poison))
    assert store.changes == []
    assert await exchange_ids(account) == {cid(1): None}


@pytest.mark.parametrize(
    "error", [ConnectionError("down"), TimeoutError(), RuntimeError("bad page")]
)
@pytest.mark.asyncio
async def test_a_reader_failure_propagates_and_changes_nothing(error: Exception) -> None:
    account, store = await account_with((cid(1), None))
    with pytest.raises(type(error)):
        await discover(account, Reader(error=error))
    assert store.changes == []
    assert not account._poisoned
    assert await exchange_ids(account) == {cid(1): None}
    assert await account.revision() == 3


@pytest.mark.parametrize("result", [(), [ex(cid(1), "X-1")], {"orders": ()}, "snapshot", object()])
@pytest.mark.asyncio
async def test_a_reader_breaking_its_contract_fails_closed(result: object) -> None:
    account, store = await account_with((cid(1), None))
    with pytest.raises(OpenOrderDiscoveryError):
        await discover(account, Reader(result=result))
    assert store.changes == []
    assert await exchange_ids(account) == {cid(1): None}


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected_before_the_read() -> None:
    account, _ = await account_with()
    reader = Reader()
    for kwargs in (
        {"account_state": object(), "reader": reader, "namespace": NS},
        {"account_state": account, "reader": object(), "namespace": NS},
        {"account_state": account, "reader": reader, "namespace": "bot01"},
    ):
        with pytest.raises(DomainValidationError):
            await discover_open_orders(**kwargs)  # type: ignore[arg-type]
    assert reader.calls == 0


def test_completion_statuses_are_the_matcher_relevant_statuses() -> None:
    assert EXCHANGE_ID_COMPLETION_STATUSES == RELEVANT_LOCAL_STATUSES


# --- blocking findings --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_local_active_order_absent_from_the_snapshot_blocks_without_inference() -> None:
    account, store = await account_with((cid(1), "X-1"), (cid(2), None))
    result = await discover(account, Reader())
    await assert_blocked_untouched(account, store, result, {cid(1): "X-1", cid(2): None})
    assert [o.client_order_id for o in result.classification.missing_local_active] == [
        cid(1),
        cid(2),
    ]
    statuses = {o.status for o in await account.orders()}
    assert statuses == {OrderStatus.UNKNOWN}  # never CANCELED / FAILED / FILLED


@pytest.mark.parametrize(
    ("client_order_id", "bucket"),
    [
        (None, "foreign"),
        ("manual-order-7", "foreign"),
        (OTHER_NS.build("o1"), "foreign"),
        (cid(9), "lost_managed"),
        ("tb1_bot01_", "identity_conflicts"),
        ("tb1_BOT01_o1", "identity_conflicts"),
        ("tb2_bot01_o1", "identity_conflicts"),
    ],
    ids=["no-cid", "unmanaged", "other-namespace", "lost", "malformed", "upper", "unsupported"],
)
@pytest.mark.asyncio
async def test_each_blocking_bucket_prevents_an_otherwise_valid_completion(
    client_order_id: str | None, bucket: str
) -> None:
    account, store = await account_with((cid(1), None))
    reader = Reader((ex(cid(1), "X-1"), ex(client_order_id, "X-2", qty="3")))
    result = await discover(account, reader)
    await assert_blocked_untouched(account, store, result, {cid(1): None})
    assert len(getattr(result.classification, bucket)) == 1
    # The completable match is reported, but not completed.
    (match,) = result.classification.managed
    assert match.exchange_id_completion_required


@pytest.mark.asyncio
async def test_a_missing_local_order_prevents_another_completion() -> None:
    account, store = await account_with((cid(1), None), (cid(2), None))
    result = await discover(account, Reader((ex(cid(1), "X-1"),)))
    await assert_blocked_untouched(account, store, result, {cid(1): None, cid(2): None})
    assert [o.client_order_id for o in result.classification.missing_local_active] == [cid(2)]


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        ((ex(cid(1), "X-1", qty="2"),), R.TERMS_MISMATCH),
        ((ex(cid(1), "X-1", side=Side.SELL),), R.TERMS_MISMATCH),
        ((ex(cid(1), "X-1", price="100.1"),), R.TERMS_MISMATCH),
        ((ex(cid(1), "Y-1"),), R.EXCHANGE_ID_MISMATCH),
        ((ex(cid(3), "X-2"),), R.REVERSE_EXCHANGE_ID_COLLISION),
    ],
    ids=["qty", "side", "price", "exchange-id-mismatch", "reverse-collision"],
)
@pytest.mark.asyncio
async def test_identity_conflicts_block_and_correct_nothing(
    snapshot: tuple[ExchangeOrder, ...], reason: R
) -> None:
    account, store = await account_with((cid(1), "X-1"), (cid(2), "X-2"), (cid(3), None))
    full = (*snapshot, *(o for o in (ex(cid(2), "X-2"), ex(cid(3), "X-3")) if o not in snapshot))
    if reason is R.REVERSE_EXCHANGE_ID_COLLISION:
        full = (ex(cid(1), "X-1"), ex(cid(3), "X-2"))
    result = await discover(account, Reader(full))
    await assert_blocked_untouched(
        account, store, result, {cid(1): "X-1", cid(2): "X-2", cid(3): None}
    )
    assert reason in {c.reason for c in result.classification.identity_conflicts}


@pytest.mark.asyncio
async def test_a_local_order_of_another_namespace_is_a_conflict_not_foreign() -> None:
    account, store = await account_with((OTHER_NS.build("o1"), None))
    result = await discover(account, Reader((ex(OTHER_NS.build("o1"), "X-1"),)))
    await assert_blocked_untouched(account, store, result, {OTHER_NS.build("o1"): None})
    assert result.classification.foreign == ()
    assert {c.reason for c in result.classification.identity_conflicts} == {R.LOCAL_UNMANAGED_ID}


@pytest.mark.asyncio
async def test_the_namespace_is_explicit() -> None:
    account, store = await account_with((cid(1), None))
    reader = Reader((ex(cid(1), "X-1"),))
    other = await discover(account, reader, namespace=OTHER_NS)
    assert other.outcome is O.BLOCKED
    assert store.changes == []
    ours = await discover(account, reader, namespace=NS)
    assert ours.outcome is O.CORRECTED


# --- clean classification -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_managed_orders_are_clean_without_a_commit() -> None:
    account, store = await account_with((cid(1), "X-1"), (cid(2), "X-2"))
    revision = await account.revision()
    result = await discover(account, Reader((ex(cid(2), "X-2"), ex(cid(1), "X-1"))))
    assert result.outcome is O.CLEAN
    assert result.exchange_ids_completed == ()
    assert (result.revision_before, result.revision_after) == (revision, revision)
    assert store.changes == []
    assert [m.local_order.client_order_id for m in result.classification.managed] == [
        cid(1),
        cid(2),
    ]


@pytest.mark.parametrize("count", [1, 2, 5, 20])
@pytest.mark.asyncio
async def test_missing_exchange_ids_are_completed_in_one_change(count: int) -> None:
    account, store = await account_with(*((cid(n), None) for n in range(1, count + 1)))
    before = {o.client_order_id: o for o in await account.orders()}
    revision = await account.revision()
    snapshot = tuple(ex(cid(n), f"X-{n}") for n in range(count, 0, -1))

    result = await discover(account, Reader(snapshot))

    assert result.outcome is O.CORRECTED
    assert result.clean
    assert (result.revision_before, result.revision_after) == (revision, revision + 1)
    expected = sorted(
        (
            dataclasses.replace(
                before[cid(n)], exchange_order_id=f"X-{n}", version=before[cid(n)].version + 1
            )
            for n in range(1, count + 1)
        ),
        key=lambda o: o.client_order_id,
    )
    assert list(result.exchange_ids_completed) == expected
    assert store.changes == [
        AccountStateChange(
            account_scope_id="acct-1",
            expected_revision=revision,
            new_revision=revision + 1,
            order_writes=tuple(expected),
        )
    ]
    # Final classification of the same snapshot: clean, nothing left to complete.
    assert result.classification.is_clean
    assert not any(m.exchange_id_completion_required for m in result.classification.managed)
    assert len(result.classification.managed) == count
    # Identity only: status, execution, terms and timestamps are kept.
    for order in await account.orders():
        old = before[order.client_order_id]
        assert dataclasses.replace(order, exchange_order_id=None, version=old.version) == old


@pytest.mark.asyncio
async def test_mixed_exact_and_missing_ids_complete_only_the_missing_ones() -> None:
    account, store = await account_with(
        (cid(1), "X-1"), (cid(2), None), (cid(3), "X-3"), (cid(4), None)
    )
    exact_before = {o.client_order_id: o for o in await account.orders()}
    revision = await account.revision()
    snapshot = tuple(ex(cid(n), f"X-{n}") for n in (4, 3, 2, 1))
    result = await discover(account, Reader(snapshot))
    assert result.outcome is O.CORRECTED
    assert [o.client_order_id for o in result.exchange_ids_completed] == [cid(2), cid(4)]
    assert len(store.changes) == 1
    assert store.changes[0].new_revision == revision + 1
    after = {o.client_order_id: o for o in await account.orders()}
    assert after[cid(1)] is exact_before[cid(1)]
    assert after[cid(3)] is exact_before[cid(3)]
    assert await exchange_ids(account) == {cid(n): f"X-{n}" for n in (1, 2, 3, 4)}


@pytest.mark.asyncio
async def test_a_second_discovery_of_the_same_snapshot_is_a_no_op() -> None:
    account, store = await account_with((cid(1), None))
    reader = Reader((ex(cid(1), "X-1"),))
    await discover(account, reader)
    store.changes.clear()
    revision = await account.revision()
    again = await discover(account, reader)
    assert again.outcome is O.CLEAN
    assert store.changes == []
    assert again.revision_after == revision


@pytest.mark.asyncio
async def test_result_is_independent_of_snapshot_and_local_order() -> None:
    snapshot = (ex(cid(1), "X-1"), ex(cid(2), "X-2"), ex(cid(3), "X-3"))
    seen = set()
    for local_order in itertools.permutations((1, 2, 3)):
        for exchange_order in itertools.permutations(snapshot):
            account, store = await account_with(*((cid(n), None) for n in local_order))
            result = await discover(account, Reader(exchange_order))
            seen.add(
                (
                    tuple(o.client_order_id for o in result.exchange_ids_completed),
                    tuple(o.client_order_id for o in store.changes[0].order_writes),
                    tuple(
                        m.exchange_order.exchange_order_id for m in result.classification.managed
                    ),
                )
            )
    assert seen == {((cid(1), cid(2), cid(3)),) * 2 + (("X-1", "X-2", "X-3"),)}


# --- store failures -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_definite_store_failure_completes_no_id() -> None:
    account, store = await account_with((cid(1), None), (cid(2), None))
    revision = await account.revision()
    store.inner.inject_commit_failure(CommitFailure.DEFINITE)
    with pytest.raises(StoreCommitError):
        await discover(account, Reader((ex(cid(1), "X-1"), ex(cid(2), "X-2"))))
    assert await exchange_ids(account) == {cid(1): None, cid(2): None}
    assert await account.revision() == revision
    assert not account._poisoned


@pytest.mark.asyncio
async def test_an_uncertain_store_failure_poisons_and_publishes_no_id() -> None:
    account, store = await account_with((cid(1), None), (cid(2), None))
    revision = await account.revision()
    store.inner.inject_commit_failure(CommitFailure.UNCERTAIN)
    reader = Reader((ex(cid(1), "X-1"), ex(cid(2), "X-2")))
    with pytest.raises(StoreUncertainError):
        await discover(account, reader)
    assert account._poisoned
    assert await exchange_ids(account) == {cid(1): None, cid(2): None}
    assert await account.revision() == revision
    with pytest.raises(AccountStatePoisonedError):
        await discover(account, reader)
    # The durable outcome is decided by the store (here: applied).
    restarted = await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=store, clock=ManualClock(at(100))
    )
    assert await exchange_ids(restarted) == {cid(1): "X-1", cid(2): "X-2"}


# --- races between the read and the classification ------------------------------------------


@pytest.mark.asyncio
async def test_an_unrelated_fill_during_the_read_does_not_prevent_the_completion() -> None:
    account, _ = await account_with((cid(1), None), (cid(2), "X-2"))
    revision = await account.revision()

    async def fill_other() -> None:
        async with account.account_lock() as locked:
            await locked.apply_fill(
                Fill(
                    exec_id="e-1",
                    exchange_order_id="X-2",
                    client_order_id=cid(2),
                    symbol=BTC,
                    side=Side.BUY,
                    price=D("100"),
                    qty=D("0.5"),
                    fee=None,
                    fee_asset=None,
                    is_maker=None,
                    exchange_ts=at(4),
                ),
                at=at(5),
            )

    result = await discover(
        account, Reader((ex(cid(1), "X-1"), ex(cid(2), "X-2")), during=fill_other)
    )
    assert result.outcome is O.CORRECTED
    assert result.revision_before == revision + 1  # classified against the current state
    assert result.revision_after == revision + 2
    assert await exchange_ids(account) == {cid(1): "X-1", cid(2): "X-2"}


@pytest.mark.asyncio
async def test_an_id_recorded_elsewhere_during_the_read_is_not_committed_again() -> None:
    account, store = await account_with((cid(1), None))

    async def ack() -> None:
        async with account.account_lock() as locked:
            await locked.record_ack(cid(1), exchange_order_id="X-1", at=at(5))

    result = await discover(account, Reader((ex(cid(1), "X-1"),), during=ack))
    assert result.outcome is O.CLEAN
    assert result.exchange_ids_completed == ()
    assert len(store.changes) == 1  # the ack only


@pytest.mark.asyncio
async def test_a_conflicting_id_recorded_during_the_read_blocks_and_is_kept() -> None:
    account, store = await account_with((cid(1), None))

    async def ack() -> None:
        async with account.account_lock() as locked:
            await locked.record_ack(cid(1), exchange_order_id="Y-1", at=at(5))

    result = await discover(account, Reader((ex(cid(1), "X-1"),), during=ack))
    assert result.outcome is O.BLOCKED
    assert {c.reason for c in result.classification.identity_conflicts} == {R.EXCHANGE_ID_MISMATCH}
    assert await exchange_ids(account) == {cid(1): "Y-1"}
    assert len(store.changes) == 1  # the ack only


@pytest.mark.asyncio
async def test_a_new_local_order_during_the_read_is_missing_from_the_stale_snapshot() -> None:
    account, _ = await account_with((cid(1), None))

    async def place() -> None:
        await place_local(account, cid(2))

    result = await discover(account, Reader((ex(cid(1), "X-1"),), during=place))
    assert result.outcome is O.BLOCKED
    assert [o.client_order_id for o in result.classification.missing_local_active] == [cid(2)]
    assert await exchange_ids(account) == {cid(1): None, cid(2): None}


@pytest.mark.asyncio
async def test_a_local_order_turning_terminal_during_the_read_is_no_longer_missing() -> None:
    account, _ = await account_with((cid(1), None), (cid(2), None))

    async def cancel() -> None:
        async with account.account_lock() as locked:
            await locked.apply_exchange_state(
                ExchangeOrderState(
                    client_order_id=cid(2),
                    exchange_order_id=None,
                    status=OrderStatus.CANCELED,
                    filled_qty=D(0),
                    avg_fill_price=None,
                    exchange_ts=at(4),
                ),
                at=at(5),
            )

    result = await discover(account, Reader((ex(cid(1), "X-1"),), during=cancel))
    assert result.outcome is O.CORRECTED
    assert result.classification.missing_local_active == ()
    assert await exchange_ids(account) == {cid(1): "X-1", cid(2): None}


@pytest.mark.asyncio
async def test_the_network_read_holds_no_account_lock() -> None:
    account, _ = await account_with((cid(1), None))
    seen: list[int] = []

    async def read_state() -> None:
        seen.append(await account.revision())  # would raise / deadlock under the lock

    await discover(account, Reader((ex(cid(1), "X-1"),), during=read_state))
    assert seen == [3]


@pytest.mark.asyncio
async def test_a_completion_waits_for_an_in_flight_mutation() -> None:
    account, store = await account_with((cid(1), None), (cid(2), "X-2"))
    store.block = asyncio.Event()

    async def hold() -> None:
        async with account.account_lock() as locked:
            await locked.set_position_qty(BTC, D("1"))

    holder = asyncio.create_task(hold())
    await asyncio.wait_for(store.entered.wait(), timeout=5)
    task = asyncio.create_task(discover(account, Reader((ex(cid(1), "X-1"), ex(cid(2), "X-2")))))
    await asyncio.sleep(0)
    assert not task.done()
    store.block.set()
    store.block = None
    await asyncio.wait_for(holder, timeout=5)
    result = await asyncio.wait_for(task, timeout=5)
    assert result.outcome is O.CORRECTED
    assert [c.order_writes != () for c in store.changes] == [False, True]


# --- the batch primitive -----------------------------------------------------------------------


def completion(n: int, exchange_order_id: str | None = None) -> ExchangeOrderIdCompletion:
    return ExchangeOrderIdCompletion(
        client_order_id=cid(n), exchange_order_id=exchange_order_id or f"X-{n}"
    )


async def complete(
    account: InMemoryAccountState,
    items: tuple[ExchangeOrderIdCompletion, ...],
    *,
    revision: int | None = None,
) -> Any:
    async with account.account_lock() as locked:
        return await locked.complete_exchange_order_ids(
            items, expected_revision=locked.revision if revision is None else revision
        )


@pytest.mark.asyncio
async def test_primitive_writes_sorted_orders_in_one_change_for_any_input_order() -> None:
    for items in itertools.permutations((completion(1), completion(2), completion(3))):
        account, store = await account_with((cid(1), None), (cid(2), None), (cid(3), None))
        state = account._state
        revision = await account.revision()
        updated = await complete(account, items)
        assert [o.client_order_id for o in updated] == [cid(1), cid(2), cid(3)]
        (change,) = store.changes
        assert change.order_writes == updated
        assert (change.expected_revision, change.new_revision) == (revision, revision + 1)
        assert account._state is not state  # copy-on-write
        assert all(o.exchange_order_id is None for o in state.orders.values())


@pytest.mark.asyncio
async def test_primitive_skips_exact_ids_and_commits_nothing_when_all_are_exact() -> None:
    account, store = await account_with((cid(1), "X-1"), (cid(2), None))
    assert [o.client_order_id for o in await complete(account, (completion(1), completion(2)))] == [
        cid(2)
    ]
    store.changes.clear()
    revision = await account.revision()
    assert await complete(account, (completion(1), completion(2))) == ()
    assert store.changes == []
    assert await account.revision() == revision


@pytest.mark.asyncio
async def test_primitive_rejects_any_unprovable_completion_and_applies_none() -> None:
    account, store = await account_with((cid(1), None), (cid(2), "Y-2"), (cid(3), "X-3"))
    async with account.account_lock() as locked:  # a terminal (non-relevant) order
        await locked.apply_exchange_state(
            ExchangeOrderState(
                client_order_id=cid(3),
                exchange_order_id="X-3",
                status=OrderStatus.CANCELED,
                filled_qty=D(0),
                avg_fill_price=None,
                exchange_ts=at(4),
            ),
            at=at(5),
        )
    await place_local(account, cid(4))
    store.changes.clear()
    revision = await account.revision()
    cases: list[tuple[tuple[ExchangeOrderIdCompletion, ...], type[Exception]]] = [
        ((completion(1), completion(2)), ExchangeIdCompletionError),  # Y-2 never replaced
        ((completion(1), completion(9)), ExchangeIdCompletionError),  # no such order
        ((completion(1), completion(3, "X-30")), ExchangeIdCompletionError),  # terminal
        ((completion(1, "X-3"),), ExchangeIdCompletionError),  # held by order 3
        ((completion(1, "Y-2"),), ExchangeIdCompletionError),  # held by order 2
        ((completion(1, "X-1"), completion(4, "X-1")), ExchangeIdCompletionError),  # dup id
        ((completion(1, "X-1"), completion(1, "X-9")), ExchangeIdCompletionError),  # dup cid
    ]
    for items, error in cases:
        with pytest.raises(error):
            await complete(account, items)
    with pytest.raises(StaleRevisionError):
        await complete(account, (completion(1),), revision=revision - 1)
    with pytest.raises(DomainValidationError):
        await complete(account, [completion(1)])  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError):
        ExchangeOrderIdCompletion(client_order_id=cid(1), exchange_order_id=" X-1")
    assert store.changes == []
    assert await account.revision() == revision
    assert (await exchange_ids(account))[cid(1)] is None
    account._poisoned = True
    with pytest.raises(AccountStatePoisonedError):
        await complete(account, (completion(1),))
    assert store.changes == []


@pytest.mark.parametrize("failure", [CommitFailure.DEFINITE, CommitFailure.UNCERTAIN])
@pytest.mark.asyncio
async def test_primitive_store_failure_publishes_nothing(failure: CommitFailure) -> None:
    account, store = await account_with((cid(1), None), (cid(2), None))
    revision = await account.revision()
    store.inner.inject_commit_failure(failure)
    with pytest.raises((StoreCommitError, StoreUncertainError)):
        await complete(account, (completion(1), completion(2)))
    assert await exchange_ids(account) == {cid(1): None, cid(2): None}
    assert await account.revision() == revision
    assert account._poisoned is (failure is CommitFailure.UNCERTAIN)


# --- simulator --------------------------------------------------------------------------------


def _spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol=BTC,
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=D("0.1"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        max_qty=D("1000000"),
        min_notional=D("0"),
    )


@pytest.mark.asyncio
async def test_simulator_end_to_end_completion_survives_restart() -> None:
    sim = SimulatedExchange(clock=ManualClock(T0), instruments=(_spec(),))
    ack = await sim.place_order(
        OrderRequest(
            client_order_id=cid(1),
            symbol=BTC,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            price=D("100"),
            qty=D("1"),
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
        )
    )
    account, store = await account_with((cid(1), None))  # the ack was lost: UNKNOWN

    first = await discover(account, sim)
    assert first.outcome is O.CORRECTED
    assert await exchange_ids(account) == {cid(1): ack.exchange_order_id}

    restarted = await InMemoryAccountState.hydrate(
        account_scope_id="acct-1", store=store, clock=ManualClock(at(100))
    )
    assert await exchange_ids(restarted) == {cid(1): ack.exchange_order_id}
    store.changes.clear()
    again = await discover(restarted, sim)
    assert again.outcome is O.CLEAN
    assert store.changes == []


@pytest.mark.asyncio
async def test_simulator_foreign_order_blocks_without_local_mutation() -> None:
    sim = SimulatedExchange(clock=ManualClock(T0), instruments=(_spec(),))
    await sim.place_order(
        OrderRequest(
            client_order_id=cid(1),
            symbol=BTC,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            price=D("100"),
            qty=D("1"),
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
        )
    )
    foreign = sim.add_external_order(
        client_order_id=None, symbol=BTC, side=Side.SELL, price=D("120"), qty=D("2")
    )
    account, store = await account_with((cid(1), None))
    result = await discover(account, sim)
    await assert_blocked_untouched(account, store, result, {cid(1): None})
    assert result.classification.foreign == (foreign,)
    assert len(await account.orders()) == 1  # nothing imported


# --- architecture -----------------------------------------------------------------------------


def test_module_never_acts_on_orders_gates_or_the_store() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in (
        "place_order",
        "cancel_order",
        "cancel_all_orders",
        "mark_exchange_reconciled",
        "SafetyController",
        "commit",
        "now",
        "Clock",
        "register_approved",
        "apply_exchange_state",
        "record_submission_outcome",
        "transition",
        "reconcile_positions",
        "accept_position_baseline",
        "SimulatedExchange",
    ):
        assert forbidden not in names, forbidden
    # Exactly one network read and one durable mutation path (calls, not docs).
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.count("list_open_orders") == 1
    assert calls.count("complete_exchange_order_ids") == 1
