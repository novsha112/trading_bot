"""Exchange reports applied to local orders, and single-shot UNKNOWN reconciliation."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.order_state import EXCHANGE_REPORTED_STATUSES, transition
from app.domain.orders import Order, OrderUpdate
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeError,
    ExchangeResponseError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.protocols import TradingClient
from app.execution import reconciliation as reconciliation_module
from app.execution import submitter as submitter_module
from app.execution.account_state import (
    AccountStateError,
    ExchangeStateMismatchError,
    InMemoryAccountState,
    LockedAccountState,
    MissingFillsError,
)
from app.execution.models import ExchangeOrderState
from app.execution.reconciliation import (
    OrderNotUnknownError,
    OrderStillUnknownError,
    UnknownOrderReconciler,
    exchange_state_from_update,
)
from app.execution.safety import SafetyController
from app.execution.submitter import OrderSubmitter
from app.persistence.memory import InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision, TradingState
from app.risk.snapshots import build_risk_snapshot

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
T3 = T0 + timedelta(seconds=3)
EXPOSURE = ExposureChange(
    reducing_qty=D("0"), increasing_qty=D("1"), worst_long_qty=D("1"), worst_short_qty=D("0")
)


def ready_safety(account: InMemoryAccountState) -> SafetyController:
    """Every recovery gate confirmed and RUNNING requested: effective RUNNING."""
    safety = SafetyController(account_state=account)
    safety.mark_hydrated()
    safety.mark_exchange_reconciled()
    safety.request_state(TradingState.RUNNING)
    return safety


def new_account(account_scope_id: str = "acct-1") -> InMemoryAccountState:
    """An account state on a fresh in-memory reference store."""
    return InMemoryAccountState(
        account_scope_id=account_scope_id, store=InMemoryAccountStateStore()
    )


class FixedClock:
    def __init__(self, at: datetime = T3) -> None:
        self.at = at

    def now(self) -> datetime:
        return self.at


def intent() -> PlaceOrderIntent:
    return PlaceOrderIntent(
        intent_id="i-1",
        strategy_id="grid-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("100"),
        qty=D("10"),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        tag=None,
        created_at=T0,
    )


async def account_with(status: OrderStatus, *, filled: str = "0") -> InMemoryAccountState:
    """Known flat BTCUSDT and order "c-1" in ``status``.

    Fills go through ``apply_fill`` (order and position stay consistent); other
    statuses are installed white-box through domain transitions, since order
    updates for them have no public producer in these tests."""
    account = new_account()
    source = intent()
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
                exposure=EXPOSURE,
            ),
            client_order_id="c-1",
            expected_revision=locked.revision,
            at=T0,
        )
        if status is S.NEW:
            return account
        await locked.mark_submitting("c-1", at=T0)
        if D(filled) > 0:
            await locked.apply_fill(fill("e-1", filled), at=T1)
        order = locked.order("c-1")
        assert order is not None
        paths: dict[OrderStatus, list[OrderStatus]] = {
            S.SUBMITTING: [],
            S.PARTIALLY_FILLED: [],
            S.FILLED: [],
            S.UNKNOWN: [S.UNKNOWN],
            S.OPEN: [S.OPEN],
            S.CANCELING: [S.OPEN, S.CANCELING] if D(filled) == 0 else [S.CANCELING],
            S.CANCELED: [S.CANCELED],
            S.EXPIRED: [S.OPEN, S.EXPIRED] if D(filled) == 0 else [S.EXPIRED],
            S.FAILED: [S.FAILED],
            S.REJECTED: [S.REJECTED],
        }
        for step in paths[status]:
            order = transition(order, step, at=T1)
        account._state.orders["c-1"] = order
    assert (await current(account)).status is status
    return account


def fill(exec_id: str, qty: str, *, exchange_order_id: str = "ex-1") -> Fill:
    return Fill(
        exec_id=exec_id,
        exchange_order_id=exchange_order_id,
        client_order_id="c-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        price=D("100"),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=T1,
    )


def report(
    status: OrderStatus,
    filled: str = "0",
    *,
    exchange_order_id: str | None = "ex-1",
    avg: str | None = None,
    client_order_id: str = "c-1",
    ts: datetime = T2,
) -> ExchangeOrderState:
    return ExchangeOrderState(
        client_order_id=client_order_id,
        exchange_order_id=exchange_order_id,
        status=status,
        filled_qty=D(filled),
        avg_fill_price=(D(avg) if avg is not None else (D("100") if D(filled) > 0 else None)),
        exchange_ts=ts,
    )


def update(status: OrderStatus, filled: str = "0", **overrides: Any) -> OrderUpdate:
    values: dict[str, Any] = {
        "client_order_id": "c-1",
        "exchange_order_id": "ex-1",
        "status": status,
        "cum_filled_qty": D(filled),
        "avg_fill_price": D("100") if D(filled) > 0 else None,
        "reject_reason": None,
        "exchange_ts": T2,
    }
    return OrderUpdate(**{**values, **overrides})


async def current(account: InMemoryAccountState) -> Order:
    order = await account.order("c-1")
    assert order is not None
    return order


async def apply(account: InMemoryAccountState, item: ExchangeOrderState) -> Order:
    async with account.account_lock() as locked:
        return await locked.apply_exchange_state(item, at=T3)


def whole_state(account: InMemoryAccountState) -> tuple[Any, ...]:
    state = account._state
    return (
        state.revision,
        dict(state.orders),
        dict(state.positions),
        dict(state.fills),
        dict(state.notionals),
    )


async def active_exposure(account: InMemoryAccountState) -> int:
    async with account.account_lock() as locked:
        snapshot = build_risk_snapshot(
            snapshot_id=f"a:{locked.revision}",
            symbol="BTCUSDT",
            trading_state=TradingState.RUNNING,
            position_qty=locked.position_qty("BTCUSDT"),
            orders=locked.active_orders("BTCUSDT"),
            account_open_order_count=locked.account_active_order_count(),
        )
    assert snapshot.open_orders is not None
    return len(snapshot.open_orders)


# --- normalized model -----------------------------------------------------------------------


def test_exchange_reported_statuses_exclude_local_lifecycle_statuses() -> None:
    assert {S.OPEN, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.REJECTED, S.EXPIRED} == (
        EXCHANGE_REPORTED_STATUSES
    )


@pytest.mark.parametrize("status", [S.NEW, S.SUBMITTING, S.CANCELING, S.UNKNOWN, S.FAILED])
def test_local_statuses_are_not_exchange_facts(status: OrderStatus) -> None:
    with pytest.raises(DomainValidationError, match="not an exchange-reported status"):
        report(status)
    with pytest.raises(DomainValidationError, match="not an exchange-reported status"):
        exchange_state_from_update(update(status))


def test_update_is_normalized_field_by_field() -> None:
    assert exchange_state_from_update(update(S.PARTIALLY_FILLED, "4")) == ExchangeOrderState(
        client_order_id="c-1",
        exchange_order_id="ex-1",
        status=S.PARTIALLY_FILLED,
        filled_qty=D("4"),
        avg_fill_price=D("100"),
        exchange_ts=T2,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"filled_qty": D("0"), "avg_fill_price": D("1")}, "avg_fill_price"),
        ({"filled_qty": D("1"), "avg_fill_price": None}, "avg_fill_price"),
        ({"filled_qty": D("-1")}, "filled_qty"),
        ({"exchange_ts": datetime(2026, 1, 15)}, "exchange_ts"),  # noqa: DTZ001
        ({"client_order_id": ""}, "client_order_id"),
    ],
)
def test_exchange_state_validation(kwargs: dict[str, Any], match: str) -> None:
    values: dict[str, Any] = {
        "client_order_id": "c-1",
        "exchange_order_id": None,
        "status": S.OPEN,
        "filled_qty": D("0"),
        "avg_fill_price": None,
        "exchange_ts": T2,
    }
    with pytest.raises(DomainValidationError, match=match):
        ExchangeOrderState(**{**values, **kwargs})


# --- apply_exchange_state: identity ---------------------------------------------------------


@pytest.mark.asyncio
async def test_report_records_a_missing_exchange_order_id() -> None:
    account = await account_with(S.UNKNOWN)
    base = await account.revision()

    order = await apply(account, report(S.OPEN, exchange_order_id="ex-9"))

    assert (order.status, order.exchange_order_id, order.last_exchange_update_ts) == (
        S.OPEN,
        "ex-9",
        T2,
    )
    assert order.updated_at == T3
    assert await account.revision() == base + 1


@pytest.mark.asyncio
async def test_report_with_another_exchange_order_id_changes_nothing() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")  # ex-1 from the fill
    before = whole_state(account)

    with pytest.raises(ExchangeStateMismatchError, match="ex-2"):
        await apply(account, report(S.PARTIALLY_FILLED, "4", exchange_order_id="ex-2"))

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_report_without_exchange_id_keeps_the_known_one() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")

    order = await apply(account, report(S.CANCELED, "4", exchange_order_id=None))

    assert (order.status, order.exchange_order_id) == (S.CANCELED, "ex-1")


@pytest.mark.asyncio
async def test_report_for_an_unknown_order_is_an_error() -> None:
    account = await account_with(S.UNKNOWN)
    before = whole_state(account)

    with pytest.raises(AccountStateError, match="c-9"):
        await apply(account, report(S.OPEN, client_order_id="c-9"))

    assert whole_state(account) == before


@pytest.mark.parametrize("status", [S.NEW, S.FAILED])
@pytest.mark.asyncio
async def test_report_for_a_never_accepted_order_is_a_mismatch(status: OrderStatus) -> None:
    account = await account_with(status)
    before = whole_state(account)

    with pytest.raises(ExchangeStateMismatchError, match="never accepted"):
        await apply(account, report(S.OPEN))

    assert whole_state(account) == before


# --- apply_exchange_state: executed quantity ------------------------------------------------


@pytest.mark.asyncio
async def test_fill_first_then_matching_report_confirms_without_position_change() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    before = whole_state(account)

    order = await apply(account, report(S.PARTIALLY_FILLED, "4"))

    assert order == before[1]["c-1"]  # same status, same id: nothing to record
    assert whole_state(account) == before
    assert await account.position_qty("BTCUSDT") == D("4")


@pytest.mark.asyncio
async def test_fill_first_then_cancel_report_releases_without_position_change() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    base = await account.revision()

    order = await apply(account, report(S.CANCELED, "4"))

    assert (order.status, order.filled_qty) == (S.CANCELED, D("4"))
    assert await account.position_qty("BTCUSDT") == D("4")
    assert await account.revision() == base + 1
    assert await active_exposure(account) == 0


@pytest.mark.asyncio
async def test_report_with_more_execution_than_applied_fills_is_missing_fills() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    before = whole_state(account)

    with pytest.raises(MissingFillsError, match="apply the missing fills first"):
        await apply(account, report(S.PARTIALLY_FILLED, "6"))

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_stale_report_with_less_execution_is_a_mismatch() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="6")
    before = whole_state(account)

    with pytest.raises(ExchangeStateMismatchError, match="below"):
        await apply(account, report(S.PARTIALLY_FILLED, "4"))

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_different_average_price_is_a_mismatch() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    before = whole_state(account)

    with pytest.raises(ExchangeStateMismatchError, match="average price"):
        await apply(account, report(S.PARTIALLY_FILLED, "4", avg="100.5"))

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_numerically_equal_values_are_consistent_under_a_low_precision_context() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    getcontext().clear_flags()
    before = (getcontext().prec, getcontext().rounding, dict(getcontext().flags))

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        order = await apply(account, report(S.CANCELED, "4.000", avg="100.00"))

    assert (order.status, order.filled_qty, order.avg_fill_price) == (S.CANCELED, D("4"), D("100"))
    assert (getcontext().prec, getcontext().rounding, dict(getcontext().flags)) == before


# --- apply_exchange_state: status monotonicity ----------------------------------------------


@pytest.mark.parametrize(
    ("local", "filled", "reported", "reported_filled"),
    [
        (S.PARTIALLY_FILLED, "4", S.OPEN, "4"),  # OPEN cannot have fills
        (S.FILLED, "10", S.OPEN, "10"),
        (S.FILLED, "10", S.PARTIALLY_FILLED, "10"),
        (S.FILLED, "10", S.CANCELED, "10"),
        (S.CANCELED, "0", S.OPEN, "0"),
        (S.EXPIRED, "0", S.OPEN, "0"),
        (S.REJECTED, "0", S.OPEN, "0"),
        (S.CANCELED, "0", S.REJECTED, "0"),
        (S.CANCELING, "4", S.OPEN, "4"),
    ],
)
@pytest.mark.asyncio
async def test_report_never_regresses_confirmed_progress(
    local: OrderStatus, filled: str, reported: OrderStatus, reported_filled: str
) -> None:
    account = await account_with(local, filled=filled)
    before = whole_state(account)

    with pytest.raises(ExchangeStateMismatchError, match="cannot move"):
        await apply(account, report(reported, reported_filled))

    assert whole_state(account) == before


@pytest.mark.parametrize("status", [S.FILLED, S.CANCELED, S.REJECTED, S.EXPIRED])
@pytest.mark.asyncio
async def test_matching_report_of_a_terminal_order_is_a_no_op(status: OrderStatus) -> None:
    filled = "10" if status is S.FILLED else "0"
    account = await account_with(status, filled=filled)
    before = whole_state(account)

    if status is S.FILLED:
        order = await apply(account, report(status, filled))
    else:
        order = await apply(account, report(status, filled, exchange_order_id=None))

    assert order.status is status
    assert whole_state(account) == before


@pytest.mark.parametrize(
    ("reported", "active"),
    [(S.OPEN, 1), (S.CANCELED, 0), (S.EXPIRED, 0), (S.REJECTED, 0)],
)
@pytest.mark.asyncio
async def test_unknown_resolves_to_a_reported_status(reported: OrderStatus, active: int) -> None:
    account = await account_with(S.UNKNOWN)
    assert await active_exposure(account) == 1

    order = await apply(account, report(reported))

    assert order.status is reported
    assert await active_exposure(account) == active
    assert await account.position_qty("BTCUSDT") == D("0")


@pytest.mark.asyncio
async def test_unknown_to_filled_needs_the_fills_first() -> None:
    account = await account_with(S.UNKNOWN)
    before = whole_state(account)

    with pytest.raises(MissingFillsError):
        await apply(account, report(S.FILLED, "10"))
    assert whole_state(account) == before

    async with account.account_lock() as locked:
        await locked.apply_fill(fill("e-1", "10"), at=T2)  # the missing fill: UNKNOWN -> FILLED
    base = await account.revision()
    order = await apply(account, report(S.FILLED, "10"))

    assert order.status is S.FILLED
    assert await account.revision() == base  # the report only confirms
    assert await account.position_qty("BTCUSDT") == D("10")


@pytest.mark.parametrize(
    ("reported", "result"),
    [(S.CANCELED, S.CANCELED), (S.EXPIRED, S.EXPIRED)],
)
@pytest.mark.asyncio
async def test_unknown_with_fills_resolves_with_the_same_execution(
    reported: OrderStatus, result: OrderStatus
) -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    async with account.account_lock() as locked:
        order = locked.order("c-1")
        assert order is not None
        account._state.orders["c-1"] = transition(order, S.UNKNOWN, at=T1)

    resolved = await apply(account, report(reported, "4"))

    assert (resolved.status, resolved.filled_qty) == (result, D("4"))
    assert await active_exposure(account) == 0


@pytest.mark.asyncio
async def test_rejected_report_with_fills_violates_the_domain_and_changes_nothing() -> None:
    account = await account_with(S.PARTIALLY_FILLED, filled="4")
    async with account.account_lock() as locked:
        order = locked.order("c-1")
        assert order is not None
        account._state.orders["c-1"] = transition(order, S.UNKNOWN, at=T1)
    before = whole_state(account)

    with pytest.raises(ExchangeStateMismatchError, match="rejected"):
        await apply(account, report(S.REJECTED, "4"))

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_identical_report_is_idempotent() -> None:
    account = await account_with(S.UNKNOWN)
    base = await account.revision()

    first = await apply(account, report(S.OPEN))
    second = await apply(account, report(S.OPEN))

    assert second == first
    assert await account.revision() == base + 1


@pytest.mark.asyncio
async def test_report_confirming_the_status_records_only_a_new_exchange_id() -> None:
    account = await account_with(S.OPEN)
    base = await account.revision()

    order = await apply(account, report(S.OPEN, exchange_order_id="ex-5"))

    assert (order.status, order.exchange_order_id) == (S.OPEN, "ex-5")
    assert await account.revision() == base + 1


@pytest.mark.asyncio
async def test_invalid_report_argument() -> None:
    account = await account_with(S.UNKNOWN)

    async with account.account_lock() as locked:
        with pytest.raises(DomainValidationError, match="ExchangeOrderState"):
            await locked.apply_exchange_state(update(S.OPEN), at=T3)  # type: ignore[arg-type]


# --- reconciler -----------------------------------------------------------------------------


class ReadClient:
    """TradingClient double for reads; placement / cancel are forbidden."""

    def __init__(
        self,
        *,
        answer: Any = None,
        error: BaseException | None = None,
        during: Callable[[OrderRef], Awaitable[None]] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.during = during
        self.gate = gate
        self.refs: list[OrderRef] = []
        self.called = asyncio.Event()

    async def place_order(self, order: OrderRequest) -> OrderAck:
        raise AssertionError("reconciliation never re-sends")

    async def cancel_order(self, order: OrderRef) -> None:
        raise AssertionError("cancel_order must not be called")

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        self.refs.append(order)
        self.called.set()
        if self.during is not None:
            await self.during(order)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.answer  # type: ignore[no-any-return]

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise AssertionError("get_open_orders must not be called")


def reconciler(account: InMemoryAccountState, client: ReadClient) -> UnknownOrderReconciler:
    return UnknownOrderReconciler(account_state=account, client=client, clock=FixedClock())


@pytest.mark.parametrize(
    ("reported", "active"),
    [(S.OPEN, 1), (S.CANCELED, 0), (S.EXPIRED, 0), (S.REJECTED, 0)],
)
@pytest.mark.asyncio
async def test_reconcile_applies_the_exchange_state(reported: OrderStatus, active: int) -> None:
    account = await account_with(S.UNKNOWN)
    client = ReadClient(answer=update(reported))
    base = await account.revision()

    order = await reconciler(account, client).reconcile(client_order_id="c-1")

    assert (order.status, order.exchange_order_id, order.updated_at) == (reported, "ex-1", T3)
    assert client.refs == [OrderRef(symbol="BTCUSDT", client_order_id="c-1")]
    assert await account.revision() == base + 1
    assert await active_exposure(account) == active


@pytest.mark.asyncio
async def test_reconcile_passes_a_known_exchange_id() -> None:
    account = await account_with(S.UNKNOWN)
    async with account.account_lock() as locked:
        await locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)
    client = ReadClient(answer=update(S.OPEN))

    await reconciler(account, client).reconcile(client_order_id="c-1")

    assert client.refs[0].exchange_order_id == "ex-1"


@pytest.mark.parametrize(
    "status",
    [
        S.NEW,
        S.SUBMITTING,
        S.OPEN,
        S.PARTIALLY_FILLED,
        S.CANCELING,
        S.FILLED,
        S.CANCELED,
        S.EXPIRED,
        S.REJECTED,
        S.FAILED,
    ],
)
@pytest.mark.asyncio
async def test_only_unknown_orders_are_reconciled(status: OrderStatus) -> None:
    filled = "10" if status is S.FILLED else "4" if status is S.PARTIALLY_FILLED else "0"
    account = await account_with(status, filled=filled)
    client = ReadClient(answer=update(S.OPEN))
    before = whole_state(account)

    with pytest.raises(OrderNotUnknownError, match=status.value):
        await reconciler(account, client).reconcile(client_order_id="c-1")

    assert client.refs == []
    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_reconcile_of_a_missing_order_reads_nothing() -> None:
    account = await account_with(S.UNKNOWN)
    client = ReadClient(answer=update(S.OPEN))

    with pytest.raises(AccountStateError, match="c-9"):
        await reconciler(account, client).reconcile(client_order_id="c-9")

    assert client.refs == []


@pytest.mark.asyncio
async def test_not_found_keeps_the_order_unknown_and_active() -> None:
    account = await account_with(S.UNKNOWN)
    client = ReadClient(answer=None)
    before = whole_state(account)

    with pytest.raises(OrderStillUnknownError, match="stays unknown"):
        await reconciler(account, client).reconcile(client_order_id="c-1")

    assert whole_state(account) == before
    assert (await current(account)).status is S.UNKNOWN
    assert await active_exposure(account) == 1


async def concurrent_fill(account: InMemoryAccountState, qty: str) -> None:
    async with account.account_lock() as locked:
        await locked.apply_fill(fill("e-1", qty), at=T2)


async def concurrent_report(account: InMemoryAccountState, status: OrderStatus) -> None:
    async with account.account_lock() as locked:
        await locked.apply_exchange_state(report(status, exchange_order_id=None), at=T2)


async def concurrent_ack(account: InMemoryAccountState) -> None:
    async with account.account_lock() as locked:
        await locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)


@pytest.mark.parametrize(
    ("event", "status", "active"),
    [
        (lambda account: concurrent_fill(account, "4"), S.PARTIALLY_FILLED, 1),
        (lambda account: concurrent_fill(account, "10"), S.FILLED, 0),
        (lambda account: concurrent_report(account, S.OPEN), S.OPEN, 1),
        (lambda account: concurrent_report(account, S.CANCELED), S.CANCELED, 0),
        (lambda account: concurrent_report(account, S.EXPIRED), S.EXPIRED, 0),
        (lambda account: concurrent_report(account, S.REJECTED), S.REJECTED, 0),
        (lambda account: concurrent_ack(account), S.UNKNOWN, 1),  # newer exchange id
    ],
    ids=["partial-fill", "full-fill", "open", "canceled", "expired", "rejected", "exchange-id"],
)
@pytest.mark.asyncio
async def test_not_found_after_newer_local_exchange_facts_is_a_mismatch(
    event: Callable[[InMemoryAccountState], Awaitable[None]], status: OrderStatus, active: int
) -> None:
    account = await account_with(S.UNKNOWN)
    base = await account.revision()

    async def meanwhile(ref: OrderRef) -> None:
        await event(account)

    client = ReadClient(answer=None, during=meanwhile)

    with pytest.raises(ExchangeStateMismatchError, match="newer exchange facts") as caught:
        await reconciler(account, client).reconcile(client_order_id="c-1")

    assert not isinstance(caught.value, OrderStillUnknownError)
    after = whole_state(account)
    order = await current(account)
    assert order.status is status  # the concurrent event's state is kept
    assert await account.revision() == base + 1  # only the concurrent event
    assert len(client.refs) == 1
    assert await active_exposure(account) == active
    # Handling "not found" itself changes nothing.
    assert whole_state(account) == after


@pytest.mark.asyncio
async def test_not_found_after_a_partial_fill_keeps_order_position_and_fills() -> None:
    account = await account_with(S.UNKNOWN)

    async def meanwhile(ref: OrderRef) -> None:
        await concurrent_fill(account, "4")

    with pytest.raises(ExchangeStateMismatchError):
        await reconciler(account, ReadClient(answer=None, during=meanwhile)).reconcile(
            client_order_id="c-1"
        )

    order = await current(account)
    assert (order.status, order.filled_qty) == (S.PARTIALLY_FILLED, D("4"))
    assert await account.position_qty("BTCUSDT") == D("4")
    assert await account.fill("e-1") == fill("e-1", "4")


@pytest.mark.asyncio
async def test_not_found_for_a_disappeared_order_is_an_invariant_error() -> None:
    account = await account_with(S.UNKNOWN)

    async def remove(ref: OrderRef) -> None:
        del account._state.orders["c-1"]  # white-box: orders are never removed

    with pytest.raises(AccountStateError, match="disappeared") as caught:
        await reconciler(account, ReadClient(answer=None, during=remove)).reconcile(
            client_order_id="c-1"
        )

    assert not isinstance(caught.value, OrderStillUnknownError | ExchangeStateMismatchError)


@pytest.mark.asyncio
async def test_concurrent_not_found_answers_while_still_unknown() -> None:
    account = await account_with(S.UNKNOWN)
    gate = asyncio.Event()
    client = ReadClient(answer=None, gate=gate)
    tool = reconciler(account, client)
    before = whole_state(account)

    first = asyncio.create_task(tool.reconcile(client_order_id="c-1"))
    second = asyncio.create_task(tool.reconcile(client_order_id="c-1"))
    await client.called.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    gate.set()
    results = await asyncio.wait_for(
        asyncio.gather(first, second, return_exceptions=True), timeout=5
    )

    assert [type(r) for r in results] == [OrderStillUnknownError, OrderStillUnknownError]
    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_concurrent_not_found_answers_around_a_fill() -> None:
    account = await account_with(S.UNKNOWN)
    first_gate, second_gate = asyncio.Event(), asyncio.Event()
    gates = [first_gate, second_gate]

    class OrderedClient(ReadClient):
        async def get_order(self, order: OrderRef) -> OrderUpdate | None:
            gate = gates[len(self.refs)]
            self.refs.append(order)
            await gate.wait()
            return None

    client = OrderedClient()
    tool = reconciler(account, client)
    first = asyncio.create_task(tool.reconcile(client_order_id="c-1"))
    second = asyncio.create_task(tool.reconcile(client_order_id="c-1"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(client.refs) == 2

    first_gate.set()  # processed while still UNKNOWN
    first_result = await asyncio.gather(first, return_exceptions=True)
    await concurrent_fill(account, "4")  # then a fill arrives
    base = await account.revision()
    second_gate.set()  # processed after the fill
    second_result = await asyncio.gather(second, return_exceptions=True)

    assert isinstance(first_result[0], OrderStillUnknownError)
    assert isinstance(second_result[0], ExchangeStateMismatchError)
    assert not isinstance(second_result[0], OrderStillUnknownError)
    assert (await current(account)).status is S.PARTIALLY_FILLED
    assert await account.revision() == base


@pytest.mark.parametrize(
    "error",
    [
        ExchangeResponseError("timeout"),
        ExchangeAmbiguousResultError("contract violation"),
        ExchangeError("other"),
        RuntimeError("unexpected"),
    ],
)
@pytest.mark.asyncio
async def test_read_error_propagates_and_changes_nothing(error: BaseException) -> None:
    account = await account_with(S.UNKNOWN)
    client = ReadClient(error=error)
    before = whole_state(account)

    with pytest.raises(type(error)) as caught:
        await reconciler(account, client).reconcile(client_order_id="c-1")

    assert caught.value is error
    assert len(client.refs) == 1  # no retry
    assert whole_state(account) == before
    assert await active_exposure(account) == 1


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (update(S.OPEN, client_order_id="c-2"), ExchangeStateMismatchError),
        (update(S.UNKNOWN), ExchangeStateMismatchError),
        (update(S.SUBMITTING), ExchangeStateMismatchError),
        ("not an update", ExchangeStateMismatchError),
        (update(S.PARTIALLY_FILLED, "4"), MissingFillsError),
    ],
)
@pytest.mark.asyncio
async def test_unusable_or_inconsistent_answer_changes_nothing(
    answer: Any, error: type[Exception]
) -> None:
    account = await account_with(S.UNKNOWN)
    before = whole_state(account)

    with pytest.raises(error):
        await reconciler(account, ReadClient(answer=answer)).reconcile(client_order_id="c-1")

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_read_happens_outside_the_account_lock() -> None:
    account = await account_with(S.UNKNOWN)
    seen: dict[str, Any] = {}

    async def during(ref: OrderRef) -> None:
        async def observe() -> None:
            async with account.account_lock() as locked:
                seen["status"] = locked.order("c-1").status  # type: ignore[union-attr]

        await asyncio.wait_for(asyncio.create_task(observe()), timeout=1)

    await reconciler(account, ReadClient(answer=update(S.OPEN), during=during)).reconcile(
        client_order_id="c-1"
    )

    assert seen["status"] is S.UNKNOWN


@pytest.mark.asyncio
async def test_fill_during_the_read_is_never_rolled_back() -> None:
    account = await account_with(S.UNKNOWN)

    async def fill_meanwhile(ref: OrderRef) -> None:
        async with account.account_lock() as locked:
            await locked.apply_fill(fill("e-1", "4"), at=T2)  # UNKNOWN -> PARTIALLY_FILLED

    client = ReadClient(answer=update(S.OPEN), during=fill_meanwhile)  # stale answer

    with pytest.raises(ExchangeStateMismatchError, match="below"):
        await reconciler(account, client).reconcile(client_order_id="c-1")

    order = await current(account)
    assert (order.status, order.filled_qty) == (S.PARTIALLY_FILLED, D("4"))
    assert await account.position_qty("BTCUSDT") == D("4")


@pytest.mark.asyncio
async def test_fill_during_the_read_with_a_matching_answer_is_confirmed() -> None:
    account = await account_with(S.UNKNOWN)

    async def fill_meanwhile(ref: OrderRef) -> None:
        async with account.account_lock() as locked:
            await locked.apply_fill(fill("e-1", "4"), at=T2)

    client = ReadClient(answer=update(S.PARTIALLY_FILLED, "4"), during=fill_meanwhile)

    order = await reconciler(account, client).reconcile(client_order_id="c-1")

    assert (order.status, order.filled_qty) == (S.PARTIALLY_FILLED, D("4"))


@pytest.mark.asyncio
async def test_concurrent_reconciles_apply_once() -> None:
    account = await account_with(S.UNKNOWN)
    gate = asyncio.Event()
    client = ReadClient(answer=update(S.OPEN), gate=gate)
    base = await account.revision()
    tool = reconciler(account, client)

    first = asyncio.create_task(tool.reconcile(client_order_id="c-1"))
    second = asyncio.create_task(tool.reconcile(client_order_id="c-1"))
    await client.called.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    gate.set()
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    assert len(client.refs) == 2  # both read; V1 does not serialize reads
    assert [r.status for r in results] == [S.OPEN, S.OPEN]
    assert await account.revision() == base + 1


# --- submitter exception boundary -----------------------------------------------------------


class AckClient:
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []

    async def place_order(self, order: OrderRequest) -> OrderAck:
        self.requests.append(order)
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="ex-1", exchange_ts=T1
        )

    async def cancel_order(self, order: OrderRef) -> None:
        raise AssertionError

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        raise AssertionError

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise AssertionError


@pytest.mark.asyncio
async def test_local_error_after_the_network_is_not_a_transport_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(self: LockedAccountState, *args: Any, **kwargs: Any) -> Order:
        raise RuntimeError("local bug while recording the ack")

    monkeypatch.setattr(LockedAccountState, "record_ack", broken)
    account = await account_with(S.NEW)
    client = AckClient()
    sender = OrderSubmitter(
        account_state=account, safety=ready_safety(account), client=client, clock=FixedClock(T1)
    )

    with pytest.raises(RuntimeError, match="local bug"):
        await sender.submit(client_order_id="c-1")

    assert len(client.requests) == 1
    # Not classified as an ambiguous transport outcome: the marker stays SUBMITTING.
    assert (await current(account)).status is S.SUBMITTING


def test_submitter_transport_handler_wraps_only_the_network_call() -> None:
    tree = ast.parse(Path(submitter_module.__file__).read_text(encoding="utf-8"))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(h.type, ast.Name) and h.type.id == "BaseException" for h in node.handlers
        )
    ]

    assert len(handlers) == 1
    (body,) = handlers[0].body
    assert isinstance(body, ast.Assign)
    call = body.value
    assert isinstance(call, ast.Await)
    assert ast.unparse(call.value) == "self._client.place_order(request)"


# --- one read per call / dependencies -------------------------------------------------------


def test_reconciler_has_no_retry_or_background_machinery() -> None:
    source = Path(reconciliation_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert not [n for n in ast.walk(tree) if isinstance(n, ast.For | ast.While | ast.AsyncFor)]
    awaited = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Await)]
    network = [call for call in awaited if call.startswith("self._client.")]
    assert network == ["self._client.get_order(ref)"]  # the only exchange call
    local = set(awaited) - set(network)
    assert local == {
        "self._classify_not_found(client_order_id, ref)",
        "locked.apply_exchange_state(report, "
        "at=change_time(self._clock, floor=current.updated_at))",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    allowed = (
        "app.domain",
        "app.execution",
        "app.exchanges.models",
        "app.exchanges.protocols",
        "app.exchanges.errors",
    )
    assert all(not n.startswith("app.") or n.startswith(allowed) for n in imports)
    assert not {n.split(".")[0] for n in imports} & {"asyncio", "time", "httpx", "logging"}


def test_read_client_satisfies_the_protocol() -> None:
    client: TradingClient = ReadClient()

    assert callable(client.get_order)
