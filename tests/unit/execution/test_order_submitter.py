"""OrderSubmitter: write-ahead SUBMITTING, one network attempt, outcome recording."""

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
from app.domain.order_state import transition
from app.domain.orders import Order, OrderUpdate
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeAuthenticationError,
    ExchangeDuplicateOrderError,
    ExchangeError,
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
    ExchangeResponseError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.protocols import TradingClient
from app.execution import requests as requests_module
from app.execution import submitter as submitter_module
from app.execution.account_state import (
    AccountStateError,
    InMemoryAccountState,
    OrderAckMismatchError,
    SubmissionOutcomeConflictError,
)
from app.execution.models import SubmissionOutcome
from app.execution.requests import order_request_from_order
from app.execution.submitter import OrderAlreadySubmittedError, OrderSubmitter
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


class SteppingClock:
    """Each read returns the next time (T1, T2, ...), so phases are distinguishable."""

    def __init__(self, *times: datetime) -> None:
        self.times = list(times) or [T1, T2, T3]
        self.calls = 0

    def now(self) -> datetime:
        value = self.times[min(self.calls, len(self.times) - 1)]
        self.calls += 1
        return value


class FailingAfterClock:
    """Returns ``first`` once, then raises: a clock failure after the network."""

    def __init__(self, first: datetime = T1) -> None:
        self.first = first
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("clock unavailable")
        return self.first


class FakeClient:
    """TradingClient double: records requests; the answer is scripted."""

    def __init__(
        self,
        *,
        error: BaseException | None = None,
        ack: Any = None,
        during: Callable[[OrderRequest], Awaitable[None]] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.error = error
        self.ack = ack
        self.during = during
        self.gate = gate
        self.requests: list[OrderRequest] = []
        self.called = asyncio.Event()

    async def place_order(self, order: OrderRequest) -> OrderAck:
        self.requests.append(order)
        self.called.set()
        if self.during is not None:
            await self.during(order)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        if self.ack is not None:
            return self.ack  # type: ignore[no-any-return]
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="ex-1", exchange_ts=T1
        )

    async def cancel_order(self, order: OrderRef) -> None:
        raise AssertionError("cancel_order must not be called")

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        raise AssertionError("get_order must not be called")

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise AssertionError("get_open_orders must not be called")


def intent(**overrides: Any) -> PlaceOrderIntent:
    values: dict[str, Any] = {
        "intent_id": "i-1",
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100.5"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "tag": None,
        "created_at": T0,
    }
    return PlaceOrderIntent(**{**values, **overrides})


async def reserved_account(**overrides: Any) -> InMemoryAccountState:
    """Known flat BTCUSDT and one approved Order(NEW) "c-1"."""
    account = InMemoryAccountState()
    source = intent(**overrides)
    async with account.account_lock() as locked:
        locked.set_position_qty("BTCUSDT", D("0"))
        locked.register_approved(
            intent=source,
            decision=RiskDecision(
                intent_id=source.intent_id,
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
    return account


def submitter(
    account: InMemoryAccountState, client: FakeClient, clock: Any = None
) -> OrderSubmitter:
    return OrderSubmitter(
        account_state=account, client=client, clock=SteppingClock() if clock is None else clock
    )


async def current(account: InMemoryAccountState) -> Order:
    order = await account.order("c-1")
    assert order is not None
    return order


def exchange_fill(exec_id: str, qty: str, *, exchange_order_id: str = "ex-1") -> Fill:
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


async def exposure_count(account: InMemoryAccountState) -> int:
    """Active orders as the next Risk snapshot sees them."""
    async with account.account_lock() as locked:
        snapshot = build_risk_snapshot(
            snapshot_id=f"acct:{locked.revision}",
            symbol="BTCUSDT",
            trading_state=TradingState.RUNNING,
            position_qty=locked.position_qty("BTCUSDT"),
            orders=locked.active_orders("BTCUSDT"),
            account_open_order_count=locked.account_active_order_count(),
        )
    assert snapshot.open_orders is not None
    return len(snapshot.open_orders)


# --- mapping --------------------------------------------------------------------------------


def make_order(**overrides: Any) -> Order:
    values: dict[str, Any] = {
        "client_order_id": "c-1",
        "exchange_order_id": None,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100.5"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "status": S.NEW,
        "filled_qty": D("0"),
        "avg_fill_price": None,
        "created_at": T0,
        "updated_at": T0,
        "last_exchange_update_ts": None,
        "version": 0,
    }
    return Order(**{**values, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"order_type": OrderType.MARKET, "price": None, "time_in_force": TimeInForce.IOC},
        {"side": Side.SELL, "reduce_only": True, "time_in_force": TimeInForce.FOK},
        {"time_in_force": TimeInForce.POST_ONLY},
        {"price": D("0.000000000000000000000000000000001"), "qty": D("123456789.123456789")},
    ],
)
def test_order_request_copies_every_field(overrides: dict[str, Any]) -> None:
    order = make_order(**overrides)
    before = make_order(**overrides)

    request = order_request_from_order(order)

    assert request == OrderRequest(
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        price=order.price,
        qty=order.qty,
        time_in_force=order.time_in_force,
        reduce_only=order.reduce_only,
    )
    assert order == before
    if order.price is not None:
        assert str(request.price) == str(order.price)
    assert str(request.qty) == str(order.qty)


def test_order_request_is_exact_under_a_low_precision_context() -> None:
    order = make_order(price=D("100.123456789"), qty=D("3.333333333"))
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        request = order_request_from_order(order)

    assert (request.price, request.qty) == (D("100.123456789"), D("3.333333333"))


def test_order_request_rejects_a_non_order() -> None:
    with pytest.raises(DomainValidationError, match="Order"):
        order_request_from_order("c-1")  # type: ignore[arg-type]


# --- happy path -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_marks_submitting_sends_once_and_records_the_ack() -> None:
    account = await reserved_account()
    client = FakeClient()
    base = await account.revision()

    order = await submitter(account, client).submit(client_order_id="c-1")

    assert client.requests == [order_request_from_order(await_order := order)]
    assert (await_order.status, await_order.exchange_order_id) == (S.SUBMITTING, "ex-1")
    assert order.updated_at == T2  # ack recorded at the second clock read
    assert order.version == 2  # NEW -> SUBMITTING, then ack metadata
    assert await account.revision() == base + 2
    assert await exposure_count(account) == 1


@pytest.mark.asyncio
async def test_submitting_is_written_before_the_network_call_and_the_lock_is_free() -> None:
    account = await reserved_account()
    base = await account.revision()
    seen: dict[str, Any] = {}

    async def during(request: OrderRequest) -> None:
        # Another task takes the account lock while the request is in flight.
        async def observe() -> None:
            async with account.account_lock() as locked:
                seen["order"] = locked.order("c-1")
                seen["revision"] = locked.revision

        await asyncio.wait_for(asyncio.create_task(observe()), timeout=1)

    await submitter(account, FakeClient(during=during)).submit(client_order_id="c-1")

    assert seen["order"].status is S.SUBMITTING
    assert seen["order"].updated_at == T1
    assert seen["revision"] == base + 1


@pytest.mark.asyncio
async def test_ack_with_a_known_id_changes_nothing_more() -> None:
    account = await reserved_account()

    async def fill_first(request: OrderRequest) -> None:
        async with account.account_lock() as locked:
            locked.apply_fill(exchange_fill("e-1", "4"), at=T1)

    base = await account.revision()
    order = await submitter(account, FakeClient(during=fill_first)).submit(client_order_id="c-1")

    # SUBMITTING, fill (sets ex-1), ack with the same id: no extra revision.
    assert (order.status, order.exchange_order_id) == (S.PARTIALLY_FILLED, "ex-1")
    assert await account.revision() == base + 2


# --- errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "active"),
    [
        (ExchangeNotSentError("down"), S.FAILED, 0),
        (ExchangeRequestValidationError("bad"), S.FAILED, 0),
        (ExchangeRejectedError("no"), S.REJECTED, 0),
        (ExchangeAuthenticationError("auth"), S.REJECTED, 0),
        (ExchangeAmbiguousResultError("timeout"), S.UNKNOWN, 1),
        (ExchangeDuplicateOrderError("dup"), S.UNKNOWN, 1),
        (ExchangeResponseError("contract violation"), S.UNKNOWN, 1),
        (ExchangeError("unclassified"), S.UNKNOWN, 1),
        (RuntimeError("unexpected"), S.UNKNOWN, 1),
    ],
)
@pytest.mark.asyncio
async def test_transport_outcome_is_recorded_and_re_raised(
    error: BaseException, status: OrderStatus, active: int
) -> None:
    account = await reserved_account()
    client = FakeClient(error=error)
    base = await account.revision()

    with pytest.raises(type(error)) as caught:
        await submitter(account, client).submit(client_order_id="c-1")

    assert caught.value is error
    order = await current(account)
    assert order.status is status
    assert order.updated_at == T2
    assert len(client.requests) == 1  # no retry
    assert await account.revision() == base + 2  # SUBMITTING, outcome
    assert await exposure_count(account) == active


@pytest.mark.asyncio
async def test_cancellation_during_the_request_leaves_the_order_unknown() -> None:
    account = await reserved_account()
    gate = asyncio.Event()
    client = FakeClient(gate=gate)
    task = asyncio.create_task(submitter(account, client).submit(client_order_id="c-1"))
    await client.called.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await current(account)).status is S.UNKNOWN
    assert await exposure_count(account) == 1


@pytest.mark.asyncio
async def test_no_retry_after_an_error() -> None:
    account = await reserved_account()
    client = FakeClient(error=ExchangeNotSentError("down"))
    sender = submitter(account, client)

    with pytest.raises(ExchangeNotSentError):
        await sender.submit(client_order_id="c-1")
    with pytest.raises(OrderAlreadySubmittedError, match="failed"):
        await sender.submit(client_order_id="c-1")

    assert len(client.requests) == 1


# --- replay ---------------------------------------------------------------------------------


async def account_in(status: OrderStatus) -> InMemoryAccountState:
    """White-box: the reserved order moved through domain transitions to ``status``."""
    account = await reserved_account()
    order = await current(account)
    paths: dict[OrderStatus, list[tuple[OrderStatus, str | None]]] = {
        S.SUBMITTING: [(S.SUBMITTING, None)],
        S.UNKNOWN: [(S.SUBMITTING, None), (S.UNKNOWN, None)],
        S.OPEN: [(S.SUBMITTING, None), (S.OPEN, None)],
        S.PARTIALLY_FILLED: [(S.SUBMITTING, None), (S.PARTIALLY_FILLED, "4")],
        S.FILLED: [(S.SUBMITTING, None), (S.FILLED, "10")],
        S.FAILED: [(S.SUBMITTING, None), (S.FAILED, None)],
        S.REJECTED: [(S.SUBMITTING, None), (S.REJECTED, None)],
        S.CANCELING: [(S.SUBMITTING, None), (S.OPEN, None), (S.CANCELING, None)],
        S.CANCELED: [(S.SUBMITTING, None), (S.OPEN, None), (S.CANCELED, None)],
        S.EXPIRED: [(S.SUBMITTING, None), (S.OPEN, None), (S.EXPIRED, None)],
    }
    for step, fill in paths[status]:
        if fill is None:
            order = transition(order, step, at=T1)
        else:
            order = transition(order, step, at=T1, filled_qty=D(fill), avg_fill_price=D("100"))
    account._state.orders["c-1"] = order
    return account


@pytest.mark.parametrize(
    "status",
    [
        S.SUBMITTING,
        S.UNKNOWN,
        S.OPEN,
        S.PARTIALLY_FILLED,
        S.FILLED,
        S.FAILED,
        S.REJECTED,
        S.CANCELING,
        S.CANCELED,
        S.EXPIRED,
    ],
)
@pytest.mark.asyncio
async def test_non_new_order_is_never_sent_again(status: OrderStatus) -> None:
    account = await account_in(status)
    client = FakeClient()
    clock = SteppingClock()
    before = (await current(account), await account.revision())

    with pytest.raises(OrderAlreadySubmittedError, match=status.value):
        await submitter(account, client, clock).submit(client_order_id="c-1")

    assert client.requests == []
    assert clock.calls == 0
    assert (await current(account), await account.revision()) == before


@pytest.mark.asyncio
async def test_unknown_order_is_an_execution_error() -> None:
    account = await reserved_account()
    client = FakeClient()

    with pytest.raises(AccountStateError, match="c-9"):
        await submitter(account, client).submit(client_order_id="c-9")

    assert client.requests == []


@pytest.mark.asyncio
async def test_concurrent_submits_of_one_order_send_one_request() -> None:
    account = await reserved_account()
    gate = asyncio.Event()
    client = FakeClient(gate=gate)
    sender = submitter(account, client)

    first = asyncio.create_task(sender.submit(client_order_id="c-1"))
    second = asyncio.create_task(sender.submit(client_order_id="c-1"))
    await client.called.wait()
    for _ in range(5):
        await asyncio.sleep(0)  # the loser has run phase A by now
    gate.set()
    results = await asyncio.wait_for(
        asyncio.gather(first, second, return_exceptions=True), timeout=5
    )

    errors = [r for r in results if isinstance(r, BaseException)]
    orders = [r for r in results if isinstance(r, Order)]
    assert len(client.requests) == 1
    assert len(orders) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], OrderAlreadySubmittedError)


# --- phase A failures -----------------------------------------------------------------------


class BrokenClock:
    def now(self) -> datetime:
        raise RuntimeError("no time")


@pytest.mark.parametrize(
    ("clock", "error"),
    [
        (BrokenClock(), RuntimeError),
        (SteppingClock(T0 - timedelta(seconds=1)), Exception),  # before the reservation
        (SteppingClock(datetime(2026, 1, 15, 12, 0)), DomainValidationError),  # noqa: DTZ001
    ],
)
@pytest.mark.asyncio
async def test_failure_before_the_marker_sends_nothing(clock: Any, error: type[Exception]) -> None:
    account = await reserved_account()
    client = FakeClient()
    before = (await current(account), await account.revision())

    with pytest.raises(error):
        await submitter(account, client, clock).submit(client_order_id="c-1")

    assert client.requests == []
    assert (await current(account), await account.revision()) == before


@pytest.mark.asyncio
async def test_unrepresentable_request_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(order: Order) -> OrderRequest:
        raise DomainValidationError("cannot be expressed")

    monkeypatch.setattr(submitter_module, "order_request_from_order", broken)
    account = await reserved_account()
    client = FakeClient()
    before = (await current(account), await account.revision())

    with pytest.raises(DomainValidationError, match="cannot be expressed"):
        await submitter(account, client).submit(client_order_id="c-1")

    assert client.requests == []
    assert (await current(account), await account.revision()) == before


# --- clock after the network ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (None, S.SUBMITTING),
        (ExchangeNotSentError("down"), S.FAILED),
        (ExchangeRejectedError("no"), S.REJECTED),
        (ExchangeAmbiguousResultError("timeout"), S.UNKNOWN),
    ],
)
@pytest.mark.asyncio
async def test_clock_failure_after_the_network_still_records_the_outcome(
    error: BaseException | None, status: OrderStatus
) -> None:
    account = await reserved_account()
    client = FakeClient(error=error)
    sender = submitter(account, client, FailingAfterClock())

    if error is None:
        await sender.submit(client_order_id="c-1")
    else:
        with pytest.raises(type(error)):
            await sender.submit(client_order_id="c-1")

    order = await current(account)
    assert order.status is status
    assert order.updated_at == T1  # fallback: the order's own time, never backwards
    if error is None:
        assert order.exchange_order_id == "ex-1"


@pytest.mark.parametrize(
    "late",
    [T0, datetime(2026, 1, 15, 12, 0), "now"],  # noqa: DTZ001
)
@pytest.mark.asyncio
async def test_invalid_outcome_time_falls_back_to_the_order_time(late: Any) -> None:
    account = await reserved_account()
    client = FakeClient(error=ExchangeAmbiguousResultError("timeout"))

    with pytest.raises(ExchangeAmbiguousResultError):
        await submitter(account, client, SteppingClock(T1, late)).submit(client_order_id="c-1")

    order = await current(account)
    assert (order.status, order.updated_at) == (S.UNKNOWN, T1)


# --- ack ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ack",
    [
        OrderAck(client_order_id="c-2", exchange_order_id="ex-1", exchange_ts=T1),
        "not an ack",
        None,
    ],
)
@pytest.mark.asyncio
async def test_foreign_or_invalid_ack_makes_the_order_unknown(ack: Any) -> None:
    account = await reserved_account()
    client = FakeClient(ack=ack if ack is not None else object())
    base = await account.revision()

    with pytest.raises(OrderAckMismatchError):
        await submitter(account, client).submit(client_order_id="c-1")

    order = await current(account)
    assert (order.status, order.exchange_order_id) == (S.UNKNOWN, None)
    assert await account.revision() == base + 2
    assert await exposure_count(account) == 1  # never released


@pytest.mark.asyncio
async def test_ack_with_another_exchange_id_after_a_fill_keeps_the_fill() -> None:
    account = await reserved_account()

    async def fill_first(request: OrderRequest) -> None:
        async with account.account_lock() as locked:
            locked.apply_fill(exchange_fill("e-1", "4", exchange_order_id="ex-7"), at=T1)

    client = FakeClient(during=fill_first)  # acks ex-1

    with pytest.raises(OrderAckMismatchError, match="ex-7"):
        await submitter(account, client).submit(client_order_id="c-1")

    order = await current(account)
    assert (order.status, order.exchange_order_id, order.filled_qty) == (
        S.PARTIALLY_FILLED,
        "ex-7",
        D("4"),
    )


@pytest.mark.parametrize(("fill_qty", "status"), [("4", S.PARTIALLY_FILLED), ("10", S.FILLED)])
@pytest.mark.asyncio
async def test_ack_after_a_fill_does_not_roll_the_order_back(
    fill_qty: str, status: OrderStatus
) -> None:
    account = await reserved_account()

    async def fill_without_id(request: OrderRequest) -> None:
        # White-box: a fill-advanced order whose exchange id is not known yet.
        async with account.account_lock() as locked:
            order = locked.order("c-1")
            assert order is not None
            account._state.orders["c-1"] = transition(
                order, status, at=T1, filled_qty=D(fill_qty), avg_fill_price=D("100")
            )

    order = await submitter(account, FakeClient(during=fill_without_id)).submit(
        client_order_id="c-1"
    )

    assert (order.status, order.filled_qty, order.exchange_order_id) == (
        status,
        D(fill_qty),
        "ex-1",
    )


# --- races with fills -----------------------------------------------------------------------


def fill_during(account: InMemoryAccountState, qty: str) -> Callable[[OrderRequest], Any]:
    async def during(request: OrderRequest) -> None:
        async with account.account_lock() as locked:
            locked.apply_fill(exchange_fill("e-1", qty), at=T1)

    return during


@pytest.mark.parametrize(("qty", "status"), [("4", S.PARTIALLY_FILLED), ("10", S.FILLED)])
@pytest.mark.asyncio
async def test_fill_while_the_request_is_in_flight(qty: str, status: OrderStatus) -> None:
    account = await reserved_account()

    order = await submitter(account, FakeClient(during=fill_during(account, qty))).submit(
        client_order_id="c-1"
    )

    assert (order.status, order.filled_qty) == (status, D(qty))
    assert await account.position_qty("BTCUSDT") == D(qty)


@pytest.mark.parametrize(
    "error",
    [ExchangeAmbiguousResultError("timeout"), ExchangeDuplicateOrderError("dup")],
)
@pytest.mark.parametrize(("qty", "status"), [("4", S.PARTIALLY_FILLED), ("10", S.FILLED)])
@pytest.mark.asyncio
async def test_ambiguous_result_after_a_fill_keeps_the_confirmed_state(
    error: BaseException, qty: str, status: OrderStatus
) -> None:
    account = await reserved_account()
    client = FakeClient(error=error, during=fill_during(account, qty))
    base = await account.revision()

    with pytest.raises(type(error)) as caught:
        await submitter(account, client).submit(client_order_id="c-1")

    assert await account.revision() == base + 2  # SUBMITTING, fill; no UNKNOWN
    assert caught.value is error
    order = await current(account)
    assert (order.status, order.filled_qty) == (status, D(qty))
    assert await account.position_qty("BTCUSDT") == D(qty)


@pytest.mark.parametrize("error", [ExchangeNotSentError("down"), ExchangeRejectedError("no")])
@pytest.mark.parametrize(("qty", "status"), [("4", S.PARTIALLY_FILLED), ("10", S.FILLED)])
@pytest.mark.asyncio
async def test_definite_failure_after_a_fill_is_a_conflict_and_keeps_the_state(
    error: BaseException, qty: str, status: OrderStatus
) -> None:
    account = await reserved_account()
    client = FakeClient(error=error, during=fill_during(account, qty))

    with pytest.raises(SubmissionOutcomeConflictError, match=status.value) as caught:
        await submitter(account, client).submit(client_order_id="c-1")

    assert caught.value.__context__ is error  # the transport error is kept
    order = await current(account)
    assert (order.status, order.filled_qty) == (status, D(qty))
    assert await account.position_qty("BTCUSDT") == D(qty)


# --- account-state operations ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_ack_is_rejected_for_orders_that_were_never_accepted() -> None:
    for status in (S.FAILED, S.REJECTED):
        account = await account_in(status)
        before = (await current(account), await account.revision())
        async with account.account_lock() as locked:
            with pytest.raises(OrderAckMismatchError, match=status.value):
                locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)
        assert (await current(account), await account.revision()) == before

    account = await reserved_account()  # NEW: never sent
    async with account.account_lock() as locked:
        with pytest.raises(OrderAckMismatchError, match="new"):
            locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)


@pytest.mark.parametrize("status", [S.UNKNOWN, S.OPEN, S.CANCELING, S.CANCELED, S.EXPIRED])
@pytest.mark.asyncio
async def test_record_ack_enriches_any_sent_state(status: OrderStatus) -> None:
    account = await account_in(status)
    base = await account.revision()

    async with account.account_lock() as locked:
        order = locked.record_ack("c-1", exchange_order_id="ex-1", at=T2)
        again = locked.record_ack("c-1", exchange_order_id="ex-1", at=T3)

    assert (order.status, order.exchange_order_id) == (status, "ex-1")
    assert again is order  # same id again: no change
    assert await account.revision() == base + 1


@pytest.mark.asyncio
async def test_record_outcome_on_a_new_order_is_an_error() -> None:
    account = await reserved_account()
    before = (await current(account), await account.revision())

    async with account.account_lock() as locked:
        with pytest.raises(AccountStateError, match="never marked SUBMITTING"):
            locked.record_submission_outcome("c-1", SubmissionOutcome.AMBIGUOUS, at=T2)

    assert (await current(account), await account.revision()) == before


# --- Decimal / dependencies -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_submission_is_exact_under_a_low_precision_context() -> None:
    account = await reserved_account(price=D("100.123456789"), qty=D("3.333333333"))
    client = FakeClient()
    getcontext().clear_flags()
    before = (getcontext().prec, getcontext().rounding, dict(getcontext().flags))

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        await submitter(account, client).submit(client_order_id="c-1")

    assert (client.requests[0].price, client.requests[0].qty) == (
        D("100.123456789"),
        D("3.333333333"),
    )
    assert (getcontext().prec, getcontext().rounding, dict(getcontext().flags)) == before


def test_fake_client_satisfies_the_trading_client_protocol() -> None:
    client: TradingClient = FakeClient()

    assert callable(client.place_order)


@pytest.mark.parametrize("module", [submitter_module, requests_module])
def test_execution_modules_use_only_exchange_abstractions(module: Any) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    allowed = (
        "app.domain",
        "app.execution",
        "app.exchanges.models",
        "app.exchanges.errors",
        "app.exchanges.protocols",
    )
    for name in imports:
        assert not name.startswith("app.") or name.startswith(allowed), name
    banned = {"httpx", "asyncio", "time", "uuid", "random", "logging", "sqlite3"}
    assert not {name.split(".")[0] for name in imports} & banned
    for word in ("bybit", "simulated", "sleep(", "datetime.now"):
        assert word not in source.lower(), word
