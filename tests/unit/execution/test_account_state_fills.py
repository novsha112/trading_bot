"""Account state: positions and atomic fill application (order + position + revision)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError, InvalidOrderTransition
from app.domain.fill_math import accumulate_execution
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.order_state import TERMINAL_STATUSES, transition
from app.domain.orders import Order
from app.execution.account_state import (
    FILLABLE_STATUSES,
    AccountStateError,
    FillApplicationError,
    FillConflictError,
    InMemoryAccountState,
)
from app.persistence.memory import InMemoryAccountStateStore
from app.risk.models import ExposureChange, RiskDecision

D = Decimal
S = OrderStatus
BUY, SELL = Side.BUY, Side.SELL
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
FLAT = D("0")
EXPOSURE = ExposureChange(
    reducing_qty=D("0"), increasing_qty=D("1"), worst_long_qty=D("1"), worst_short_qty=D("0")
)


def new_account(account_scope_id: str = "acct-1") -> InMemoryAccountState:
    """An account state on a fresh in-memory reference store."""
    return InMemoryAccountState(
        account_scope_id=account_scope_id, store=InMemoryAccountStateStore()
    )


def intent(intent_id: str, **overrides: Any) -> PlaceOrderIntent:
    values: dict[str, Any] = {
        "intent_id": intent_id,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "tag": None,
        "created_at": T0,
    }
    return PlaceOrderIntent(**{**values, **overrides})


def fill(
    exec_id: str = "e-1",
    *,
    client_order_id: str | None = "c-1",
    qty: str = "4",
    price: str = "100",
    side: Side = BUY,
    symbol: str = "BTCUSDT",
    exchange_order_id: str = "ex-1",
    **overrides: Any,
) -> Fill:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": exchange_order_id,
        "client_order_id": client_order_id,
        "symbol": symbol,
        "side": side,
        "price": D(price),
        "qty": D(qty),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "exchange_ts": T1,
    }
    return Fill(**{**values, **overrides})


async def account_with(
    *,
    position: Decimal | None = FLAT,
    submitted: bool = True,
    client_order_id: str = "c-1",
    **intent_overrides: Any,
) -> InMemoryAccountState:
    """One reserved order (SUBMITTING by default) and the BTCUSDT position."""
    account = new_account()
    await add_order(account, client_order_id, submitted=submitted, **intent_overrides)
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", position)
    return account


async def add_order(
    account: InMemoryAccountState, client_order_id: str, *, submitted: bool = True, **overrides: Any
) -> None:
    source = intent(f"i-{client_order_id}", **overrides)
    async with account.account_lock() as locked:
        await locked.register_approved(
            intent=source,
            decision=RiskDecision(
                intent_id=source.intent_id,
                snapshot_id="s",
                policy_id="p",
                approved=True,
                reasons=(),
                exposure=EXPOSURE,
            ),
            client_order_id=client_order_id,
            expected_revision=locked.revision,
            at=T0,
        )
        if submitted:
            await locked.mark_submitting(client_order_id, at=T0)


async def apply(account: InMemoryAccountState, item: Fill, *, at: datetime = T2) -> Order:
    async with account.account_lock() as locked:
        return await locked.apply_fill(item, at=at)


def whole_state(account: InMemoryAccountState) -> tuple[Any, ...]:
    """Every part of the local state (white-box, for atomicity checks)."""
    state = account._state
    return (
        state.revision,
        dict(state.orders),
        dict(state.notionals),
        dict(state.positions),
        dict(state.fills),
        dict(state.placements),
    )


def inject(account: InMemoryAccountState, order: Order) -> None:
    """White-box: install a domain-valid order state that has no public path yet
    (OPEN / CANCELING / UNKNOWN / terminal need exchange updates, not in scope)."""
    account._state.orders[order.client_order_id] = order


# --- positions ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_is_unknown_until_set() -> None:
    account = new_account()

    assert await account.position_qty("BTCUSDT") is None
    assert await account.revision() == 0


@pytest.mark.parametrize("qty", ["0", "2.5", "-2.5"])
@pytest.mark.asyncio
async def test_known_flat_long_and_short_positions(qty: str) -> None:
    account = new_account()

    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", D(qty))
        assert locked.revision == 1

    assert await account.position_qty("BTCUSDT") == D(qty)
    assert await account.position_qty("ETHUSDT") is None  # other symbols stay unknown


@pytest.mark.asyncio
async def test_position_revision_changes_only_with_the_value() -> None:
    account = new_account()
    revisions: list[int] = []

    async with account.account_lock() as locked:
        for value in (None, D("0"), D("0.0"), D("1"), D("1"), None, None):
            await locked.set_position_qty("BTCUSDT", value)
            revisions.append(locked.revision)

    assert revisions == [0, 1, 1, 2, 2, 3, 3]
    assert await account.position_qty("BTCUSDT") is None


@pytest.mark.parametrize(
    ("symbol", "qty", "match"),
    [("", D("0"), "symbol"), ("BTCUSDT", 0, "position"), ("BTCUSDT", D("Infinity"), "position")],
)
@pytest.mark.asyncio
async def test_invalid_position_is_rejected(symbol: str, qty: object, match: str) -> None:
    account = new_account()

    async with account.account_lock() as locked:
        with pytest.raises(DomainValidationError, match=match):
            await locked.set_position_qty(symbol, qty)  # type: ignore[arg-type]
        assert locked.revision == 0


# --- mark_submitting ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_submitting_is_a_write_ahead_transition() -> None:
    account = await account_with(submitted=False)
    before = await account.revision()

    async with account.account_lock() as locked:
        order = await locked.mark_submitting("c-1", at=T1)

    assert (order.status, order.version, order.updated_at) == (S.SUBMITTING, 1, T1)
    assert await account.order("c-1") == order
    assert await account.revision() == before + 1
    assert len(await account.active_orders("BTCUSDT")) == 1  # still active exposure


@pytest.mark.asyncio
async def test_mark_submitting_rejects_unknown_and_repeated_orders() -> None:
    account = await account_with()
    before = whole_state(account)

    async with account.account_lock() as locked:
        with pytest.raises(AccountStateError, match="c-9"):
            await locked.mark_submitting("c-9", at=T1)
        with pytest.raises(InvalidOrderTransition, match="submitting -> submitting"):
            await locked.mark_submitting("c-1", at=T1)

    assert whole_state(account) == before


# --- fill application -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_updates_order_position_and_revision_together() -> None:
    account = await account_with()
    before = await account.revision()

    order = await apply(account, fill(qty="4", price="100.5"))

    assert (order.status, order.filled_qty, order.avg_fill_price) == (
        S.PARTIALLY_FILLED,
        D("4"),
        D("100.5"),
    )
    assert (order.exchange_order_id, order.last_exchange_update_ts, order.updated_at) == (
        "ex-1",
        T1,
        T2,
    )
    assert order.version == 2  # NEW -> SUBMITTING -> PARTIALLY_FILLED
    assert await account.order("c-1") == order
    assert await account.position_qty("BTCUSDT") == D("4")
    assert await account.fill("e-1") == fill(qty="4", price="100.5")
    assert await account.revision() == before + 1
    active = await account.active_orders("BTCUSDT")
    assert [o.qty - o.filled_qty for o in active] == [D("6")]


@pytest.mark.asyncio
async def test_completing_fill_makes_the_order_filled_and_inactive() -> None:
    account = await account_with()

    await apply(account, fill("e-1", qty="4"))
    order = await apply(account, fill("e-2", qty="6", price="101"))

    assert (order.status, order.filled_qty) == (S.FILLED, D("10"))
    assert order.avg_fill_price == D("100.6")
    assert await account.position_qty("BTCUSDT") == D("10")
    assert await account.active_orders("BTCUSDT") == ()
    assert await account.account_active_order_count() == 0


@pytest.mark.asyncio
async def test_average_price_uses_the_shared_exact_rule() -> None:
    account = await account_with(qty=D("3"))

    await apply(account, fill("e-1", qty="1", price="100"))
    await apply(account, fill("e-2", qty="1", price="101"))
    order = await apply(account, fill("e-3", qty="1", price="103"))

    expected = accumulate_execution(
        filled_qty=D("2"), filled_notional=D("201"), price=D("103"), qty=D("1")
    )
    assert order.avg_fill_price == expected.avg_fill_price
    assert str(order.avg_fill_price) == "101.3333333333333333333333333333333333333"


@pytest.mark.parametrize(
    ("position", "side", "qty", "expected"),
    [
        ("0", BUY, "4", "4"),
        ("0", SELL, "4", "-4"),
        ("3", BUY, "4", "7"),
        ("5", SELL, "4", "1"),
        ("4", SELL, "4", "0"),
        ("3", SELL, "4", "-1"),
        ("-3", SELL, "4", "-7"),
        ("-5", BUY, "4", "-1"),
        ("-4", BUY, "4", "0"),
        ("-3", BUY, "4", "1"),
    ],
)
@pytest.mark.asyncio
async def test_signed_position_change(position: str, side: Side, qty: str, expected: str) -> None:
    account = await account_with(position=D(position), side=side)

    await apply(account, fill(side=side, qty=qty))

    assert await account.position_qty("BTCUSDT") == D(expected)


@pytest.mark.asyncio
async def test_fill_changes_only_its_own_symbol() -> None:
    account = await account_with()
    async with account.account_lock() as locked:
        await locked.set_position_qty("ETHUSDT", D("-2"))

    await apply(account, fill(qty="1"))

    assert await account.position_qty("ETHUSDT") == D("-2")


@pytest.mark.parametrize(
    ("position", "side", "qty", "expected"),
    [("3", SELL, "2", "1"), ("3", SELL, "3", "0"), ("-3", BUY, "3", "0")],
)
@pytest.mark.asyncio
async def test_valid_reduce_only_fill(position: str, side: Side, qty: str, expected: str) -> None:
    account = await account_with(position=D(position), side=side, reduce_only=True)

    order = await apply(account, fill(side=side, qty=qty))

    assert order.filled_qty == D(qty)
    assert await account.position_qty("BTCUSDT") == D(expected)


@pytest.mark.asyncio
async def test_fill_with_unknown_position_updates_the_order_only() -> None:
    account = await account_with(position=None, reduce_only=True, side=SELL)
    before = await account.revision()

    order = await apply(account, fill(side=SELL, qty="2"))

    assert order.filled_qty == D("2")
    assert await account.position_qty("BTCUSDT") is None  # unknown + delta = unknown
    assert await account.revision() == before + 1


# --- replay / conflict ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_fill_replay_changes_nothing() -> None:
    account = await account_with()
    first = await apply(account, fill(qty="4"))
    before = whole_state(account)

    again = await apply(account, fill(qty="4"), at=T2 + timedelta(seconds=5))

    assert again == first
    assert whole_state(account) == before


@pytest.mark.parametrize(
    "changes",
    [{"qty": "5"}, {"price": "101"}, {"exchange_ts": T2}, {"client_order_id": "c-2"}],
)
@pytest.mark.asyncio
async def test_same_exec_id_with_different_data_is_a_conflict(changes: dict[str, Any]) -> None:
    account = await account_with()
    await add_order(account, "c-2")
    await apply(account, fill(qty="4"))
    before = whole_state(account)

    with pytest.raises(FillConflictError, match="e-1"):
        await apply(account, fill(**changes))

    assert whole_state(account) == before


# --- statuses -------------------------------------------------------------------------------


def test_fillable_statuses_match_the_state_machine() -> None:
    assert {S.SUBMITTING, S.OPEN, S.PARTIALLY_FILLED, S.CANCELING, S.UNKNOWN} == FILLABLE_STATUSES
    assert not FILLABLE_STATUSES & TERMINAL_STATUSES
    assert S.NEW not in FILLABLE_STATUSES


async def account_in_status(status: OrderStatus) -> InMemoryAccountState:
    """A SUBMITTING order moved (through domain transitions) to ``status``."""
    account = await account_with()
    order = await account.order("c-1")
    assert order is not None
    paths = {
        S.SUBMITTING: [],
        S.OPEN: [S.OPEN],
        S.CANCELING: [S.OPEN, S.CANCELING],
        S.UNKNOWN: [S.UNKNOWN],
        S.FAILED: [S.FAILED],
        S.REJECTED: [S.REJECTED],
        S.CANCELED: [S.OPEN, S.CANCELED],
        S.EXPIRED: [S.OPEN, S.EXPIRED],
    }
    for step in paths[status]:
        order = transition(order, step, at=T1)
    inject(account, order)
    return account


@pytest.mark.parametrize(
    ("status", "partial", "full"),
    [
        (S.SUBMITTING, S.PARTIALLY_FILLED, S.FILLED),
        (S.OPEN, S.PARTIALLY_FILLED, S.FILLED),
        (S.CANCELING, S.CANCELING, S.FILLED),  # a fill racing a cancel request
        (S.UNKNOWN, S.PARTIALLY_FILLED, S.FILLED),
    ],
)
@pytest.mark.asyncio
async def test_fill_status_matrix(
    status: OrderStatus, partial: OrderStatus, full: OrderStatus
) -> None:
    partial_account = await account_in_status(status)
    full_account = await account_in_status(status)

    assert (await apply(partial_account, fill(qty="4"))).status is partial
    assert (await apply(full_account, fill(qty="10"))).status is full


@pytest.mark.asyncio
async def test_partially_filled_order_accepts_further_fills() -> None:
    account = await account_with()
    await apply(account, fill("e-1", qty="4"))

    order = await apply(account, fill("e-2", qty="1"))

    assert (order.status, order.filled_qty) == (S.PARTIALLY_FILLED, D("5"))


@pytest.mark.asyncio
async def test_new_order_cannot_receive_a_fill() -> None:
    account = await account_with(submitted=False)
    before = whole_state(account)

    with pytest.raises(FillApplicationError, match="is new"):
        await apply(account, fill())

    assert whole_state(account) == before


@pytest.mark.parametrize("status", [S.FAILED, S.REJECTED, S.CANCELED, S.EXPIRED])
@pytest.mark.asyncio
async def test_terminal_order_cannot_receive_a_fill(status: OrderStatus) -> None:
    account = await account_in_status(status)
    before = whole_state(account)

    with pytest.raises(FillApplicationError, match=status.value):
        await apply(account, fill())

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_filled_order_cannot_receive_another_fill() -> None:
    account = await account_with()
    await apply(account, fill("e-1", qty="10"))
    before = whole_state(account)

    with pytest.raises(FillApplicationError, match="filled"):
        await apply(account, fill("e-2", qty="1"))

    assert whole_state(account) == before


# --- failure atomicity ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "bad_fill", "error", "match"),
    [
        ({}, fill(qty="10.000001"), FillApplicationError, "overfills"),
        (
            {"position": D("3"), "side": SELL, "reduce_only": True},
            fill(side=SELL, qty="4"),
            FillApplicationError,
            "reverse",
        ),
        (
            {"position": D("3"), "side": BUY, "reduce_only": True},
            fill(side=BUY, qty="1"),
            FillApplicationError,
            "same side",
        ),
        (
            {"position": D("0"), "side": SELL, "reduce_only": True},
            fill(side=SELL, qty="1"),
            FillApplicationError,
            "without an open position",
        ),
        ({}, fill(client_order_id=None), FillApplicationError, "foreign"),
        ({}, fill(client_order_id="c-9"), FillApplicationError, "no local order c-9"),
        ({}, fill(symbol="ETHUSDT"), FillApplicationError, "does not match"),
        ({}, fill(side=SELL), FillApplicationError, "does not match"),
        ({"position": D("1E+100")}, fill(qty="1E-100", price="1"), FillApplicationError, "exactly"),
    ],
)
@pytest.mark.asyncio
async def test_failed_fill_changes_nothing(
    setup: dict[str, Any], bad_fill: Fill, error: type[Exception], match: str
) -> None:
    account = await account_with(**setup)
    before = whole_state(account)

    with pytest.raises(error, match=match):
        await apply(account, bad_fill)

    assert whole_state(account) == before
    assert await account.fill(bad_fill.exec_id) is None


@pytest.mark.asyncio
async def test_exchange_order_id_mismatch_changes_nothing() -> None:
    account = await account_with()
    await apply(account, fill("e-1", qty="1", exchange_order_id="ex-1"))
    before = whole_state(account)

    with pytest.raises(FillApplicationError, match="ex-2"):
        await apply(account, fill("e-2", qty="1", exchange_order_id="ex-2"))

    assert whole_state(account) == before


@pytest.mark.asyncio
async def test_fill_time_before_the_order_update_changes_nothing() -> None:
    account = await account_with()
    before = whole_state(account)

    with pytest.raises(InvalidOrderTransition, match="backwards"):
        await apply(account, fill(), at=T0 - timedelta(seconds=1))

    assert whole_state(account) == before


@pytest.mark.parametrize(("item", "at"), [("fill", T2), (None, T2)])
@pytest.mark.asyncio
async def test_invalid_fill_arguments(item: object, at: datetime) -> None:
    account = await account_with()
    before = whole_state(account)

    async with account.account_lock() as locked:
        with pytest.raises(DomainValidationError, match="Fill"):
            await locked.apply_fill(item, at=at)  # type: ignore[arg-type]
        with pytest.raises(DomainValidationError, match="at"):
            await locked.apply_fill(fill(), at=datetime(2026, 1, 15, 12, 0))  # noqa: DTZ001

    assert whole_state(account) == before


# --- Decimal context ------------------------------------------------------------------------


async def awkward_fills() -> tuple[Any, ...]:
    account = await account_with(position=D("-0.067"), qty=D("3.4"))
    await apply(account, fill("e-1", qty="1.111111111", price="100.3"))
    order = await apply(account, fill("e-2", qty="0.222222222", price="99.7"))
    return order, await account.position_qty("BTCUSDT"), account._state.notionals["c-1"]


@pytest.mark.asyncio
async def test_fill_path_is_exact_under_a_low_precision_context() -> None:
    expected = await awkward_fills()
    getcontext().clear_flags()
    context_before = getcontext()
    state_before = (context_before.prec, context_before.rounding, dict(context_before.flags))
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        result = await awkward_fills()

    assert result == expected
    order, position, notional = result
    assert order.filled_qty == D("1.333333333")
    assert position == D("1.266333333")
    assert notional == D("111.4444444333") + D("22.1555555334")
    after = getcontext()
    assert (after.prec, after.rounding, dict(after.flags)) == state_before
