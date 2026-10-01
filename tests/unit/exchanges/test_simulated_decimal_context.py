"""Simulation results must not depend on the process-global decimal context.

Quantities here have more significant digits than the low test precision (2), so
any operation that silently used the global context (e.g. unary minus or abs()
on a Decimal, which round to it) would change them: -Decimal("3.333") is -3.3 at
precision 2.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from typing import Any, TypeVar

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.fills import Fill
from app.domain.instrument import InstrumentSpec
from app.domain.positions import Position
from app.exchanges.models import OrderRef, OrderRequest
from app.exchanges.simulated import SimulatedExchange
from app.exchanges.simulated_accounting import CashState, EquityState, SimulatedCashConfig
from app.exchanges.simulated_fees import LiquidityRole, SimulatedFeePolicy, TradingFeeSchedule
from app.exchanges.simulated_positions import MarkQuote, SimulatedPositionLedger

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
R = TypeVar("R")


@contextmanager
def low_precision() -> Iterator[None]:
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        yield


def in_both_contexts(run: Callable[[], R]) -> R:
    """Run in the default and a low-precision context; results must be equal."""
    baseline = run()
    with low_precision():
        low = run()
    assert low == baseline
    return baseline


async def in_both_contexts_async(run: Callable[[], Awaitable[R]]) -> R:
    baseline = await run()
    with low_precision():
        low = await run()
    assert low == baseline
    return baseline


def test_the_low_context_really_breaks_naive_unary_operations() -> None:
    # The failure mode these tests guard against.
    with low_precision():
        assert -D("3.333") == D("-3.4")  # ROUND_UP away from zero
        assert abs(D("-3.333")) == D("3.4")
        assert D("3.333").copy_negate() == D("-3.333")
        assert D("-3.333").copy_abs() == D("3.333")


# --- position ledger ------------------------------------------------------------------


def fill(side: Side, qty: str, price: str, exec_id: str, symbol: str = "BTCUSDT") -> Fill:
    return Fill(
        exec_id=exec_id,
        exchange_order_id="SIM-1",
        client_order_id="c",
        symbol=symbol,
        side=side,
        price=D(price),
        qty=D(qty),
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=T0,
    )


def ledger_run(*fills: tuple[Side, str, str], mark: str | None = None) -> Position:
    ledger = SimulatedPositionLedger()
    for i, (side, qty, price) in enumerate(fills):
        ledger.apply_fill(fill(side, qty, price, f"e{i}"))
    quote = None if mark is None else MarkQuote(price=D(mark), at=T0)
    p = ledger.get_position("BTCUSDT", mark=quote)
    assert p is not None
    return p


def test_sell_opening_keeps_exact_quantity() -> None:
    p = in_both_contexts(lambda: ledger_run((Side.SELL, "3.333", "100")))

    assert (p.qty, p.entry_price) == (D("-3.333"), D("100"))  # not -3.3 / -3.4


def test_long_valuation() -> None:
    p = in_both_contexts(lambda: ledger_run((Side.BUY, "3.333", "100"), mark="101.37"))

    assert (p.qty, p.entry_price, p.unrealized_pnl) == (D("3.333"), D("100"), D("4.56621"))


def test_short_valuation() -> None:
    p = in_both_contexts(lambda: ledger_run((Side.SELL, "3.333", "100"), mark="98.63"))

    assert (p.qty, p.entry_price, p.unrealized_pnl) == (D("-3.333"), D("100"), D("4.56621"))


@pytest.mark.parametrize(
    ("fills", "expected"),
    [
        (
            ((Side.BUY, "7.777", "100"), (Side.SELL, "3.333", "110")),
            (D("4.444"), D("100"), D("33.33")),
        ),
        (
            ((Side.SELL, "7.777", "100"), (Side.BUY, "3.333", "90")),
            (D("-4.444"), D("100"), D("33.33")),
        ),
    ],
    ids=["long", "short"],
)
def test_partial_close(
    fills: tuple[tuple[Side, str, str], ...], expected: tuple[Decimal, ...]
) -> None:
    p = in_both_contexts(lambda: ledger_run(*fills, mark="105"))

    assert (p.qty, p.entry_price, p.realized_pnl) == expected
    sign = 1 if p.qty > 0 else -1
    assert p.unrealized_pnl == (D("105") - D("100")) * D("4.444") * sign


@pytest.mark.parametrize(
    ("fills", "expected"),
    [
        (
            ((Side.BUY, "3.333", "100"), (Side.SELL, "5.555", "110")),
            (D("-2.222"), D("110"), D("33.33")),
        ),
        (
            ((Side.SELL, "3.333", "100"), (Side.BUY, "5.555", "90")),
            (D("2.222"), D("90"), D("33.33")),
        ),
    ],
    ids=["long-to-short", "short-to-long"],
)
def test_reversal(fills: tuple[tuple[Side, str, str], ...], expected: tuple[Decimal, ...]) -> None:
    p = in_both_contexts(lambda: ledger_run(*fills))

    assert (p.qty, p.entry_price, p.realized_pnl) == expected


def test_weighted_repeating_basis_valuation() -> None:
    p = in_both_contexts(
        lambda: ledger_run((Side.BUY, "3.333", "100"), (Side.BUY, "1.111", "110"), mark="105.5")
    )

    # basis = (333.3 + 122.21) / 4.444; unrealized = 105.5 * 4.444 - 455.51 = 13.332
    assert p.qty == D("4.444")
    assert p.unrealized_pnl == D("13.332")


def test_batch_delta_and_signed_qty() -> None:
    from fractions import Fraction

    def run() -> tuple[tuple[Fraction | None, ...], Decimal]:
        ledger = SimulatedPositionLedger()
        batch = ledger.begin_batch()
        deltas = [
            batch.apply(fill(Side.BUY, "7.777", "100", "a")),
            batch.apply(fill(Side.SELL, "3.333", "110", "b")),
            batch.apply(fill(Side.SELL, "5.555", "95", "c")),
        ]
        return tuple(deltas), batch.signed_qty("BTCUSDT")

    deltas, signed = in_both_contexts(run)

    assert signed == D("-1.111")
    assert deltas == (Fraction(0), Fraction(3333, 100), Fraction(-2222, 100))


# --- simulator --------------------------------------------------------------------------

SPEC = InstrumentSpec(
    symbol="BTCUSDT",
    base_asset="BTC",
    quote_asset="USDT",
    tick_size=D("0.01"),
    qty_step=D("0.001"),
    min_qty=D("0.001"),
    max_qty=D("1000"),
    min_notional=D("0"),
)


def new_exchange(*, cash: bool = False) -> SimulatedExchange:
    kwargs: dict[str, Any] = {}
    if cash:
        kwargs = {
            "fees": SimulatedFeePolicy(
                schedule=TradingFeeSchedule(
                    maker_rate=D("-0.0001"), taker_rate=D("0.00055"), fee_asset="USDT"
                ),
                liquidity_role=LiquidityRole.TAKER,
            ),
            "cash": SimulatedCashConfig(asset="USDT", starting_cash=D("10000")),
        }
    return SimulatedExchange(clock=ManualClock(T0), instruments=(SPEC,), **kwargs)


def order(cid: str, side: Side, qty: str, price: str, *, reduce_only: bool = False) -> OrderRequest:
    return OrderRequest(
        client_order_id=cid,
        symbol="BTCUSDT",
        side=side,
        order_type=OrderType.LIMIT,
        price=D(price),
        qty=D(qty),
        time_in_force=TimeInForce.GTC,
        reduce_only=reduce_only,
    )


async def trade(exchange: SimulatedExchange, req: OrderRequest, price: str) -> list[Fill]:
    await exchange.place_order(req)
    return list(
        await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D(price))
    )


async def status(exchange: SimulatedExchange, cid: str) -> tuple[OrderStatus, Decimal]:
    update = await exchange.get_order(OrderRef(symbol="BTCUSDT", client_order_id=cid))
    assert update is not None
    return update.status, update.cum_filled_qty


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("open_side", "ro_side", "price"), [(Side.BUY, Side.SELL, "110"), (Side.SELL, Side.BUY, "90")]
)
async def test_reduce_only_never_exceeds_exact_position(
    open_side: Side, ro_side: Side, price: str
) -> None:
    async def run() -> tuple[object, ...]:
        exchange = new_exchange()
        await trade(exchange, order("open", open_side, "3.333", "100"), "100")
        fills = await trade(exchange, order("ro", ro_side, "7.777", price, reduce_only=True), price)
        position = await exchange.get_position(symbol="BTCUSDT")
        return [f.qty for f in fills], position, await status(exchange, "ro")

    fills, position, ro_status = await in_both_contexts_async(run)

    assert fills == [D("3.333")]  # never 3.4 (over-reduce) or 3.3 (under)
    assert isinstance(position, Position)
    assert position.qty == 0  # flat, no reversal
    assert ro_status == (OrderStatus.CANCELED, D("3.333"))


@pytest.mark.asyncio
async def test_multiple_reduce_only_orders_sum_to_exact_position() -> None:
    async def run() -> tuple[object, ...]:
        exchange = new_exchange()
        await trade(exchange, order("open", Side.BUY, "7.777", "100"), "100")
        await exchange.place_order(order("a", Side.SELL, "3.333", "110", reduce_only=True))
        await exchange.place_order(order("b", Side.SELL, "5.555", "110", reduce_only=True))
        fills = await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("110"))
        position = await exchange.get_position(symbol="BTCUSDT")
        return [(f.client_order_id, f.qty) for f in fills], position

    fills, position = await in_both_contexts_async(run)

    assert fills == [("a", D("3.333")), ("b", D("4.444"))]
    assert sum(q for _, q in fills) == D("7.777")
    assert isinstance(position, Position)
    assert position.qty == 0


@pytest.mark.asyncio
async def test_cash_accounting_is_context_independent() -> None:
    async def run() -> tuple[object, ...]:
        exchange = new_exchange(cash=True)
        await trade(exchange, order("open", Side.BUY, "7.777", "100.01"), "100.01")
        await trade(exchange, order("c1", Side.SELL, "3.333", "110.37"), "110.37")
        await trade(exchange, order("c2", Side.SELL, "5.555", "95.13"), "95.13")  # reversal
        return await exchange.get_cash_state(), await exchange.get_position(symbol="BTCUSDT")

    cash, position = await in_both_contexts_async(run)

    assert isinstance(position, Position)
    assert position.qty == D("-1.111")
    assert cash is not None


@pytest.mark.asyncio
async def test_mark_valuation_in_the_simulator_is_context_independent() -> None:
    async def run() -> tuple[object, ...]:
        exchange = new_exchange()
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("104.57"))
        await trade(exchange, order("a", Side.BUY, "3.333", "100"), "100")
        await trade(exchange, order("b", Side.BUY, "1.111", "110"), "110")
        after_increase = await exchange.get_position(symbol="BTCUSDT")
        await trade(exchange, order("c", Side.SELL, "2.222", "108"), "108")
        after_partial = await exchange.get_position(symbol="BTCUSDT")
        await trade(exchange, order("d", Side.SELL, "5.555", "103"), "103")
        after_reversal = await exchange.get_position(symbol="BTCUSDT")
        return after_increase, after_partial, after_reversal

    _, after_partial, after_reversal = await in_both_contexts_async(run)

    assert isinstance(after_partial, Position)
    assert isinstance(after_reversal, Position)
    assert after_partial.qty == D("2.222")
    assert after_reversal.qty == D("-3.333")
    assert after_reversal.unrealized_pnl == (D("103") - D("104.57")) * D("3.333")


@pytest.mark.asyncio
async def test_simulation_does_not_modify_the_global_context() -> None:
    def snapshot() -> tuple[object, ...]:
        context = getcontext()
        return context.prec, context.rounding, dict(context.traps), context.Emin, context.Emax

    before = snapshot()
    exchange = new_exchange(cash=True)
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("101.37"))
    await trade(exchange, order("open", Side.BUY, "7.777", "100"), "100")
    await trade(exchange, order("ro", Side.SELL, "9.999", "110", reduce_only=True), "110")
    await exchange.get_position(symbol="BTCUSDT")
    await exchange.get_cash_state()

    assert snapshot() == before


@pytest.mark.asyncio
async def test_equity_scenario_is_context_independent() -> None:
    async def run() -> tuple[object, ...]:
        exchange = new_exchange(cash=True)
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("101.37"))
        await trade(exchange, order("a", Side.BUY, "3.333", "100"), "100")
        await trade(exchange, order("b", Side.BUY, "7.777", "100.01"), "100.01")
        await trade(exchange, order("c", Side.SELL, "5.555", "103.33"), "103.33")
        await exchange.place_order(order("ro", Side.SELL, "9.999", "104.44", reduce_only=True))
        await exchange.fill_crossed_limit_orders(
            symbol="BTCUSDT", execution_price=D("104.44"), available_qty=D("3.333")
        )
        return (
            await exchange.get_position(symbol="BTCUSDT"),
            await exchange.get_cash_state(),
            await exchange.get_equity_state(),
        )

    before = (getcontext().prec, getcontext().rounding, dict(getcontext().traps))
    position, cash, equity = await in_both_contexts_async(run)
    after = (getcontext().prec, getcontext().rounding, dict(getcontext().traps))

    assert before == after
    assert isinstance(position, Position)
    assert position.qty == D("2.222")  # 3.333 + 7.777 - 5.555 - 3.333
    assert isinstance(cash, CashState)
    assert isinstance(equity, EquityState)
    assert position.unrealized_pnl is not None
    # All values here are terminating, so the published fields add up exactly.
    assert cash.cash == D("10000") + position.realized_pnl - cash.trading_fees
    assert equity.unrealized_pnl == position.unrealized_pnl
    assert equity.equity == cash.cash + position.unrealized_pnl
