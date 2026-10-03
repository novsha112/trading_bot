"""Simulator-specific recovery reads: ordering, cursors, failure injection,
duplicate scenarios, external orders and the read-only protocol boundary."""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.instrument import InstrumentSpec
from app.exchanges import protocols as protocols_module
from app.exchanges.errors import (
    ExchangeDuplicateOrderError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
    ExchangeResponseError,
)
from app.exchanges.models import OrderRequest
from app.exchanges.protocols import ExchangeStateReader, TradingClient
from app.exchanges.recovery import ExchangeExecution, ExecutionKind, ExecutionQuery
from app.exchanges.simulated import SimulatedExchange

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
APP = Path(__file__).resolve().parents[3] / "app"


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def spec(symbol: str = "BTCUSDT") -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol,
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        tick_size=D("0.1"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        max_qty=D("1000000"),
        min_notional=D("0"),
    )


def exchange(page_size: int = 2, **kwargs: Any) -> tuple[SimulatedExchange, ManualClock]:
    clock = ManualClock(T0)
    return (
        SimulatedExchange(
            clock=clock,
            instruments=(spec("BTCUSDT"), spec("ETHUSDT")),
            execution_page_size=page_size,
            **kwargs,
        ),
        clock,
    )


def request(cid: str = "bot-1", **overrides: Any) -> OrderRequest:
    values: dict[str, Any] = {
        "client_order_id": cid,
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100"),
        "qty": D("2"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
    }
    return OrderRequest(**{**values, **overrides})


def raw(exec_id: str, ts: int, **overrides: Any) -> ExchangeExecution:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": "X-1",
        "client_order_id": None,
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D("100"),
        "qty": D("1"),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "kind": ExecutionKind.TRADE,
        "exchange_ts": at(ts),
    }
    return ExchangeExecution(**{**values, **overrides})


def query(**overrides: Any) -> ExecutionQuery:
    values: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "exchange_order_id": None,
        "start": at(-100),
        "end": at(100),
    }
    return ExecutionQuery(**{**values, **overrides})


# --- open orders ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_snapshot_has_exact_notional_and_clock_time() -> None:
    sim, clock = exchange()
    await sim.place_order(request("bot-1", qty=D("3")))
    clock.advance(timedelta(seconds=5))
    await sim.fill_crossed_limit_orders(
        symbol="BTCUSDT", execution_price=D("99.9"), available_qty=D("1")
    )
    clock.advance(timedelta(seconds=5))

    snapshot = await sim.list_open_orders()

    (order,) = snapshot.orders
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.cum_filled_notional == D("99.9")
    assert order.avg_fill_price == D("99.9")
    assert (order.created_ts, order.updated_ts) == (at(0), at(5))
    assert snapshot.server_ts == at(10)


@pytest.mark.asyncio
async def test_open_snapshot_order_is_deterministic() -> None:
    sim, clock = exchange()
    await sim.place_order(request("bot-b"))
    sim.add_external_order(
        client_order_id=None, symbol="XRPUSDT", side=Side.SELL, price=D("1"), qty=D("5")
    )
    clock.advance(timedelta(seconds=1))
    await sim.place_order(request("bot-a"))

    snapshot = await sim.list_open_orders()

    assert [o.created_ts for o in snapshot.orders] == sorted(o.created_ts for o in snapshot.orders)
    assert [o.client_order_id for o in snapshot.orders] == ["bot-b", None, "bot-a"]


@pytest.mark.asyncio
async def test_external_orders_are_listed_but_never_matched() -> None:
    sim, _ = exchange()
    sim.add_external_order(
        client_order_id="manual-1", symbol="BTCUSDT", side=Side.BUY, price=D("200"), qty=D("1")
    )

    fills = await sim.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("100"))

    assert fills == ()
    assert [o.client_order_id for o in (await sim.list_open_orders()).orders] == ["manual-1"]
    assert await sim.get_open_orders(symbol="BTCUSDT") == ()  # TradingClient view unchanged


@pytest.mark.asyncio
async def test_client_ids_stay_unique_between_bot_and_external_orders() -> None:
    sim, _ = exchange()
    await sim.place_order(request("bot-1"))
    with pytest.raises(ExchangeDuplicateOrderError):
        sim.add_external_order(
            client_order_id="bot-1", symbol="BTCUSDT", side=Side.BUY, price=D("1"), qty=D("1")
        )
    sim.add_external_order(
        client_order_id="manual-1", symbol="BTCUSDT", side=Side.BUY, price=D("1"), qty=D("1")
    )
    with pytest.raises(ExchangeDuplicateOrderError):
        sim.add_external_order(
            client_order_id="manual-1", symbol="ETHUSDT", side=Side.BUY, price=D("1"), qty=D("1")
        )
    with pytest.raises(ExchangeDuplicateOrderError):
        await sim.place_order(request("manual-1"))


@pytest.mark.parametrize(
    "overrides",
    [{"price": D("0")}, {"qty": 1}, {"symbol": ""}, {"client_order_id": ""}],
)
def test_invalid_external_orders_are_refused(overrides: dict[str, Any]) -> None:
    sim, _ = exchange()
    values: dict[str, Any] = {
        "client_order_id": None,
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D("1"),
        "qty": D("1"),
    }
    with pytest.raises(ExchangeRequestValidationError):
        sim.add_external_order(**{**values, **overrides})


# --- positions ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_snapshot_restriction_can_be_lifted() -> None:
    sim, clock = exchange()
    assert (await sim.get_position_snapshot()).positions == ()
    await sim.place_order(request("bot-1"))
    await sim.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("100"))
    clock.advance(timedelta(seconds=3))

    sim.restrict_position_snapshot(("ETHUSDT",))
    partial = await sim.get_position_snapshot()
    sim.restrict_position_snapshot(None)
    full = await sim.get_position_snapshot()

    assert (partial.positions, partial.complete) == ((), False)
    assert full.complete is True
    assert [(p.symbol, p.qty) for p in full.positions] == [("BTCUSDT", D("2"))]
    assert full.server_ts == at(3)


@pytest.mark.parametrize("symbols", [["BTCUSDT"], ("",), (" X",)])
def test_invalid_position_restriction_is_refused(symbols: Any) -> None:
    sim, _ = exchange()
    with pytest.raises((ValueError, ExchangeRequestValidationError)):
        sim.restrict_position_snapshot(symbols)


# --- executions -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fills_are_recorded_as_trade_executions_with_their_fields() -> None:
    sim, _ = exchange(page_size=10)
    await sim.place_order(request("bot-1"))
    (fill,) = await sim.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("99"))

    page = await sim.list_executions(query())

    (execution,) = page.executions
    assert execution == ExchangeExecution(
        exec_id=fill.exec_id,
        exchange_order_id=fill.exchange_order_id,
        client_order_id="bot-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        price=D("99"),
        qty=D("2"),
        fee=None,
        fee_asset=None,
        is_maker=None,
        kind=ExecutionKind.TRADE,
        exchange_ts=fill.exchange_ts,
    )


@pytest.mark.asyncio
async def test_simulator_orders_executions_by_time_then_exec_id() -> None:
    sim, _ = exchange(page_size=10)
    for exec_id, ts in (("b", 2), ("c", 1), ("a", 2), ("d", 0)):
        sim.record_external_execution(raw(exec_id, ts))

    page = await sim.list_executions(query())

    assert [e.exec_id for e in page.executions] == ["d", "c", "a", "b"]


@pytest.mark.asyncio
async def test_cursors_are_opaque_and_bound_to_their_query() -> None:
    sim, _ = exchange(page_size=1)
    sim.record_external_execution(raw("e-1", 1))
    sim.record_external_execution(raw("e-2", 2))
    first = await sim.list_executions(query())
    cursor = first.next_cursor
    assert cursor is not None

    second = await sim.list_executions(query(), cursor=cursor)
    repeat = await sim.list_executions(query(), cursor=cursor)  # a cursor can be reused

    assert (
        [e.exec_id for e in second.executions] == ["e-2"] == [e.exec_id for e in repeat.executions]
    )
    assert second.next_cursor is None
    with pytest.raises(ExchangeRejectedError, match="another query"):
        await sim.list_executions(query(exchange_order_id="X-1"), cursor=cursor)
    with pytest.raises(ExchangeRejectedError, match="unknown"):
        await sim.list_executions(query(), cursor=cursor + "x")
    with pytest.raises(ExchangeRejectedError, match="unknown"):
        await sim.list_executions(query(), cursor=5)  # type: ignore[arg-type]
    with pytest.raises(ExchangeRequestValidationError):
        await sim.list_executions(object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_page_failures_are_deterministic_and_replaceable() -> None:
    sim, _ = exchange(page_size=1)
    for n in range(3):
        sim.record_external_execution(raw(f"e-{n}", n))
    sim.set_execution_page_failures(frozenset({1}))

    for _ in range(3):
        with pytest.raises(ExchangeResponseError, match="page 1"):
            await sim.list_executions(query())

    sim.set_execution_page_failures(frozenset())
    assert [e.exec_id for e in (await sim.list_executions(query())).executions] == ["e-0"]


@pytest.mark.parametrize("pages", [{1}, frozenset({0}), frozenset({True}), frozenset({"1"})])
def test_invalid_page_failure_configuration_is_refused(pages: Any) -> None:
    sim, _ = exchange()
    with pytest.raises(ValueError, match="frozenset"):
        sim.set_execution_page_failures(pages)


@pytest.mark.parametrize("size", [0, -1, 1.0, True, "2"])
def test_page_size_must_be_a_positive_int(size: Any) -> None:
    with pytest.raises(ValueError, match="execution_page_size"):
        exchange(page_size=size)


@pytest.mark.asyncio
async def test_same_exec_id_on_different_pages_is_reported_as_is() -> None:
    sim, _ = exchange(page_size=1)
    sim.record_external_execution(raw("dup", 1))
    sim.record_external_execution(raw("dup", 2))  # identical payload except time
    sim.record_external_execution(raw("dup2", 3))
    sim.record_external_execution(raw("dup2", 4, qty=D("7")))  # conflicting payload

    first = await sim.list_executions(query())
    pages = [first]
    while pages[-1].next_cursor is not None:
        pages.append(await sim.list_executions(query(), cursor=pages[-1].next_cursor))

    assert [e.exec_id for p in pages for e in p.executions] == ["dup", "dup", "dup2", "dup2"]
    assert [e.qty for p in pages for e in p.executions][-2:] == [D("1"), D("7")]


@pytest.mark.asyncio
async def test_same_exec_id_within_one_page_is_a_response_error() -> None:
    sim, _ = exchange(page_size=2)
    sim.record_external_execution(raw("dup", 1))
    sim.record_external_execution(raw("dup", 1))

    with pytest.raises(ExchangeResponseError, match="more than once"):
        await sim.list_executions(query())


def test_record_external_execution_requires_the_dto() -> None:
    sim, _ = exchange()
    with pytest.raises(ExchangeRequestValidationError):
        sim.record_external_execution(object())  # type: ignore[arg-type]


# --- boundaries -----------------------------------------------------------------------------


def test_trading_client_is_unchanged() -> None:
    members = {name for name, _ in inspect.getmembers(TradingClient) if not name.startswith("_")}
    assert members == {"place_order", "cancel_order", "get_order", "get_open_orders"}
    assert str(inspect.signature(TradingClient.get_order)) == (
        "(self, order: 'OrderRef') -> 'OrderUpdate | None'"
    )


def test_state_reader_is_read_only() -> None:
    members = {
        name for name, _ in inspect.getmembers(ExchangeStateReader) if not name.startswith("_")
    }
    assert members == {"list_open_orders", "get_position_snapshot", "list_executions"}
    source = inspect.getsource(protocols_module.ExchangeStateReader)
    for word in ("place_order", "cancel_order", "def cancel", "def place"):
        assert word not in source


def test_exchanges_package_never_imports_execution_services_or_persistence() -> None:
    for path in sorted((APP / "exchanges").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                assert not name.startswith(("app.execution", "app.services", "app.persistence")), (
                    path,
                    name,
                )


def test_bybit_adapter_has_no_recovery_reads() -> None:
    for path in sorted((APP / "exchanges" / "bybit").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "recovery" not in source, path
