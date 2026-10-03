"""Behavioral contract of ``ExchangeStateReader``, reusable by any implementation.

``ExchangeStateReaderContract`` holds the tests; an implementation subclasses it
(as ``Test...``) and provides ``make_harness(page_size)``: a ``ReaderHarness``
that can put its reader's exchange into the states a test needs (a simulator
directly, a future adapter through recorded fixtures). Nothing here is tied to a
particular exchange. Ordering of executions is NOT part of the contract: tests
compare sets / multisets, never response order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.instrument import InstrumentSpec
from app.exchanges.errors import ExchangeRejectedError, ExchangeResponseError
from app.exchanges.models import OrderRef, OrderRequest
from app.exchanges.protocols import ExchangeStateReader
from app.exchanges.recovery import (
    ExchangeExecution,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
)
from app.exchanges.simulated import SimulatedExchange

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


class ReaderHarness(Protocol):
    """Puts the reader's exchange into a test state (implementation-specific)."""

    @property
    def reader(self) -> ExchangeStateReader: ...

    async def place(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: Side,
        price: Decimal,
        qty: Decimal,
        reduce_only: bool = False,
    ) -> None: ...

    async def cancel(self, *, client_order_id: str, symbol: str) -> None: ...

    async def fill(self, *, symbol: str, price: Decimal, qty: Decimal | None = None) -> None: ...

    def external_order(
        self, *, client_order_id: str | None, symbol: str, side: Side, price: Decimal, qty: Decimal
    ) -> None: ...

    def record_execution(self, execution: ExchangeExecution) -> None: ...

    def fail_page(self, page: int) -> None: ...

    def partial_positions(self, symbols: tuple[str, ...]) -> None: ...

    def advance(self, seconds: int) -> None: ...


async def collect(reader: ExchangeStateReader, query: ExecutionQuery) -> list[ExecutionPage]:
    """All pages of ``query`` (bounded, so a broken cursor cannot loop forever)."""
    pages = [await reader.list_executions(query)]
    while pages[-1].next_cursor is not None:
        assert len(pages) < 1000
        pages.append(await reader.list_executions(query, cursor=pages[-1].next_cursor))
    return pages


def ids(pages: Sequence[ExecutionPage]) -> Counter[str]:
    return Counter(e.exec_id for page in pages for e in page.executions)


def raw(exec_id: str, ts: int, **overrides: object) -> ExchangeExecution:
    values: dict[str, object] = {
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
    return ExchangeExecution(**{**values, **overrides})  # type: ignore[arg-type]


def query(start: int = 0, end: int = 100, **overrides: object) -> ExecutionQuery:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "exchange_order_id": None,
        "start": at(start),
        "end": at(end),
    }
    return ExecutionQuery(**{**values, **overrides})  # type: ignore[arg-type]


class ExchangeStateReaderContract:
    """Subclass as ``Test<Implementation>`` and implement ``make_harness``."""

    def make_harness(self, page_size: int) -> ReaderHarness:
        raise NotImplementedError

    # --- open orders -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_open_snapshot_is_the_complete_open_set(self) -> None:
        h = self.make_harness(3)
        await h.place(
            client_order_id="bot-1", symbol="BTCUSDT", side=Side.BUY, price=D("100"), qty=D("2")
        )
        await h.place(
            client_order_id="bot-2", symbol="ETHUSDT", side=Side.SELL, price=D("50"), qty=D("1")
        )
        await h.place(
            client_order_id="bot-3", symbol="BTCUSDT", side=Side.BUY, price=D("90"), qty=D("1")
        )
        await h.place(
            client_order_id="bot-4", symbol="BTCUSDT", side=Side.BUY, price=D("99"), qty=D("1")
        )
        h.external_order(
            client_order_id="manual-1", symbol="SOLUSDT", side=Side.BUY, price=D("10"), qty=D("1")
        )
        h.external_order(
            client_order_id=None, symbol="XRPUSDT", side=Side.SELL, price=D("1"), qty=D("5")
        )
        await h.cancel(client_order_id="bot-3", symbol="BTCUSDT")
        h.advance(1)
        await h.fill(symbol="BTCUSDT", price=D("98"), qty=D("1"))  # bot-1 PF, bot-4 untouched
        h.advance(1)

        snapshot = await h.reader.list_open_orders()

        by_client = {o.client_order_id: o for o in snapshot.orders}
        assert set(by_client) == {"bot-1", "bot-2", "bot-4", "manual-1", None}
        assert by_client["bot-1"].status is S.PARTIALLY_FILLED
        assert by_client["bot-1"].cum_filled_qty == D("1")
        assert all(o.status in (S.OPEN, S.PARTIALLY_FILLED) for o in snapshot.orders)
        assert all(o.updated_ts <= snapshot.server_ts for o in snapshot.orders)

    @pytest.mark.asyncio
    async def test_open_snapshot_preserves_full_terms(self) -> None:
        h = self.make_harness(3)
        await h.place(
            client_order_id="bot-1",
            symbol="BTCUSDT",
            side=Side.SELL,
            price=D("101.5"),
            qty=D("2.500"),
        )
        h.external_order(
            client_order_id=None, symbol="ETHUSDT", side=Side.BUY, price=D("7"), qty=D("3")
        )

        snapshot = await h.reader.list_open_orders()

        mine = next(o for o in snapshot.orders if o.client_order_id == "bot-1")
        assert (mine.symbol, mine.side, mine.order_type, mine.price, mine.qty) == (
            "BTCUSDT",
            Side.SELL,
            OrderType.LIMIT,
            D("101.5"),
            D("2.500"),
        )
        assert repr(mine.qty) == "Decimal('2.500')"
        assert (mine.time_in_force, mine.reduce_only, mine.status) == (
            TimeInForce.GTC,
            False,
            S.OPEN,
        )
        assert mine.cum_filled_qty == 0
        assert mine.exchange_order_id
        foreign = next(o for o in snapshot.orders if o.client_order_id is None)
        assert (foreign.symbol, foreign.side, foreign.qty) == ("ETHUSDT", Side.BUY, D("3"))

    @pytest.mark.asyncio
    async def test_empty_exchange_has_an_empty_open_snapshot(self) -> None:
        snapshot = await self.make_harness(3).reader.list_open_orders()
        assert snapshot.orders == ()

    # --- positions ---------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_position_snapshot_has_signed_and_zero_quantities(self) -> None:
        h = self.make_harness(3)
        await h.place(
            client_order_id="b-1", symbol="BTCUSDT", side=Side.BUY, price=D("100"), qty=D("2")
        )
        await h.place(
            client_order_id="s-1", symbol="ETHUSDT", side=Side.SELL, price=D("50"), qty=D("3")
        )
        await h.place(
            client_order_id="b-2", symbol="SOLUSDT", side=Side.BUY, price=D("10"), qty=D("1")
        )
        await h.fill(symbol="BTCUSDT", price=D("100"))
        await h.fill(symbol="ETHUSDT", price=D("50"))
        await h.fill(symbol="SOLUSDT", price=D("10"))
        await h.place(
            client_order_id="s-2", symbol="SOLUSDT", side=Side.SELL, price=D("10"), qty=D("1")
        )
        await h.fill(symbol="SOLUSDT", price=D("10"))

        snapshot = await h.reader.get_position_snapshot()

        assert snapshot.complete is True
        assert {p.symbol: p.qty for p in snapshot.positions} == {
            "BTCUSDT": D("2"),
            "ETHUSDT": D("-3"),
            "SOLUSDT": D("0"),
        }

    @pytest.mark.asyncio
    async def test_partial_position_snapshot_says_so(self) -> None:
        h = self.make_harness(3)
        await h.place(
            client_order_id="b-1", symbol="BTCUSDT", side=Side.BUY, price=D("100"), qty=D("2")
        )
        await h.place(
            client_order_id="b-2", symbol="ETHUSDT", side=Side.BUY, price=D("50"), qty=D("1")
        )
        await h.fill(symbol="BTCUSDT", price=D("100"))
        await h.fill(symbol="ETHUSDT", price=D("50"))
        h.partial_positions(("ETHUSDT",))

        snapshot = await h.reader.get_position_snapshot()

        assert snapshot.complete is False
        # BTCUSDT is absent and therefore UNKNOWN (not flat) in a partial snapshot.
        assert {p.symbol for p in snapshot.positions} == {"ETHUSDT"}

    # --- executions --------------------------------------------------------------------

    @pytest.mark.parametrize("page_size", [1, 2, 3])
    @pytest.mark.asyncio
    async def test_pagination_returns_every_matching_execution_once(self, page_size: int) -> None:
        h = self.make_harness(page_size)
        for n in range(7):
            h.record_execution(raw(f"e-{n}", ts=10 + n))
        h.record_execution(raw("other-symbol", ts=12, symbol="ETHUSDT"))
        q = query()

        pages = await collect(h.reader, q)

        assert ids(pages) == Counter(f"e-{n}" for n in range(7))
        assert len(pages) == -(-7 // page_size)
        assert all(page.query == q for page in pages)  # exact echo on every page
        assert all(page.next_cursor is not None for page in pages[:-1])
        assert pages[-1].next_cursor is None
        assert all(len(page.executions) <= page_size for page in pages)

    @pytest.mark.asyncio
    async def test_window_bounds_are_inclusive(self) -> None:
        h = self.make_harness(2)
        for ts in (9, 10, 15, 20, 21):
            h.record_execution(raw(f"e-{ts}", ts=ts))

        pages = await collect(h.reader, query(start=10, end=20))

        assert set(ids(pages)) == {"e-10", "e-15", "e-20"}

    @pytest.mark.asyncio
    async def test_order_filter_selects_one_exchange_order(self) -> None:
        h = self.make_harness(2)
        h.record_execution(raw("a-1", ts=1, exchange_order_id="X-A"))
        h.record_execution(raw("b-1", ts=2, exchange_order_id="X-B"))
        h.record_execution(raw("a-2", ts=3, exchange_order_id="X-A"))

        pages = await collect(h.reader, query(exchange_order_id="X-A"))

        assert set(ids(pages)) == {"a-1", "a-2"}

    @pytest.mark.asyncio
    async def test_empty_result_is_one_exhausted_page(self) -> None:
        pages = await collect(self.make_harness(2).reader, query())
        assert len(pages) == 1
        assert pages[0].executions == ()
        assert pages[0].next_cursor is None

    @pytest.mark.asyncio
    async def test_unknown_cursor_is_an_error_not_an_empty_page(self) -> None:
        h = self.make_harness(1)
        h.record_execution(raw("e-1", ts=1))
        h.record_execution(raw("e-2", ts=2))
        first = await h.reader.list_executions(query())
        assert first.next_cursor is not None

        with pytest.raises(ExchangeRejectedError):
            await h.reader.list_executions(query(), cursor="no-such-cursor")
        with pytest.raises(ExchangeRejectedError):
            await h.reader.list_executions(query(end=50), cursor=first.next_cursor)

    @pytest.mark.asyncio
    async def test_injected_page_failure_is_repeatable_and_returns_nothing(self) -> None:
        h = self.make_harness(2)
        for n in range(5):
            h.record_execution(raw(f"e-{n}", ts=n + 1))
        h.fail_page(2)
        q = query()
        first = await h.reader.list_executions(q)
        assert first.next_cursor is not None

        for _ in range(2):
            with pytest.raises(ExchangeResponseError):
                await h.reader.list_executions(q, cursor=first.next_cursor)
        # The first page is still valid; the history is incomplete, never "done".
        again = await h.reader.list_executions(q)
        assert again.next_cursor is not None

    @pytest.mark.asyncio
    async def test_order_fills_are_trade_executions_of_that_order(self) -> None:
        h = self.make_harness(2)
        await h.place(
            client_order_id="bot-1", symbol="BTCUSDT", side=Side.BUY, price=D("100"), qty=D("3")
        )
        await h.fill(symbol="BTCUSDT", price=D("99"), qty=D("1"))
        h.advance(1)
        await h.fill(symbol="BTCUSDT", price=D("98"), qty=D("2"))
        snapshot = await h.reader.list_open_orders()
        assert snapshot.orders == ()  # FILLED: no longer open

        pages = await collect(h.reader, query(start=-10, end=100))

        trades = [e for page in pages for e in page.executions]
        assert len(trades) == 2
        assert {e.kind for e in trades} == {ExecutionKind.TRADE}
        assert {e.client_order_id for e in trades} == {"bot-1"}
        assert sum((e.qty for e in trades), D(0)) == D("3")
        assert len({e.exchange_order_id for e in trades}) == 1


# --- the simulator --------------------------------------------------------------------------


def _spec(symbol: str) -> InstrumentSpec:
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


class SimulatorHarness:
    def __init__(self, page_size: int) -> None:
        self.clock = ManualClock(T0)
        self.exchange = SimulatedExchange(
            clock=self.clock,
            instruments=tuple(_spec(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")),
            execution_page_size=page_size,
        )
        self._failing: set[int] = set()

    @property
    def reader(self) -> ExchangeStateReader:
        return self.exchange

    async def place(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: Side,
        price: Decimal,
        qty: Decimal,
        reduce_only: bool = False,
    ) -> None:
        await self.exchange.place_order(
            OrderRequest(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                qty=qty,
                time_in_force=TimeInForce.GTC,
                reduce_only=reduce_only,
            )
        )

    async def cancel(self, *, client_order_id: str, symbol: str) -> None:
        await self.exchange.cancel_order(OrderRef(symbol=symbol, client_order_id=client_order_id))

    async def fill(self, *, symbol: str, price: Decimal, qty: Decimal | None = None) -> None:
        await self.exchange.fill_crossed_limit_orders(
            symbol=symbol, execution_price=price, available_qty=qty
        )

    def external_order(
        self, *, client_order_id: str | None, symbol: str, side: Side, price: Decimal, qty: Decimal
    ) -> None:
        self.exchange.add_external_order(
            client_order_id=client_order_id, symbol=symbol, side=side, price=price, qty=qty
        )

    def record_execution(self, execution: ExchangeExecution) -> None:
        self.exchange.record_external_execution(execution)

    def fail_page(self, page: int) -> None:
        self._failing.add(page)
        self.exchange.set_execution_page_failures(frozenset(self._failing))

    def partial_positions(self, symbols: tuple[str, ...]) -> None:
        self.exchange.restrict_position_snapshot(symbols)

    def advance(self, seconds: int) -> None:
        self.clock.advance(timedelta(seconds=seconds))


class TestSimulatedExchangeStateReader(ExchangeStateReaderContract):
    def make_harness(self, page_size: int) -> ReaderHarness:
        return SimulatorHarness(page_size)


def test_simulator_satisfies_the_reader_protocol() -> None:
    reader: ExchangeStateReader = SimulatorHarness(1).exchange
    assert callable(reader.list_executions)
