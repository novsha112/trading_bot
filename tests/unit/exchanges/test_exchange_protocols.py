"""Static conformance of adapters to the exchange protocols.

mypy is the main proof: assigning a fake adapter to a protocol-typed variable
fails type checking if a signature does not match. The fakes are test doubles,
not runtime implementations.
"""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.domain.balances import Balance
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.instrument import InstrumentSpec
from app.domain.market import Ticker
from app.domain.orders import Order, OrderUpdate
from app.domain.positions import Position
from app.exchanges import protocols
from app.exchanges.models import OrderAck
from app.exchanges.protocols import AccountClient, MarketDataClient, TradingClient

TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
D = Decimal


def submitting_order() -> Order:
    return Order(
        client_order_id="grid1-buy-0001",
        exchange_order_id=None,
        strategy_id="grid-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=D("65000"),
        qty=D("0.001"),
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=False,
        status=OrderStatus.SUBMITTING,
        filled_qty=D("0"),
        avg_fill_price=None,
        created_at=TS,
        updated_at=TS,
        last_exchange_update_ts=None,
        version=1,
    )


class FakeMarketData:
    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        return InstrumentSpec(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USDT",
            tick_size=D("0.1"),
            qty_step=D("0.001"),
            min_qty=D("0.001"),
            max_qty=D("100"),
            min_notional=D("5"),
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        return Ticker(
            symbol=symbol,
            last_price=D("65000"),
            mark_price=None,
            best_bid=None,
            best_ask=None,
            funding_rate=None,
            next_funding_at=None,
            exchange_ts=TS,
            received_ts=TS,
        )


class FakeAccount:
    async def get_balances(self) -> tuple[Balance, ...]:
        return ()

    async def get_positions(self) -> tuple[Position, ...]:
        return ()


class FakeTrading:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def place_order(self, order: Order) -> OrderAck:
        self.sent.append(order.client_order_id)
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="o-1", exchange_ts=None
        )

    async def cancel_order(self, order: Order) -> None:
        self.sent.append(f"cancel:{order.client_order_id}")

    async def get_order(self, *, symbol: str, client_order_id: str) -> OrderUpdate | None:
        return None

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        return ()


# Structural typing: these assignments are checked by mypy (strict).
MARKET: MarketDataClient = FakeMarketData()
ACCOUNT: AccountClient = FakeAccount()
TRADING: TradingClient = FakeTrading()


async def submit(client: TradingClient, order: Order) -> OrderAck:
    """A consumer written only against the protocol."""
    return await client.place_order(order)


@pytest.mark.asyncio
async def test_consumer_uses_protocol_only() -> None:
    fake = FakeTrading()
    order = submitting_order()

    ack = await submit(fake, order)

    assert ack.client_order_id == order.client_order_id
    assert fake.sent == [order.client_order_id]
    assert await fake.get_order(symbol="BTCUSDT", client_order_id="missing") is None


@pytest.mark.asyncio
async def test_read_fakes() -> None:
    assert (await MARKET.get_instrument("BTCUSDT")).symbol == "BTCUSDT"
    assert (await MARKET.get_ticker("BTCUSDT")).last_price == D("65000")
    assert await ACCOUNT.get_balances() == ()
    assert await ACCOUNT.get_positions() == ()


@pytest.mark.parametrize("protocol", [MarketDataClient, AccountClient, TradingClient])
def test_protocol_methods_are_async(protocol: type) -> None:
    methods = {
        name: member
        for name, member in vars(protocol).items()
        if callable(member) and not name.startswith("_")
    }
    assert methods
    for name, member in methods.items():
        assert inspect.iscoroutinefunction(member), name


@pytest.mark.parametrize("protocol", [MarketDataClient, AccountClient, TradingClient])
def test_protocols_are_static_only(protocol: type) -> None:
    # No @runtime_checkable: conformance is verified by mypy, not isinstance().
    assert not getattr(protocol, "_is_runtime_protocol", False)


def test_contracts_contain_no_exchange_specific_names() -> None:
    package = Path(protocols.__file__).parent
    forbidden = re.compile(r"bybit|binance|okx|pybit|ccxt|orderLinkId|positionIdx", re.IGNORECASE)
    for path in package.glob("*.py"):
        assert not forbidden.search(path.read_text(encoding="utf-8")), path.name
