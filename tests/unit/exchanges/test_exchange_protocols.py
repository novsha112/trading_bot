"""Static conformance of adapters to the exchange protocols.

mypy (strict) is the proof: fakes are assigned to protocol-typed variables, and
wrong adapters are assigned with ``type: ignore[assignment]``. Because unused
ignores are errors in strict mode, mypy fails if a wrong adapter ever starts to
conform. The fakes are test doubles, not runtime implementations.
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
from app.domain.intents import PlaceOrderIntent
from app.domain.market import Ticker
from app.domain.orders import Order, OrderUpdate
from app.domain.positions import Position
from app.exchanges import protocols
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.protocols import AccountClient, MarketDataClient, TradingClient

TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
D = Decimal

REQUEST = OrderRequest(
    client_order_id="grid1-buy-0001",
    symbol="BTCUSDT",
    side=Side.BUY,
    order_type=OrderType.LIMIT,
    price=D("65000"),
    qty=D("0.001"),
    time_in_force=TimeInForce.POST_ONLY,
    reduce_only=False,
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
    """Remembers orders by client order id, like an exchange does."""

    def __init__(self) -> None:
        self.orders: dict[str, OrderUpdate] = {}
        self.canceled: list[OrderRef] = []

    async def place_order(self, order: OrderRequest) -> OrderAck:
        self.orders[order.client_order_id] = OrderUpdate(
            client_order_id=order.client_order_id,
            exchange_order_id="o-1",
            status=OrderStatus.OPEN,
            cum_filled_qty=D("0"),
            avg_fill_price=None,
            reject_reason=None,
            exchange_ts=TS,
        )
        return OrderAck(
            client_order_id=order.client_order_id, exchange_order_id="o-1", exchange_ts=None
        )

    async def cancel_order(self, order: OrderRef) -> None:
        self.canceled.append(order)

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        return self.orders.get(order.client_order_id)

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        return tuple(self.orders.values())


class AcceptsDomainOrder:
    async def place_order(self, order: Order) -> OrderAck:
        raise NotImplementedError

    async def cancel_order(self, order: OrderRef) -> None:
        raise NotImplementedError

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        raise NotImplementedError

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise NotImplementedError


class AcceptsIntent:
    async def place_order(self, order: PlaceOrderIntent) -> OrderAck:
        raise NotImplementedError

    async def cancel_order(self, order: OrderRef) -> None:
        raise NotImplementedError

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        raise NotImplementedError

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise NotImplementedError


class CancelsDomainOrder:
    async def place_order(self, order: OrderRequest) -> OrderAck:
        raise NotImplementedError

    async def cancel_order(self, order: Order) -> None:
        raise NotImplementedError

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        raise NotImplementedError

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        raise NotImplementedError


# Conforming adapters (checked by mypy).
MARKET: MarketDataClient = FakeMarketData()
ACCOUNT: AccountClient = FakeAccount()
TRADING: TradingClient = FakeTrading()

# Non-conforming adapters: mypy must reject each of these assignments.
WRONG_ORDER: TradingClient = AcceptsDomainOrder()  # type: ignore[assignment]
WRONG_INTENT: TradingClient = AcceptsIntent()  # type: ignore[assignment]
WRONG_CANCEL: TradingClient = CancelsDomainOrder()  # type: ignore[assignment]


async def reconcile_ambiguous_placement(
    client: TradingClient, request: OrderRequest
) -> OrderUpdate | None:
    """After an ambiguous placement only the client order id is known."""
    return await client.get_order(
        OrderRef(symbol=request.symbol, client_order_id=request.client_order_id)
    )


@pytest.mark.asyncio
async def test_client_order_id_exists_before_placement_and_resolves_ambiguity() -> None:
    fake = FakeTrading()
    assert REQUEST.client_order_id == "grid1-buy-0001"  # known before any network call

    ack = await fake.place_order(REQUEST)
    assert ack.client_order_id == REQUEST.client_order_id

    found = await reconcile_ambiguous_placement(fake, REQUEST)
    assert found is not None
    assert found.client_order_id == REQUEST.client_order_id
    missing = await fake.get_order(OrderRef(symbol="BTCUSDT", client_order_id="never-sent"))
    assert missing is None


@pytest.mark.asyncio
async def test_cancel_before_and_after_ack() -> None:
    fake = FakeTrading()
    await fake.cancel_order(OrderRef(symbol="BTCUSDT", client_order_id="c-1"))
    await fake.cancel_order(
        OrderRef(symbol="BTCUSDT", client_order_id="c-1", exchange_order_id="o-1")
    )
    assert [ref.exchange_order_id for ref in fake.canceled] == [None, "o-1"]


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
    assert not getattr(protocol, "_is_runtime_protocol", False)


def test_trading_client_takes_no_order_or_intent() -> None:
    hints = {
        name: inspect.get_annotations(member, eval_str=True)
        for name, member in vars(TradingClient).items()
        if callable(member) and not name.startswith("_")
    }
    used = {t for annotations in hints.values() for t in annotations.values()}
    assert Order not in used
    assert PlaceOrderIntent not in used
    assert hints["place_order"]["order"] is OrderRequest
    assert hints["cancel_order"]["order"] is OrderRef
    assert hints["get_order"]["order"] is OrderRef


def test_contracts_contain_no_exchange_specific_names() -> None:
    package = Path(protocols.__file__).parent
    forbidden = re.compile(r"bybit|binance|okx|pybit|ccxt|orderLinkId|positionIdx", re.IGNORECASE)
    for path in package.glob("*.py"):
        assert not forbidden.search(path.read_text(encoding="utf-8")), path.name
