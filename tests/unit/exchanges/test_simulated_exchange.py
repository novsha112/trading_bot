"""Deterministic simulated exchange: order lifecycle without network or credentials.

No fills, matching, fees, balances or positions yet: a LIMIT order rests OPEN until
it is canceled.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.fills import Fill
from app.domain.instrument import InstrumentSpec
from app.domain.orders import OrderUpdate
from app.exchanges import simulated
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeDuplicateOrderError,
    ExchangeError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.protocols import TradingClient
from app.exchanges.simulated import SimulatedExchange
from app.exchanges.simulated_accounting import SimulatedCashConfig
from app.exchanges.simulated_fees import (
    LiquidityRole,
    SimulatedFeePolicy,
    TradingFeeSchedule,
)

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

LIMIT: dict[str, Any] = {
    "client_order_id": "grid1-buy-0001",
    "symbol": "BTCUSDT",
    "side": Side.BUY,
    "order_type": OrderType.LIMIT,
    "price": D("65000.5"),
    "qty": D("0.001"),
    "time_in_force": TimeInForce.GTC,
    "reduce_only": False,
}


def spec(symbol: str = "BTCUSDT", **overrides: Any) -> InstrumentSpec:
    values: dict[str, Any] = {
        "symbol": symbol,
        "base_asset": symbol.removesuffix("USDT"),
        "quote_asset": "USDT",
        "tick_size": D("0.1"),
        "qty_step": D("0.001"),
        "min_qty": D("0.001"),
        "max_qty": D("1000000"),
        "min_notional": D("0"),
    }
    return InstrumentSpec(**{**values, **overrides})


# Permissive specs for lifecycle tests; instrument rules have their own tests.
SPECS = (spec("BTCUSDT"), spec("ETHUSDT"), spec("SOLUSDT"))


def request(**overrides: Any) -> OrderRequest:
    return OrderRequest(**{**LIMIT, **overrides})


def ref(
    client_order_id: str = "grid1-buy-0001",
    symbol: str = "BTCUSDT",
    exchange_order_id: str | None = None,
) -> OrderRef:
    return OrderRef(
        symbol=symbol, client_order_id=client_order_id, exchange_order_id=exchange_order_id
    )


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(T0)


@pytest.fixture
def exchange(clock: ManualClock) -> SimulatedExchange:
    return SimulatedExchange(clock=clock, instruments=SPECS)


# --- contract -----------------------------------------------------------------


def test_implements_trading_client(clock: ManualClock) -> None:
    # mypy strict is the proof (structural Protocol, no runtime_checkable).
    client: TradingClient = SimulatedExchange(clock=clock, instruments=SPECS)
    assert client is not None


def test_constructor_requires_keyword_clock(clock: ManualClock) -> None:
    with pytest.raises(TypeError):
        SimulatedExchange(clock)  # type: ignore[call-arg]


def test_repr_is_compact_and_deterministic(exchange: SimulatedExchange) -> None:
    assert repr(exchange) == "SimulatedExchange(orders=0)"


# --- place_order: LIMIT -------------------------------------------------------


@pytest.mark.asyncio
async def test_limit_becomes_open(exchange: SimulatedExchange) -> None:
    ack = await exchange.place_order(request())

    assert ack == OrderAck(
        client_order_id="grid1-buy-0001", exchange_order_id="SIM-0000000001", exchange_ts=T0
    )
    update = await exchange.get_order(ref())
    assert update == OrderUpdate(
        client_order_id="grid1-buy-0001",
        exchange_order_id="SIM-0000000001",
        status=OrderStatus.OPEN,
        cum_filled_qty=D("0"),
        avg_fill_price=None,
        reject_reason=None,
        exchange_ts=T0,
    )


@pytest.mark.asyncio
async def test_post_only_limit_becomes_open(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(time_in_force=TimeInForce.POST_ONLY))

    update = await exchange.get_order(ref())
    assert update is not None
    assert update.status is OrderStatus.OPEN


@pytest.mark.asyncio
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
async def test_both_sides_accepted(exchange: SimulatedExchange, side: Side) -> None:
    await exchange.place_order(request(side=side))

    assert len(await exchange.get_open_orders(symbol="BTCUSDT")) == 1


# --- place_order: deterministic IDs -------------------------------------------


@pytest.mark.asyncio
async def test_exchange_order_ids_are_a_deterministic_sequence(
    exchange: SimulatedExchange,
) -> None:
    acks = [await exchange.place_order(request(client_order_id=f"c-{i}")) for i in range(3)]

    assert [a.exchange_order_id for a in acks] == [
        "SIM-0000000001",
        "SIM-0000000002",
        "SIM-0000000003",
    ]


@pytest.mark.asyncio
async def test_two_different_orders_get_different_ids(exchange: SimulatedExchange) -> None:
    a = await exchange.place_order(request(client_order_id="a"))
    b = await exchange.place_order(request(client_order_id="b", price=D("64000")))

    assert a.exchange_order_id != b.exchange_order_id


@pytest.mark.asyncio
async def test_same_scenario_on_two_instances_gives_identical_results() -> None:
    async def run() -> list[object]:
        exchange = SimulatedExchange(clock=ManualClock(T0), instruments=SPECS)
        out: list[object] = []
        out.append(await exchange.place_order(request(client_order_id="a")))
        out.append(await exchange.place_order(request(client_order_id="b")))
        await exchange.cancel_order(ref("a"))
        out.append(await exchange.get_order(ref("a")))
        out.append(await exchange.get_open_orders(symbol="BTCUSDT"))
        return out

    assert await run() == await run()


@pytest.mark.asyncio
async def test_instances_do_not_share_state_or_sequence() -> None:
    first = SimulatedExchange(clock=ManualClock(T0), instruments=SPECS)
    second = SimulatedExchange(clock=ManualClock(T0), instruments=SPECS)

    await first.place_order(request(client_order_id="a"))
    await first.place_order(request(client_order_id="b"))
    ack = await second.place_order(request(client_order_id="z"))

    assert ack.exchange_order_id == "SIM-0000000001"
    assert await second.get_order(ref("a")) is None
    assert await first.get_order(ref("z")) is None
    assert len(await first.get_open_orders(symbol="BTCUSDT")) == 2
    assert len(await second.get_open_orders(symbol="BTCUSDT")) == 1


@pytest.mark.asyncio
async def test_refused_requests_do_not_consume_ids(exchange: SimulatedExchange) -> None:
    with pytest.raises(ExchangeRejectedError):
        await exchange.place_order(
            request(client_order_id="m", order_type=OrderType.MARKET, price=None)
        )
    ack = await exchange.place_order(request(client_order_id="l"))

    assert ack.exchange_order_id == "SIM-0000000001"


# --- place_order: idempotency -------------------------------------------------


@pytest.mark.asyncio
async def test_identical_duplicate_returns_the_original_ack(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    first = await exchange.place_order(request())
    clock.advance(timedelta(seconds=5))

    second = await exchange.place_order(request())

    assert second == first  # same exchange_order_id and original exchange_ts
    assert len(await exchange.get_open_orders(symbol="BTCUSDT")) == 1
    assert repr(exchange) == "SimulatedExchange(orders=1)"


@pytest.mark.asyncio
async def test_value_equal_decimals_count_as_identical(exchange: SimulatedExchange) -> None:
    first = await exchange.place_order(request(price=D("65000.5"), qty=D("0.001")))
    second = await exchange.place_order(request(price=D("65000.50"), qty=D("1E-3")))

    assert second == first


@pytest.mark.asyncio
async def test_identical_duplicate_after_cancel_does_not_reopen(
    exchange: SimulatedExchange,
) -> None:
    first = await exchange.place_order(request())
    await exchange.cancel_order(ref())

    second = await exchange.place_order(request())

    assert second == first
    update = await exchange.get_order(ref())
    assert update is not None
    assert update.status is OrderStatus.CANCELED
    assert await exchange.get_open_orders(symbol="BTCUSDT") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"price": D("65001")},
        {"qty": D("0.002")},
        {"side": Side.SELL},
        {"time_in_force": TimeInForce.POST_ONLY},
        {"symbol": "ETHUSDT"},
        {"reduce_only": True},
    ],
)
async def test_same_client_id_with_changed_payload_is_refused(
    exchange: SimulatedExchange, change: dict[str, Any]
) -> None:
    original = await exchange.place_order(request())

    with pytest.raises(ExchangeDuplicateOrderError, match="grid1-buy-0001"):
        await exchange.place_order(request(**change))

    # The existing order is untouched and nothing new was created.
    update = await exchange.get_order(ref())
    assert update is not None
    assert update.exchange_order_id == original.exchange_order_id
    assert update.status is OrderStatus.OPEN
    assert repr(exchange) == "SimulatedExchange(orders=1)"


def test_duplicate_error_is_not_a_rejection_nor_ambiguous() -> None:
    assert issubclass(ExchangeDuplicateOrderError, ExchangeError)
    assert not issubclass(ExchangeDuplicateOrderError, ExchangeRejectedError)
    assert not issubclass(ExchangeDuplicateOrderError, ExchangeAmbiguousResultError)


@pytest.mark.asyncio
async def test_duplicate_keeps_original_order_and_does_not_consume_an_id(
    exchange: SimulatedExchange,
) -> None:
    original = await exchange.place_order(request())

    with pytest.raises(ExchangeDuplicateOrderError) as excinfo:
        await exchange.place_order(request(price=D("64000"), qty=D("0.5")))
    assert not isinstance(excinfo.value, ExchangeRejectedError)

    update = await exchange.get_order(ref())
    assert update == OrderUpdate(
        client_order_id="grid1-buy-0001",
        exchange_order_id=original.exchange_order_id,
        status=OrderStatus.OPEN,
        cum_filled_qty=D("0"),
        avg_fill_price=None,
        reject_reason=None,
        exchange_ts=T0,
    )
    assert original.exchange_order_id == "SIM-0000000001"
    following = await exchange.place_order(request(client_order_id="next"))
    assert following.exchange_order_id == "SIM-0000000002"


# --- place_order: refused -----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tif", [TimeInForce.IOC, TimeInForce.GTC, TimeInForce.FOK])
async def test_market_is_refused_until_a_price_model_exists(
    exchange: SimulatedExchange, tif: TimeInForce
) -> None:
    order = request(order_type=OrderType.MARKET, price=None, time_in_force=tif)

    with pytest.raises(ExchangeRejectedError, match="market"):
        await exchange.place_order(order)

    assert await exchange.get_order(ref()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tif", [TimeInForce.IOC, TimeInForce.FOK])
async def test_immediate_limit_tif_is_refused_until_matching_exists(
    exchange: SimulatedExchange, tif: TimeInForce
) -> None:
    # IOC / FOK never rest on the book; without matching their outcome is unknown.
    with pytest.raises(ExchangeRejectedError, match="time_in_force"):
        await exchange.place_order(request(time_in_force=tif))

    assert await exchange.get_order(ref()) is None


@pytest.mark.asyncio
async def test_reduce_only_is_refused_until_positions_exist(exchange: SimulatedExchange) -> None:
    with pytest.raises(ExchangeRejectedError, match="reduce_only"):
        await exchange.place_order(request(reduce_only=True))

    assert await exchange.get_order(ref()) is None


@pytest.mark.asyncio
async def test_refused_client_id_can_be_used_by_a_valid_request(
    exchange: SimulatedExchange,
) -> None:
    with pytest.raises(ExchangeRejectedError):
        await exchange.place_order(request(reduce_only=True))

    ack = await exchange.place_order(request())

    assert ack.client_order_id == "grid1-buy-0001"


@pytest.mark.asyncio
async def test_non_order_request_is_refused_locally(exchange: SimulatedExchange) -> None:
    with pytest.raises(ExchangeRequestValidationError):
        await exchange.place_order(LIMIT)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_place_order_does_not_mutate_request(exchange: SimulatedExchange) -> None:
    order = request()
    before = dataclasses.astuple(order)

    await exchange.place_order(order)
    await exchange.cancel_order(ref())

    assert dataclasses.astuple(order) == before


# --- cancel_order -------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_open_becomes_canceled(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    await exchange.place_order(request())
    clock.advance(timedelta(seconds=3))

    await exchange.cancel_order(ref())

    update = await exchange.get_order(ref())
    assert update == OrderUpdate(
        client_order_id="grid1-buy-0001",
        exchange_order_id="SIM-0000000001",
        status=OrderStatus.CANCELED,
        cum_filled_qty=D("0"),
        avg_fill_price=None,
        reject_reason=None,
        exchange_ts=T0 + timedelta(seconds=3),
    )


@pytest.mark.asyncio
async def test_cancel_with_matching_exchange_order_id(exchange: SimulatedExchange) -> None:
    ack = await exchange.place_order(request())

    await exchange.cancel_order(ref(exchange_order_id=ack.exchange_order_id))

    update = await exchange.get_order(ref())
    assert update is not None
    assert update.status is OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_repeated_cancel_is_rejected_without_changing_state(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    await exchange.place_order(request())
    await exchange.cancel_order(ref())
    canceled = await exchange.get_order(ref())
    clock.advance(timedelta(seconds=10))

    for _ in range(2):
        with pytest.raises(ExchangeRejectedError, match="canceled"):
            await exchange.cancel_order(ref())

    assert await exchange.get_order(ref()) == canceled  # timestamp unchanged too


@pytest.mark.asyncio
async def test_cancel_unknown_order_is_rejected(exchange: SimulatedExchange) -> None:
    with pytest.raises(ExchangeRejectedError, match="not found"):
        await exchange.cancel_order(ref("missing"))

    assert repr(exchange) == "SimulatedExchange(orders=0)"


@pytest.mark.asyncio
async def test_cancel_with_wrong_symbol_is_rejected_and_order_stays_open(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request())

    with pytest.raises(ExchangeRejectedError, match="not found"):
        await exchange.cancel_order(ref(symbol="ETHUSDT"))

    update = await exchange.get_order(ref())
    assert update is not None
    assert update.status is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_cancel_with_mismatched_exchange_order_id_is_rejected(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request())

    with pytest.raises(ExchangeRejectedError, match="exchange_order_id"):
        await exchange.cancel_order(ref(exchange_order_id="SIM-0000000099"))

    update = await exchange.get_order(ref())
    assert update is not None
    assert update.status is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_cancel_non_ref_is_refused_locally(exchange: SimulatedExchange) -> None:
    with pytest.raises(ExchangeRequestValidationError):
        await exchange.cancel_order("grid1-buy-0001")  # type: ignore[arg-type]


# --- get_order ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_known_order_by_client_id_and_by_both_ids(
    exchange: SimulatedExchange,
) -> None:
    ack = await exchange.place_order(request())

    by_client = await exchange.get_order(ref())
    by_both = await exchange.get_order(ref(exchange_order_id=ack.exchange_order_id))

    assert by_client is not None
    assert by_client == by_both
    assert by_client.exchange_order_id == ack.exchange_order_id


@pytest.mark.asyncio
async def test_get_unknown_order_is_none(exchange: SimulatedExchange) -> None:
    assert await exchange.get_order(ref("missing")) is None


@pytest.mark.asyncio
async def test_get_with_wrong_symbol_is_none(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request())

    assert await exchange.get_order(ref(symbol="ETHUSDT")) is None


@pytest.mark.asyncio
async def test_get_with_mismatched_exchange_order_id_is_rejected(
    exchange: SimulatedExchange,
) -> None:
    # Not None: the order exists, the caller's reference is contradictory. None
    # would let reconciliation conclude "never placed" and risk a duplicate.
    await exchange.place_order(request())

    with pytest.raises(ExchangeRejectedError, match="exchange_order_id"):
        await exchange.get_order(ref(exchange_order_id="SIM-0000000099"))


@pytest.mark.asyncio
async def test_get_with_exchange_order_id_of_another_order_is_rejected(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request(client_order_id="a"))
    other = await exchange.place_order(request(client_order_id="b"))

    with pytest.raises(ExchangeRejectedError, match="exchange_order_id"):
        await exchange.get_order(ref("a", exchange_order_id=other.exchange_order_id))


@pytest.mark.asyncio
async def test_get_non_ref_is_refused_locally(exchange: SimulatedExchange) -> None:
    with pytest.raises(ExchangeRequestValidationError):
        await exchange.get_order(None)  # type: ignore[arg-type]


# --- get_open_orders ----------------------------------------------------------


@pytest.mark.asyncio
async def test_open_orders_only_active_and_only_symbol(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="btc-1"))
    await exchange.place_order(request(client_order_id="btc-2"))
    await exchange.place_order(request(client_order_id="eth-1", symbol="ETHUSDT"))
    await exchange.cancel_order(ref("btc-1"))

    btc = await exchange.get_open_orders(symbol="BTCUSDT")
    eth = await exchange.get_open_orders(symbol="ETHUSDT")

    assert [u.client_order_id for u in btc] == ["btc-2"]
    assert [u.client_order_id for u in eth] == ["eth-1"]
    assert all(u.status is OrderStatus.OPEN for u in (*btc, *eth))
    assert await exchange.get_open_orders(symbol="SOLUSDT") == ()


@pytest.mark.asyncio
async def test_open_orders_ordered_by_created_at_then_client_id(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    # Same timestamp: client_order_id decides, not insertion order.
    await exchange.place_order(request(client_order_id="c"))
    await exchange.place_order(request(client_order_id="a"))
    clock.advance(timedelta(seconds=1))
    await exchange.place_order(request(client_order_id="0-later"))
    await exchange.place_order(request(client_order_id="b-later"))

    first = await exchange.get_open_orders(symbol="BTCUSDT")
    second = await exchange.get_open_orders(symbol="BTCUSDT")

    assert [u.client_order_id for u in first] == ["a", "c", "0-later", "b-later"]
    assert first == second
    assert isinstance(first, tuple)


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["", " BTCUSDT", None])
async def test_open_orders_invalid_symbol_refused_locally(
    exchange: SimulatedExchange, symbol: object
) -> None:
    with pytest.raises(ExchangeRequestValidationError):
        await exchange.get_open_orders(symbol=symbol)  # type: ignore[arg-type]


# --- clock --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timestamps_come_from_the_injected_clock(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    clock.set(T0 + timedelta(minutes=1))
    ack = await exchange.place_order(request())
    clock.set(T0 + timedelta(minutes=2))
    open_update = (await exchange.get_open_orders(symbol="BTCUSDT"))[0]
    clock.set(T0 + timedelta(minutes=3))
    await exchange.cancel_order(ref())
    clock.set(T0 + timedelta(minutes=4))
    canceled = await exchange.get_order(ref())

    assert ack.exchange_ts == T0 + timedelta(minutes=1)
    assert open_update.exchange_ts == T0 + timedelta(minutes=1)  # last change, not query
    assert canceled is not None
    assert canceled.exchange_ts == T0 + timedelta(minutes=3)


# --- errors -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulator_never_raises_ambiguous(exchange: SimulatedExchange) -> None:
    calls: list[Any] = [
        exchange.place_order(request(order_type=OrderType.MARKET, price=None)),
        exchange.cancel_order(ref("missing")),
    ]
    for call in calls:
        with pytest.raises(ExchangeError) as excinfo:
            await call
        assert not isinstance(excinfo.value, ExchangeAmbiguousResultError)


# --- module boundaries --------------------------------------------------------


def _imports() -> set[str]:
    source = Path(simulated.__file__).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_network_randomness_system_time_or_bybit_imports() -> None:
    forbidden_tops = {
        "random",
        "secrets",
        "uuid",
        "time",
        "socket",
        "ssl",
        "http",
        "urllib",
        "httpx",
        "asyncio",
        "os",
        "logging",
        "structlog",
        "pydantic",
    }
    for name in _imports():
        assert name.split(".")[0] not in forbidden_tops, name
        assert "bybit" not in name, name
        assert not name.startswith(("app.config", "app.persistence", "app.strategies")), name


def test_no_system_clock_or_credentials_in_source() -> None:
    source = Path(simulated.__file__).read_text(encoding="utf-8")

    for banned in ("datetime.now", "utcnow", "time.time", "sleep(", "uuid", "random"):
        assert banned not in source, banned
    for banned in ("api_key", "api_secret", "credential", "BYBIT"):
        assert banned.lower() not in source.lower(), banned


# === deterministic limit fills ================================================


class CountingClock:
    """ManualClock wrapper that counts reads, to pin when time is taken."""

    def __init__(self, start: datetime) -> None:
        self.inner = ManualClock(start)
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        return self.inner.now()


async def fill_at(exchange: SimulatedExchange, price: str, symbol: str = "BTCUSDT") -> list[Fill]:
    return list(await exchange.fill_crossed_limit_orders(symbol=symbol, execution_price=D(price)))


async def status_of(
    exchange: SimulatedExchange, client_order_id: str, symbol: str = "BTCUSDT"
) -> OrderStatus:
    update = await exchange.get_order(ref(client_order_id, symbol=symbol))
    assert update is not None
    return update.status


# --- crossing -----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "limit", "execution", "crossed"),
    [
        (Side.BUY, "100", "95", True),  # below limit
        (Side.BUY, "100", "100", True),  # equal
        (Side.BUY, "100", "100.0000001", False),  # above
        (Side.SELL, "100", "105", True),  # above limit
        (Side.SELL, "100", "100", True),  # equal
        (Side.SELL, "100", "99.9999999", False),  # below
    ],
)
async def test_crossing_rules(
    exchange: SimulatedExchange, side: Side, limit: str, execution: str, crossed: bool
) -> None:
    await exchange.place_order(request(side=side, price=D(limit)))

    fills = await fill_at(exchange, execution)

    assert len(fills) == (1 if crossed else 0)
    expected = OrderStatus.FILLED if crossed else OrderStatus.OPEN
    assert await status_of(exchange, "grid1-buy-0001") is expected


# --- fill contents ------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_is_exact_with_unknown_fee_and_liquidity_role(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    ack = await exchange.place_order(request(price=D("100"), qty=D("0.25")))
    clock.advance(timedelta(seconds=7))

    fills = await fill_at(exchange, "95")

    assert fills == [
        Fill(
            exec_id="SIM-EXEC-0000000001",
            exchange_order_id=ack.exchange_order_id,
            client_order_id="grid1-buy-0001",
            symbol="BTCUSDT",
            side=Side.BUY,
            price=D("95"),  # execution price, not the limit price
            qty=D("0.25"),
            fee=None,
            fee_asset=None,
            is_maker=None,
            exchange_ts=T0 + timedelta(seconds=7),
        )
    ]


@pytest.mark.asyncio
async def test_sell_fill_side_and_price(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(side=Side.SELL, price=D("100")))

    (fill,) = await fill_at(exchange, "101.5")

    assert fill.side is Side.SELL
    assert fill.price == D("101.5")


@pytest.mark.asyncio
async def test_post_only_fill_does_not_claim_maker(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(time_in_force=TimeInForce.POST_ONLY, price=D("100")))

    (fill,) = await fill_at(exchange, "100")

    assert fill.is_maker is None
    assert fill.fee is None
    assert fill.fee_asset is None


# --- state after fill ---------------------------------------------------------


@pytest.mark.asyncio
async def test_order_update_after_fill(exchange: SimulatedExchange, clock: ManualClock) -> None:
    ack = await exchange.place_order(request(price=D("100"), qty=D("0.25")))
    clock.advance(timedelta(seconds=2))
    await fill_at(exchange, "99.5")
    clock.advance(timedelta(seconds=30))

    update = await exchange.get_order(ref())

    assert update == OrderUpdate(
        client_order_id="grid1-buy-0001",
        exchange_order_id=ack.exchange_order_id,
        status=OrderStatus.FILLED,
        cum_filled_qty=D("0.25"),
        avg_fill_price=D("99.5"),
        reject_reason=None,
        exchange_ts=T0 + timedelta(seconds=2),
    )
    assert await exchange.get_open_orders(symbol="BTCUSDT") == ()


@pytest.mark.asyncio
async def test_filled_order_never_fills_twice(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100")))
    await fill_at(exchange, "90")

    assert await fill_at(exchange, "90") == []
    assert await fill_at(exchange, "80") == []
    update = await exchange.get_order(ref())
    assert update is not None
    assert update.avg_fill_price == D("90")


@pytest.mark.asyncio
async def test_canceled_order_never_fills(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100")))
    await exchange.cancel_order(ref())

    assert await fill_at(exchange, "50") == []
    assert await status_of(exchange, "grid1-buy-0001") is OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_filled_order_cannot_be_canceled(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100")))
    await fill_at(exchange, "100")
    before = await exchange.get_order(ref())

    with pytest.raises(ExchangeRejectedError, match="filled"):
        await exchange.cancel_order(ref())

    assert await exchange.get_order(ref()) == before


@pytest.mark.asyncio
async def test_other_symbols_are_untouched(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="btc", price=D("100")))
    await exchange.place_order(request(client_order_id="eth", symbol="ETHUSDT", price=D("100")))

    fills = await fill_at(exchange, "50", symbol="ETHUSDT")

    assert [f.client_order_id for f in fills] == ["eth"]
    assert await status_of(exchange, "btc") is OrderStatus.OPEN
    assert await status_of(exchange, "eth", symbol="ETHUSDT") is OrderStatus.FILLED


# --- batches ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_order_ids_and_single_timestamp(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    # Same created_at for "c" and "a": client_order_id decides.
    await exchange.place_order(request(client_order_id="c", price=D("100")))
    await exchange.place_order(request(client_order_id="a", side=Side.SELL, price=D("90")))
    clock.advance(timedelta(seconds=1))
    await exchange.place_order(request(client_order_id="0-later", price=D("96")))
    await exchange.place_order(request(client_order_id="not-crossed", price=D("94")))
    await exchange.place_order(request(client_order_id="sell-no", side=Side.SELL, price=D("96")))
    clock.advance(timedelta(seconds=1))

    fills = await fill_at(exchange, "95")

    assert [f.client_order_id for f in fills] == ["a", "c", "0-later"]
    assert [f.exec_id for f in fills] == [
        "SIM-EXEC-0000000001",
        "SIM-EXEC-0000000002",
        "SIM-EXEC-0000000003",
    ]
    batch_ts = T0 + timedelta(seconds=2)
    assert {f.exchange_ts for f in fills} == {batch_ts}
    for client_order_id in ("a", "c", "0-later"):
        update = await exchange.get_order(ref(client_order_id))
        assert update is not None
        assert update.exchange_ts == batch_ts
    open_ids = [u.client_order_id for u in await exchange.get_open_orders(symbol="BTCUSDT")]
    assert open_ids == ["not-crossed", "sell-no"]


@pytest.mark.asyncio
async def test_exec_sequence_continues_across_batches_and_is_independent_of_order_ids(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100")))
    await exchange.place_order(request(client_order_id="b", price=D("90")))
    await exchange.place_order(request(client_order_id="c", price=D("80")))

    first = await fill_at(exchange, "100")
    second = await fill_at(exchange, "85")

    assert [f.exec_id for f in first] == ["SIM-EXEC-0000000001"]
    assert [f.exec_id for f in second] == ["SIM-EXEC-0000000002"]
    assert [f.exchange_order_id for f in (*first, *second)] == ["SIM-0000000001", "SIM-0000000002"]


@pytest.mark.asyncio
async def test_calls_without_fills_do_not_consume_exec_ids(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="open", price=D("100")))
    await exchange.place_order(request(client_order_id="canceled", price=D("200")))
    await exchange.cancel_order(ref("canceled"))

    assert await fill_at(exchange, "150") == []  # crosses only the canceled order
    assert await fill_at(exchange, "101") == []  # crosses nothing
    with pytest.raises(ExchangeRequestValidationError):
        await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("0"))

    (fill,) = await fill_at(exchange, "100")
    assert fill.exec_id == "SIM-EXEC-0000000001"


# --- clock --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clock_read_once_per_batch_and_not_without_fills() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await exchange.place_order(request(client_order_id="a", price=D("100")))
    await exchange.place_order(request(client_order_id="b", price=D("100")))
    clock.calls = 0

    await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("101"))
    assert clock.calls == 0  # nothing crossed
    with pytest.raises(ExchangeRequestValidationError):
        await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("-1"))
    assert clock.calls == 0  # invalid input

    fills = await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("99"))
    assert len(fills) == 2
    assert clock.calls == 1


# --- validation ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        D("0"),
        D("-0"),
        D("-1"),
        D("NaN"),
        D("sNaN"),
        D("Infinity"),
        D("-Infinity"),
        1.5,
        100,
        True,
        "100",
        None,
    ],
)
async def test_invalid_execution_price_changes_nothing(
    exchange: SimulatedExchange, value: object
) -> None:
    await exchange.place_order(request(price=D("100")))

    with pytest.raises(ExchangeRequestValidationError, match="execution_price"):
        await exchange.fill_crossed_limit_orders(
            symbol="BTCUSDT",
            execution_price=value,  # type: ignore[arg-type]
        )

    assert await status_of(exchange, "grid1-buy-0001") is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_decimal_subclass_execution_price_rejected(exchange: SimulatedExchange) -> None:
    class Weird(Decimal):
        pass

    await exchange.place_order(request(price=D("100")))

    with pytest.raises(ExchangeRequestValidationError, match="execution_price"):
        await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=Weird("50"))

    assert await status_of(exchange, "grid1-buy-0001") is OrderStatus.OPEN


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["", " BTCUSDT", None, 1])
async def test_invalid_symbol_refused_like_get_open_orders(
    exchange: SimulatedExchange, symbol: object
) -> None:
    await exchange.place_order(request(price=D("100")))

    with pytest.raises(ExchangeRequestValidationError, match="symbol"):
        await exchange.fill_crossed_limit_orders(
            symbol=symbol,  # type: ignore[arg-type]
            execution_price=D("50"),
        )

    assert await status_of(exchange, "grid1-buy-0001") is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_execution_price_with_many_digits_is_kept_exactly(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request(price=D("100")))

    (fill,) = await fill_at(exchange, "99.123456789012345678901234567890123")

    assert fill.price == D("99.123456789012345678901234567890123")
    assert str(fill.price) == "99.123456789012345678901234567890123"


# --- atomicity ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_batch_preparation_leaves_no_partial_state(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100")))
    await exchange.place_order(request(client_order_id="b", price=D("100")))
    real_build = simulated._build_fill
    calls: list[str] = []

    def failing_second(record: Any, **kwargs: Any) -> Fill:
        calls.append(record.request.client_order_id)
        if len(calls) == 2:
            raise RuntimeError("injected failure while preparing the second fill")
        return real_build(record, **kwargs)

    monkeypatch.setattr(simulated, "_build_fill", failing_second)
    with pytest.raises(RuntimeError, match="injected"):
        await fill_at(exchange, "99")
    monkeypatch.setattr(simulated, "_build_fill", real_build)

    assert calls == ["a", "b"]
    assert await status_of(exchange, "a") is OrderStatus.OPEN
    assert await status_of(exchange, "b") is OrderStatus.OPEN
    assert len(await exchange.get_open_orders(symbol="BTCUSDT")) == 2

    fills = await fill_at(exchange, "99")
    assert [f.exec_id for f in fills] == ["SIM-EXEC-0000000001", "SIM-EXEC-0000000002"]


# --- idempotency after fill ---------------------------------------------------


@pytest.mark.asyncio
async def test_identical_retry_after_fill_returns_original_ack(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    original = await exchange.place_order(request(price=D("100")))
    clock.advance(timedelta(seconds=1))
    await fill_at(exchange, "100")
    filled = await exchange.get_order(ref())
    clock.advance(timedelta(seconds=1))

    again = await exchange.place_order(request(price=D("100")))

    assert again == original
    assert await exchange.get_order(ref()) == filled
    assert await fill_at(exchange, "1") == []
    with pytest.raises(ExchangeDuplicateOrderError):
        await exchange.place_order(request(price=D("101")))
    assert await exchange.get_order(ref()) == filled


# --- determinism --------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_fill_scenario_gives_identical_results() -> None:
    async def run() -> tuple[object, ...]:
        clock = ManualClock(T0)
        exchange = SimulatedExchange(clock=clock, instruments=SPECS)
        await exchange.place_order(request(client_order_id="b", price=D("100")))
        await exchange.place_order(request(client_order_id="a", side=Side.SELL, price=D("90")))
        clock.advance(timedelta(seconds=1))
        await exchange.place_order(request(client_order_id="c", price=D("80")))
        clock.advance(timedelta(seconds=1))
        first = await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("95"))
        clock.advance(timedelta(seconds=1))
        second = await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("79"))
        states = [await exchange.get_order(ref(c)) for c in ("a", "b", "c")]
        return first, second, states

    assert await run() == await run()


# === deterministic partial fills ==============================================


async def partial(
    exchange: SimulatedExchange, price: str, available: str, symbol: str = "BTCUSDT"
) -> list[Fill]:
    return list(
        await exchange.fill_crossed_limit_orders(
            symbol=symbol, execution_price=D(price), available_qty=D(available)
        )
    )


async def update_of(
    exchange: SimulatedExchange, client_order_id: str = "grid1-buy-0001"
) -> OrderUpdate:
    update = await exchange.get_order(ref(client_order_id))
    assert update is not None
    return update


# --- single order lifecycle ---------------------------------------------------


@pytest.mark.asyncio
async def test_partial_lifecycle_3_4_3(exchange: SimulatedExchange, clock: ManualClock) -> None:
    ack = await exchange.place_order(request(price=D("100"), qty=D("10")))

    clock.advance(timedelta(seconds=1))
    first = await partial(exchange, "100", "3")
    after_first = await update_of(exchange)
    clock.advance(timedelta(seconds=1))
    second = await partial(exchange, "100", "4")
    after_second = await update_of(exchange)
    clock.advance(timedelta(seconds=1))
    third = await partial(exchange, "100", "100")
    after_third = await update_of(exchange)

    assert [f.qty for f in (*first, *second, *third)] == [D("3"), D("4"), D("3")]
    assert [f.exec_id for f in (*first, *second, *third)] == [
        "SIM-EXEC-0000000001",
        "SIM-EXEC-0000000002",
        "SIM-EXEC-0000000003",
    ]
    assert after_first == OrderUpdate(
        client_order_id="grid1-buy-0001",
        exchange_order_id=ack.exchange_order_id,
        status=OrderStatus.PARTIALLY_FILLED,
        cum_filled_qty=D("3"),
        avg_fill_price=D("100"),
        reject_reason=None,
        exchange_ts=T0 + timedelta(seconds=1),
    )
    assert (after_second.status, after_second.cum_filled_qty) == (
        OrderStatus.PARTIALLY_FILLED,
        D("7"),
    )
    assert after_second.exchange_ts == T0 + timedelta(seconds=2)
    assert (after_third.status, after_third.cum_filled_qty) == (OrderStatus.FILLED, D("10"))
    assert after_third.exchange_ts == T0 + timedelta(seconds=3)
    assert await partial(exchange, "100", "100") == []


@pytest.mark.asyncio
async def test_fill_qty_is_per_execution_and_price_is_execution_price(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("10")))

    (a,) = await partial(exchange, "99", "2.5")
    (b,) = await partial(exchange, "98", "2.5")

    assert (a.qty, a.price) == (D("2.5"), D("99"))
    assert (b.qty, b.price) == (D("2.5"), D("98"))  # not the average
    assert (a.fee, a.fee_asset, a.is_maker) == (None, None, None)
    assert (b.fee, b.fee_asset, b.is_maker) == (None, None, None)


# --- weighted average ---------------------------------------------------------


@pytest.mark.asyncio
async def test_weighted_average_across_partial_fills(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("10")))

    await partial(exchange, "100", "3")  # 300
    assert (await update_of(exchange)).avg_fill_price == D("100")
    await partial(exchange, "95", "4")  # 380 -> 680 / 7
    mid = (await update_of(exchange)).avg_fill_price
    await partial(exchange, "90", "3")  # 270 -> 950 / 10
    final = await update_of(exchange)

    assert mid == D("97.14285714285714285714285714285714285714")  # 680/7, 40 digits
    assert final.avg_fill_price == D("95")  # exact: computed from the exact notional
    assert final.status is OrderStatus.FILLED


@pytest.mark.asyncio
async def test_average_is_exact_when_finite(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("4")))

    await partial(exchange, "99.5", "1")
    await partial(exchange, "99.25", "3")

    assert (await update_of(exchange)).avg_fill_price == D("99.3125")


@pytest.mark.asyncio
async def test_repeating_average_rounds_half_even_to_40_significant_digits(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("3")))

    await partial(exchange, "100", "1")
    await partial(exchange, "100", "1")
    await partial(exchange, "99", "1")  # 299 / 3 = 99.666...

    avg = (await update_of(exchange)).avg_fill_price
    assert avg == D("99.66666666666666666666666666666666666667")
    assert avg is not None
    assert len(avg.as_tuple().digits) == 40


@pytest.mark.asyncio
async def test_no_drift_from_chained_rounding(exchange: SimulatedExchange) -> None:
    # Each intermediate average is rounded, but the next one comes from the exact
    # notional: three thirds at 100, 100, 99 then 101 must give exactly 100.
    await exchange.place_order(request(price=D("101"), qty=D("4")))

    for price in ("100", "100", "99", "101"):
        await partial(exchange, price, "1")

    assert (await update_of(exchange)).avg_fill_price == D("100")


@pytest.mark.asyncio
async def test_global_decimal_context_does_not_change_results() -> None:
    async def run() -> tuple[object, ...]:
        exchange = SimulatedExchange(clock=ManualClock(T0), instruments=SPECS)
        await exchange.place_order(request(price=D("100"), qty=D("3")))
        fills = [
            *await partial(exchange, "100.123456789", "1"),
            *await partial(exchange, "99.987654321", "1.5"),
            *await partial(exchange, "99", "0.5"),
        ]
        return tuple(fills), await update_of(exchange)

    baseline = await run()
    with localcontext() as context:
        context.prec = 3
        context.rounding = "ROUND_DOWN"
        under_tiny_context = await run()

    assert under_tiny_context == baseline
    assert str(baseline[1].avg_fill_price) == str(under_tiny_context[1].avg_fill_price)  # type: ignore[attr-defined]


# --- multiple orders / allocation --------------------------------------------


@pytest.mark.asyncio
async def test_allocation_follows_created_at_then_client_id(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))

    fills = await partial(exchange, "100", "7")

    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("5")), ("b", D("2"))]
    assert (await update_of(exchange, "a")).status is OrderStatus.FILLED
    b = await update_of(exchange, "b")
    assert (b.status, b.cum_filled_qty) == (OrderStatus.PARTIALLY_FILLED, D("2"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("available", "expected"),
    [
        ("3", [("a", "3")]),  # less than the first remaining
        ("5", [("a", "5")]),  # exactly the first remaining
        ("9", [("a", "5"), ("b", "4")]),  # exactly the sum
        ("50", [("a", "5"), ("b", "4")]),  # more than the sum
    ],
)
async def test_available_qty_boundaries(
    exchange: SimulatedExchange, available: str, expected: list[tuple[str, str]]
) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("4")))

    fills = await partial(exchange, "100", available)

    assert [(f.client_order_id, f.qty) for f in fills] == [(c, D(q)) for c, q in expected]
    assert [f.exec_id for f in fills] == [f"SIM-EXEC-{i:010d}" for i in range(1, len(expected) + 1)]


@pytest.mark.asyncio
async def test_partially_filled_order_keeps_its_place_and_crossing_rules(
    exchange: SimulatedExchange,
) -> None:
    await exchange.place_order(request(client_order_id="buy", price=D("100"), qty=D("5")))
    await exchange.place_order(
        request(client_order_id="sell", side=Side.SELL, price=D("110"), qty=D("5"))
    )
    await partial(exchange, "100", "2")  # buy -> 2/5

    assert await partial(exchange, "105", "100") == []  # crosses neither
    fills = await partial(exchange, "110", "100")  # only the sell crosses

    assert [(f.client_order_id, f.qty) for f in fills] == [("sell", D("5"))]
    buy = await update_of(exchange, "buy")
    assert (buy.status, buy.cum_filled_qty) == (OrderStatus.PARTIALLY_FILLED, D("2"))


@pytest.mark.asyncio
async def test_unlimited_mode_fills_remaining_of_partial(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("10")))
    await partial(exchange, "100", "4")

    (fill,) = await fill_at(exchange, "90")  # available_qty=None

    assert fill.qty == D("6")
    final = await update_of(exchange)
    assert (final.status, final.cum_filled_qty, final.avg_fill_price) == (
        OrderStatus.FILLED,
        D("10"),
        D("94"),  # (400 + 540) / 10
    )


@pytest.mark.asyncio
async def test_explicit_none_keeps_full_fill_behavior(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("4")))

    fills = await exchange.fill_crossed_limit_orders(
        symbol="BTCUSDT", execution_price=D("100"), available_qty=None
    )

    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("5")), ("b", D("4"))]


# --- active orders / cancel ---------------------------------------------------


@pytest.mark.asyncio
async def test_partially_filled_is_open(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="p", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="o", price=D("90"), qty=D("5")))
    await exchange.place_order(request(client_order_id="f", price=D("100"), qty=D("1")))
    await partial(exchange, "100", "3")  # f? no: order is (created_at, id) -> f, p

    open_orders = await exchange.get_open_orders(symbol="BTCUSDT")

    assert [(u.client_order_id, u.status) for u in open_orders] == [
        ("o", OrderStatus.OPEN),
        ("p", OrderStatus.PARTIALLY_FILLED),
    ]


@pytest.mark.asyncio
async def test_cancel_partial_keeps_fill_history(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("10")))
    await partial(exchange, "99", "4")
    clock.advance(timedelta(seconds=5))

    await exchange.cancel_order(ref())
    canceled = await update_of(exchange)

    assert (canceled.status, canceled.cum_filled_qty, canceled.avg_fill_price) == (
        OrderStatus.CANCELED,
        D("4"),
        D("99"),
    )
    assert canceled.exchange_ts == T0 + timedelta(seconds=5)
    assert await partial(exchange, "50", "100") == []
    assert await fill_at(exchange, "50") == []
    assert await update_of(exchange) == canceled
    assert await exchange.get_open_orders(symbol="BTCUSDT") == ()
    with pytest.raises(ExchangeRejectedError, match="canceled"):
        await exchange.cancel_order(ref())


@pytest.mark.asyncio
async def test_identical_retry_after_partial_returns_original_ack(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    original = await exchange.place_order(request(price=D("100"), qty=D("10")))
    await partial(exchange, "100", "4")
    before = await update_of(exchange)
    clock.advance(timedelta(seconds=1))

    assert await exchange.place_order(request(price=D("100"), qty=D("10"))) == original
    assert await update_of(exchange) == before
    with pytest.raises(ExchangeDuplicateOrderError):
        await exchange.place_order(request(price=D("100"), qty=D("6")))
    assert await update_of(exchange) == before


# --- validation ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        D("0"),
        D("-0"),
        D("-1"),
        D("NaN"),
        D("sNaN"),
        D("Infinity"),
        D("-Infinity"),
        1.5,
        3,
        True,
        "3",
    ],
)
async def test_invalid_available_qty_changes_nothing(value: object) -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await exchange.place_order(request(price=D("100")))
    clock.calls = 0

    with pytest.raises(ExchangeRequestValidationError, match="available_qty"):
        await exchange.fill_crossed_limit_orders(
            symbol="BTCUSDT",
            execution_price=D("100"),
            available_qty=value,  # type: ignore[arg-type]
        )

    assert clock.calls == 0
    assert await status_of(exchange, "grid1-buy-0001") is OrderStatus.OPEN
    (fill,) = await fill_at(exchange, "100")
    assert fill.exec_id == "SIM-EXEC-0000000001"


@pytest.mark.asyncio
async def test_decimal_subclass_available_qty_rejected(exchange: SimulatedExchange) -> None:
    class Weird(Decimal):
        pass

    await exchange.place_order(request(price=D("100")))

    with pytest.raises(ExchangeRequestValidationError, match="available_qty"):
        await exchange.fill_crossed_limit_orders(
            symbol="BTCUSDT", execution_price=D("100"), available_qty=Weird("1")
        )


# --- clock / atomicity / invariants -------------------------------------------


@pytest.mark.asyncio
async def test_one_clock_read_per_partial_batch() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
    clock.calls = 0

    fills = await exchange.fill_crossed_limit_orders(
        symbol="BTCUSDT", execution_price=D("100"), available_qty=D("7")
    )
    assert clock.calls == 1
    assert len({f.exchange_ts for f in fills}) == 1
    a = await update_of(exchange, "a")
    b = await update_of(exchange, "b")
    assert a.exchange_ts == b.exchange_ts == fills[0].exchange_ts

    await exchange.fill_crossed_limit_orders(
        symbol="BTCUSDT", execution_price=D("101"), available_qty=D("7")
    )
    assert clock.calls == 1  # nothing crossed


@pytest.mark.asyncio
async def test_failed_partial_batch_preparation_leaves_no_state(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="c", price=D("100"), qty=D("5")))
    await partial(exchange, "100", "2")  # a -> 2/5, uses SIM-EXEC-1
    before = [await update_of(exchange, c) for c in ("a", "b", "c")]
    real_build = simulated._build_fill
    calls: list[str] = []

    def failing_third(record: Any, **kwargs: Any) -> Fill:
        calls.append(record.request.client_order_id)
        if len(calls) == 3:
            raise RuntimeError("injected failure while preparing the third fill")
        return real_build(record, **kwargs)

    monkeypatch.setattr(simulated, "_build_fill", failing_third)
    with pytest.raises(RuntimeError, match="injected"):
        await partial(exchange, "100", "11")
    monkeypatch.setattr(simulated, "_build_fill", real_build)

    assert calls == ["a", "b", "c"]
    assert [await update_of(exchange, c) for c in ("a", "b", "c")] == before
    fills = await partial(exchange, "100", "11")
    assert [(f.client_order_id, f.qty, f.exec_id) for f in fills] == [
        ("a", D("3"), "SIM-EXEC-0000000002"),
        ("b", D("5"), "SIM-EXEC-0000000003"),
        ("c", D("3"), "SIM-EXEC-0000000004"),
    ]


@pytest.mark.asyncio
async def test_corrupted_internal_state_fails_before_commit(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
    await partial(exchange, "100", "2")  # a -> PARTIALLY_FILLED 2/5
    # Corrupt "a": partially filled with nothing remaining.
    object.__setattr__(exchange._orders["a"], "filled_qty", D("5"))
    b_before = await update_of(exchange, "b")

    with pytest.raises(RuntimeError, match="invariant"):
        await partial(exchange, "100", "100")

    assert await update_of(exchange, "b") == b_before
    assert exchange._orders["a"].filled_qty == D("5")  # not silently repaired


def test_record_invariants_reject_inconsistent_states() -> None:
    order = request(price=D("100"), qty=D("5"))
    base: dict[str, Any] = {
        "request": order,
        "exchange_order_id": "SIM-1",
        "created_at": T0,
        "updated_at": T0,
    }
    bad = [
        (OrderStatus.OPEN, D("1"), D("100"), D("100")),
        (OrderStatus.OPEN, D("0"), D("100"), D("0")),
        (OrderStatus.PARTIALLY_FILLED, D("0"), None, D("0")),
        (OrderStatus.PARTIALLY_FILLED, D("5"), D("100"), D("500")),
        (OrderStatus.PARTIALLY_FILLED, D("2"), None, D("200")),
        (OrderStatus.FILLED, D("4"), D("100"), D("400")),
        (OrderStatus.FILLED, D("5"), None, D("500")),
        (OrderStatus.CANCELED, D("5"), D("100"), D("500")),
        (OrderStatus.CANCELED, D("2"), None, D("200")),
        (OrderStatus.CANCELED, D("0"), D("100"), D("0")),
        (OrderStatus.PARTIALLY_FILLED, D("2"), D("100"), D("0")),
    ]
    for status, filled, avg, notional in bad:
        with pytest.raises(RuntimeError, match="invariant"):
            simulated._SimulatedOrder(
                **base,
                status=status,
                filled_qty=filled,
                avg_fill_price=avg,
                filled_notional=notional,
            )
    for status, filled, avg, notional in [
        (OrderStatus.OPEN, D("0"), None, D("0")),
        (OrderStatus.PARTIALLY_FILLED, D("2"), D("100"), D("200")),
        (OrderStatus.FILLED, D("5"), D("100"), D("500")),
        (OrderStatus.CANCELED, D("0"), None, D("0")),
        (OrderStatus.CANCELED, D("2"), D("100"), D("200")),
    ]:
        simulated._SimulatedOrder(
            **base, status=status, filled_qty=filled, avg_fill_price=avg, filled_notional=notional
        )


# --- determinism --------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_partial_scenario_gives_identical_results() -> None:
    async def run() -> tuple[object, ...]:
        clock = ManualClock(T0)
        exchange = SimulatedExchange(clock=clock, instruments=SPECS)
        await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
        await exchange.place_order(
            request(client_order_id="a", side=Side.SELL, price=D("90"), qty=D("3"))
        )
        clock.advance(timedelta(seconds=1))
        first = await partial(exchange, "95", "4")
        clock.advance(timedelta(seconds=1))
        second = await partial(exchange, "93.3", "2.25")
        clock.advance(timedelta(seconds=1))
        third = await fill_at(exchange, "91")
        states = [await exchange.get_order(ref(c)) for c in ("a", "b")]
        return first, second, third, states

    assert await run() == await run()


# === instrument specs: pre-trade validation ===================================

STRICT = spec(
    "BTCUSDT",
    tick_size=D("0.10"),
    qty_step=D("0.1"),
    min_qty=D("0.2"),
    max_qty=D("50"),
    min_notional=D("5"),
)


def strict_exchange(clock: Any = None) -> SimulatedExchange:
    return SimulatedExchange(clock=clock or ManualClock(T0), instruments=(STRICT,))


def strict_order(**overrides: Any) -> OrderRequest:
    return request(**{"price": D("100.00"), "qty": D("1.0"), **overrides})


# --- constructor --------------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_accepts_several_instruments(clock: ManualClock) -> None:
    exchange = SimulatedExchange(clock=clock, instruments=(spec("BTCUSDT"), spec("ETHUSDT")))

    await exchange.place_order(request(client_order_id="b"))
    await exchange.place_order(request(client_order_id="e", symbol="ETHUSDT"))

    assert repr(exchange) == "SimulatedExchange(orders=2)"


def test_constructor_rejects_duplicate_symbols(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="duplicate instrument BTCUSDT"):
        SimulatedExchange(
            clock=clock, instruments=(spec("BTCUSDT"), spec("BTCUSDT", tick_size=D("1")))
        )


@pytest.mark.parametrize("bad", [("BTCUSDT",), (None,), ({"symbol": "BTCUSDT"},)])
def test_constructor_rejects_non_specs(clock: ManualClock, bad: tuple[Any, ...]) -> None:
    with pytest.raises(TypeError, match="InstrumentSpec"):
        SimulatedExchange(clock=clock, instruments=bad)


@pytest.mark.asyncio
async def test_caller_collection_mutation_does_not_change_registry(clock: ManualClock) -> None:
    specs = [spec("BTCUSDT")]
    exchange = SimulatedExchange(clock=clock, instruments=specs)  # type: ignore[arg-type]
    specs.clear()
    specs.append(spec("ETHUSDT"))

    await exchange.place_order(request())
    with pytest.raises(ExchangeRejectedError, match="unknown instrument"):
        await exchange.place_order(request(client_order_id="e", symbol="ETHUSDT"))


@pytest.mark.asyncio
async def test_empty_registry_constructs_but_place_fails_closed(clock: ManualClock) -> None:
    exchange = SimulatedExchange(clock=clock)

    with pytest.raises(ExchangeRejectedError, match="unknown instrument BTCUSDT"):
        await exchange.place_order(request())

    assert repr(exchange) == "SimulatedExchange(orders=0)"


# --- place_order rules --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "accepted"),
    [
        ({"price": D("100.00")}, True),
        ({"price": D("100.10")}, True),
        ({"price": D("100.1")}, True),
        ({"price": D("100.05")}, False),  # tick misaligned
        ({"qty": D("1.0")}, True),
        ({"qty": D("1.05")}, False),  # step misaligned
        ({"qty": D("0.2")}, True),  # exact min qty
        ({"qty": D("0.1")}, False),  # below min qty
        ({"qty": D("50")}, True),  # exact max qty
        ({"qty": D("50.1")}, False),  # above max qty
        ({"price": D("25.00"), "qty": D("0.2")}, True),  # notional exactly 5
        ({"price": D("24.90"), "qty": D("0.2")}, False),  # notional 4.98
        ({"time_in_force": TimeInForce.POST_ONLY}, True),
    ],
)
async def test_instrument_rules(overrides: dict[str, Any], accepted: bool) -> None:
    exchange = strict_exchange()
    order = strict_order(**overrides)

    if accepted:
        ack = await exchange.place_order(order)
        assert ack.exchange_order_id == "SIM-0000000001"
    else:
        with pytest.raises(ExchangeRejectedError):
            await exchange.place_order(order)
        assert await exchange.get_order(ref()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"symbol": "ETHUSDT"}, "unknown instrument ETHUSDT"),
        ({"price": D("100.05")}, "tick_size"),
        ({"qty": D("1.05")}, "qty_step"),
        ({"qty": D("0.1")}, "min_qty"),
        ({"qty": D("50.1")}, "max_qty"),
        ({"price": D("24.90"), "qty": D("0.2")}, "min_notional"),
    ],
)
async def test_rejection_is_side_effect_free(overrides: dict[str, Any], reason: str) -> None:
    clock = CountingClock(T0)
    exchange = strict_exchange(clock)

    with pytest.raises(ExchangeRejectedError, match=reason) as excinfo:
        await exchange.place_order(strict_order(**overrides))

    assert not isinstance(excinfo.value, ExchangeRequestValidationError)
    assert clock.calls == 0
    assert repr(exchange) == "SimulatedExchange(orders=0)"
    ack = await exchange.place_order(strict_order(client_order_id="valid"))
    assert ack.exchange_order_id == "SIM-0000000001"  # no id consumed


@pytest.mark.asyncio
async def test_rules_ignore_the_global_decimal_context() -> None:
    cases: list[tuple[dict[str, Any], bool]] = [
        ({"price": D("100.10"), "qty": D("1.2")}, True),
        ({"price": D("100.05")}, False),
        ({"qty": D("1.05")}, False),
        ({"qty": D("50")}, True),
        ({"qty": D("50.1")}, False),
        ({"price": D("25.00"), "qty": D("0.2")}, True),
        ({"price": D("24.90"), "qty": D("0.2")}, False),
        ({"price": D("12345.60"), "qty": D("49.9")}, True),
    ]

    async def outcomes() -> list[bool]:
        results = []
        for i, (overrides, _) in enumerate(cases):
            exchange = strict_exchange()
            try:
                await exchange.place_order(strict_order(client_order_id=f"c{i}", **overrides))
                results.append(True)
            except ExchangeRejectedError:
                results.append(False)
        return results

    baseline = await outcomes()
    with localcontext() as context:
        context.prec = 2
        context.rounding = "ROUND_UP"
        low_precision = await outcomes()

    assert baseline == [accepted for _, accepted in cases]
    assert low_precision == baseline


# --- idempotency ordering -----------------------------------------------------


@pytest.mark.asyncio
async def test_identical_retry_returns_ack_before_any_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = strict_exchange()
    original = await exchange.place_order(strict_order())

    def must_not_run(*_: Any, **__: Any) -> None:
        raise AssertionError("instrument rules must not run for an existing client id")

    monkeypatch.setattr(simulated, "_check_instrument_rules", must_not_run)

    assert await exchange.place_order(strict_order()) == original
    with pytest.raises(ExchangeDuplicateOrderError):
        await exchange.place_order(strict_order(price=D("100.05")))  # changed + invalid


@pytest.mark.asyncio
async def test_rejected_client_id_can_be_reused_by_a_valid_request() -> None:
    # A rejected order was never created, so its client id is not taken.
    exchange = strict_exchange()
    with pytest.raises(ExchangeRejectedError):
        await exchange.place_order(strict_order(price=D("100.05")))

    ack = await exchange.place_order(strict_order())

    assert ack.client_order_id == "grid1-buy-0001"
    assert ack.exchange_order_id == "SIM-0000000001"


# --- fills respect qty_step -----------------------------------------------------


async def strict_partial(exchange: SimulatedExchange, available: str) -> list[Fill]:
    return list(
        await exchange.fill_crossed_limit_orders(
            symbol="BTCUSDT", execution_price=D("100"), available_qty=D(available)
        )
    )


@pytest.mark.asyncio
async def test_fill_qty_rounds_down_to_step() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order())

    (fill,) = await strict_partial(exchange, "0.15")

    assert fill.qty == D("0.1")
    update = await update_of(exchange)
    assert (update.status, update.cum_filled_qty) == (OrderStatus.PARTIALLY_FILLED, D("0.1"))


@pytest.mark.asyncio
async def test_budget_below_step_creates_no_fill_and_no_exec_id() -> None:
    clock = CountingClock(T0)
    exchange = strict_exchange(clock)
    await exchange.place_order(strict_order())
    clock.calls = 0

    assert await strict_partial(exchange, "0.09") == []

    assert clock.calls == 0
    assert (await update_of(exchange)).status is OrderStatus.OPEN
    (fill,) = await strict_partial(exchange, "0.1")
    assert fill.exec_id == "SIM-EXEC-0000000001"


@pytest.mark.asyncio
async def test_budget_across_orders_yields_only_step_multiples() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order(client_order_id="a", qty=D("0.2")))
    await exchange.place_order(strict_order(client_order_id="b", qty=D("1.0")))
    await exchange.place_order(strict_order(client_order_id="c", qty=D("1.0")))

    fills = await strict_partial(exchange, "0.25")  # a: 0.2, b: 0.05 -> 0

    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("0.2"))]
    assert [f.exec_id for f in fills] == ["SIM-EXEC-0000000001"]
    fills = await strict_partial(exchange, "0.35")  # b: 0.3, then 0.05 unused
    assert [(f.client_order_id, f.qty) for f in fills] == [("b", D("0.3"))]


@pytest.mark.asyncio
async def test_final_remainder_closes_exactly_and_stays_aligned() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order(qty=D("1.0")))

    quantities = []
    for available in ("0.15", "0.37", "0.29", "5"):
        quantities += [f.qty for f in await strict_partial(exchange, available)]
        cum = (await update_of(exchange)).cum_filled_qty
        assert is_multiple(cum, D("0.1"))

    assert quantities == [D("0.1"), D("0.3"), D("0.2"), D("0.4")]
    final = await update_of(exchange)
    assert (final.status, final.cum_filled_qty) == (OrderStatus.FILLED, D("1.0"))


def is_multiple(value: Decimal, step: Decimal) -> bool:
    return value % step == 0


@pytest.mark.asyncio
async def test_unlimited_fill_after_partial_closes_remainder() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order(qty=D("1.0")))
    await strict_partial(exchange, "0.45")  # 0.4

    (fill,) = await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("100"))

    assert fill.qty == D("0.6")
    assert (await update_of(exchange)).status is OrderStatus.FILLED


@pytest.mark.asyncio
async def test_execution_price_is_not_tick_enforced() -> None:
    # The external execution price is a simulation input, not a new order price.
    exchange = strict_exchange()
    await exchange.place_order(strict_order())

    (fill,) = await exchange.fill_crossed_limit_orders(
        symbol="BTCUSDT", execution_price=D("99.987")
    )

    assert fill.price == D("99.987")


# --- invariants -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_misaligned_remaining_qty_is_an_invariant_failure() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order(client_order_id="a", qty=D("1.0")))
    await exchange.place_order(strict_order(client_order_id="b", qty=D("1.0")))
    await strict_partial(exchange, "0.1")  # a -> 0.1 / 1.0
    object.__setattr__(exchange._orders["a"], "filled_qty", D("0.15"))
    b_before = await update_of(exchange, "b")

    with pytest.raises(RuntimeError, match="invariant"):
        await strict_partial(exchange, "5")

    assert await update_of(exchange, "b") == b_before
    assert exchange._orders["a"].filled_qty == D("0.15")  # not repaired


@pytest.mark.asyncio
async def test_order_without_registered_instrument_is_an_invariant_failure() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order())
    exchange._instruments.clear()  # simulate corrupted simulator state

    with pytest.raises(RuntimeError, match="invariant"):
        await strict_partial(exchange, "5")

    assert (await update_of(exchange)).status is OrderStatus.OPEN


# === position accounting integration ==========================================


async def position_of(exchange: SimulatedExchange, symbol: str = "BTCUSDT") -> Any:
    return await exchange.get_position(symbol=symbol)


@pytest.mark.asyncio
async def test_no_position_before_fills(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100")))

    assert await position_of(exchange) is None
    assert await position_of(exchange, "UNKNOWNUSDT") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["", " BTCUSDT", None, 1])
async def test_get_position_malformed_symbol(exchange: SimulatedExchange, symbol: object) -> None:
    with pytest.raises(ExchangeRequestValidationError, match="symbol"):
        await exchange.get_position(symbol=symbol)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_partial_fills_move_position_by_fill_qty(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    await exchange.place_order(request(price=D("100"), qty=D("10")))
    clock.advance(timedelta(seconds=1))
    await partial(exchange, "100", "3")
    first = await position_of(exchange)
    clock.advance(timedelta(seconds=1))
    await partial(exchange, "94", "3")
    second = await position_of(exchange)
    clock.advance(timedelta(seconds=1))
    await fill_at(exchange, "91")  # remaining 4
    full = await position_of(exchange)

    assert (first.qty, first.entry_price, first.updated_at) == (
        D("3"),
        D("100"),
        T0 + timedelta(seconds=1),
    )
    assert (second.qty, second.entry_price) == (D("6"), D("97"))
    assert (full.qty, full.entry_price, full.realized_pnl) == (D("10"), D("94.6"), D("0"))
    assert full.updated_at == T0 + timedelta(seconds=3)
    assert (full.mark_price, full.unrealized_pnl) == (None, None)


@pytest.mark.asyncio
async def test_sell_fills_close_and_reverse(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("10")))
    await fill_at(exchange, "100")
    await exchange.place_order(
        request(client_order_id="s1", side=Side.SELL, price=D("110"), qty=D("4"))
    )
    await exchange.place_order(
        request(client_order_id="s2", side=Side.SELL, price=D("120"), qty=D("6"))
    )
    await exchange.place_order(
        request(client_order_id="s3", side=Side.SELL, price=D("130"), qty=D("5"))
    )

    await fill_at(exchange, "110")  # s1: partial close +40
    partial_close = await position_of(exchange)
    await fill_at(exchange, "120")  # s2: exact close +120
    flat = await position_of(exchange)
    await fill_at(exchange, "130")  # s3: opens short
    short = await position_of(exchange)

    assert (partial_close.qty, partial_close.entry_price, partial_close.realized_pnl) == (
        D("6"),
        D("100"),
        D("40"),
    )
    assert (flat.qty, flat.entry_price, flat.realized_pnl, flat.unrealized_pnl) == (
        D("0"),
        None,
        D("160"),
        D("0"),
    )
    assert (short.qty, short.entry_price, short.realized_pnl) == (D("-5"), D("130"), D("160"))


@pytest.mark.asyncio
async def test_reversal_within_one_order(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("10")))
    await fill_at(exchange, "100")
    await exchange.place_order(
        request(client_order_id="s", side=Side.SELL, price=D("110"), qty=D("15"))
    )

    await fill_at(exchange, "110")

    p = await position_of(exchange)
    assert (p.qty, p.entry_price, p.realized_pnl) == (D("-5"), D("110"), D("100"))


@pytest.mark.asyncio
async def test_batch_fills_apply_in_batch_order(exchange: SimulatedExchange) -> None:
    # Same created_at: "a" (SELL 8) is processed before "b" (BUY 5).
    await exchange.place_order(
        request(client_order_id="a", side=Side.SELL, price=D("90"), qty=D("8"))
    )
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))

    fills = await fill_at(exchange, "95")

    assert [f.client_order_id for f in fills] == ["a", "b"]
    p = await position_of(exchange)
    # SELL 8 @95 opens short; BUY 5 @95 closes 5 at entry -> realized 0.
    assert (p.qty, p.entry_price, p.realized_pnl) == (D("-3"), D("95"), D("0"))


@pytest.mark.asyncio
async def test_non_fills_do_not_touch_positions(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(client_order_id="canceled", price=D("100")))
    await exchange.cancel_order(ref("canceled"))
    await exchange.place_order(request(client_order_id="far", price=D("50")))
    with pytest.raises(ExchangeRejectedError):
        await exchange.place_order(request(client_order_id="bad", reduce_only=True))

    await fill_at(exchange, "90")  # crosses only the canceled order

    assert await position_of(exchange) is None


@pytest.mark.asyncio
async def test_budget_below_step_does_not_touch_positions() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order())

    await strict_partial(exchange, "0.05")

    assert await position_of(exchange) is None


@pytest.mark.asyncio
async def test_get_position_does_not_read_the_clock() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await exchange.place_order(request(price=D("100")))
    await fill_at(exchange, "100")
    clock.calls = 0

    p1 = await position_of(exchange)
    p2 = await position_of(exchange)
    await position_of(exchange, "ETHUSDT")

    assert clock.calls == 0
    assert p1 == p2


@pytest.mark.asyncio
async def test_failed_fill_preparation_leaves_positions_unchanged(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
    await partial(exchange, "100", "2")
    before = (await position_of(exchange), [await update_of(exchange, c) for c in "ab"])
    real_build = simulated._build_fill
    calls: list[str] = []

    def failing_second(record: Any, **kwargs: Any) -> Fill:
        calls.append(record.request.client_order_id)
        if len(calls) == 2:
            raise RuntimeError("injected fill failure")
        return real_build(record, **kwargs)

    monkeypatch.setattr(simulated, "_build_fill", failing_second)
    with pytest.raises(RuntimeError, match="injected"):
        await fill_at(exchange, "100")
    monkeypatch.undo()

    assert (await position_of(exchange), [await update_of(exchange, c) for c in "ab"]) == before
    fills = await fill_at(exchange, "100")
    assert [f.exec_id for f in fills] == ["SIM-EXEC-0000000002", "SIM-EXEC-0000000003"]


@pytest.mark.asyncio
async def test_failed_position_preparation_leaves_everything_unchanged(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.exchanges import simulated_positions

    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("5")))
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("5")))
    await partial(exchange, "100", "2")
    before = (await position_of(exchange), [await update_of(exchange, c) for c in "ab"])
    real_apply = simulated_positions._apply_fill
    calls: list[str] = []

    def failing_second(state: Any, fill: Fill) -> Any:
        calls.append(fill.exec_id)
        if len(calls) == 2:
            raise simulated_positions.PositionAccountingError("injected position failure")
        return real_apply(state, fill)

    monkeypatch.setattr(simulated_positions, "_apply_fill", failing_second)
    with pytest.raises(simulated_positions.PositionAccountingError, match="injected"):
        await fill_at(exchange, "100")
    monkeypatch.undo()

    assert calls == ["SIM-EXEC-0000000002", "SIM-EXEC-0000000003"]
    assert (await position_of(exchange), [await update_of(exchange, c) for c in "ab"]) == before
    fills = await fill_at(exchange, "100")
    assert [f.exec_id for f in fills] == ["SIM-EXEC-0000000002", "SIM-EXEC-0000000003"]
    p = await position_of(exchange)
    assert (p.qty, p.entry_price) == (D("10"), D("100"))


# === reduce-only limit orders =================================================


async def open_position(
    exchange: SimulatedExchange, side: Side, qty: str, price: str = "100", cid: str = "open"
) -> None:
    await exchange.place_order(request(client_order_id=cid, side=side, price=D(price), qty=D(qty)))
    await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D(price))


def ro(cid: str, side: Side, qty: str, price: str, **overrides: Any) -> OrderRequest:
    return request(
        client_order_id=cid, side=side, qty=D(qty), price=D(price), reduce_only=True, **overrides
    )


async def qty_of(exchange: SimulatedExchange) -> Decimal:
    p = await position_of(exchange)
    return D("0") if p is None else p.qty


# --- placement ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("position_side", "ro_side", "accepted"),
    [
        (Side.BUY, Side.SELL, True),  # long reduced by sell
        (Side.BUY, Side.BUY, False),  # would increase long
        (Side.SELL, Side.BUY, True),  # short reduced by buy
        (Side.SELL, Side.SELL, False),  # would increase short
    ],
)
async def test_reduce_only_placement_direction(
    exchange: SimulatedExchange, position_side: Side, ro_side: Side, accepted: bool
) -> None:
    await open_position(exchange, position_side, "5")
    order = ro("ro", ro_side, "2", "100")

    if accepted:
        ack = await exchange.place_order(order)
        assert ack.exchange_order_id == "SIM-0000000002"
    else:
        with pytest.raises(ExchangeRejectedError, match="reduce_only"):
            await exchange.place_order(order)
        assert await exchange.get_order(ref("ro")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
async def test_reduce_only_without_position_is_rejected(
    exchange: SimulatedExchange, side: Side
) -> None:
    with pytest.raises(ExchangeRejectedError, match="reduce_only"):
        await exchange.place_order(ro("ro", side, "1", "100"))


@pytest.mark.asyncio
async def test_reduce_only_on_flat_position_is_rejected(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("5"), price=D("100"))
    )
    await fill_at(exchange, "100")
    assert (await position_of(exchange)).qty == 0

    with pytest.raises(ExchangeRejectedError, match="reduce_only"):
        await exchange.place_order(ro("ro", Side.SELL, "1", "100"))


@pytest.mark.asyncio
async def test_oversized_reduce_only_is_accepted_at_placement(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "5")

    await exchange.place_order(ro("ro", Side.SELL, "10", "110"))

    assert (await update_of(exchange, "ro")).status is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_reduce_only_rejection_is_side_effect_free() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await open_position(exchange, Side.BUY, "5")
    clock.calls = 0

    with pytest.raises(ExchangeRejectedError, match="reduce_only"):
        await exchange.place_order(ro("ro", Side.BUY, "1", "100"))

    assert clock.calls == 0
    assert repr(exchange) == "SimulatedExchange(orders=1)"
    ack = await exchange.place_order(request(client_order_id="next", price=D("90")))
    assert ack.exchange_order_id == "SIM-0000000002"


@pytest.mark.asyncio
async def test_reduce_only_still_passes_instrument_rules() -> None:
    exchange = strict_exchange()
    await exchange.place_order(strict_order(client_order_id="open", qty=D("5")))
    await exchange.fill_crossed_limit_orders(symbol="BTCUSDT", execution_price=D("100"))

    with pytest.raises(ExchangeRejectedError, match="tick_size"):
        await exchange.place_order(
            strict_order(client_order_id="ro", side=Side.SELL, price=D("100.05"), reduce_only=True)
        )
    with pytest.raises(ExchangeRejectedError, match="min_qty"):
        await exchange.place_order(
            strict_order(client_order_id="ro", side=Side.SELL, qty=D("0.1"), reduce_only=True)
        )


@pytest.mark.asyncio
async def test_reduce_only_retry_before_position_validation(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "5")
    original = await exchange.place_order(ro("ro", Side.SELL, "5", "110"))
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("5"), price=D("105"))
    )
    await fill_at(exchange, "105")  # flat; "ro" (limit 110) not crossed
    assert (await position_of(exchange)).qty == 0

    assert await exchange.place_order(ro("ro", Side.SELL, "5", "110")) == original
    with pytest.raises(ExchangeDuplicateOrderError):
        await exchange.place_order(ro("ro", Side.SELL, "4", "110"))


# --- execution --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reduce_only_partial_lifecycle(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("ro", Side.SELL, "7", "110"))

    (f1,) = await partial(exchange, "110", "3")
    after1 = await update_of(exchange, "ro")
    assert (f1.qty, after1.status, await qty_of(exchange)) == (
        D("3"),
        OrderStatus.PARTIALLY_FILLED,
        D("7"),
    )
    assert [u.client_order_id for u in await exchange.get_open_orders(symbol="BTCUSDT")] == ["ro"]

    (f2,) = await partial(exchange, "110", "2")
    after2 = await update_of(exchange, "ro")
    assert (f2.qty, after2.status, await qty_of(exchange)) == (
        D("2"),
        OrderStatus.PARTIALLY_FILLED,
        D("5"),
    )

    (f3,) = await partial(exchange, "110", "10")
    after3 = await update_of(exchange, "ro")
    assert (f3.qty, after3.status, after3.cum_filled_qty, await qty_of(exchange)) == (
        D("2"),
        OrderStatus.FILLED,
        D("7"),
        D("3"),
    )


@pytest.mark.asyncio
async def test_reduce_only_exact_close(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.SELL, "5")
    await exchange.place_order(ro("ro", Side.BUY, "5", "90"))

    (fill,) = await fill_at(exchange, "90")

    assert fill.qty == D("5")
    assert (await update_of(exchange, "ro")).status is OrderStatus.FILLED
    p = await position_of(exchange)
    assert (p.qty, p.realized_pnl) == (D("0"), D("50"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("position_side", "ro_side", "price"),
    [(Side.BUY, Side.SELL, "110"), (Side.SELL, Side.BUY, "90")],
)
async def test_oversized_reduce_only_never_reverses(
    exchange: SimulatedExchange, clock: ManualClock, position_side: Side, ro_side: Side, price: str
) -> None:
    await open_position(exchange, position_side, "5")
    await exchange.place_order(ro("ro", ro_side, "10", price))
    clock.advance(timedelta(seconds=1))

    fills = await exchange.fill_crossed_limit_orders(
        symbol="BTCUSDT", execution_price=D(price), available_qty=D("10")
    )

    assert [(f.client_order_id, f.qty) for f in fills] == [("ro", D("5"))]
    assert await qty_of(exchange) == D("0")  # flat, never the other side
    update = await update_of(exchange, "ro")
    assert update == OrderUpdate(
        client_order_id="ro",
        exchange_order_id="SIM-0000000002",
        status=OrderStatus.CANCELED,
        cum_filled_qty=D("5"),
        avg_fill_price=D(price),
        reject_reason=None,
        exchange_ts=T0 + timedelta(seconds=1),
    )
    assert await exchange.get_open_orders(symbol="BTCUSDT") == ()


@pytest.mark.asyncio
async def test_two_oversized_reduce_only_orders_never_over_reduce(
    exchange: SimulatedExchange,
) -> None:
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "7", "110"))
    await exchange.place_order(ro("b", Side.SELL, "7", "110"))

    fills = await fill_at(exchange, "110")

    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("7")), ("b", D("3"))]
    assert sum(f.qty for f in fills) == D("10")
    assert await qty_of(exchange) == D("0")
    a, b = await update_of(exchange, "a"), await update_of(exchange, "b")
    assert (a.status, a.cum_filled_qty) == (OrderStatus.FILLED, D("7"))
    assert (b.status, b.cum_filled_qty) == (OrderStatus.CANCELED, D("3"))


@pytest.mark.asyncio
async def test_reduce_only_on_flat_at_execution_is_canceled_without_fill() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "5", "110"))
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("5"), price=D("105"))
    )
    await fill_at(exchange, "105")
    clock.inner.advance(timedelta(seconds=9))
    clock.calls = 0

    fills = await fill_at(exchange, "110")

    assert fills == []
    assert clock.calls == 1  # an auto-cancel is a state change
    update = await update_of(exchange, "ro")
    assert (update.status, update.cum_filled_qty, update.avg_fill_price) == (
        OrderStatus.CANCELED,
        D("0"),
        None,
    )
    assert update.exchange_ts == T0 + timedelta(seconds=9)
    assert await exchange.get_open_orders(symbol="BTCUSDT") == ()
    await exchange.place_order(request(client_order_id="n", price=D("100")))
    (fill,) = await fill_at(exchange, "100")
    assert fill.exec_id == "SIM-EXEC-0000000003"  # no exec id spent on the cancel


@pytest.mark.asyncio
async def test_reduce_only_after_reversal_is_canceled_without_fill(
    exchange: SimulatedExchange,
) -> None:
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "5", "110"))
    await exchange.place_order(
        request(client_order_id="rev", side=Side.SELL, qty=D("8"), price=D("105"))
    )
    await fill_at(exchange, "105")
    assert await qty_of(exchange) == D("-3")

    assert await fill_at(exchange, "110") == []

    assert (await update_of(exchange, "ro")).status is OrderStatus.CANCELED
    assert await qty_of(exchange) == D("-3")  # the new short is not reduced by a sell


@pytest.mark.asyncio
async def test_auto_cancel_does_not_consume_liquidity(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "5", "100"))
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("5"), price=D("95"))
    )
    await fill_at(exchange, "97")  # only "close" crosses -> flat
    await exchange.place_order(
        request(client_order_id="z-buy", side=Side.BUY, qty=D("5"), price=D("105"))
    )

    fills = await partial(exchange, "100", "2")  # "ro" first (older), then "z-buy"

    assert [(f.client_order_id, f.qty, f.exec_id) for f in fills] == [
        ("z-buy", D("2"), "SIM-EXEC-0000000003")
    ]
    assert (await update_of(exchange, "ro")).status is OrderStatus.CANCELED
    assert await qty_of(exchange) == D("2")


@pytest.mark.asyncio
async def test_normal_then_reduce_only_in_one_batch(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="a", side=Side.SELL, qty=D("6"), price=D("100"))
    )
    await exchange.place_order(ro("b", Side.SELL, "6", "100"))

    fills = await fill_at(exchange, "100")

    # "b" sees the position after "a": only 4 left to reduce.
    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("6")), ("b", D("4"))]
    assert await qty_of(exchange) == D("0")
    assert (await update_of(exchange, "b")).status is OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_reduce_only_then_normal_in_one_batch(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "6", "100"))
    await exchange.place_order(
        request(client_order_id="b", side=Side.SELL, qty=D("6"), price=D("100"))
    )

    fills = await fill_at(exchange, "100")

    # No special priority: the normal order may still reverse the position.
    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("6")), ("b", D("6"))]
    assert await qty_of(exchange) == D("-2")
    assert (await update_of(exchange, "a")).status is OrderStatus.FILLED


@pytest.mark.asyncio
async def test_post_only_reduce_only(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "3", "110", time_in_force=TimeInForce.POST_ONLY))

    (fill,) = await fill_at(exchange, "110")

    assert (fill.qty, fill.is_maker, fill.fee, fill.fee_asset) == (D("3"), None, None, None)
    assert await qty_of(exchange) == D("2")


@pytest.mark.asyncio
async def test_reduce_only_average_survives_auto_cancel(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("ro", Side.SELL, "10", "100"))
    await partial(exchange, "100", "2")
    await partial(exchange, "110", "3")
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("5"), price=D("90"))
    )
    await fill_at(exchange, "95")  # only "close" crosses -> flat

    assert await fill_at(exchange, "100") == []

    update = await update_of(exchange, "ro")
    assert (update.status, update.cum_filled_qty, update.avg_fill_price) == (
        OrderStatus.CANCELED,
        D("5"),
        D("106"),
    )


@pytest.mark.asyncio
async def test_manual_cancel_of_reduce_only(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "3", "110"))

    await exchange.cancel_order(ref("ro"))

    assert (await update_of(exchange, "ro")).status is OrderStatus.CANCELED
    with pytest.raises(ExchangeRejectedError, match="canceled"):
        await exchange.cancel_order(ref("ro"))
    assert await fill_at(exchange, "120") == []
    assert await qty_of(exchange) == D("5")


# --- atomicity ----------------------------------------------------------------------


async def snapshot(exchange: SimulatedExchange, *cids: str) -> tuple[object, ...]:
    return (
        await position_of(exchange),
        [await update_of(exchange, c) for c in cids],
        exchange._exec_sequence,
    )


@pytest.mark.asyncio
async def test_failure_after_prepared_reduce_only_fill(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "7", "110"))
    await exchange.place_order(
        request(client_order_id="b", side=Side.SELL, qty=D("2"), price=D("110"))
    )
    before = await snapshot(exchange, "a", "b")
    real_build = simulated._build_fill
    calls: list[str] = []

    def failing_second(record: Any, **kwargs: Any) -> Fill:
        calls.append(record.request.client_order_id)
        if len(calls) == 2:
            raise RuntimeError("injected")
        return real_build(record, **kwargs)

    monkeypatch.setattr(simulated, "_build_fill", failing_second)
    with pytest.raises(RuntimeError, match="injected"):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert calls == ["a", "b"]
    assert await snapshot(exchange, "a", "b") == before


@pytest.mark.asyncio
async def test_failure_after_prepared_auto_cancel(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "5", "100"))
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("5"), price=D("95"))
    )
    await fill_at(exchange, "97")  # flat
    await exchange.place_order(request(client_order_id="z-buy", qty=D("5"), price=D("105")))
    before = await snapshot(exchange, "ro", "z-buy")

    def failing(record: Any, **kwargs: Any) -> Fill:
        raise RuntimeError("injected after auto-cancel")

    monkeypatch.setattr(simulated, "_build_fill", failing)
    with pytest.raises(RuntimeError, match="injected"):
        await fill_at(exchange, "100")
    monkeypatch.undo()

    assert await snapshot(exchange, "ro", "z-buy") == before
    assert (await update_of(exchange, "ro")).status is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_failure_in_position_preparation_after_earlier_order(
    exchange: SimulatedExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.exchanges import simulated_positions

    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="a", side=Side.SELL, qty=D("4"), price=D("110"))
    )
    await exchange.place_order(ro("b", Side.SELL, "4", "110"))
    before = await snapshot(exchange, "a", "b")
    real_apply = simulated_positions._apply_fill
    calls: list[str] = []

    def failing_second(state: Any, fill: Fill) -> Any:
        calls.append(fill.client_order_id or "")
        if len(calls) == 2:
            raise simulated_positions.PositionAccountingError("injected")
        return real_apply(state, fill)

    monkeypatch.setattr(simulated_positions, "_apply_fill", failing_second)
    with pytest.raises(simulated_positions.PositionAccountingError, match="injected"):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert calls == ["a", "b"]
    assert await snapshot(exchange, "a", "b") == before
    fills = await fill_at(exchange, "110")
    assert [(f.client_order_id, f.qty) for f in fills] == [("a", D("4")), ("b", D("4"))]


# === trading fees ===============================================================


def fee_exchange(
    role: LiquidityRole = LiquidityRole.TAKER,
    *,
    maker: str = "-0.0001",
    taker: str = "0.001",
    clock: Any = None,
) -> SimulatedExchange:
    policy = SimulatedFeePolicy(
        schedule=TradingFeeSchedule(maker_rate=D(maker), taker_rate=D(taker), fee_asset="USDT"),
        liquidity_role=role,
    )
    return SimulatedExchange(clock=clock or ManualClock(T0), instruments=SPECS, fees=policy)


@pytest.mark.asyncio
async def test_no_fee_mode_keeps_unknown_fee_metadata(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100")))

    (fill,) = await fill_at(exchange, "100")

    assert (fill.fee, fill.fee_asset, fill.is_maker) == (None, None, None)


def test_fees_argument_must_be_a_policy(clock: ManualClock) -> None:
    with pytest.raises(TypeError, match="SimulatedFeePolicy"):
        SimulatedExchange(clock=clock, instruments=SPECS, fees="0.001")  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "fee", "is_maker"),
    [(LiquidityRole.TAKER, D("0.5"), False), (LiquidityRole.MAKER, D("-0.05"), True)],
)
async def test_fill_carries_fee_for_the_configured_role(
    role: LiquidityRole, fee: Decimal, is_maker: bool
) -> None:
    exchange = fee_exchange(role)
    await exchange.place_order(request(price=D("100"), qty=D("5")))

    (fill,) = await fill_at(exchange, "100")

    assert (fill.fee, fill.fee_asset, fill.is_maker) == (fee, "USDT", is_maker)


@pytest.mark.asyncio
async def test_post_only_does_not_imply_maker() -> None:
    exchange = fee_exchange(LiquidityRole.TAKER)
    await exchange.place_order(request(price=D("100"), time_in_force=TimeInForce.POST_ONLY))

    (fill,) = await fill_at(exchange, "100")

    assert fill.is_maker is False  # the configured assumption, not a book reconstruction


@pytest.mark.asyncio
async def test_partial_fills_each_pay_their_own_fee() -> None:
    exchange = fee_exchange()
    await exchange.place_order(request(price=D("110"), qty=D("10")))

    f1 = await partial(exchange, "100", "2")
    f2 = await partial(exchange, "110", "3")
    f3 = await partial(exchange, "90", "5")

    assert [f.fee for f in (*f1, *f2, *f3)] == [D("0.2"), D("0.33"), D("0.45")]
    update = await update_of(exchange)
    assert (update.status, update.cum_filled_qty) == (OrderStatus.FILLED, D("10"))


@pytest.mark.asyncio
async def test_zero_fee_is_known_and_distinct_from_unknown() -> None:
    exchange = fee_exchange(taker="0")
    await exchange.place_order(request(price=D("100")))

    (fill,) = await fill_at(exchange, "100")

    assert fill.fee == 0
    assert fill.fee is not None
    assert fill.fee_asset == "USDT"


@pytest.mark.asyncio
async def test_reduce_only_fee_and_auto_cancel_without_fee() -> None:
    exchange = fee_exchange()
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "10", "110"))

    fills = await fill_at(exchange, "110")

    assert [(f.client_order_id, f.qty, f.fee) for f in fills] == [("ro", D("5"), D("0.55"))]
    assert (await update_of(exchange, "ro")).status is OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_position_realized_pnl_stays_gross_with_fees() -> None:
    exchange = fee_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("10"), price=D("110"))
    )

    (close,) = await fill_at(exchange, "110")

    p = await position_of(exchange)
    assert (p.qty, p.realized_pnl) == (D("0"), D("100"))  # not 100 - fees
    assert close.fee == D("1.1")


@pytest.mark.asyncio
async def test_fee_scenario_is_deterministic() -> None:
    async def run() -> tuple[object, ...]:
        exchange = fee_exchange(LiquidityRole.MAKER)
        await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("3")))
        await exchange.place_order(
            request(client_order_id="b", side=Side.SELL, price=D("90"), qty=D("2"))
        )
        fills = [*await partial(exchange, "95", "4"), *await fill_at(exchange, "91")]
        return tuple(fills), await position_of(exchange)

    assert await run() == await run()


@pytest.mark.asyncio
async def test_failed_fee_calculation_leaves_everything_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = fee_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "4", "110"))
    await exchange.place_order(
        request(client_order_id="b", side=Side.SELL, qty=D("2"), price=D("110"))
    )
    await exchange.place_order(ro("c", Side.SELL, "20", "110"))
    before = await snapshot(exchange, "a", "b", "c")
    real_fee = SimulatedFeePolicy.fill_fee
    calls: list[Decimal] = []

    def failing_third(self: SimulatedFeePolicy, *, price: Decimal, qty: Decimal) -> Any:
        calls.append(qty)
        if len(calls) == 3:
            raise ArithmeticError("injected fee failure")
        return real_fee(self, price=price, qty=qty)

    monkeypatch.setattr(SimulatedFeePolicy, "fill_fee", failing_third)
    with pytest.raises(ArithmeticError, match="injected"):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert calls == [D("4"), D("2"), D("4")]
    assert await snapshot(exchange, "a", "b", "c") == before
    fills = await fill_at(exchange, "110")
    assert [(f.client_order_id, f.qty, f.fee) for f in fills] == [
        ("a", D("4"), D("0.44")),
        ("b", D("2"), D("0.22")),
        ("c", D("4"), D("0.44")),
    ]
    assert (await update_of(exchange, "c")).status is OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_retry_and_refill_do_not_create_fees() -> None:
    exchange = fee_exchange()
    original = await exchange.place_order(request(price=D("100")))
    (fill,) = await fill_at(exchange, "100")

    assert await exchange.place_order(request(price=D("100"))) == original
    assert await fill_at(exchange, "100") == []
    assert fill.fee == D("0.0001")


# === cash accounting ============================================================


def cash_exchange(
    *,
    starting: str = "10000",
    taker: str = "0.001",
    maker: str = "-0.0001",
    role: LiquidityRole = LiquidityRole.TAKER,
    clock: Any = None,
) -> SimulatedExchange:
    return SimulatedExchange(
        clock=clock or ManualClock(T0),
        instruments=SPECS,
        fees=SimulatedFeePolicy(
            schedule=TradingFeeSchedule(maker_rate=D(maker), taker_rate=D(taker), fee_asset="USDT"),
            liquidity_role=role,
        ),
        cash=SimulatedCashConfig(asset="USDT", starting_cash=D(starting)),
    )


async def cash_parts(exchange: SimulatedExchange) -> tuple[Decimal, ...]:
    state = await exchange.get_cash_state()
    assert state is not None
    return state.gross_realized_pnl, state.trading_fees, state.cash


@pytest.mark.asyncio
async def test_no_accounting_mode_has_no_cash_state(exchange: SimulatedExchange) -> None:
    await exchange.place_order(request(price=D("100")))
    await fill_at(exchange, "100")

    assert await exchange.get_cash_state() is None


def test_cash_requires_a_fee_policy(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="fee policy"):
        SimulatedExchange(
            clock=clock,
            instruments=SPECS,
            cash=SimulatedCashConfig(asset="USDT", starting_cash=D("1")),
        )


def test_cash_and_fee_asset_must_match(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="fee asset"):
        SimulatedExchange(
            clock=clock,
            instruments=SPECS,
            fees=SimulatedFeePolicy(
                schedule=TradingFeeSchedule(maker_rate=D("0"), taker_rate=D("0"), fee_asset="USDC"),
                liquidity_role=LiquidityRole.TAKER,
            ),
            cash=SimulatedCashConfig(asset="USDT", starting_cash=D("1")),
        )


def test_instrument_quote_asset_must_match_cash_asset(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="quote asset"):
        SimulatedExchange(
            clock=clock,
            instruments=(spec("BTCUSDT"), spec("BTCUSDC", quote_asset="USDC")),
            fees=SimulatedFeePolicy(
                schedule=TradingFeeSchedule(maker_rate=D("0"), taker_rate=D("0"), fee_asset="USDT"),
                liquidity_role=LiquidityRole.TAKER,
            ),
            cash=SimulatedCashConfig(asset="USDT", starting_cash=D("1")),
        )


def test_cash_argument_type(clock: ManualClock) -> None:
    with pytest.raises(TypeError, match="SimulatedCashConfig"):
        SimulatedExchange(clock=clock, instruments=SPECS, cash="10000")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_initial_cash_state_and_read_does_not_use_clock() -> None:
    clock = CountingClock(T0)
    exchange = cash_exchange(clock=clock)

    state = await exchange.get_cash_state()

    assert state is not None
    assert (state.asset, state.starting_cash, state.cash) == ("USDT", D("10000"), D("10000"))
    assert clock.calls == 0


@pytest.mark.asyncio
async def test_opening_moves_cash_by_fee_only() -> None:
    exchange = cash_exchange()

    await open_position(exchange, Side.BUY, "10")  # notional 1000, fee 1

    assert await cash_parts(exchange) == (D("0"), D("1"), D("9999"))


@pytest.mark.asyncio
async def test_close_with_profit_and_fees() -> None:
    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("10"), price=D("110"))
    )

    await fill_at(exchange, "110")

    assert await cash_parts(exchange) == (D("100"), D("2.1"), D("10097.9"))
    p = await position_of(exchange)
    assert p.realized_pnl == D("100")  # position stays gross


@pytest.mark.asyncio
async def test_close_with_loss() -> None:
    exchange = cash_exchange(taker="0")
    await open_position(exchange, Side.SELL, "10")
    await exchange.place_order(
        request(client_order_id="close", side=Side.BUY, qty=D("10"), price=D("110"))
    )

    await fill_at(exchange, "110")

    assert await cash_parts(exchange) == (D("-100"), D("0"), D("9900"))


@pytest.mark.asyncio
async def test_partial_closes_and_reversal() -> None:
    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")  # fee 1
    await exchange.place_order(
        request(client_order_id="s1", side=Side.SELL, qty=D("4"), price=D("110"))
    )
    await fill_at(exchange, "110")  # +40, fee 0.44
    assert await cash_parts(exchange) == (D("40"), D("1.44"), D("10038.56"))
    await exchange.place_order(
        request(client_order_id="s2", side=Side.SELL, qty=D("9"), price=D("90"))
    )

    await fill_at(exchange, "90")  # closes 6 @100 -> -60, opens short 3; fee on all 9

    assert await cash_parts(exchange) == (D("-20"), D("2.25"), D("9977.75"))
    assert (await position_of(exchange)).qty == D("-3")


@pytest.mark.asyncio
async def test_reduce_only_cash_and_auto_cancel() -> None:
    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "5")  # fee 0.5
    await exchange.place_order(ro("ro", Side.SELL, "10", "110"))

    await fill_at(exchange, "110")  # fills 5: +50, fee 0.55; remainder canceled

    assert await cash_parts(exchange) == (D("50"), D("1.05"), D("10048.95"))
    assert (await update_of(exchange, "ro")).status is OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_maker_rebate_increases_cash() -> None:
    exchange = cash_exchange(role=LiquidityRole.MAKER)

    await open_position(exchange, Side.BUY, "10")  # fee -0.1

    assert await cash_parts(exchange) == (D("0"), D("-0.1"), D("10000.1"))


@pytest.mark.asyncio
async def test_multiple_fills_in_one_batch() -> None:
    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")  # fee 1
    await exchange.place_order(
        request(client_order_id="a", side=Side.SELL, qty=D("4"), price=D("110"))
    )
    await exchange.place_order(
        request(client_order_id="b", side=Side.SELL, qty=D("6"), price=D("110"))
    )

    await fill_at(exchange, "120")  # +80 and +120, fees 0.48 + 0.72

    assert await cash_parts(exchange) == (D("200"), D("2.2"), D("10197.8"))


@pytest.mark.asyncio
async def test_failed_fee_calculation_leaves_cash_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "4", "110"))
    await exchange.place_order(ro("b", Side.SELL, "20", "110"))
    before = (await snapshot(exchange, "a", "b"), await exchange.get_cash_state())
    real_fee = SimulatedFeePolicy.fill_fee
    calls: list[Decimal] = []

    def failing_second(self: SimulatedFeePolicy, *, price: Decimal, qty: Decimal) -> Any:
        calls.append(qty)
        if len(calls) == 2:
            raise ArithmeticError("injected")
        return real_fee(self, price=price, qty=qty)

    monkeypatch.setattr(SimulatedFeePolicy, "fill_fee", failing_second)
    with pytest.raises(ArithmeticError):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert (await snapshot(exchange, "a", "b"), await exchange.get_cash_state()) == before


@pytest.mark.asyncio
async def test_failed_position_preparation_leaves_cash_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.exchanges import simulated_positions

    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="a", side=Side.SELL, qty=D("4"), price=D("110"))
    )
    await exchange.place_order(ro("b", Side.SELL, "4", "110"))
    before = (await snapshot(exchange, "a", "b"), await exchange.get_cash_state())
    real_apply = simulated_positions._apply_fill
    calls: list[str] = []

    def failing_second(state: Any, fill: Fill) -> Any:
        calls.append(fill.exec_id)
        if len(calls) == 2:
            raise simulated_positions.PositionAccountingError("injected")
        return real_apply(state, fill)

    monkeypatch.setattr(simulated_positions, "_apply_fill", failing_second)
    with pytest.raises(simulated_positions.PositionAccountingError):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert (await snapshot(exchange, "a", "b"), await exchange.get_cash_state()) == before


@pytest.mark.asyncio
async def test_failed_accounting_preparation_leaves_everything_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.exchanges import simulated_accounting

    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "4", "110"))
    await exchange.place_order(ro("b", Side.SELL, "20", "110"))  # partial + auto-cancel
    before = (await snapshot(exchange, "a", "b"), await exchange.get_cash_state())
    real_apply = simulated_accounting.CashBatch.apply
    calls: list[str] = []

    def failing_second(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs["exec_id"])
        if len(calls) == 2:
            raise simulated_accounting.CashAccountingError("injected")
        real_apply(self, **kwargs)

    monkeypatch.setattr(simulated_accounting.CashBatch, "apply", failing_second)
    with pytest.raises(simulated_accounting.CashAccountingError, match="injected"):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert calls == ["SIM-EXEC-0000000002", "SIM-EXEC-0000000003"]
    assert (await snapshot(exchange, "a", "b"), await exchange.get_cash_state()) == before
    assert (await update_of(exchange, "b")).status is OrderStatus.OPEN  # no auto-cancel


@pytest.mark.asyncio
async def test_cash_scenario_is_deterministic() -> None:
    async def run() -> tuple[object, ...]:
        exchange = cash_exchange()
        await open_position(exchange, Side.BUY, "3")
        await exchange.place_order(
            request(client_order_id="s", side=Side.SELL, qty=D("5"), price=D("101"))
        )
        await partial(exchange, "103.3", "2")
        await fill_at(exchange, "99.7")
        return await position_of(exchange), await exchange.get_cash_state()

    assert await run() == await run()


# === mark-to-market ===============================================================


class ScriptedClock:
    """Returns the given times in order (may go backwards) to test stream ordering."""

    def __init__(self, *seconds: int) -> None:
        self.times = [T0 + timedelta(seconds=s) for s in seconds]
        self.calls = 0

    def now(self) -> datetime:
        self.calls += 1
        return self.times.pop(0)


async def set_mark(exchange: SimulatedExchange, price: str, symbol: str = "BTCUSDT") -> Any:
    return await exchange.set_mark_price(symbol=symbol, mark_price=D(price))


@pytest.mark.asyncio
async def test_mark_before_position_is_stored_and_applied_on_open(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    assert await set_mark(exchange, "110") is None
    assert await position_of(exchange) is None  # no artificial position

    clock.advance(timedelta(seconds=1))
    await open_position(exchange, Side.BUY, "10")

    p = await position_of(exchange)
    assert (p.qty, p.entry_price, p.mark_price, p.unrealized_pnl) == (
        D("10"),
        D("100"),
        D("110"),
        D("100"),
    )
    assert p.updated_at == T0 + timedelta(seconds=1)


@pytest.mark.asyncio
async def test_no_mark_keeps_unknown_unrealized(exchange: SimulatedExchange) -> None:
    await open_position(exchange, Side.BUY, "10")

    p = await position_of(exchange)
    assert (p.mark_price, p.unrealized_pnl) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "marks"),
    [
        (Side.BUY, [("90", "-100"), ("100", "0"), ("110", "100")]),
        (Side.SELL, [("90", "100"), ("100", "0"), ("110", "-100")]),
    ],
)
async def test_mark_updates_revalue_only(
    exchange: SimulatedExchange, clock: ManualClock, side: Side, marks: list[tuple[str, str]]
) -> None:
    await open_position(exchange, side, "10")
    before = await position_of(exchange)

    for i, (price, unrealized) in enumerate(marks, start=1):
        clock.advance(timedelta(seconds=1))
        returned = await set_mark(exchange, price)
        p = await position_of(exchange)
        assert returned == p
        assert (p.mark_price, p.unrealized_pnl) == (D(price), D(unrealized))
        assert (p.qty, p.entry_price, p.realized_pnl) == (
            before.qty,
            before.entry_price,
            before.realized_pnl,
        )
        assert p.updated_at == T0 + timedelta(seconds=i)
    assert exchange._exec_sequence == 1
    assert exchange._sequence == 1


@pytest.mark.asyncio
async def test_mark_validation_and_registry(exchange: SimulatedExchange) -> None:
    for bad in (
        D("0"),
        D("-1"),
        D("NaN"),
        D("sNaN"),
        D("Infinity"),
        D("-Infinity"),
        1,
        1.5,
        True,
        "1",
    ):
        with pytest.raises(ExchangeRequestValidationError, match="mark_price"):
            await exchange.set_mark_price(symbol="BTCUSDT", mark_price=bad)  # type: ignore[arg-type]
    with pytest.raises(ExchangeRequestValidationError, match="symbol"):
        await exchange.set_mark_price(symbol=" BTCUSDT", mark_price=D("1"))
    with pytest.raises(ExchangeRejectedError, match="unknown instrument"):
        await exchange.set_mark_price(symbol="DOGEUSDT", mark_price=D("1"))

    class Sub(Decimal):
        pass

    with pytest.raises(ExchangeRequestValidationError, match="mark_price"):
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=Sub("1"))


@pytest.mark.asyncio
async def test_invalid_mark_does_not_read_clock() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)

    with pytest.raises(ExchangeRequestValidationError):
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("0"))
    with pytest.raises(ExchangeRejectedError):
        await exchange.set_mark_price(symbol="DOGEUSDT", mark_price=D("1"))

    assert clock.calls == 0


@pytest.mark.asyncio
async def test_fill_after_mark_and_average_change_use_exact_basis(
    exchange: SimulatedExchange,
) -> None:
    await set_mark(exchange, "120")
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("2")))
    await fill_at(exchange, "100")
    await exchange.place_order(request(client_order_id="b", price=D("110"), qty=D("1")))
    await fill_at(exchange, "110")

    p = await position_of(exchange)
    assert p.unrealized_pnl == D("50")  # 360 - 310, from the exact basis


@pytest.mark.asyncio
async def test_partial_close_revalues_remaining(exchange: SimulatedExchange) -> None:
    await set_mark(exchange, "110")
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="s", side=Side.SELL, qty=D("4"), price=D("105"))
    )

    await fill_at(exchange, "105")

    p = await position_of(exchange)
    assert (p.qty, p.realized_pnl, p.mark_price, p.unrealized_pnl) == (
        D("6"),
        D("20"),
        D("110"),
        D("60"),
    )


@pytest.mark.asyncio
async def test_exact_close_and_reopen_keep_mark(exchange: SimulatedExchange) -> None:
    await set_mark(exchange, "105")
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("10"), price=D("100"))
    )
    await fill_at(exchange, "100")

    flat = await position_of(exchange)
    assert (flat.qty, flat.mark_price, flat.unrealized_pnl) == (D("0"), D("105"), D("0"))

    await exchange.place_order(
        request(client_order_id="reopen", side=Side.SELL, qty=D("2"), price=D("100"))
    )
    await fill_at(exchange, "100")
    short = await position_of(exchange)
    assert (short.qty, short.unrealized_pnl) == (D("-2"), D("-10"))


@pytest.mark.asyncio
async def test_reversal_revalues_new_side(exchange: SimulatedExchange) -> None:
    await set_mark(exchange, "105")
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="rev", side=Side.SELL, qty=D("15"), price=D("110"))
    )

    await fill_at(exchange, "110")

    p = await position_of(exchange)
    assert (p.qty, p.entry_price, p.realized_pnl, p.unrealized_pnl) == (
        D("-5"),
        D("110"),
        D("100"),
        D("25"),
    )


@pytest.mark.asyncio
async def test_batch_fills_see_the_same_stored_mark(exchange: SimulatedExchange) -> None:
    await set_mark(exchange, "100")
    await exchange.place_order(request(client_order_id="a", price=D("96"), qty=D("1")))
    await exchange.place_order(request(client_order_id="b", price=D("96"), qty=D("2")))

    await fill_at(exchange, "95")

    p = await position_of(exchange)
    assert (p.qty, p.mark_price, p.unrealized_pnl) == (D("3"), D("100"), D("15"))


@pytest.mark.asyncio
async def test_reduce_only_with_mark_still_never_reverses(exchange: SimulatedExchange) -> None:
    await set_mark(exchange, "120")
    await open_position(exchange, Side.BUY, "5")
    await exchange.place_order(ro("ro", Side.SELL, "10", "110"))

    await fill_at(exchange, "110")

    p = await position_of(exchange)
    assert (p.qty, p.mark_price, p.unrealized_pnl) == (D("0"), D("120"), D("0"))


@pytest.mark.asyncio
async def test_marks_do_not_touch_cash_fees_or_realized() -> None:
    clock = ManualClock(T0)
    exchange = cash_exchange(clock=clock)
    await open_position(exchange, Side.BUY, "10")  # fee 1
    cash_before = await exchange.get_cash_state()

    for price in ("90", "130", "100.5"):
        clock.advance(timedelta(seconds=1))
        await set_mark(exchange, price)
        assert await exchange.get_cash_state() == cash_before

    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("10"), price=D("110"))
    )
    (close,) = await fill_at(exchange, "110")
    assert close.fee == D("1.1")
    assert await cash_parts(exchange) == (D("100"), D("2.1"), D("10097.9"))
    p = await position_of(exchange)
    assert (p.realized_pnl, p.unrealized_pnl, p.mark_price) == (D("100"), D("0"), D("100.5"))


@pytest.mark.asyncio
async def test_same_price_new_timestamp_is_a_new_event(
    exchange: SimulatedExchange, clock: ManualClock
) -> None:
    await open_position(exchange, Side.BUY, "1")
    first = await set_mark(exchange, "101")
    again_same_time = await set_mark(exchange, "101")
    clock.advance(timedelta(seconds=3))
    later = await set_mark(exchange, "101")

    assert again_same_time == first
    assert later.updated_at == T0 + timedelta(seconds=3)
    assert later.unrealized_pnl == first.unrealized_pnl


@pytest.mark.asyncio
async def test_mark_reads_do_not_use_clock() -> None:
    clock = CountingClock(T0)
    exchange = SimulatedExchange(clock=clock, instruments=SPECS)
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("1"))
    assert clock.calls == 1  # one read per mark update
    await exchange.get_position(symbol="BTCUSDT")
    assert clock.calls == 1


# --- timestamp streams ----------------------------------------------------------------


async def scripted_exchange(*seconds: int) -> SimulatedExchange:
    exchange = SimulatedExchange(clock=ScriptedClock(*seconds), instruments=SPECS)
    return exchange


@pytest.mark.asyncio
async def test_fill_then_mark_then_fill() -> None:  # A
    exchange = await scripted_exchange(0, 1, 2, 3, 4)
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("1")))  # t0
    await fill_at(exchange, "100")  # t1
    await set_mark(exchange, "105")  # t2
    await exchange.place_order(request(client_order_id="b", price=D("100"), qty=D("1")))  # t3
    await fill_at(exchange, "100")  # t4

    p = await position_of(exchange)
    assert (p.qty, p.unrealized_pnl, p.updated_at) == (D("2"), D("10"), T0 + timedelta(seconds=4))


@pytest.mark.asyncio
async def test_newer_mark_does_not_make_an_older_fill_stale() -> None:  # B
    exchange = await scripted_exchange(0, 9, 5)
    await exchange.place_order(request(price=D("100"), qty=D("1")))  # t0
    await set_mark(exchange, "105")  # t9

    (fill,) = await fill_at(exchange, "100")  # t5 < mark t9: still a valid fill

    assert fill.exchange_ts == T0 + timedelta(seconds=5)
    p = await position_of(exchange)
    assert (p.qty, p.unrealized_pnl) == (D("1"), D("5"))
    assert p.updated_at == T0 + timedelta(seconds=9)  # latest of fill and mark


@pytest.mark.asyncio
async def test_older_fill_after_newer_fill_is_still_rejected() -> None:  # C
    exchange = await scripted_exchange(0, 1, 5, 3)
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("1")))  # t0
    await exchange.place_order(request(client_order_id="b", price=D("90"), qty=D("1")))  # t1
    await fill_at(exchange, "100")  # t5: fills a
    before = await snapshot(exchange, "a", "b")

    with pytest.raises(RuntimeError, match="older"):
        await fill_at(exchange, "90")  # t3: fill of b is older than t5

    assert await snapshot(exchange, "a", "b") == before


@pytest.mark.asyncio
async def test_older_mark_is_rejected_without_mutation() -> None:  # D
    exchange = await scripted_exchange(0, 1, 9, 4)
    await exchange.place_order(request(price=D("100"), qty=D("1")))
    await fill_at(exchange, "100")
    await set_mark(exchange, "105")  # t9
    before = await position_of(exchange)

    with pytest.raises(RuntimeError, match="stale mark"):
        await set_mark(exchange, "90")  # t4

    assert await position_of(exchange) == before


@pytest.mark.asyncio
async def test_equal_timestamps_for_fills_and_marks() -> None:  # E, F
    exchange = await scripted_exchange(0, 0, 5, 5, 5, 5)
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("1")))
    await exchange.place_order(request(client_order_id="b", price=D("90"), qty=D("1")))
    await fill_at(exchange, "100")  # t5
    await fill_at(exchange, "90")  # t5, equal: allowed
    await set_mark(exchange, "95")  # t5
    p = await set_mark(exchange, "96")  # t5, equal: allowed

    assert (p.qty, p.mark_price, p.unrealized_pnl) == (D("2"), D("96"), D("2"))


# --- atomicity ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_valuation_in_fill_batch_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.exchanges import simulated_positions

    exchange = cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "20", "110"))
    before = (await snapshot(exchange, "a"), await exchange.get_cash_state())

    def failing(state: Any, mark: Any) -> Any:
        raise ArithmeticError("injected valuation failure")

    monkeypatch.setattr(simulated_positions, "_to_position", failing)
    with pytest.raises(ArithmeticError, match="injected"):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert (await snapshot(exchange, "a"), await exchange.get_cash_state()) == before
    assert (await update_of(exchange, "a")).status is OrderStatus.OPEN


@pytest.mark.asyncio
async def test_failed_revaluation_keeps_old_mark(
    exchange: SimulatedExchange, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.exchanges import simulated_positions

    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("105"))
    await open_position(exchange, Side.BUY, "10")
    before = await position_of(exchange)
    clock.advance(timedelta(seconds=1))

    def failing(state: Any, mark: Any) -> Any:
        raise ArithmeticError("injected revaluation failure")

    monkeypatch.setattr(simulated_positions, "_to_position", failing)
    with pytest.raises(ArithmeticError, match="injected"):
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("90"))
    monkeypatch.undo()

    assert await position_of(exchange) == before
    assert before.mark_price == D("105")


# === equity read model ============================================================


async def equity_parts(exchange: SimulatedExchange) -> tuple[Decimal, Decimal, Decimal] | None:
    state = await exchange.get_equity_state()
    return None if state is None else (state.cash, state.unrealized_pnl, state.equity)


def no_fee_cash_exchange(clock: Any = None) -> SimulatedExchange:
    return cash_exchange(taker="0", maker="0", clock=clock)


@pytest.mark.asyncio
async def test_no_cash_accounting_means_no_equity(exchange: SimulatedExchange) -> None:
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))
    await open_position(exchange, Side.BUY, "1")

    assert await exchange.get_equity_state() is None


@pytest.mark.asyncio
async def test_equity_without_positions_is_cash() -> None:
    exchange = cash_exchange()

    state = await exchange.get_equity_state()

    assert state is not None
    assert (state.asset, state.cash, state.unrealized_pnl, state.equity) == (
        "USDT",
        D("10000"),
        D("0"),
        D("10000"),
    )


@pytest.mark.asyncio
async def test_flat_positions_need_no_mark() -> None:
    exchange = no_fee_cash_exchange()
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(
        request(client_order_id="close", side=Side.SELL, qty=D("10"), price=D("110"))
    )
    await fill_at(exchange, "110")

    assert await equity_parts(exchange) == (D("10100"), D("0"), D("10100"))


@pytest.mark.asyncio
async def test_open_position_without_mark_makes_equity_unknown() -> None:
    exchange = cash_exchange()
    await open_position(exchange, Side.BUY, "10")

    assert await exchange.get_equity_state() is None
    assert await exchange.get_cash_state() is not None  # cash itself stays known


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "price", "unrealized"),
    [
        (Side.BUY, "110", "100"),
        (Side.BUY, "90", "-100"),
        (Side.BUY, "100", "0"),
        (Side.SELL, "90", "100"),
        (Side.SELL, "110", "-100"),
        (Side.SELL, "100", "0"),
    ],
)
async def test_equity_long_and_short(side: Side, price: str, unrealized: str) -> None:
    exchange = no_fee_cash_exchange()
    await open_position(exchange, side, "10")
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D(price))

    assert await equity_parts(exchange) == (
        D("10000"),
        D(unrealized),
        D("10000") + D(unrealized),
    )


@pytest.mark.asyncio
async def test_multiple_positions_all_marked_or_unknown() -> None:
    exchange = no_fee_cash_exchange()
    await open_position(exchange, Side.BUY, "10")  # BTC long @100
    await exchange.place_order(
        request(client_order_id="eth", symbol="ETHUSDT", side=Side.SELL, qty=D("2"), price=D("50"))
    )
    await exchange.fill_crossed_limit_orders(symbol="ETHUSDT", execution_price=D("50"))
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))  # +100

    assert await exchange.get_equity_state() is None  # ETH has no mark

    await exchange.set_mark_price(symbol="ETHUSDT", mark_price=D("70"))  # short: -40
    assert await equity_parts(exchange) == (D("10000"), D("60"), D("10060"))


@pytest.mark.asyncio
async def test_mark_updates_move_equity_not_cash() -> None:
    clock = ManualClock(T0)
    exchange = cash_exchange(clock=clock)
    await open_position(exchange, Side.BUY, "10")  # fee 1 -> cash 9999

    results = []
    for price in ("100", "110", "90"):
        clock.advance(timedelta(seconds=1))
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D(price))
        results.append(await equity_parts(exchange))

    assert results == [
        (D("9999"), D("0"), D("9999")),
        (D("9999"), D("100"), D("10099")),
        (D("9999"), D("-100"), D("9899")),
    ]


@pytest.mark.asyncio
async def test_partial_close_transfers_unrealized_to_cash() -> None:
    exchange = no_fee_cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))
    await open_position(exchange, Side.BUY, "10")
    assert await equity_parts(exchange) == (D("10000"), D("100"), D("10100"))
    await exchange.place_order(
        request(client_order_id="s", side=Side.SELL, qty=D("4"), price=D("110"))
    )

    await fill_at(exchange, "110")

    assert await equity_parts(exchange) == (D("10040"), D("60"), D("10100"))


@pytest.mark.asyncio
async def test_partial_close_with_fee_reduces_equity_by_the_fee_only() -> None:
    exchange = cash_exchange()  # taker 0.001
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))
    await open_position(exchange, Side.BUY, "10")  # fee 1
    before = await equity_parts(exchange)
    await exchange.place_order(
        request(client_order_id="s", side=Side.SELL, qty=D("4"), price=D("110"))
    )

    (fill,) = await fill_at(exchange, "110")  # fee 0.44

    after = await equity_parts(exchange)
    assert before == (D("9999"), D("100"), D("10099"))
    assert after == (D("10038.56"), D("60"), D("10098.56"))
    assert before[2] - after[2] == fill.fee


@pytest.mark.asyncio
@pytest.mark.parametrize(("close", "equity"), [("110", "10100"), ("105", "10050")])
async def test_exact_close_at_or_away_from_mark(close: str, equity: str) -> None:
    exchange = no_fee_cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))
    await open_position(exchange, Side.BUY, "10")
    assert (await equity_parts(exchange))[2] == D("10100")  # type: ignore[index]
    await exchange.place_order(
        request(client_order_id="c", side=Side.SELL, qty=D("10"), price=D(close))
    )

    await fill_at(exchange, close)

    assert await equity_parts(exchange) == (D(equity), D("0"), D(equity))


@pytest.mark.asyncio
async def test_reversal_equity() -> None:
    exchange = cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("105"))
    await open_position(exchange, Side.BUY, "10")  # fee 1
    await exchange.place_order(
        request(client_order_id="rev", side=Side.SELL, qty=D("15"), price=D("110"))
    )

    await fill_at(exchange, "110")  # realized +100, fee 1.65, short 5 @110 -> +25

    assert await equity_parts(exchange) == (D("10097.35"), D("25"), D("10122.35"))


@pytest.mark.asyncio
async def test_reduce_only_partial_close_equity() -> None:
    exchange = no_fee_cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("120"))
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("ro", Side.SELL, "4", "110"))

    await fill_at(exchange, "110")  # +40 realized; 6 left at mark 120 -> +120

    assert await equity_parts(exchange) == (D("10040"), D("120"), D("10160"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "maker", "taker", "expected"),
    [
        (LiquidityRole.TAKER, "0", "0", "10000"),  # zero fee
        (LiquidityRole.TAKER, "0", "0.001", "9999"),  # fee
        (LiquidityRole.MAKER, "-0.0001", "0.001", "10000.1"),  # rebate
    ],
)
async def test_fees_and_rebates_reach_equity_through_cash(
    role: LiquidityRole, maker: str, taker: str, expected: str
) -> None:
    exchange = cash_exchange(role=role, maker=maker, taker=taker)
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("100"))

    await open_position(exchange, Side.BUY, "10")

    assert await equity_parts(exchange) == (D(expected), D("0"), D(expected))


@pytest.mark.asyncio
async def test_equity_with_repeating_basis_is_exact() -> None:
    exchange = no_fee_cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("120"))
    await exchange.place_order(request(client_order_id="a", price=D("100"), qty=D("2")))
    await fill_at(exchange, "100")
    await exchange.place_order(request(client_order_id="b", price=D("110"), qty=D("1")))
    await fill_at(exchange, "110")  # basis 310/3
    await exchange.place_order(
        request(client_order_id="s", side=Side.SELL, qty=D("1"), price=D("120"))
    )

    await fill_at(exchange, "120")  # realized 50/3, remaining 2 @310/3 -> 100/3

    state = await exchange.get_equity_state()
    assert state is not None
    assert state.equity == D("10050")  # exact 10000 + 50/3 + 100/3
    assert state.cash == D("10016.66666666666666666666666666666666667")
    assert state.unrealized_pnl == D("33.33333333333333333333333333333333333333")


@pytest.mark.asyncio
async def test_equity_ignores_the_global_decimal_context() -> None:
    async def run() -> object:
        exchange = cash_exchange()
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("101.37"))
        await open_position(exchange, Side.BUY, "3.333")
        return await exchange.get_equity_state()

    baseline = await run()
    with localcontext() as context:
        context.prec = 2
        context.rounding = "ROUND_UP"
        low = await run()

    assert low == baseline


@pytest.mark.asyncio
async def test_equity_reads_are_pure() -> None:
    clock = CountingClock(T0)
    exchange = cash_exchange(clock=clock)
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("105"))
    await open_position(exchange, Side.BUY, "10")
    before = (
        await exchange.get_cash_state(),
        await position_of(exchange),
        exchange._marks.copy(),
        exchange._sequence,
        exchange._exec_sequence,
    )
    clock.calls = 0

    first = await exchange.get_equity_state()
    second = await exchange.get_equity_state()

    assert first == second
    assert clock.calls == 0
    assert (
        await exchange.get_cash_state(),
        await position_of(exchange),
        exchange._marks.copy(),
        exchange._sequence,
        exchange._exec_sequence,
    ) == before


@pytest.mark.asyncio
async def test_failed_fill_batch_preserves_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = cash_exchange()
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("110"))
    await open_position(exchange, Side.BUY, "10")
    await exchange.place_order(ro("a", Side.SELL, "4", "110"))
    await exchange.place_order(ro("b", Side.SELL, "20", "110"))
    before = await exchange.get_equity_state()
    real_fee = SimulatedFeePolicy.fill_fee
    calls: list[Decimal] = []

    def failing_second(self: SimulatedFeePolicy, *, price: Decimal, qty: Decimal) -> Any:
        calls.append(qty)
        if len(calls) == 2:
            raise ArithmeticError("injected")
        return real_fee(self, price=price, qty=qty)

    monkeypatch.setattr(SimulatedFeePolicy, "fill_fee", failing_second)
    with pytest.raises(ArithmeticError):
        await fill_at(exchange, "110")
    monkeypatch.undo()

    assert await exchange.get_equity_state() == before


@pytest.mark.asyncio
async def test_failed_mark_update_preserves_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.exchanges import simulated_positions

    clock = ManualClock(T0)
    exchange = cash_exchange(clock=clock)
    await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("105"))
    await open_position(exchange, Side.BUY, "10")
    before = await exchange.get_equity_state()
    clock.advance(timedelta(seconds=1))

    def failing(state: Any, mark: Any) -> Any:
        raise ArithmeticError("injected")

    monkeypatch.setattr(simulated_positions, "_to_position", failing)
    with pytest.raises(ArithmeticError):
        await exchange.set_mark_price(symbol="BTCUSDT", mark_price=D("90"))
    monkeypatch.undo()

    assert await exchange.get_equity_state() == before
