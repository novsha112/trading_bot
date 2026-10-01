"""Deterministic simulated exchange: order lifecycle without network or credentials.

No fills, matching, fees, balances or positions yet: a LIMIT order rests OPEN until
it is canceled.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.clock import ManualClock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.fills import Fill
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
    return SimulatedExchange(clock=clock)


# --- contract -----------------------------------------------------------------


def test_implements_trading_client(clock: ManualClock) -> None:
    # mypy strict is the proof (structural Protocol, no runtime_checkable).
    client: TradingClient = SimulatedExchange(clock=clock)
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
        exchange = SimulatedExchange(clock=ManualClock(T0))
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
    first = SimulatedExchange(clock=ManualClock(T0))
    second = SimulatedExchange(clock=ManualClock(T0))

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
    exchange = SimulatedExchange(clock=clock)
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
        exchange = SimulatedExchange(clock=clock)
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
