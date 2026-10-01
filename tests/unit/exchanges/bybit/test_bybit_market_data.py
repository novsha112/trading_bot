"""Bybit public market data adapter, tested against a mocked HTTP transport only.

Response shapes follow the official Bybit V5 docs (bybit-exchange/docs, master):
GET /v5/market/instruments-info and GET /v5/market/tickers, category=linear.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest

from app.domain.clock import ManualClock
from app.domain.instrument import InstrumentSpec
from app.domain.market import Ticker
from app.exchanges.bybit.endpoints import BYBIT_MAINNET_REST_URL, BYBIT_TESTNET_REST_URL
from app.exchanges.bybit.market_data import BybitMarketDataClient
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeError,
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeResponseError,
)
from app.exchanges.protocols import MarketDataClient

D = Decimal
BASE = "https://bybit.example"
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
RESPONSE_MS = 1707186451514

INSTRUMENT: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "contractType": "LinearPerpetual",
    "status": "Trading",
    "baseCoin": "BTC",
    "quoteCoin": "USDT",
    "launchTime": "1585526400000",
    "deliveryTime": "0",
    "deliveryFeeRate": "",
    "priceScale": "2",
    "leverageFilter": {"minLeverage": "1", "maxLeverage": "100.00", "leverageStep": "0.01"},
    "priceFilter": {"minPrice": "0.10", "maxPrice": "199999.80", "tickSize": "0.10"},
    "lotSizeFilter": {
        "minNotionalValue": "5",
        "maxOrderQty": "1190.000",
        "maxMktOrderQty": "500.000",
        "minOrderQty": "0.001",
        "qtyStep": "0.001",
        "postOnlyMaxOrderQty": "1190.000",
    },
    "unifiedMarginTrade": True,
    "fundingInterval": 480,
    "settleCoin": "USDT",
    "copyTrading": "both",
    "upperFundingRate": "0.005",
    "lowerFundingRate": "-0.005",
    "isPreListing": False,
    "preListingInfo": None,
}

TICKER: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "lastPrice": "65000.50",
    "indexPrice": "64990.12",
    "markPrice": "64995.30",
    "prevPrice24h": "64000.00",
    "price24hPcnt": "0.0156",
    "highPrice24h": "65500.00",
    "lowPrice24h": "63800.00",
    "prevPrice1h": "64900.00",
    "openInterest": "50000.123",
    "openInterestValue": "3249837621.35",
    "turnover24h": "1234567890.1234",
    "volume24h": "19000.123",
    "fundingRate": "0.0001",
    "nextFundingTime": "1768478400000",
    "predictedDeliveryPrice": "",
    "basisRate": "",
    "deliveryFeeRate": "",
    "deliveryTime": "0",
    "ask1Size": "1.234",
    "bid1Price": "65000.40",
    "ask1Price": "65000.60",
    "bid1Size": "2.345",
    "basis": "",
    "preOpenPrice": "",
    "preQty": "",
    "curPreListingPhase": "",
    "fundingIntervalHour": "8",
    "basisRateYear": "",
    "fundingCap": "0.005",
}


def envelope(
    items: list[dict[str, Any]], *, category: str = "linear", **overrides: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {"category": category, "list": items, "nextPageCursor": ""},
        "retExtInfo": {},
        "time": RESPONSE_MS,
    }
    body.update(overrides)
    return body


class Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []


Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(NOW)


def make_adapter(
    recorder: Recorder, clock: ManualClock, handler: Handler
) -> tuple[BybitMarketDataClient, httpx.AsyncClient]:
    def record(request: httpx.Request) -> httpx.Response:
        recorder.requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return BybitMarketDataClient(client=client, base_url=BASE, clock=clock), client


def json_response(body: Any, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, text=json.dumps(body))


def text_response(text: str, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, text=text)


def with_instrument(**changes: Any) -> dict[str, Any]:
    item = copy.deepcopy(INSTRUMENT)
    for path, value in changes.items():
        *parents, leaf = path.split(".")
        node = item
        for key in parents:
            node = node[key]
        if value is _DELETE:
            del node[leaf]
        else:
            node[leaf] = value
    return item


def with_ticker(**changes: Any) -> dict[str, Any]:
    item = copy.deepcopy(TICKER)
    for key, value in changes.items():
        if value is _DELETE:
            del item[key]
        else:
            item[key] = value
    return item


_DELETE = object()


async def get_instrument(
    recorder: Recorder, clock: ManualClock, body: Any, status: int = 200
) -> InstrumentSpec:
    adapter, client = make_adapter(recorder, clock, json_response(body, status))
    async with client:
        return await adapter.get_instrument("BTCUSDT")


async def get_ticker(
    recorder: Recorder, clock: ManualClock, body: Any, status: int = 200
) -> Ticker:
    adapter, client = make_adapter(recorder, clock, json_response(body, status))
    async with client:
        return await adapter.get_ticker("BTCUSDT")


# --- Static conformance ------------------------------------------------------------------


def _conforms(client: httpx.AsyncClient, clock: ManualClock) -> MarketDataClient:
    return BybitMarketDataClient(client=client, base_url=BASE, clock=clock)  # checked by mypy


def test_endpoint_constants() -> None:
    assert BYBIT_TESTNET_REST_URL == "https://api-testnet.bybit.com"
    assert BYBIT_MAINNET_REST_URL == "https://api.bybit.com"


@pytest.mark.parametrize(
    "base_url", ["http://api-testnet.bybit.com", "", "api.bybit.com", "https://x/", 1]
)
def test_base_url_must_be_https_without_trailing_slash(base_url: Any, clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="base_url"):
        BybitMarketDataClient(client=httpx.AsyncClient(), base_url=base_url, clock=clock)


# --- Instrument --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_instrument_mapping(recorder: Recorder, clock: ManualClock) -> None:
    spec = await get_instrument(recorder, clock, envelope([INSTRUMENT]))

    assert spec == InstrumentSpec(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=D("0.10"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        max_qty=D("500.000"),  # min(maxOrderQty, maxMktOrderQty)
        min_notional=D("5"),
    )
    assert str(spec.tick_size) == "0.10"  # exact string-to-Decimal, no float detour


@pytest.mark.asyncio
async def test_instrument_max_qty_is_the_lower_limit(
    recorder: Recorder, clock: ManualClock
) -> None:
    item = with_instrument(
        **{"lotSizeFilter.maxOrderQty": "100", "lotSizeFilter.maxMktOrderQty": "300"}
    )
    assert (await get_instrument(recorder, clock, envelope([item]))).max_qty == D("100")


@pytest.mark.asyncio
async def test_instrument_request(recorder: Recorder, clock: ManualClock) -> None:
    await get_instrument(recorder, clock, envelope([INSTRUMENT]))

    [request] = recorder.requests
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "bybit.example"
    assert request.url.path == "/v5/market/instruments-info"
    assert dict(request.url.params) == {"category": "linear", "symbol": "BTCUSDT"}
    assert_no_auth(request)


@pytest.mark.asyncio
async def test_unknown_symbol(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeRejectedError, match="not found"):
        await get_instrument(recorder, clock, envelope([]))


@pytest.mark.asyncio
async def test_wrong_symbol_in_response(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeResponseError, match="does not contain the requested symbol"):
        await get_instrument(recorder, clock, envelope([with_instrument(symbol="ETHUSDT")]))


@pytest.mark.asyncio
async def test_duplicate_symbol_in_response(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeResponseError, match="more than one"):
        await get_instrument(recorder, clock, envelope([INSTRUMENT, INSTRUMENT]))


@pytest.mark.asyncio
async def test_symbol_is_not_case_normalized(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeResponseError, match="does not contain the requested symbol"):
        await get_instrument(recorder, clock, envelope([with_instrument(symbol="btcusdt")]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contractType": "LinearFutures"}, "contractType"),
        ({"contractType": "InversePerpetual"}, "contractType"),
        ({"quoteCoin": "USDC", "settleCoin": "USDC"}, "quoteCoin"),
        ({"settleCoin": "USDC"}, "settleCoin"),
        ({"status": "PreLaunch"}, "status"),
        ({"status": "Closed"}, "status"),
        ({"status": "Delivering"}, "status"),
        ({"isPreListing": True}, "pre-listing"),
    ],
)
async def test_unsupported_instrument_rejected(
    recorder: Recorder, clock: ManualClock, changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ExchangeRejectedError, match=f"unsupported instrument.*{message}"):
        await get_instrument(recorder, clock, envelope([with_instrument(**changes)]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "symbol",
        "contractType",
        "status",
        "baseCoin",
        "quoteCoin",
        "settleCoin",
        "priceFilter",
        "priceFilter.tickSize",
        "lotSizeFilter",
        "lotSizeFilter.qtyStep",
        "lotSizeFilter.minOrderQty",
        "lotSizeFilter.maxOrderQty",
        "lotSizeFilter.maxMktOrderQty",
        "lotSizeFilter.minNotionalValue",
    ],
)
async def test_missing_instrument_field(recorder: Recorder, clock: ManualClock, path: str) -> None:
    with pytest.raises(ExchangeResponseError, match=path.split(".")[-1]):
        await get_instrument(recorder, clock, envelope([with_instrument(**{path: _DELETE})]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value", ["", "abc", "1,5", "NaN", "Infinity", "-Infinity", " 0.1", 0.1, 1, True, None, ["0.1"]]
)
async def test_malformed_instrument_decimal(
    recorder: Recorder, clock: ManualClock, value: Any
) -> None:
    with pytest.raises(ExchangeResponseError, match="tickSize"):
        await get_instrument(
            recorder, clock, envelope([with_instrument(**{"priceFilter.tickSize": value})])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "value"), [("priceFilter.tickSize", "0"), ("lotSizeFilter.qtyStep", "-0.001")]
)
async def test_domain_invariants_apply(
    recorder: Recorder, clock: ManualClock, path: str, value: str
) -> None:
    with pytest.raises(ExchangeResponseError, match="must be > 0"):
        await get_instrument(recorder, clock, envelope([with_instrument(**{path: value})]))


# --- Ticker ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticker_mapping(recorder: Recorder, clock: ManualClock) -> None:
    ticker = await get_ticker(recorder, clock, envelope([TICKER]))

    assert ticker == Ticker(
        symbol="BTCUSDT",
        last_price=D("65000.50"),
        mark_price=D("64995.30"),
        best_bid=D("65000.40"),
        best_ask=D("65000.60"),
        funding_rate=D("0.0001"),
        next_funding_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        exchange_ts=datetime(2024, 2, 6, 2, 27, 31, 514000, tzinfo=UTC),
        received_ts=NOW,
    )


@pytest.mark.asyncio
async def test_ticker_request(recorder: Recorder, clock: ManualClock) -> None:
    await get_ticker(recorder, clock, envelope([TICKER]))

    [request] = recorder.requests
    assert request.method == "GET"
    assert request.url.path == "/v5/market/tickers"
    assert dict(request.url.params) == {"category": "linear", "symbol": "BTCUSDT"}
    assert_no_auth(request)


@pytest.mark.asyncio
async def test_received_ts_comes_from_clock_after_response(
    recorder: Recorder, clock: ManualClock
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        clock.advance(timedelta(milliseconds=250))  # time passes while waiting for the answer
        return httpx.Response(200, text=json.dumps(envelope([TICKER])))

    adapter, client = make_adapter(recorder, clock, handler)
    async with client:
        ticker = await adapter.get_ticker("BTCUSDT")
    assert ticker.received_ts == NOW + timedelta(milliseconds=250)


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", ["0.000375", "-0.0075", "0", "-0"])
async def test_funding_rate_any_sign(recorder: Recorder, clock: ManualClock, rate: str) -> None:
    ticker = await get_ticker(recorder, clock, envelope([with_ticker(fundingRate=rate)]))
    assert ticker.funding_rate == D(rate)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "attribute"),
    [
        ("markPrice", "mark_price"),
        ("bid1Price", "best_bid"),
        ("ask1Price", "best_ask"),
        ("fundingRate", "funding_rate"),
        ("nextFundingTime", "next_funding_at"),
    ],
)
async def test_empty_optional_fields_become_none(
    recorder: Recorder, clock: ManualClock, field: str, attribute: str
) -> None:
    ticker = await get_ticker(recorder, clock, envelope([with_ticker(**{field: ""})]))
    assert getattr(ticker, attribute) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "symbol",
        "lastPrice",
        "markPrice",
        "bid1Price",
        "ask1Price",
        "fundingRate",
        "nextFundingTime",
    ],
)
async def test_missing_ticker_field(recorder: Recorder, clock: ManualClock, field: str) -> None:
    # A documented field that is absent means the response shape changed: fail closed.
    with pytest.raises(ExchangeResponseError, match=field):
        await get_ticker(recorder, clock, envelope([with_ticker(**{field: _DELETE})]))


@pytest.mark.asyncio
async def test_empty_last_price_is_an_error(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeResponseError, match="lastPrice"):
        await get_ticker(recorder, clock, envelope([with_ticker(lastPrice="")]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lastPrice", "abc"),
        ("lastPrice", 65000.5),
        ("markPrice", "NaN"),
        ("bid1Price", 1),
        ("fundingRate", True),
    ],
)
async def test_malformed_ticker_decimal(
    recorder: Recorder, clock: ManualClock, field: str, value: Any
) -> None:
    with pytest.raises(ExchangeResponseError, match=field):
        await get_ticker(recorder, clock, envelope([with_ticker(**{field: value})]))


@pytest.mark.asyncio
async def test_crossed_book_is_rejected_by_domain(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeResponseError, match="best_bid"):
        await get_ticker(
            recorder, clock, envelope([with_ticker(bid1Price="65001", ask1Price="65000")])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["abc", "1.5", "-1", "1e12", 1768478400000, " 1768478400000"])
async def test_malformed_next_funding_time(
    recorder: Recorder, clock: ManualClock, value: Any
) -> None:
    with pytest.raises(ExchangeResponseError, match="nextFundingTime"):
        await get_ticker(recorder, clock, envelope([with_ticker(nextFundingTime=value)]))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["1707186451514", 1707186451514.0, -1, True, None])
async def test_malformed_envelope_time(recorder: Recorder, clock: ManualClock, value: Any) -> None:
    with pytest.raises(ExchangeResponseError, match="time"):
        await get_ticker(recorder, clock, envelope([TICKER], time=value))


@pytest.mark.asyncio
async def test_ticker_wrong_and_duplicate_symbol(recorder: Recorder, clock: ManualClock) -> None:
    with pytest.raises(ExchangeResponseError, match="does not contain the requested symbol"):
        await get_ticker(recorder, clock, envelope([with_ticker(symbol="ETHUSDT")]))
    with pytest.raises(ExchangeResponseError, match="more than one"):
        await get_ticker(recorder, clock, envelope([TICKER, TICKER]))
    with pytest.raises(ExchangeRejectedError, match="not found"):
        await get_ticker(recorder, clock, envelope([]))


# --- Envelope and transport errors -------------------------------------------------------

GETTERS = ["get_instrument", "get_ticker"]


async def call(adapter: BybitMarketDataClient, getter: str) -> object:
    if getter == "get_instrument":
        return await adapter.get_instrument("BTCUSDT")
    return await adapter.get_ticker("BTCUSDT")


@pytest.mark.asyncio
@pytest.mark.parametrize("getter", GETTERS)
@pytest.mark.parametrize(
    ("handler", "error", "message"),
    [
        (
            json_response(envelope([], retCode=10001, retMsg="params error")),
            ExchangeRejectedError,
            "10001",
        ),
        (
            json_response(envelope([], retCode=10006, retMsg="Too many visits")),
            ExchangeRejectedError,
            "10006",
        ),
        (json_response({"retCode": "0", "retMsg": "OK"}), ExchangeResponseError, "retCode"),
        (json_response(envelope([], category="spot")), ExchangeResponseError, "category"),
        (json_response({**envelope([]), "result": []}), ExchangeResponseError, "result"),
        (
            json_response({**envelope([]), "result": {"category": "linear"}}),
            ExchangeResponseError,
            "list",
        ),
        (
            json_response({**envelope([]), "result": {"category": "linear", "list": {}}}),
            ExchangeResponseError,
            "list",
        ),
        (json_response(envelope(["BTCUSDT"])), ExchangeResponseError, "list item"),  # type: ignore[list-item]
        (json_response([1, 2]), ExchangeResponseError, "JSON object"),
        (text_response("<html>Bad gateway</html>"), ExchangeResponseError, "not valid JSON"),
        (text_response('{"retCode": NaN}'), ExchangeResponseError, "not valid JSON"),
        (text_response(""), ExchangeResponseError, "not valid JSON"),
        (json_response(envelope([]), 403), ExchangeRejectedError, "HTTP 403"),
        (json_response(envelope([]), 429), ExchangeRejectedError, "HTTP 429"),
        (json_response(envelope([]), 404), ExchangeRejectedError, "HTTP 404"),
        (json_response(envelope([]), 500), ExchangeResponseError, "HTTP 500"),
        (json_response(envelope([]), 503), ExchangeResponseError, "HTTP 503"),
    ],
)
async def test_envelope_and_status_errors(
    recorder: Recorder,
    clock: ManualClock,
    getter: str,
    handler: Handler,
    error: type[ExchangeError],
    message: str,
) -> None:
    adapter, client = make_adapter(recorder, clock, handler)
    async with client:
        with pytest.raises(error, match=message) as info:
            await call(adapter, getter)
    assert info.value.__cause__ is None or isinstance(info.value.__cause__, Exception)


def raising(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("getter", GETTERS)
@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (httpx.ConnectError("connection refused"), ExchangeNotSentError),
        (httpx.ConnectTimeout("connect timeout"), ExchangeNotSentError),
        (httpx.PoolTimeout("pool timeout"), ExchangeNotSentError),
        (httpx.ReadTimeout("read timeout"), ExchangeResponseError),
        (httpx.WriteTimeout("write timeout"), ExchangeResponseError),
        (httpx.ReadError("connection reset"), ExchangeResponseError),
        (httpx.RemoteProtocolError("server disconnected"), ExchangeResponseError),
    ],
)
async def test_transport_errors(
    recorder: Recorder, clock: ManualClock, getter: str, exc: Exception, error: type[ExchangeError]
) -> None:
    adapter, client = make_adapter(recorder, clock, raising(exc))
    async with client:
        with pytest.raises(error):
            await call(adapter, getter)


@pytest.mark.asyncio
@pytest.mark.parametrize("getter", GETTERS)
async def test_read_errors_are_never_ambiguous(
    recorder: Recorder, clock: ManualClock, getter: str
) -> None:
    adapter, client = make_adapter(recorder, clock, raising(httpx.ReadTimeout("timeout")))
    async with client:
        with pytest.raises(ExchangeError) as info:
            await call(adapter, getter)
    assert not isinstance(info.value, ExchangeAmbiguousResultError)


@pytest.mark.asyncio
async def test_error_messages_do_not_contain_response_body(
    recorder: Recorder, clock: ManualClock
) -> None:
    body = "<html>" + "x" * 5000 + " internal details </html>"
    adapter, client = make_adapter(recorder, clock, text_response(body, 502))
    async with client:
        with pytest.raises(ExchangeResponseError) as info:
            await adapter.get_ticker("BTCUSDT")
    assert "internal details" not in str(info.value)
    assert len(str(info.value)) < 300


@pytest.mark.asyncio
async def test_long_ret_msg_is_truncated(recorder: Recorder, clock: ManualClock) -> None:
    adapter, client = make_adapter(
        recorder, clock, json_response(envelope([], retCode=10001, retMsg="m" * 1000))
    )
    async with client:
        with pytest.raises(ExchangeRejectedError) as info:
            await adapter.get_ticker("BTCUSDT")
    assert len(str(info.value)) < 300


# --- Client ownership and no authentication ----------------------------------------------


@pytest.mark.asyncio
async def test_adapter_does_not_close_injected_client(
    recorder: Recorder, clock: ManualClock
) -> None:
    adapter, client = make_adapter(recorder, clock, json_response(envelope([TICKER])))
    await adapter.get_ticker("BTCUSDT")
    assert not client.is_closed
    await client.aclose()


def assert_no_auth(request: httpx.Request) -> None:
    headers = {name.lower() for name in request.headers}
    assert "authorization" not in headers
    assert not [h for h in headers if h.startswith("x-bapi")]
    params = {name.lower() for name in request.url.params}
    assert not params & {"api_key", "apikey", "sign", "signature", "timestamp", "recv_window"}
    assert request.content == b""


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to use a real transport fails the test."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("real network access in a unit test")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
