"""Bybit private REST transport: HMAC signing and outcome classification.

Mocked HTTP transport only. Expected signatures were computed independently with
``openssl dgst -sha256 -hmac`` over the literal pre-sign strings below, following
the official Bybit V5 rule (docs/v5/guide.mdx, "Create A Request"):
GET: timestamp + api_key + recv_window + queryString;
POST: timestamp + api_key + recv_window + jsonBodyString; HMAC_SHA256, lowercase hex.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest

from app.domain.clock import ManualClock
from app.exchanges.bybit.credentials import BybitCredentials
from app.exchanges.bybit.private_rest import BybitPrivateRestTransport, BybitResponse
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeAuthenticationError,
    ExchangeError,
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeResponseError,
)

API_KEY = "TEST_API_KEY_DO_NOT_LEAK_7f3a"
API_SECRET = "TEST_API_SECRET_DO_NOT_LEAK_9c1e"
BASE = "https://bybit.example"
# 2026-01-15T12:00:00.123456Z -> 1768478400123 ms (sub-millisecond part truncated)
NOW = datetime(2026, 1, 15, 12, 0, 0, 123456, tzinfo=UTC)
TIMESTAMP = "1768478400123"
SERVER_MS = 1768478400200

GET_QUERY = "category=linear&symbol=BTCUSDT"
GET_SIGNATURE = "4435601c14cd8c952b4d9b82f6d064c311675113b44e17b03679bd94ab679a06"
POST_BODY = (
    '{"category":"linear","symbol":"BTCUSDT","side":"Buy","orderType":"Limit",'
    '"qty":"0.001","price":"65000.5","orderLinkId":"grid1-buy-0001","reduceOnly":false}'
)
POST_SIGNATURE = "687b5788ed1bf14786a4550beac4a89b6620463794ee473c0ac29bcfc98952b7"
EMPTY_QUERY_SIGNATURE = "c562f356bba2e2322bd31bec2c2e3b07db1473752a33ad2c9f0035ca24aa1a72"
SPECIAL_QUERY = "symbol=BTCUSDT&cursor=a%20b%2Fc%26d%3D%C3%A9~&limit=20&category=linear"
SPECIAL_SIGNATURE = "169e56023364e63d8d2ca50ed7abfb962d5abf5aff2db373c1db199787d452fc"

Handler = Callable[[httpx.Request], httpx.Response]


def ok(result: Any = None, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {"orderId": "o-1", "orderLinkId": "grid1-buy-0001"} if result is None else result,
        "retExtInfo": {},
        "time": SERVER_MS,
    }
    body.update(overrides)
    return body


def json_response(body: Any, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, text=json.dumps(body))


def text_response(text: str, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, text=text)


def raising(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


class Harness:
    def __init__(self, handler: Handler, recv_window_ms: int = 5000) -> None:
        self.requests: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(record))
        self.transport = BybitPrivateRestTransport(
            client=self.client,
            base_url=BASE,
            credentials=BybitCredentials(api_key=API_KEY, api_secret=API_SECRET),
            clock=ManualClock(NOW),
            recv_window_ms=recv_window_ms,
        )


async def do_get(
    handler: Handler, params: dict[str, Any] | None = None
) -> tuple[Harness, BybitResponse]:
    harness = Harness(handler)
    async with harness.client:
        response = await harness.transport.get(
            "/v5/order/realtime",
            params={"category": "linear", "symbol": "BTCUSDT"} if params is None else params,
        )
    return harness, response


async def do_post(
    handler: Handler, body: dict[str, Any] | None = None
) -> tuple[Harness, BybitResponse]:
    harness = Harness(handler)
    async with harness.client:
        response = await harness.transport.post_mutating(
            "/v5/order/create", body=json.loads(POST_BODY) if body is None else body
        )
    return harness, response


def assert_no_secret_in_request(request: httpx.Request) -> None:
    raw = (
        str(request.url).encode()
        + b"".join(name + b": " + value for name, value in request.headers.raw)
        + request.content
    )
    assert API_SECRET.encode() not in raw


def assert_no_credentials(text: str) -> None:
    assert API_KEY not in text
    assert API_SECRET not in text


# --- Credentials -------------------------------------------------------------------------


def test_credentials_repr_is_masked() -> None:
    credentials = BybitCredentials(api_key=API_KEY, api_secret=API_SECRET)
    for text in (repr(credentials), str(credentials), f"{credentials}", f"{credentials!r}"):
        assert_no_credentials(text)


@pytest.mark.parametrize("field", ["api_key", "api_secret"])
@pytest.mark.parametrize("value", ["", "   ", f" {API_SECRET}", f"{API_SECRET}\n", None, 1])
def test_credentials_validated_without_echo(field: str, value: Any) -> None:
    values: dict[str, Any] = {"api_key": API_KEY, "api_secret": API_SECRET, field: value}
    with pytest.raises(ValueError, match=field) as info:
        BybitCredentials(**values)
    assert_no_credentials(str(info.value))


# --- Construction ------------------------------------------------------------------------


def make(**overrides: Any) -> BybitPrivateRestTransport:
    values: dict[str, Any] = {
        "client": httpx.AsyncClient(),
        "base_url": BASE,
        "credentials": BybitCredentials(api_key=API_KEY, api_secret=API_SECRET),
        "clock": ManualClock(NOW),
    }
    values.update(overrides)
    return BybitPrivateRestTransport(**values)


def test_transport_repr_has_no_credentials() -> None:
    transport = make()
    assert_no_credentials(repr(transport))
    assert_no_credentials(str(transport))
    assert "bybit.example" in repr(transport)


@pytest.mark.parametrize(
    "base_url", ["http://bybit.example", "https://bybit.example/", "", "bybit.example", None]
)
def test_base_url_validated(base_url: Any) -> None:
    with pytest.raises(ValueError, match="base_url"):
        make(base_url=base_url)


@pytest.mark.parametrize("value", [0, -1, True, 5000.0, "5000", None])
def test_recv_window_validated(value: Any) -> None:
    with pytest.raises(ValueError, match="recv_window_ms"):
        make(recv_window_ms=value)


def test_recv_window_default_is_documented_5000() -> None:
    assert make()._recv_window_ms == 5000


def test_credentials_type_required() -> None:
    with pytest.raises(TypeError, match="credentials"):
        make(credentials=(API_KEY, API_SECRET))


# --- Signing vectors ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_signing_vector() -> None:
    harness, _ = await do_get(json_response(ok()))
    [request] = harness.requests

    assert request.method == "GET"
    assert request.url.path == "/v5/order/realtime"
    assert request.url.query == GET_QUERY.encode()
    assert request.headers["X-BAPI-API-KEY"] == API_KEY
    assert request.headers["X-BAPI-TIMESTAMP"] == TIMESTAMP
    assert request.headers["X-BAPI-RECV-WINDOW"] == "5000"
    assert request.headers["X-BAPI-SIGN"] == GET_SIGNATURE
    assert "content-type" not in request.headers
    assert request.content == b""
    assert_no_secret_in_request(request)


@pytest.mark.asyncio
async def test_post_signing_vector_and_exact_body() -> None:
    harness, _ = await do_post(json_response(ok()))
    [request] = harness.requests

    assert request.method == "POST"
    assert request.url.path == "/v5/order/create"
    assert request.url.query == b""
    assert request.content == POST_BODY.encode()  # exactly the signed text
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-BAPI-API-KEY"] == API_KEY
    assert request.headers["X-BAPI-TIMESTAMP"] == TIMESTAMP
    assert request.headers["X-BAPI-RECV-WINDOW"] == "5000"
    assert request.headers["X-BAPI-SIGN"] == POST_SIGNATURE
    assert_no_secret_in_request(request)


@pytest.mark.asyncio
async def test_empty_query_signs_empty_string() -> None:
    harness, _ = await do_get(json_response(ok()), params={})
    [request] = harness.requests
    assert request.url.query == b""
    assert str(request.url) == f"{BASE}/v5/order/realtime"
    assert request.headers["X-BAPI-SIGN"] == EMPTY_QUERY_SIGNATURE


@pytest.mark.asyncio
async def test_query_order_special_characters_and_no_double_encoding() -> None:
    params = {"symbol": "BTCUSDT", "cursor": "a b/c&d=é~", "limit": 20, "category": "linear"}
    harness, _ = await do_get(json_response(ok()), params=params)
    [request] = harness.requests
    # Caller order is kept (no sorting), values are percent-encoded once, and httpx
    # sends exactly the signed string.
    assert request.url.query == SPECIAL_QUERY.encode()
    assert request.headers["X-BAPI-SIGN"] == SPECIAL_SIGNATURE


@pytest.mark.asyncio
async def test_literal_percent_is_encoded_once() -> None:
    harness, _ = await do_get(json_response(ok()), params={"cursor": "50%"})
    assert harness.requests[0].url.query == b"cursor=50%25"


@pytest.mark.asyncio
async def test_timestamp_and_recv_window_from_dependencies() -> None:
    harness = Harness(json_response(ok()), recv_window_ms=2500)
    clock = harness.transport._clock
    assert isinstance(clock, ManualClock)
    async with harness.client:
        await harness.transport.get("/v5/order/realtime", params={})
        clock.advance(timedelta(milliseconds=1, microseconds=999))
        await harness.transport.get("/v5/order/realtime", params={})
    stamps = [r.headers["X-BAPI-TIMESTAMP"] for r in harness.requests]
    assert stamps == ["1768478400123", "1768478400125"]
    assert {r.headers["X-BAPI-RECV-WINDOW"] for r in harness.requests} == {"2500"}


@pytest.mark.asyncio
async def test_post_body_serialization_policy() -> None:
    body = {"b": "é", "a": 1, "nested": {"z": [True, None, "x"]}}
    harness, _ = await do_post(json_response(ok()), body=body)
    # Caller key order kept, compact separators, ASCII-only escapes.
    assert harness.requests[0].content == b'{"b":"\\u00e9","a":1,"nested":{"z":[true,null,"x"]}}'


@pytest.mark.parametrize(
    "params",
    [
        {"qty": 0.5},
        {"qty": Decimal("0.5")},
        {"reduceOnly": True},
        {"symbol": None},
        {"": "x"},
        {"bad key": "x"},
        {"k": ["a"]},
    ],
)
@pytest.mark.asyncio
async def test_unsupported_query_values_rejected_before_sending(params: dict[str, Any]) -> None:
    harness = Harness(json_response(ok()))
    async with harness.client:
        with pytest.raises(ExchangeNotSentError):
            await harness.transport.get("/v5/order/realtime", params=params)
    assert harness.requests == []


@pytest.mark.parametrize(
    "body",
    [
        {"qty": 0.5},
        {"qty": Decimal("0.5")},
        {"qty": float("nan")},
        {1: "x"},
        {"nested": {"x": 1.5}},
        {"items": [Decimal("1")]},
        {"tags": {"a"}},
    ],
)
@pytest.mark.asyncio
async def test_unsupported_body_values_rejected_before_sending(body: dict[Any, Any]) -> None:
    harness = Harness(json_response(ok()))
    async with harness.client:
        with pytest.raises(ExchangeNotSentError):
            await harness.transport.post_mutating("/v5/order/create", body=body)
    assert harness.requests == []


# --- Path safety -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/v5/order/create",
        "//evil.example/v5/order/create",
        "v5/order/create",
        "/v4/order/create",
        "/v5",
        "/v5/",
        "/v5/../order",
        "/v5/order/../../x",
        "/v5//order",
        "/v5/order/create?category=linear",
        "/v5/order/create#x",
        "/v5/%2e%2e/x",
        "/v5/order create",
        "/v5/order/créate",
    ],
)
@pytest.mark.asyncio
async def test_unsafe_paths_rejected_before_sending(path: str) -> None:
    harness = Harness(json_response(ok()))
    async with harness.client:
        with pytest.raises(ExchangeNotSentError, match="path"):
            await harness.transport.get(path, params={})
        with pytest.raises(ExchangeNotSentError, match="path"):
            await harness.transport.post_mutating(path, body={})
    assert harness.requests == []


# --- Success -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_returns_result_ext_info_and_server_time() -> None:
    _, response = await do_post(json_response(ok(retExtInfo={"list": [{"code": 0, "msg": "OK"}]})))
    assert response == BybitResponse(
        result={"orderId": "o-1", "orderLinkId": "grid1-buy-0001"},
        ret_ext_info={"list": [{"code": 0, "msg": "OK"}]},
        server_time=datetime(2026, 1, 15, 12, 0, 0, 200000, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_numbers_in_result_never_become_float() -> None:
    _, response = await do_get(json_response(ok(result={"price": 0.1})))
    assert isinstance(response.result["price"], Decimal)


# --- GET error classification (read-only: never ambiguous) ------------------------------

GET_CASES: list[tuple[Handler, type[ExchangeError], str]] = [
    (raising(httpx.ConnectError("refused")), ExchangeNotSentError, "not sent"),
    (raising(httpx.ConnectTimeout("t")), ExchangeNotSentError, "not sent"),
    (raising(httpx.PoolTimeout("t")), ExchangeNotSentError, "not sent"),
    (raising(httpx.ReadTimeout("t")), ExchangeResponseError, "ReadTimeout"),
    (raising(httpx.WriteError("t")), ExchangeResponseError, "WriteError"),
    (raising(httpx.RemoteProtocolError("t")), ExchangeResponseError, "RemoteProtocolError"),
    (raising(httpx.ProxyError("t")), ExchangeResponseError, "ProxyError"),
    (json_response(ok(), 400), ExchangeRejectedError, "HTTP 400"),
    (json_response(ok(), 404), ExchangeRejectedError, "HTTP 404"),
    (json_response(ok(), 401), ExchangeAuthenticationError, "HTTP 401"),
    (json_response(ok(), 403), ExchangeResponseError, "HTTP 403"),
    (json_response(ok(), 408), ExchangeResponseError, "HTTP 408"),
    (json_response(ok(), 429), ExchangeResponseError, "HTTP 429"),
    (json_response(ok(), 500), ExchangeResponseError, "HTTP 500"),
    (text_response("<html>oops</html>"), ExchangeResponseError, "not valid JSON"),
    (json_response([1]), ExchangeResponseError, "JSON object"),
    (json_response(ok(time="1")), ExchangeResponseError, "time"),
    (json_response(ok(result=[])), ExchangeResponseError, "result"),
    (json_response(ok(retExtInfo=None)), ExchangeResponseError, "retExtInfo"),
    (json_response(ok(retCode=10003)), ExchangeAuthenticationError, "10003"),
    (json_response(ok(retCode=10004)), ExchangeAuthenticationError, "10004"),
    (json_response(ok(retCode=10005)), ExchangeAuthenticationError, "10005"),
    (json_response(ok(retCode=10007)), ExchangeAuthenticationError, "10007"),
    (json_response(ok(retCode=10010)), ExchangeAuthenticationError, "10010"),
    (json_response(ok(retCode=33004)), ExchangeAuthenticationError, "33004"),
    (json_response(ok(retCode=10001)), ExchangeRejectedError, "10001"),
    (json_response(ok(retCode=10002)), ExchangeRejectedError, "10002"),
    (json_response(ok(retCode=10000)), ExchangeResponseError, "10000"),
    (json_response(ok(retCode=10006)), ExchangeResponseError, "10006"),
    (json_response(ok(retCode=10016)), ExchangeResponseError, "10016"),
    (json_response(ok(retCode=429)), ExchangeResponseError, "retCode 429"),
    (json_response(ok(retCode=99999)), ExchangeResponseError, "99999"),
]


@pytest.mark.parametrize(("handler", "error", "message"), GET_CASES)
@pytest.mark.asyncio
async def test_get_error_classification(
    handler: Handler, error: type[ExchangeError], message: str
) -> None:
    with pytest.raises(error, match=message) as info:
        await do_get(handler)
    assert type(info.value) is error
    assert not isinstance(info.value, ExchangeAmbiguousResultError)
    assert_no_credentials(str(info.value))
    assert_no_credentials(repr(info.value))


# --- Mutating POST error classification --------------------------------------------------

POST_CASES: list[tuple[Handler, type[ExchangeError], str]] = [
    # Provably not sent: the connection was never established / acquired.
    (raising(httpx.ConnectError("refused")), ExchangeNotSentError, "not sent"),
    (raising(httpx.ConnectTimeout("t")), ExchangeNotSentError, "not sent"),
    (raising(httpx.PoolTimeout("t")), ExchangeNotSentError, "not sent"),
    # Possibly sent: outcome unknown.
    (raising(httpx.ReadTimeout("t")), ExchangeAmbiguousResultError, "ReadTimeout"),
    (raising(httpx.WriteTimeout("t")), ExchangeAmbiguousResultError, "WriteTimeout"),
    (raising(httpx.WriteError("t")), ExchangeAmbiguousResultError, "WriteError"),
    (raising(httpx.ReadError("t")), ExchangeAmbiguousResultError, "ReadError"),
    (raising(httpx.RemoteProtocolError("t")), ExchangeAmbiguousResultError, "RemoteProtocolError"),
    (raising(httpx.ProxyError("t")), ExchangeAmbiguousResultError, "ProxyError"),
    (json_response(ok(), 400), ExchangeRejectedError, "HTTP 400"),
    (json_response(ok(), 404), ExchangeRejectedError, "HTTP 404"),
    (json_response(ok(), 401), ExchangeAuthenticationError, "HTTP 401"),
    (json_response(ok(), 403), ExchangeAmbiguousResultError, "HTTP 403"),
    (json_response(ok(), 408), ExchangeAmbiguousResultError, "HTTP 408"),
    (json_response(ok(), 429), ExchangeAmbiguousResultError, "HTTP 429"),
    (json_response(ok(), 500), ExchangeAmbiguousResultError, "HTTP 500"),
    (json_response(ok(), 502), ExchangeAmbiguousResultError, "HTTP 502"),
    (json_response(ok(), 302), ExchangeAmbiguousResultError, "HTTP 302"),
    (text_response("<html>oops</html>"), ExchangeAmbiguousResultError, "not valid JSON"),
    (text_response(""), ExchangeAmbiguousResultError, "not valid JSON"),
    (json_response({"retCode": 0}), ExchangeAmbiguousResultError, "time"),
    (json_response(ok(result="done")), ExchangeAmbiguousResultError, "result"),
    (json_response(ok(retCode=10003)), ExchangeAuthenticationError, "10003"),
    (json_response(ok(retCode=10004)), ExchangeAuthenticationError, "10004"),
    (json_response(ok(retCode=10005)), ExchangeAuthenticationError, "10005"),
    (json_response(ok(retCode=10001)), ExchangeRejectedError, "10001"),
    (json_response(ok(retCode=10002)), ExchangeRejectedError, "10002"),
    (json_response(ok(retCode=10000)), ExchangeAmbiguousResultError, "10000"),
    (json_response(ok(retCode=10006)), ExchangeAmbiguousResultError, "10006"),
    (json_response(ok(retCode=10016)), ExchangeAmbiguousResultError, "10016"),
    (json_response(ok(retCode=429)), ExchangeAmbiguousResultError, "retCode 429"),
    (json_response(ok(retCode=99999)), ExchangeAmbiguousResultError, "99999"),
    (json_response(ok(retCode="0")), ExchangeAmbiguousResultError, "retCode"),
]


@pytest.mark.parametrize(("handler", "error", "message"), POST_CASES)
@pytest.mark.asyncio
async def test_post_error_classification(
    handler: Handler, error: type[ExchangeError], message: str
) -> None:
    with pytest.raises(error, match=message) as info:
        await do_post(handler)
    assert type(info.value) is error
    assert not isinstance(info.value, ExchangeResponseError)  # read-only category
    assert_no_credentials(str(info.value))
    assert_no_credentials(repr(info.value))


# --- Secret leakage ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("do", [do_get, do_post])
async def test_ret_msg_echoing_credentials_is_redacted(do: Any) -> None:
    body = ok(retCode=10003, retMsg=f"API key {API_KEY} invalid, secret {API_SECRET}")
    with pytest.raises(ExchangeAuthenticationError) as info:
        await do(json_response(body))
    assert_no_credentials(str(info.value))
    assert "[REDACTED]" in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("do", [do_get, do_post])
async def test_error_never_contains_response_body(do: Any) -> None:
    body = f"<html>{API_KEY} {API_SECRET} internal trace</html>"
    with pytest.raises(ExchangeError) as info:
        await do(text_response(body, 502))
    assert "internal trace" not in str(info.value)
    assert_no_credentials(str(info.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("do", [do_get, do_post])
async def test_secret_never_sent(do: Any) -> None:
    harness, _ = await do(json_response(ok()))
    for request in harness.requests:
        assert_no_secret_in_request(request)


# --- Ownership, no network ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_does_not_close_injected_client() -> None:
    harness = Harness(json_response(ok()))
    await harness.transport.get("/v5/order/realtime", params={})
    assert not harness.client.is_closed
    await harness.client.aclose()


def test_pre_send_classification_matches_installed_httpcore() -> None:
    # Pre-send semantics were verified against httpcore 1.x (ConnectError/ConnectTimeout
    # only while establishing a connection, PoolTimeout while waiting for one).
    import httpcore

    assert httpcore.__version__.startswith("1.")


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("real network access in a unit test")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
