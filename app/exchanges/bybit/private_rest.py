"""Bybit V5 private REST transport: HMAC signing and outcome classification.

Official contract (bybit-exchange/docs, docs/v5/guide.mdx):
* headers ``X-BAPI-API-KEY``, ``X-BAPI-TIMESTAMP`` (UTC ms), ``X-BAPI-SIGN``,
  ``X-BAPI-RECV-WINDOW`` (ms, documented default 5000);
* string to sign: GET ``timestamp + api_key + recv_window + queryString``,
  POST ``timestamp + api_key + recv_window + jsonBodyString``;
* HMAC_SHA256 with the API secret, lowercase hex.

The signed query string / JSON body is built once and sent byte-for-byte as
signed. Business endpoints (orders, account) are not implemented here.

Outcome classification (duplicate-order prevention over saved reconciliations):
* ``post_mutating``:
  - ``ExchangeNotSentError`` only when not sending is proven before entering the
    HTTP send path: local validation (path, body, serialization) or httpx
    ``PoolTimeout`` (documented as waiting to acquire a pool connection).
  - Any other transport failure (connect, read, write, protocol, proxy, unknown)
    and any non-2xx HTTP status: ``ExchangeAmbiguousResultError``; an HTTP error
    status is not a documented Bybit business outcome.
  - HTTP 2xx with a valid envelope: documented auth codes -> authentication error,
    10001 / 10002 -> rejected, everything else non-zero -> ambiguous; a malformed
    body -> ambiguous.
  After an ambiguous outcome the order is reconciled by client order id; it is
  never re-sent blindly.
* ``get`` (read-only, no side effects): never ambiguous; connection failures are
  "not sent", a missing or broken answer is ``ExchangeResponseError``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final, NoReturn
from urllib.parse import quote

import httpx

from app.domain.clock import Clock
from app.domain.errors import DomainError
from app.domain.validation import require_utc, utc_from_ms
from app.exchanges.bybit.credentials import BybitCredentials
from app.exchanges.bybit.types import JsonValue
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeAuthenticationError,
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeResponseError,
)

DEFAULT_RECV_WINDOW_MS: Final = 5000

# retCode values (docs/v5/error, UTA) whose meaning we rely on.
AUTH_RET_CODES: Final = frozenset(
    {
        10003,  # API key is invalid
        10004,  # Error sign
        10005,  # Permission denied
        10007,  # User authentication failed
        10010,  # Unmatched IP
        33004,  # (Derivatives) API key has expired
    }
)
# The request was refused before processing (bad parameters, timestamp outside
# recv_window): definitive, also for mutating requests.
REQUEST_ERROR_RET_CODES: Final = frozenset({10001, 10002})
# Everything else (10000 server timeout, 10006 rate limit, 10016 server error,
# 429 high load, unknown codes) is "no definitive answer": a response error for
# reads, ambiguous for mutating requests.

# For read-only GET only: HTTP statuses that describe the request itself
# (docs/v5/error): 400 malformed, 404 path not found -> refused; 401 ->
# authentication refused. Mutating POST treats every non-2xx as ambiguous.
_READ_REQUEST_ERROR_HTTP: Final = frozenset({400, 404})
_READ_AUTH_HTTP: Final = frozenset({401})

_PATH: Final = re.compile(r"/v5(?:/[A-Za-z0-9_-]+)+\Z")
_PARAM_NAME: Final = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
# Read-only requests: no connection means nothing was read; no side effect either way.
_READ_NOT_SENT_ERRORS: Final = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
# Mutating requests: only waiting for a pool connection (public httpx semantics:
# "Timed out waiting to acquire a connection from the pool") precedes any send.
_MUTATING_NOT_SENT_ERRORS: Final = (httpx.PoolTimeout,)
_MAX_MESSAGE: Final = 120
_REDACTED: Final = "[REDACTED]"

QueryValue = str | int


@dataclass(frozen=True, slots=True)
class BybitResponse:
    """A successful (``retCode == 0``) private response."""

    result: dict[str, Any]
    ret_ext_info: dict[str, Any]
    """Extended info; batch endpoints report per-item outcomes here."""
    server_time: datetime
    """Bybit server response timestamp (envelope ``time``)."""


def _unix_millis(moment: datetime) -> int:
    """UTC datetime -> integer Unix milliseconds (sub-millisecond part truncated)."""
    delta = require_utc(moment, "now") - _EPOCH
    millis = (delta.days * 86_400 + delta.seconds) * 1000 + delta.microseconds // 1000
    if millis < 0:
        raise ValueError("clock time before the Unix epoch")
    return millis


def _encode_query(params: Mapping[str, QueryValue]) -> str:
    """Canonical query string: caller order, values percent-encoded once (UTF-8,
    only RFC 3986 unreserved characters left as is)."""
    parts: list[str] = []
    for name, value in params.items():
        if not isinstance(name, str) or not _PARAM_NAME.match(name):
            raise ExchangeNotSentError("Bybit: invalid query parameter name")
        if isinstance(value, bool) or not isinstance(value, str | int):
            # bool / float / Decimal have no documented query representation:
            # callers convert them to the documented string form first.
            raise ExchangeNotSentError(f"Bybit: unsupported value type for query parameter {name}")
        parts.append(f"{name}={quote(str(value), safe='')}")
    return "&".join(parts)


def _check_json(value: object, where: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExchangeNotSentError(f"Bybit: non-string key in {where}")
            _check_json(item, f"{where}.{key}")
        return
    # float / Decimal / anything else: numbers must be sent as documented strings.
    raise ExchangeNotSentError(f"Bybit: unsupported JSON value type in {where}")


def _encode_body(body: Mapping[str, JsonValue]) -> str:
    """Canonical JSON body: caller key order, compact separators, ASCII-only."""
    _check_json(dict(body), "body")
    return json.dumps(dict(body), separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant {name}")


class BybitPrivateRestTransport:
    """Signed requests to Bybit V5 private REST endpoints.

    The HTTP client is injected and owned by the caller; it is never closed here.
    Timestamps come only from the injected ``Clock``.
    """

    __slots__ = ("_base_url", "_client", "_clock", "_credentials", "_recv_window_ms")

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        credentials: BybitCredentials,
        clock: Clock,
        recv_window_ms: int = DEFAULT_RECV_WINDOW_MS,
    ) -> None:
        if (
            not isinstance(base_url, str)
            or not base_url.startswith("https://")
            or base_url.endswith("/")
            or len(base_url) <= len("https://")
        ):
            raise ValueError("base_url must be an https:// URL without a trailing slash")
        if not isinstance(credentials, BybitCredentials):
            raise TypeError("credentials must be BybitCredentials")
        if type(recv_window_ms) is not int or recv_window_ms <= 0:
            raise ValueError("recv_window_ms must be an int > 0")
        self._client = client
        self._base_url = base_url
        self._credentials = credentials
        self._clock = clock
        self._recv_window_ms = recv_window_ms

    def __repr__(self) -> str:
        return (
            f"BybitPrivateRestTransport(base_url={self._base_url!r}, "
            f"recv_window_ms={self._recv_window_ms})"
        )

    async def get(self, path: str, *, params: Mapping[str, QueryValue]) -> BybitResponse:
        """Signed read-only GET. Never raises ExchangeAmbiguousResultError."""
        self._check_path(path)
        query = _encode_query(params)
        url = f"{self._base_url}{path}?{query}" if query else f"{self._base_url}{path}"
        headers = self._auth_headers(query)
        return await self._send("GET", path, url, headers, None, mutating=False)

    async def post_mutating(self, path: str, *, body: Mapping[str, JsonValue]) -> BybitResponse:
        """Signed POST that may change exchange state (e.g. create/cancel order).

        Without a definitive answer the outcome is ExchangeAmbiguousResultError.
        """
        self._check_path(path)
        payload = _encode_body(body)
        headers = self._auth_headers(payload)
        headers["Content-Type"] = "application/json"
        return await self._send(
            "POST", path, f"{self._base_url}{path}", headers, payload.encode("ascii"), mutating=True
        )

    # --- internals ---------------------------------------------------------------------

    @staticmethod
    def _check_path(path: str) -> None:
        # Only relative /v5/... paths: the API key can never go to another host.
        if not isinstance(path, str) or not _PATH.match(path):
            raise ExchangeNotSentError("Bybit: invalid private API path")

    def _auth_headers(self, signed_part: str) -> dict[str, str]:
        timestamp = str(_unix_millis(self._clock.now()))
        recv_window = str(self._recv_window_ms)
        api_key = self._credentials.api_key
        payload = f"{timestamp}{api_key}{recv_window}{signed_part}"
        signature = hmac.new(
            self._credentials.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

    def _sanitize(self, text: str) -> str:
        text = text[:_MAX_MESSAGE].replace("\n", " ")
        for value in (self._credentials.api_secret, self._credentials.api_key):
            text = text.replace(value, _REDACTED)
        return text

    async def _send(
        self,
        method: str,
        path: str,
        url: str,
        headers: dict[str, str],
        content: bytes | None,
        *,
        mutating: bool,
    ) -> BybitResponse:
        where = f"Bybit private {method} {path}"

        def no_answer(reason: str) -> NoReturn:
            if mutating:
                raise ExchangeAmbiguousResultError(f"{where}: outcome unknown, {reason}")
            raise ExchangeResponseError(f"{where}: {reason}")

        not_sent_errors = _MUTATING_NOT_SENT_ERRORS if mutating else _READ_NOT_SENT_ERRORS
        try:
            response = await self._client.request(method, url, headers=headers, content=content)
        except not_sent_errors as exc:
            raise ExchangeNotSentError(f"{where}: not sent ({type(exc).__name__})") from None
        except httpx.RequestError as exc:
            no_answer(f"no response ({type(exc).__name__})")

        status = response.status_code
        if not 200 <= status < 300:
            if mutating:
                # An HTTP error status is not a documented Bybit business outcome.
                no_answer(f"HTTP {status}")
            if status in _READ_AUTH_HTTP:
                raise ExchangeAuthenticationError(f"{where}: authentication refused, HTTP {status}")
            if status in _READ_REQUEST_ERROR_HTTP:
                raise ExchangeRejectedError(f"{where}: request refused, HTTP {status}")
            no_answer(f"HTTP {status}")

        try:
            body = json.loads(
                response.content, parse_float=Decimal, parse_constant=_reject_constant
            )
        except ValueError:
            no_answer("response is not valid JSON")
        if not isinstance(body, dict):
            no_answer("response is not a JSON object")

        ret_code = body.get("retCode")
        if type(ret_code) is not int:
            no_answer("field retCode is missing or not an integer")
        if ret_code != 0:
            ret_msg = body.get("retMsg")
            detail = self._sanitize(ret_msg) if isinstance(ret_msg, str) else ""
            if ret_code in AUTH_RET_CODES:
                raise ExchangeAuthenticationError(f"{where}: retCode {ret_code}: {detail}")
            if ret_code in REQUEST_ERROR_RET_CODES:
                raise ExchangeRejectedError(f"{where}: retCode {ret_code}: {detail}")
            no_answer(f"retCode {ret_code}: {detail}")

        time_ms = body.get("time")
        if type(time_ms) is not int:
            no_answer("field time is missing or not an integer")
        try:
            server_time = utc_from_ms(time_ms)
        except DomainError:
            no_answer("field time is not a valid millisecond timestamp")
        result = body.get("result")
        if not isinstance(result, dict):
            no_answer("field result is missing or not an object")
        ret_ext_info = body.get("retExtInfo")
        if not isinstance(ret_ext_info, dict):
            no_answer("field retExtInfo is missing or not an object")
        return BybitResponse(result=result, ret_ext_info=ret_ext_info, server_time=server_time)
