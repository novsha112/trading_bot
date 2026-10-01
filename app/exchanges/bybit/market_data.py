"""Bybit V5 public market data over REST (``MarketDataClient``).

Official contract (bybit-exchange/docs, docs/v5):
* ``GET /v5/market/instruments-info?category=linear&symbol=...``
* ``GET /v5/market/tickers?category=linear&symbol=...``
* envelope: ``retCode`` (0 = success), ``retMsg``, ``result``, ``retExtInfo``,
  ``time`` (Bybit server response timestamp, ms; not a market-event timestamp).
* error codes (docs/v5/error): only the codes whose meaning we rely on are listed
  below; any other non-zero ``retCode`` is treated as "no valid answer", never as
  a client mistake.

v1 supports USDT-settled linear perpetuals only (``contractType=LinearPerpetual``,
``quoteCoin=settleCoin=USDT``, ``status=Trading``, not pre-listing); anything else
is rejected. Numbers arrive as JSON strings and are parsed to ``Decimal`` directly;
a JSON number in their place is rejected. No credentials, signing or private
endpoints are involved.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final, NoReturn

import httpx

from app.domain.clock import Clock
from app.domain.errors import DomainError
from app.domain.instrument import InstrumentSpec
from app.domain.market import Ticker
from app.domain.validation import utc_from_ms
from app.exchanges.errors import (
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeResponseError,
)

_CATEGORY: Final = "linear"
_INSTRUMENTS_PATH: Final = "/v5/market/instruments-info"
_TICKERS_PATH: Final = "/v5/market/tickers"

_SUPPORTED_CONTRACT_TYPE: Final = "LinearPerpetual"
_SUPPORTED_SETTLEMENT: Final = "USDT"
_TRADING_STATUS: Final = "Trading"

# Plain decimal strings as Bybit sends them ("0.10", "-0.0075"): no exponent,
# no whitespace, no underscores, no NaN/Infinity.
_DECIMAL_TEXT: Final = re.compile(r"-?[0-9]+(?:\.[0-9]+)?\Z")
_MILLIS_TEXT: Final = re.compile(r"[0-9]+\Z")
_MAX_MESSAGE: Final = 120

# retCode values whose meaning is "the request itself is wrong" (definitive).
REQUEST_ERROR_RET_CODES: Final = frozenset({10001})  # Request parameter error
# retCode values meaning "no normal result right now" (infrastructure / load).
TRANSIENT_RET_CODES: Final = frozenset(
    {
        10000,  # Server Timeout
        10006,  # Too many visits (API rate limit)
        10016,  # Server error
        429,  # High server load
    }
)
# HTTP statuses that describe a malformed or unsupported request (docs/v5/error):
# 400 bad request, 404 path not found. Everything else that is not 2xx (403 edge /
# region / IP rate limit, 408, 429, 5xx, unknown) is "no valid answer".
_REQUEST_ERROR_HTTP_STATUSES: Final = frozenset({400, 404})


def _fail(message: str) -> NoReturn:
    raise ExchangeResponseError(f"Bybit: {message}")


def _require(obj: dict[str, Any], key: str) -> Any:
    if key not in obj:
        _fail(f"missing field {key}")
    return obj[key]


def _object(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require(obj, key)
    if not isinstance(value, dict):
        _fail(f"field {key} is not an object")
    return value


def _text(obj: dict[str, Any], key: str) -> str:
    value = _require(obj, key)
    if not isinstance(value, str) or not value:
        _fail(f"field {key} must be a non-empty string")
    return value


def _decimal(obj: dict[str, Any], key: str) -> Decimal:
    value = _require(obj, key)
    if not isinstance(value, str) or not _DECIMAL_TEXT.match(value):
        _fail(f"field {key} is not a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation:
        _fail(f"field {key} is not a decimal string")
    if not result.is_finite():
        _fail(f"field {key} is not finite")
    return result


def _optional_decimal(obj: dict[str, Any], key: str) -> Decimal | None:
    # The field must be present; an empty string means "no value" (e.g. empty book side).
    if _require(obj, key) == "":
        return None
    return _decimal(obj, key)


def _optional_millis(obj: dict[str, Any], key: str) -> datetime | None:
    # "" and "0" mean "no value"; 0 is never turned into 1970-01-01.
    value = _require(obj, key)
    if value == "":
        return None
    if not isinstance(value, str) or not _MILLIS_TEXT.match(value):
        _fail(f"field {key} is not a millisecond timestamp string")
    millis = int(value)
    if millis == 0:
        return None
    return _utc(millis, key)


def _positive_decimal(obj: dict[str, Any], key: str) -> Decimal:
    # For documented fields that are validated but not projected into a domain model
    # (the domain cannot check them).
    value = _decimal(obj, key)
    if value <= 0:
        _fail(f"field {key} must be > 0")
    return value


def _utc(ms: int, key: str) -> datetime:
    try:
        return utc_from_ms(ms)
    except DomainError:
        _fail(f"field {key} is not a valid millisecond timestamp")


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant {name}")


class BybitMarketDataClient:
    """Public Bybit V5 market data for USDT linear perpetuals.

    The HTTP client is injected and owned by the caller (timeouts, connection
    pooling and closing are configured there); this adapter never closes it.
    ``base_url`` is chosen by the composition layer (see ``endpoints``).
    """

    def __init__(self, *, client: httpx.AsyncClient, base_url: str, clock: Clock) -> None:
        if (
            not isinstance(base_url, str)
            or not base_url.startswith("https://")
            or base_url.endswith("/")
            or len(base_url) <= len("https://")
        ):
            raise ValueError("base_url must be an https:// URL without a trailing slash")
        self._client = client
        self._base_url = base_url
        self._clock = clock

    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        items, _, _ = await self._get(_INSTRUMENTS_PATH, symbol)
        item = self._select(items, symbol)

        contract_type = _text(item, "contractType")
        status = _text(item, "status")
        base_asset = _text(item, "baseCoin")
        quote_asset = _text(item, "quoteCoin")
        settle_asset = _text(item, "settleCoin")
        price_filter = _object(item, "priceFilter")
        lot_filter = _object(item, "lotSizeFilter")
        tick_size = _decimal(price_filter, "tickSize")
        qty_step = _decimal(lot_filter, "qtyStep")
        min_qty = _decimal(lot_filter, "minOrderQty")
        max_qty = _decimal(lot_filter, "maxOrderQty")
        # maxMktOrderQty is the separate limit for MARKET orders. It is part of the
        # documented contract, so it is validated (a broken value means a broken
        # response), but InstrumentSpec has no market-order limit yet: it is not
        # projected until a consumer of market orders needs it.
        _positive_decimal(lot_filter, "maxMktOrderQty")
        min_notional = _decimal(lot_filter, "minNotionalValue")
        # Documented for every linear instrument (false for regular contracts):
        # a missing field is a schema change, not "false".
        pre_listing = _require(item, "isPreListing")
        if not isinstance(pre_listing, bool):
            _fail("field isPreListing is not a boolean")

        unsupported = (
            ("contractType", contract_type != _SUPPORTED_CONTRACT_TYPE, contract_type),
            ("quoteCoin", quote_asset != _SUPPORTED_SETTLEMENT, quote_asset),
            ("settleCoin", settle_asset != _SUPPORTED_SETTLEMENT, settle_asset),
            ("status", status != _TRADING_STATUS, status),
        )
        for field, is_unsupported, value in unsupported:
            if is_unsupported:
                raise ExchangeRejectedError(
                    f"Bybit: unsupported instrument {symbol}: {field}={value}"
                )
        if pre_listing:
            raise ExchangeRejectedError(
                f"Bybit: unsupported instrument {symbol}: pre-listing contract"
            )

        try:
            return InstrumentSpec(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                tick_size=tick_size,
                qty_step=qty_step,
                min_qty=min_qty,
                # Maximum quantity of a limit / post-only order (maxOrderQty).
                max_qty=max_qty,
                min_notional=min_notional,
            )
        except DomainError as exc:
            _fail(f"invalid instrument data for {symbol}: {exc}")

    async def get_ticker(self, symbol: str) -> Ticker:
        items, response_time, received_ts = await self._get(_TICKERS_PATH, symbol)
        item = self._select(items, symbol)

        last_price = _decimal(item, "lastPrice")
        mark_price = _optional_decimal(item, "markPrice")
        best_bid = _optional_decimal(item, "bid1Price")
        best_ask = _optional_decimal(item, "ask1Price")
        funding_rate = _optional_decimal(item, "fundingRate")
        next_funding_at = _optional_millis(item, "nextFundingTime")
        try:
            return Ticker(
                symbol=symbol,
                last_price=last_price,
                mark_price=mark_price,
                best_bid=best_bid,
                best_ask=best_ask,
                funding_rate=funding_rate,
                next_funding_at=next_funding_at,
                # The ticker item has no timestamp of its own. This is the Bybit server
                # response timestamp (envelope ``time``), not a market-event timestamp.
                exchange_ts=response_time,
                received_ts=received_ts,
            )
        except DomainError as exc:
            _fail(f"invalid ticker data for {symbol}: {exc}")

    async def _get(self, path: str, symbol: str) -> tuple[list[dict[str, Any]], datetime, datetime]:
        """GET a public endpoint and validate the common envelope.

        Returns the ``result.list`` items, the envelope ``time`` and the local
        receive time.
        """
        try:
            response = await self._client.get(
                f"{self._base_url}{path}", params={"category": _CATEGORY, "symbol": symbol}
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # No connection was established: nothing reached the exchange.
            raise ExchangeNotSentError(f"Bybit: request not sent ({type(exc).__name__})") from None
        except httpx.RequestError as exc:
            raise ExchangeResponseError(f"Bybit: no response ({type(exc).__name__})") from None
        received_ts = self._clock.now()

        status = response.status_code
        if status in _REQUEST_ERROR_HTTP_STATUSES:
            raise ExchangeRejectedError(f"Bybit: request refused with HTTP {status}")
        if not 200 <= status < 300:
            _fail(f"no valid answer, HTTP {status}")

        try:
            body = json.loads(
                response.content, parse_float=Decimal, parse_constant=_reject_constant
            )
        except ValueError:
            _fail("response is not valid JSON")
        if not isinstance(body, dict):
            _fail("response is not a JSON object")

        ret_code = _require(body, "retCode")
        if type(ret_code) is not int:
            _fail("field retCode is not an integer")
        if ret_code != 0:
            ret_msg = body.get("retMsg")
            detail = ret_msg[:_MAX_MESSAGE].replace("\n", " ") if isinstance(ret_msg, str) else ""
            if ret_code in REQUEST_ERROR_RET_CODES:
                raise ExchangeRejectedError(f"Bybit: retCode {ret_code}: {detail}")
            # Known transient codes and unknown codes alike: do not claim the caller
            # made a wrong request when we do not know that.
            kind = "transient" if ret_code in TRANSIENT_RET_CODES else "unrecognized"
            _fail(f"{kind} retCode {ret_code}: {detail}")

        time_ms = _require(body, "time")
        if type(time_ms) is not int:
            _fail("field time is not an integer")
        response_time = _utc(time_ms, "time")

        result = _object(body, "result")
        if _require(result, "category") != _CATEGORY:
            _fail("result category is not linear")
        items = _require(result, "list")
        if not isinstance(items, list):
            _fail("field list is not an array")
        if not all(isinstance(item, dict) for item in items):
            _fail("a list item is not an object")
        return items, response_time, received_ts

    @staticmethod
    def _select(items: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
        """The single item for exactly ``symbol`` (no case normalization)."""
        if not items:
            raise ExchangeRejectedError(f"Bybit: symbol {symbol} not found in category linear")
        matches = [item for item in items if item.get("symbol") == symbol]
        if len(matches) > 1:
            _fail(f"response contains more than one item for {symbol}")
        if not matches:
            _fail(f"response does not contain the requested symbol {symbol}")
        return matches[0]
