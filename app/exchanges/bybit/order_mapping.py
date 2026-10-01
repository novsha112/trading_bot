"""Pure mapping of an ``OrderRequest`` to a Bybit V5 ``POST /v5/order/create`` body.

Official contract (bybit-exchange/docs, branch master: docs/v5/order/create-order.mdx,
docs/v5/enum.mdx):
* ``category``: ``linear`` (scope: USDT linear perpetuals, one-way mode);
* ``symbol``: "uppercase only"; sent exactly as given, never normalized;
* ``side``: ``Buy`` / ``Sell``; ``orderType``: ``Limit`` / ``Market``;
* ``qty`` / ``price``: strings; ``price`` only for limit orders ("Market order
  will ignore this field");
* ``timeInForce``: ``GTC`` / ``IOC`` / ``FOK`` / ``PostOnly``; "Market order will
  always use IOC", so a market order with any other value is not expressible and
  is refused instead of being silently changed by the exchange;
* ``reduceOnly``: JSON boolean;
* ``orderLinkId``: "A max of 36 characters. Combinations of numbers, letters
  (upper and lower cases), dashes, and underscores are supported."

Not sent: ``positionIdx`` (required only in hedge mode; one-way is the default),
TP/SL, trigger, SMP, MMP, slippage, ``orderFilter``, broker and RPI fields.

Decimals are written in canonical plain notation: no exponent, no float, no
trailing fractional zeros, no rounding (``1.500`` -> ``"1.5"``, ``1E+8`` ->
``"100000000"``, ``1E-8`` -> ``"0.00000001"``). Equal values always give equal
text. Alignment to tick size / qty step is a pre-trade check, not done here.

Every call returns a new dict owned by the caller (the transport serializes and
signs it immediately). Failures raise ``ExchangeRequestValidationError``: nothing
was sent. No I/O, no logging.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final, TypeVar

from app.domain.enums import OrderType, Side, TimeInForce
from app.exchanges.bybit.types import JsonValue
from app.exchanges.errors import ExchangeRequestValidationError
from app.exchanges.models import OrderRequest

CATEGORY_LINEAR: Final = "linear"

_SIDES: Final[dict[Side, str]] = {Side.BUY: "Buy", Side.SELL: "Sell"}
_ORDER_TYPES: Final[dict[OrderType, str]] = {OrderType.LIMIT: "Limit", OrderType.MARKET: "Market"}
_LIMIT_TIME_IN_FORCE: Final[dict[TimeInForce, str]] = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.POST_ONLY: "PostOnly",
}
_MARKET_TIME_IN_FORCE: Final[dict[TimeInForce, str]] = {TimeInForce.IOC: "IOC"}

_ORDER_LINK_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,36}")
_SYMBOL: Final = re.compile(r"[A-Z0-9]+")
# Local safety bound, not a Bybit limit: real prices / quantities are far shorter.
# Refuses absurd magnitudes (e.g. 1E+999999999) before expanding them to text.
MAX_DECIMAL_TEXT_LENGTH: Final = 64

_E = TypeVar("_E", Side, OrderType, TimeInForce)


def _fail(message: str) -> ExchangeRequestValidationError:
    return ExchangeRequestValidationError(f"Bybit order request: {message}")


def _enum_value(value: object, enum_type: type[_E], table: dict[_E, str], field: str) -> str:
    # Exact enum members only: a StrEnum compares equal to its raw string, which
    # must not slip through a dict lookup.
    if not isinstance(value, enum_type) or value not in table:
        raise _fail(f"unsupported {field}")
    return table[value]


def _plain_decimal(value: object, field: str) -> str:
    """Canonical plain-notation text of a finite positive Decimal, without any
    arithmetic context (no rounding, no exponent, no trailing fractional zeros)."""
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise _fail(f"{field} must be a finite Decimal > 0")
    _, digit_tuple, exponent = value.as_tuple()
    if not isinstance(exponent, int):  # unreachable for finite values
        raise _fail(f"{field} must be a finite Decimal > 0")
    digits = "".join(map(str, digit_tuple)).lstrip("0")
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped
    if exponent >= 0:
        length = len(digits) + exponent
    elif len(digits) > -exponent:
        length = len(digits) + 1
    else:
        length = 2 + -exponent
    if length > MAX_DECIMAL_TEXT_LENGTH:
        raise _fail(f"{field} exceeds {MAX_DECIMAL_TEXT_LENGTH} characters in plain notation")
    if exponent >= 0:
        return digits + "0" * exponent
    point = len(digits) + exponent
    if point > 0:
        return f"{digits[:point]}.{digits[point:]}"
    return "0." + "0" * -point + digits


def map_order_request(order: OrderRequest) -> dict[str, JsonValue]:
    """Build the documented create-order body for one linear USDT perpetual order."""
    if not isinstance(order, OrderRequest):
        raise _fail("expected an OrderRequest")

    symbol = order.symbol
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
        raise _fail("symbol must be uppercase letters and digits only")
    link_id = order.client_order_id
    if not isinstance(link_id, str) or not _ORDER_LINK_ID.fullmatch(link_id):
        raise _fail("orderLinkId must be 1-36 characters of [A-Za-z0-9_-]")
    if type(order.reduce_only) is not bool:
        raise _fail("reduceOnly must be a bool")

    side = _enum_value(order.side, Side, _SIDES, "side")
    order_type = _enum_value(order.order_type, OrderType, _ORDER_TYPES, "orderType")
    is_limit = order.order_type is OrderType.LIMIT
    time_in_force = _enum_value(
        order.time_in_force,
        TimeInForce,
        _LIMIT_TIME_IN_FORCE if is_limit else _MARKET_TIME_IN_FORCE,
        "timeInForce" if is_limit else "timeInForce for a market order (always IOC)",
    )
    qty = _plain_decimal(order.qty, "qty")

    body: dict[str, JsonValue] = {
        "category": CATEGORY_LINEAR,
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "qty": qty,
    }
    if is_limit:
        body["price"] = _plain_decimal(order.price, "price")
    elif order.price is not None:
        raise _fail("price must be absent for a market order")
    body["timeInForce"] = time_in_force
    body["reduceOnly"] = order.reduce_only
    body["orderLinkId"] = link_id
    return body
