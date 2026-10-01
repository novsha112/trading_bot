"""Pure mapping OrderRequest -> Bybit V5 ``POST /v5/order/create`` body.

Official contract (bybit-exchange/docs, branch master, docs/v5/order/create-order.mdx
and docs/v5/enum.mdx). No transport, no httpx, no network: the mapping is a pure
function of the request.
"""

from __future__ import annotations

import ast
import json
import tracemalloc
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderType, Side, TimeInForce
from app.exchanges.bybit import order_mapping
from app.exchanges.bybit.order_mapping import map_order_request
from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
    ExchangeResponseError,
)
from app.exchanges.models import OrderRequest

D = Decimal

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
MARKET: dict[str, Any] = {
    **LIMIT,
    "order_type": OrderType.MARKET,
    "price": None,
    "time_in_force": TimeInForce.IOC,
}


def request(base: dict[str, Any] = LIMIT, **overrides: Any) -> OrderRequest:
    return OrderRequest(**{**base, **overrides})


def tampered(order: OrderRequest, **fields: Any) -> OrderRequest:
    """Bypass OrderRequest validation (frozen dataclass) to prove the mapper
    re-checks what it sends instead of trusting construction-time validation."""
    for name, value in fields.items():
        object.__setattr__(order, name, value)
    return order


# --- complete payloads --------------------------------------------------------


def test_limit_payload_exact() -> None:
    body = map_order_request(request())

    assert body == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": "0.001",
        "price": "65000.5",
        "timeInForce": "GTC",
        "reduceOnly": False,
        "orderLinkId": "grid1-buy-0001",
    }
    # Deterministic key order: the transport signs the body in this order.
    assert list(body) == [
        "category",
        "symbol",
        "side",
        "orderType",
        "qty",
        "price",
        "timeInForce",
        "reduceOnly",
        "orderLinkId",
    ]


def test_market_payload_exact_has_no_price() -> None:
    body = map_order_request(request(MARKET, side=Side.SELL, qty=D("2")))

    assert body == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Sell",
        "orderType": "Market",
        "qty": "2",
        "timeInForce": "IOC",
        "reduceOnly": False,
        "orderLinkId": "grid1-buy-0001",
    }
    assert "price" not in body


def test_only_documented_scope_fields_are_sent() -> None:
    # One-way linear USDT perpetual: no positionIdx (required only in hedge mode),
    # no TP/SL, trigger, SMP, MMP, slippage, orderFilter, broker or RPI fields.
    allowed = {
        "category",
        "symbol",
        "side",
        "orderType",
        "qty",
        "price",
        "timeInForce",
        "reduceOnly",
        "orderLinkId",
    }
    for order in (request(), request(MARKET), request(reduce_only=True)):
        assert set(map_order_request(order)) <= allowed


def test_payload_is_json_native_and_serializable() -> None:
    body = map_order_request(request(reduce_only=True))

    for value in body.values():
        assert type(value) in (str, bool)
    encoded = json.dumps(body, separators=(",", ":"), allow_nan=False)
    assert '"reduceOnly":true' in encoded
    assert '"qty":"0.001"' in encoded
    assert '"price":"65000.5"' in encoded


# --- enum mapping -------------------------------------------------------------


@pytest.mark.parametrize(("side", "expected"), [(Side.BUY, "Buy"), (Side.SELL, "Sell")])
def test_side(side: Side, expected: str) -> None:
    assert map_order_request(request(side=side))["side"] == expected


@pytest.mark.parametrize(
    ("tif", "expected"),
    [
        (TimeInForce.GTC, "GTC"),
        (TimeInForce.IOC, "IOC"),
        (TimeInForce.FOK, "FOK"),
        (TimeInForce.POST_ONLY, "PostOnly"),
    ],
)
def test_limit_time_in_force(tif: TimeInForce, expected: str) -> None:
    body = map_order_request(request(time_in_force=tif))

    assert body["orderType"] == "Limit"
    assert body["timeInForce"] == expected


def test_limit_gtc_is_sent_explicitly() -> None:
    # Bybit defaults to GTC when omitted; the payload never relies on defaults.
    assert map_order_request(request(time_in_force=TimeInForce.GTC))["timeInForce"] == "GTC"


@pytest.mark.parametrize("tif", [TimeInForce.GTC, TimeInForce.FOK])
def test_market_with_non_ioc_time_in_force_fails_closed(tif: TimeInForce) -> None:
    # Docs: "Market order will always use IOC". GTC / FOK would be silently replaced
    # by the exchange, so the requested semantics are not expressible.
    with pytest.raises(ExchangeRequestValidationError, match="timeInForce"):
        map_order_request(request(MARKET, time_in_force=tif))


def test_market_post_only_fails_closed_even_if_construction_was_bypassed() -> None:
    order = tampered(request(MARKET), time_in_force=TimeInForce.POST_ONLY)

    with pytest.raises(ExchangeRequestValidationError):
        map_order_request(order)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("side", "buy"),
        ("side", "Buy"),
        ("side", None),
        ("order_type", "limit"),
        ("order_type", "Limit"),
        ("time_in_force", "gtc"),
        ("time_in_force", "PostOnly"),
        ("time_in_force", "RPI"),
    ],
)
def test_raw_strings_instead_of_enums_fail_closed(field: str, value: object) -> None:
    order = tampered(request(), **{field: value})

    with pytest.raises(ExchangeRequestValidationError):
        map_order_request(order)


# --- price / qty --------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (D("1"), "1"),
        (D("0.001"), "0.001"),
        (D("65000.5"), "65000.5"),
        (D("1.500"), "1.5"),
        (D("100.000"), "100"),
        (D("100"), "100"),
        (D("1E+8"), "100000000"),
        (D("1.0E+2"), "100"),
        (D("1E-8"), "0.00000001"),
        (D("12345E-9"), "0.000012345"),
        (D("0.10"), "0.1"),
        (D("1E+1"), "10"),
        (D("123456789012345678901234567890.123456789"), "123456789012345678901234567890.123456789"),
    ],
)
def test_decimal_canonical_plain_notation(value: Decimal, expected: str) -> None:
    limit = map_order_request(request(price=value, qty=value))
    market = map_order_request(request(MARKET, qty=value))

    assert limit["price"] == expected
    assert limit["qty"] == expected
    assert market["qty"] == expected


def test_equal_decimals_map_to_identical_text() -> None:
    a = map_order_request(request(price=D("65000.50"), qty=D("1E-3")))
    b = map_order_request(request(price=D("65000.5"), qty=D("0.0010")))

    assert a == b


def test_decimal_formatting_ignores_the_ambient_context() -> None:
    # More significant digits than a tiny context precision: never rounded.
    value = D("1.23456789")
    with localcontext() as context:
        context.prec = 3
        body = map_order_request(request(price=value, qty=value))

    assert body["price"] == "1.23456789"
    assert body["qty"] == "1.23456789"


@pytest.mark.parametrize("field", ["price", "qty"])
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
        1,
        "1",
        True,
        None,
    ],
)
def test_invalid_numbers_fail_closed(field: str, value: object) -> None:
    order = tampered(request(), **{field: value})

    with pytest.raises(ExchangeRequestValidationError, match=field):
        map_order_request(order)


@pytest.mark.parametrize("field", ["price", "qty"])
def test_decimal_subclass_rejected(field: str) -> None:
    class Weird(Decimal):
        def __str__(self) -> str:
            return "1e3"

    order = tampered(request(), **{field: Weird("1")})

    with pytest.raises(ExchangeRequestValidationError, match=field):
        map_order_request(order)


@pytest.mark.parametrize("field", ["price", "qty"])
@pytest.mark.parametrize("value", [D("1E+999999999"), D("1E-999999999"), D("123456789E+999999990")])
def test_pathological_exponent_fails_before_materializing_text(field: str, value: Decimal) -> None:
    order = request(LIMIT, **{field: value})

    tracemalloc.start()
    try:
        with pytest.raises(ExchangeRequestValidationError) as excinfo:
            map_order_request(order)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # A plain-notation string would need ~1 GB; the check runs on as_tuple() only.
    assert peak < 1_000_000
    assert field in str(excinfo.value)
    assert "decimal representation exceeds local safety limit" in str(excinfo.value)


def test_market_price_present_fails_closed() -> None:
    order = tampered(request(MARKET), price=D("65000"))

    with pytest.raises(ExchangeRequestValidationError, match="price"):
        map_order_request(order)


def test_limit_price_missing_fails_closed() -> None:
    order = tampered(request(), price=None)

    with pytest.raises(ExchangeRequestValidationError, match="price"):
        map_order_request(order)


def test_no_rounding_to_any_tick_or_step() -> None:
    # Alignment to tickSize / qtyStep is a pre-trade check, not the mapper's job:
    # the value is sent exactly as requested.
    body = map_order_request(request(price=D("65000.123456789"), qty=D("0.0012345")))

    assert body["price"] == "65000.123456789"
    assert body["qty"] == "0.0012345"


# --- reduceOnly ---------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_reduce_only_is_json_bool(value: bool) -> None:
    sent = map_order_request(request(reduce_only=value))["reduceOnly"]

    assert sent is value


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_reduce_only_non_bool_fails_closed(value: object) -> None:
    order = tampered(request(), reduce_only=value)

    with pytest.raises(ExchangeRequestValidationError, match="reduceOnly"):
        map_order_request(order)


# --- orderLinkId --------------------------------------------------------------


@pytest.mark.parametrize(
    "link_id",
    [
        "a",
        "Z",
        "0",
        "-",
        "_",
        "x" * 36,
        "grid1-buy-0001",
        "ABC_def-123",
        "0123456789abcdefABCDEF-_0123456789ab",
    ],
)
def test_order_link_id_accepted_unchanged(link_id: str) -> None:
    assert map_order_request(request(client_order_id=link_id))["orderLinkId"] == link_id


@pytest.mark.parametrize(
    "link_id",
    [
        "x" * 37,
        "x" * 100,
        "grid 1",
        "grid.1",
        "grid:1",
        "grid/1",
        "grid+1",
        "grid#1",
        "ordér",
        "\uff11\uff12\uff13",  # full-width digits
        "abc\n",
        "abc\x00",
    ],
)
def test_order_link_id_outside_documented_charset_fails_closed(link_id: str) -> None:
    order = tampered(request(), client_order_id=link_id)

    with pytest.raises(ExchangeRequestValidationError, match="orderLinkId"):
        map_order_request(order)


@pytest.mark.parametrize("link_id", ["", " abc", "abc ", None, 123])
def test_order_link_id_invalid_after_bypass_fails_closed(link_id: object) -> None:
    order = tampered(request(), client_order_id=link_id)

    with pytest.raises(ExchangeRequestValidationError, match="orderLinkId"):
        map_order_request(order)


def test_order_link_id_never_truncated_or_normalized() -> None:
    link_id = "Grid_A-" + "9" * 29  # exactly 36

    assert map_order_request(request(client_order_id=link_id))["orderLinkId"] == link_id


# --- symbol / category --------------------------------------------------------


@pytest.mark.parametrize("symbol", ["BTCUSDT", "1000PEPEUSDT", "ETHUSDT"])
def test_symbol_sent_exactly(symbol: str) -> None:
    body = map_order_request(request(symbol=symbol))

    assert body["symbol"] == symbol
    assert body["category"] == "linear"


@pytest.mark.parametrize("symbol", ["btcusdt", "BtcUSDT", "BTCUSDt", "ethusdt"])
def test_lowercase_symbol_fails_closed_without_normalizing(symbol: str) -> None:
    # Docs: "Symbol name, like BTCUSDT, uppercase only". The mapper never
    # upper-cases: a wrong symbol is a caller bug, not something to repair silently.
    order = request(symbol=symbol)

    with pytest.raises(ExchangeRequestValidationError, match="symbol"):
        map_order_request(order)


@pytest.mark.parametrize("symbol", [" BTCUSDT", "BTCUSDT ", "BTCUSDT\n", "\tBTCUSDT"])
def test_symbol_with_surrounding_whitespace_fails_closed(symbol: str) -> None:
    order = tampered(request(), symbol=symbol)

    with pytest.raises(ExchangeRequestValidationError, match="symbol"):
        map_order_request(order)


@pytest.mark.parametrize("symbol", ["BTC-USDT", "BTC/USDT", "BTC_USDT", "BTC-26DEC25", "123"])
def test_uppercase_symbol_with_punctuation_is_not_rejected_by_the_mapper(symbol: str) -> None:
    # The docs define no character set for symbol, only "uppercase only". Whether
    # the symbol exists and is a USDT linear perpetual is proven by InstrumentSpec /
    # preflight, not guessed here.
    assert map_order_request(request(symbol=symbol))["symbol"] == symbol


@pytest.mark.parametrize("symbol", ["", None, 1])
def test_symbol_invalid_after_bypass_fails_closed(symbol: object) -> None:
    order = tampered(request(), symbol=symbol)

    with pytest.raises(ExchangeRequestValidationError, match="symbol"):
        map_order_request(order)


# --- error category, purity, determinism --------------------------------------


def test_validation_error_is_a_pre_send_not_sent_error() -> None:
    assert issubclass(ExchangeRequestValidationError, ExchangeNotSentError)
    for other in (ExchangeAmbiguousResultError, ExchangeRejectedError, ExchangeResponseError):
        assert not issubclass(ExchangeRequestValidationError, other)
    with pytest.raises(ExchangeNotSentError):
        map_order_request(request(MARKET, time_in_force=TimeInForce.FOK))


def test_non_order_request_fails_closed() -> None:
    with pytest.raises(ExchangeRequestValidationError):
        map_order_request(LIMIT)  # type: ignore[arg-type]


def test_input_is_not_mutated() -> None:
    order = request()
    before = (
        order.client_order_id,
        order.symbol,
        order.side,
        order.order_type,
        order.price,
        order.qty,
        order.time_in_force,
        order.reduce_only,
    )

    map_order_request(order)

    after = (
        order.client_order_id,
        order.symbol,
        order.side,
        order.order_type,
        order.price,
        order.qty,
        order.time_in_force,
        order.reduce_only,
    )
    assert before == after


def test_each_call_returns_a_fresh_independent_dict() -> None:
    order = request()
    first = map_order_request(order)
    first["qty"] = "999"
    first["extra"] = "x"

    second = map_order_request(order)

    assert second["qty"] == "0.001"
    assert "extra" not in second
    assert first is not second


def test_deterministic_across_calls() -> None:
    order = request(time_in_force=TimeInForce.POST_ONLY, reduce_only=True)
    bodies = [map_order_request(order) for _ in range(5)]
    encoded = {json.dumps(body, separators=(",", ":")) for body in bodies}

    assert len(encoded) == 1


def test_error_messages_do_not_echo_long_input() -> None:
    order = request(client_order_id="x" * 500)

    with pytest.raises(ExchangeRequestValidationError) as excinfo:
        map_order_request(order)

    assert "x" * 37 not in str(excinfo.value)


def test_mapping_module_has_no_transport_or_logging_dependencies() -> None:
    source = Path(order_mapping.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("httpx", "logging", "structlog", "pydantic", "app.config", "private_rest")
    for name in imported:
        assert not any(part in name for part in forbidden), name


def test_safety_limit_is_a_documented_internal_bound() -> None:
    assert order_mapping.MAX_DECIMAL_TEXT_LENGTH == 1024


@pytest.mark.parametrize(
    "value",
    [D("1E+70"), D("1E-70"), D("1" * 64 + ".5"), D("9" * 200), D("0." + "0" * 300 + "1")],
)
def test_values_longer_than_64_characters_are_serialized(value: Decimal) -> None:
    # 64 was never a Bybit rule; such values are only bounded by the local limit.
    text = map_order_request(request(LIMIT, qty=value, price=value))["qty"]

    assert isinstance(text, str)
    assert len(text) > 64
    assert "E" not in text
    assert "e" not in text
    assert D(text) == value


LIMIT_LENGTH = 1024


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (D(f"1E+{LIMIT_LENGTH - 1}"), True),  # "1" + 1023 zeros = 1024 characters
        (D(f"1E+{LIMIT_LENGTH}"), False),  # 1025 characters
        (D(f"1E-{LIMIT_LENGTH - 2}"), True),  # "0." + 1021 zeros + "1" = 1024
        (D(f"1E-{LIMIT_LENGTH - 1}"), False),  # 1025 characters
        (D("1" * (LIMIT_LENGTH - 2) + ".1"), True),  # 1024 characters
        (D("1" * (LIMIT_LENGTH - 1) + ".1"), False),  # 1025 characters
        (D("7" * LIMIT_LENGTH), True),  # 1024 characters, integer
        (D("7" * (LIMIT_LENGTH + 1)), False),  # 1025 characters, integer
    ],
)
@pytest.mark.parametrize("field", ["price", "qty"])
def test_local_safety_limit_boundary_is_exact(field: str, value: Decimal, accepted: bool) -> None:
    order = request(LIMIT, **{field: value})

    if accepted:
        text = map_order_request(order)[field]
        assert isinstance(text, str)
        assert len(text) == LIMIT_LENGTH
        assert D(text) == value
    else:
        with pytest.raises(ExchangeRequestValidationError) as excinfo:
            map_order_request(order)
        message = str(excinfo.value)
        assert field in message
        assert "decimal representation exceeds local safety limit" in message
        assert "Bybit limit" not in message
