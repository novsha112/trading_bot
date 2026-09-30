"""Ticker and Candle market snapshots."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from app.domain.errors import DomainValidationError
from app.domain.market import Candle, Ticker

D = Decimal
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NAIVE = datetime(2026, 1, 15, 12, 0)  # noqa: DTZ001 - deliberately naive
PLUS_TWO = datetime(2026, 1, 15, 12, 0, tzinfo=timezone(timedelta(hours=2)))
BAD_DECIMALS: list[Any] = [1.5, 1, True, "1", D("NaN"), D("sNaN"), D("Infinity")]

TICKER: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "last_price": D("65000.5"),
    "mark_price": D("65001"),
    "best_bid": D("65000.4"),
    "best_ask": D("65000.6"),
    "funding_rate": D("0.0001"),
    "next_funding_at": TS + timedelta(hours=8),
    "exchange_ts": TS,
    "received_ts": TS + timedelta(milliseconds=15),
}

CANDLE: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "interval": timedelta(minutes=1),
    "open_time": TS,
    "close_time": TS + timedelta(minutes=1),
    "open_price": D("100"),
    "high_price": D("110"),
    "low_price": D("95"),
    "close_price": D("105"),
    "volume": D("12.5"),
    "is_closed": True,
}


def ticker(**overrides: Any) -> Ticker:
    return Ticker(**{**TICKER, **overrides})


def candle(**overrides: Any) -> Candle:
    return Candle(**{**CANDLE, **overrides})


@pytest.mark.parametrize(("model", "values"), [(Ticker, TICKER), (Candle, CANDLE)])
def test_structure_frozen_slotted_keyword_only(model: Any, values: dict[str, Any]) -> None:
    instance = model(**values)
    for name, value in values.items():
        assert getattr(instance, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.symbol = "ETHUSDT"
    assert not hasattr(instance, "__dict__")
    with pytest.raises(TypeError):
        model(*values.values())


# --- Ticker ------------------------------------------------------------------------------


def test_ticker_optional_fields_may_be_none() -> None:
    t = ticker(
        mark_price=None, best_bid=None, best_ask=None, funding_rate=None, next_funding_at=None
    )
    assert t.mark_price is None
    assert t.best_bid is None
    assert t.best_ask is None
    assert t.funding_rate is None
    assert t.next_funding_at is None


@pytest.mark.parametrize("field", ["last_price", "mark_price", "best_bid", "best_ask"])
@pytest.mark.parametrize("value", [D("0"), D("-1")])
def test_ticker_prices_must_be_positive(field: str, value: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be > 0"):
        ticker(**{field: value})


@pytest.mark.parametrize("field", ["last_price", "mark_price", "funding_rate"])
@pytest.mark.parametrize("value", BAD_DECIMALS)
def test_ticker_rejects_non_decimal(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be"):
        ticker(**{field: value})


def test_ticker_last_price_is_required() -> None:
    with pytest.raises(DomainValidationError, match=r"^last_price must be a Decimal"):
        ticker(last_price=None)


def test_ticker_crossed_book_rejected() -> None:
    with pytest.raises(DomainValidationError, match=r"^best_bid .* must be <= best_ask"):
        ticker(best_bid=D("65000.7"), best_ask=D("65000.6"))


def test_ticker_locked_book_and_single_side_accepted() -> None:
    assert ticker(best_bid=D("65000.5"), best_ask=D("65000.5")).best_bid == D("65000.5")
    assert ticker(best_bid=D("70000"), best_ask=None).best_ask is None
    assert ticker(best_bid=None, best_ask=D("1")).best_bid is None


@pytest.mark.parametrize("rate", [D("-0.0075"), D("0"), D("0.0075")])
def test_ticker_funding_rate_any_sign(rate: Decimal) -> None:
    assert ticker(funding_rate=rate).funding_rate == rate


@pytest.mark.parametrize("field", ["exchange_ts", "received_ts", "next_funding_at"])
@pytest.mark.parametrize("value", [NAIVE, PLUS_TWO, 1768478400000])
def test_ticker_timestamps_must_be_utc(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        ticker(**{field: value})


def test_ticker_received_before_exchange_ts_is_allowed() -> None:
    # Clock skew between the exchange and the host is possible.
    t = ticker(received_ts=TS - timedelta(milliseconds=50))
    assert t.received_ts < t.exchange_ts


@pytest.mark.parametrize("value", ["", " BTCUSDT", None])
def test_ticker_symbol_validated(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^symbol "):
        ticker(symbol=value)


# --- Candle ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prices",
    [
        {"open_price": D("5"), "high_price": D("5"), "low_price": D("5"), "close_price": D("5")},
        {
            "open_price": D("110"),
            "high_price": D("110"),
            "low_price": D("95"),
            "close_price": D("95"),
        },
        {
            "open_price": D("95"),
            "high_price": D("110"),
            "low_price": D("95"),
            "close_price": D("110"),
        },
    ],
)
def test_candle_boundary_ohlc_accepted(prices: dict[str, Decimal]) -> None:
    assert candle(**prices).high_price == prices["high_price"]


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ({"high_price": D("99")}, "high_price .* must be >= open_price"),
        ({"high_price": D("104"), "open_price": D("100")}, "high_price .* must be >= close_price"),
        ({"low_price": D("101")}, "low_price .* must be <= open_price"),
        (
            {"open_price": D("100"), "close_price": D("98"), "low_price": D("99")},
            "low_price .* must be <= close_price",
        ),
    ],
)
def test_candle_ohlc_relations(prices: dict[str, Decimal], message: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{message}"):
        candle(**prices)


@pytest.mark.parametrize("field", ["open_price", "high_price", "low_price", "close_price"])
@pytest.mark.parametrize("value", [D("0"), D("-1"), *BAD_DECIMALS])
def test_candle_prices_must_be_positive_decimal(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be"):
        candle(**{field: value})


def test_candle_zero_volume_accepted_negative_rejected() -> None:
    assert candle(volume=D("0")).volume == 0
    with pytest.raises(DomainValidationError, match=r"^volume must be >= 0"):
        candle(volume=D("-0.1"))
    with pytest.raises(DomainValidationError, match=r"^volume must be"):
        candle(volume=1.5)


@pytest.mark.parametrize("value", [timedelta(0), timedelta(seconds=-60), 60, None])
def test_candle_interval_must_be_positive_timedelta(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^interval must be"):
        candle(interval=value)


def test_candle_close_time_must_be_after_open_time() -> None:
    with pytest.raises(DomainValidationError, match=r"^close_time .* must be > open_time"):
        candle(close_time=TS)


def test_candle_interval_not_tied_to_time_bounds() -> None:
    # Providers mark candle boundaries differently (e.g. close = open + interval - 1ms).
    c = candle(close_time=TS + timedelta(minutes=1) - timedelta(milliseconds=1))
    assert c.close_time - c.open_time != c.interval


@pytest.mark.parametrize("field", ["open_time", "close_time"])
@pytest.mark.parametrize("value", [NAIVE, PLUS_TWO])
def test_candle_timestamps_must_be_utc(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} "):
        candle(**{field: value})


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_candle_is_closed_must_be_bool(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^is_closed must be a bool"):
        candle(is_closed=value)
