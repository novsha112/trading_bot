"""Market data snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from app.domain.errors import DomainValidationError
from app.domain.validation import (
    require_bool,
    require_decimal,
    require_non_negative,
    require_positive,
    require_text,
    require_utc,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Ticker:
    """Latest prices of one instrument.

    ``received_ts`` may be earlier than ``exchange_ts``: clocks of the exchange and
    the host can drift, so no ordering between them is enforced here.
    """

    symbol: str
    last_price: Decimal
    mark_price: Decimal | None
    best_bid: Decimal | None
    best_ask: Decimal | None
    funding_rate: Decimal | None
    """Current funding rate; any sign (negative means shorts pay longs)."""
    next_funding_at: datetime | None
    exchange_ts: datetime
    received_ts: datetime

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        require_positive(self.last_price, "last_price")
        if self.mark_price is not None:
            require_positive(self.mark_price, "mark_price")
        if self.best_bid is not None:
            require_positive(self.best_bid, "best_bid")
        if self.best_ask is not None:
            require_positive(self.best_ask, "best_ask")
        if (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid > self.best_ask
        ):
            raise DomainValidationError(
                f"best_bid ({self.best_bid}) must be <= best_ask ({self.best_ask})"
            )
        if self.funding_rate is not None:
            require_decimal(self.funding_rate, "funding_rate")
        if self.next_funding_at is not None:
            require_utc(self.next_funding_at, "next_funding_at")
        require_utc(self.exchange_ts, "exchange_ts")
        require_utc(self.received_ts, "received_ts")


@dataclass(frozen=True, slots=True, kw_only=True)
class Candle:
    """OHLCV bar.

    ``close_time - open_time`` is not required to equal ``interval``: providers
    mark candle boundaries differently (e.g. close = next open - 1 ms).
    """

    symbol: str
    interval: timedelta
    open_time: datetime
    close_time: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    is_closed: bool
    """False for a bar that is still forming; its values will change."""

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        if not isinstance(self.interval, timedelta):
            raise DomainValidationError(
                f"interval must be a timedelta, got {type(self.interval).__name__}"
            )
        if self.interval <= timedelta(0):
            raise DomainValidationError(f"interval must be > 0, got {self.interval}")
        require_utc(self.open_time, "open_time")
        require_utc(self.close_time, "close_time")
        if self.close_time <= self.open_time:
            raise DomainValidationError(
                f"close_time ({self.close_time.isoformat()}) must be > "
                f"open_time ({self.open_time.isoformat()})"
            )
        require_positive(self.open_price, "open_price")
        require_positive(self.high_price, "high_price")
        require_positive(self.low_price, "low_price")
        require_positive(self.close_price, "close_price")
        # low <= high follows from the four checks below.
        for name, bound in (("open_price", self.open_price), ("close_price", self.close_price)):
            if self.high_price < bound:
                raise DomainValidationError(
                    f"high_price ({self.high_price}) must be >= {name} ({bound})"
                )
            if self.low_price > bound:
                raise DomainValidationError(
                    f"low_price ({self.low_price}) must be <= {name} ({bound})"
                )
        require_non_negative(self.volume, "volume")
        require_bool(self.is_closed, "is_closed")
