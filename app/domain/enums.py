"""Domain enumerations.

Values are our own, lowercase and exchange-neutral ("buy", not Bybit's "Buy").
They are stored in the database and written to logs, so changing a value is a
data migration. Exchange adapters map exchange values to these explicitly.
"""

from __future__ import annotations

from enum import StrEnum


class TradingMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


class Side(StrEnum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(StrEnum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    # Maker-only: the exchange cancels the order instead of letting it take liquidity.
    # Modeled as a time-in-force (not a separate flag) so that invalid combinations
    # such as "post-only + IOC" cannot be expressed.
    POST_ONLY = "post_only"


class OrderStatus(StrEnum):
    """Order lifecycle states (docs/ARCHITECTURE.md, section 8)."""

    NEW = "new"
    SUBMITTING = "submitting"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    CANCELING = "canceling"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"
    # Submission outcome is unknown (e.g. timeout after sending): must be resolved
    # through the exchange, never by blindly re-sending the order.
    UNKNOWN = "unknown"


class RoundingDirection(StrEnum):
    """Explicit direction for rounding to a tick size / quantity step."""

    DOWN = "down"
    UP = "up"


class PositionSide(StrEnum):
    """Direction of a position, derived from its signed quantity."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class GridMode(StrEnum):
    """Direction of a grid: which side builds the position."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class GridSpacing(StrEnum):
    """How grid levels are distributed between the lower and upper price."""

    ARITHMETIC = "arithmetic"  # equal price difference between levels
    GEOMETRIC = "geometric"  # equal price ratio between levels
