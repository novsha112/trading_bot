"""Guards for values that enter domain models.

Money, prices, quantities, rates and leverage are ``Decimal`` only. ``float``
is rejected because it cannot represent most decimal prices exactly; ``bool``
and ``int`` are rejected so that a wrong type never slips through silently.
Non-finite decimals (NaN, sNaN, +/-Infinity) are rejected because comparisons
with NaN are always False and would quietly disable limit checks.

Timestamps are timezone-aware ``datetime`` objects with a UTC offset of zero.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final

from app.domain.errors import DomainValidationError

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_ZERO_OFFSET: Final = timedelta(0)


def require_decimal(value: object, field: str) -> Decimal:
    """Return ``value`` if it is a finite ``Decimal``."""
    if not isinstance(value, Decimal):
        raise DomainValidationError(f"{field} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise DomainValidationError(f"{field} must be finite, got {value}")
    return value


def require_positive(value: object, field: str) -> Decimal:
    """Return ``value`` if it is a finite ``Decimal`` greater than zero."""
    decimal = require_decimal(value, field)
    if decimal <= 0:
        raise DomainValidationError(f"{field} must be > 0, got {decimal}")
    return decimal


def require_non_negative(value: object, field: str) -> Decimal:
    """Return ``value`` if it is a finite ``Decimal`` greater than or equal to zero."""
    decimal = require_decimal(value, field)
    if decimal < 0:
        raise DomainValidationError(f"{field} must be >= 0, got {decimal}")
    return decimal


def require_text(value: object, field: str) -> str:
    """Return ``value`` if it is a non-empty ``str`` without surrounding whitespace.

    Identifiers are not normalized: " BTCUSDT" is a producer bug, not a symbol.
    """
    if not isinstance(value, str):
        raise DomainValidationError(f"{field} must be a str, got {type(value).__name__}")
    if not value or value != value.strip():
        raise DomainValidationError(
            f"{field} must be non-empty and without surrounding whitespace, got {value!r}"
        )
    return value


def require_utc(value: object, field: str) -> datetime:
    """Return ``value`` if it is a timezone-aware ``datetime`` with UTC offset 0.

    Values are not converted: a non-UTC timestamp means the producer (usually an
    exchange adapter) skipped normalization, which must be fixed at the source.
    """
    if not isinstance(value, datetime):
        raise DomainValidationError(f"{field} must be a datetime, got {type(value).__name__}")
    offset = value.utcoffset()
    if offset is None:
        raise DomainValidationError(f"{field} must be timezone-aware")
    if offset != _ZERO_OFFSET:
        raise DomainValidationError(f"{field} must be in UTC (offset 0), got offset {offset}")
    return value


def utc_from_ms(ms: int) -> datetime:
    """Convert Unix epoch milliseconds to a UTC ``datetime`` exactly.

    Integer ``timedelta`` arithmetic is used instead of ``fromtimestamp(ms / 1000)``,
    which goes through ``float`` and can shift the millisecond part.
    """
    if not isinstance(ms, int) or isinstance(ms, bool):
        raise DomainValidationError(f"ms must be an int, got {type(ms).__name__}")
    if ms < 0:
        raise DomainValidationError(f"ms must be >= 0, got {ms}")
    try:
        return _EPOCH + timedelta(milliseconds=ms)
    except OverflowError:
        raise DomainValidationError(
            f"ms is out of the supported datetime range, got {ms}"
        ) from None
