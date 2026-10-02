"""Lossless storage codecs for exact Decimal and UTC datetime values (stdlib only).

Decimal (docs/ARCHITECTURE.md 11.0): "canonical" means a deterministic, lossless
storage text. ``encode_decimal`` accepts only an exact, finite ``Decimal`` and
writes its scientific string (upper-case exponent), so the coefficient digits,
the exponent (scale, trailing zeros) and the sign of zero all survive:
``decode_decimal(encode_decimal(x)).as_tuple() == x.as_tuple()``. Nothing is
normalized or rounded and no precision limit is imposed. ``decode_decimal``
accepts any strict, finite decimal spelling in ASCII (sign, digits, optional
point, optional exponent) and nothing else: no surrounding whitespace (storage
corruption is never "repaired"), no NaN / Infinity, no underscores, no non-ASCII
digits. Neither direction reads or changes the global decimal context.

UTC datetime: ``encode_utc_datetime`` accepts only an exact, timezone-aware
``datetime`` whose offset is zero and writes one fixed form,
``YYYY-MM-DDTHH:MM:SS.ffffff+00:00`` (always six microsecond digits). A non-UTC
offset is rejected, never converted: the producer must normalize at the source.
``decode_utc_datetime`` accepts exactly that form and returns an aware datetime
in ``datetime.UTC``.

Every malformed or unsupported value raises ``PersistenceCodecError``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal
from typing import Final

# Decimal construction from text is exact and does not use the context; the
# context below is used only to write the exponent marker deterministically.
_TEXT_CONTEXT: Final = Context(capitals=1)
_DECIMAL_TEXT: Final = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", re.ASCII
)
_UTC_TEXT: Final = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}\+00:00", re.ASCII
)
_UTC_SUFFIX: Final = "+00:00"
_ZERO_OFFSET: Final = timedelta(0)


class PersistenceCodecError(ValueError):
    """A value cannot be encoded for, or decoded from, storage."""


def encode_decimal(value: Decimal) -> str:
    """Lossless storage text of an exact, finite ``Decimal``."""
    if type(value) is not Decimal:
        raise PersistenceCodecError(f"expected an exact Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise PersistenceCodecError(f"Decimal must be finite, got {value}")
    return _TEXT_CONTEXT.to_sci_string(value)


def decode_decimal(value: str) -> Decimal:
    """The exact ``Decimal`` of strict, finite decimal text."""
    if type(value) is not str:
        raise PersistenceCodecError(f"expected str, got {type(value).__name__}")
    if _DECIMAL_TEXT.fullmatch(value) is None:
        raise PersistenceCodecError(f"malformed decimal text: {value!r}")
    decoded = Decimal(value)
    if not decoded.is_finite():  # pragma: no cover - excluded by the pattern
        raise PersistenceCodecError(f"Decimal must be finite, got {value!r}")
    return decoded


def encode_utc_datetime(value: datetime) -> str:
    """Fixed-form storage text of an exact, aware UTC ``datetime``."""
    if type(value) is not datetime:
        raise PersistenceCodecError(f"expected an exact datetime, got {type(value).__name__}")
    offset = value.utcoffset()
    if offset is None:
        raise PersistenceCodecError("datetime must be timezone-aware")
    if offset != _ZERO_OFFSET:
        raise PersistenceCodecError(f"datetime must be in UTC (offset 0), got offset {offset}")
    return value.replace(tzinfo=None).isoformat(timespec="microseconds") + _UTC_SUFFIX


def decode_utc_datetime(value: str) -> datetime:
    """The aware UTC ``datetime`` of text written by ``encode_utc_datetime``."""
    if type(value) is not str:
        raise PersistenceCodecError(f"expected str, got {type(value).__name__}")
    if _UTC_TEXT.fullmatch(value) is None:
        raise PersistenceCodecError(f"malformed UTC datetime text: {value!r}")
    try:
        naive = datetime.fromisoformat(value.removesuffix(_UTC_SUFFIX))
    except ValueError:
        raise PersistenceCodecError(f"invalid UTC datetime: {value!r}") from None
    return naive.replace(tzinfo=UTC)
