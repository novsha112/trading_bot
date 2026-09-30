"""Decimal and UTC guards shared by all domain models."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from app.domain.errors import DomainError, DomainValidationError
from app.domain.validation import (
    require_decimal,
    require_non_negative,
    require_positive,
    require_utc,
    utc_from_ms,
)

NON_FINITE = [
    Decimal("NaN"),
    Decimal("-NaN"),
    Decimal("sNaN"),
    Decimal("Infinity"),
    Decimal("-Infinity"),
]
NOT_DECIMAL: list[object] = [1.5, 0.0, float("nan"), True, False, 1, 0, "1.5", None, b"1"]


def test_validation_error_is_value_error_and_domain_error() -> None:
    assert issubclass(DomainValidationError, ValueError)
    assert issubclass(DomainValidationError, DomainError)


# --- require_decimal ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [Decimal("0"), Decimal("1.5"), Decimal("-2"), Decimal("1E-18")])
def test_require_decimal_returns_same_value(value: Decimal) -> None:
    assert require_decimal(value, "price") is value


@pytest.mark.parametrize("value", NOT_DECIMAL)
def test_require_decimal_rejects_non_decimal_types(value: object) -> None:
    with pytest.raises(DomainValidationError, match=r"^price must be a Decimal"):
        require_decimal(value, "price")


@pytest.mark.parametrize("value", NON_FINITE)
def test_require_decimal_rejects_non_finite(value: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=r"^qty must be finite"):
        require_decimal(value, "qty")


# --- require_positive / require_non_negative ----------------------------------------------


@pytest.mark.parametrize("value", [Decimal("0.00000001"), Decimal("1"), Decimal("65000.5")])
def test_require_positive_accepts_positive(value: Decimal) -> None:
    assert require_positive(value, "price") is value


@pytest.mark.parametrize("value", [Decimal("0"), Decimal("-0"), Decimal("-0.1")])
def test_require_positive_rejects_zero_and_negative(value: Decimal) -> None:
    with pytest.raises(DomainValidationError, match=r"^price must be > 0"):
        require_positive(value, "price")


@pytest.mark.parametrize("value", [Decimal("0"), Decimal("-0"), Decimal("3")])
def test_require_non_negative_accepts_zero_and_positive(value: Decimal) -> None:
    assert require_non_negative(value, "filled_qty") is value


def test_require_non_negative_rejects_negative() -> None:
    with pytest.raises(DomainValidationError, match=r"^filled_qty must be >= 0"):
        require_non_negative(Decimal("-0.001"), "filled_qty")


@pytest.mark.parametrize("guard", [require_positive, require_non_negative])
@pytest.mark.parametrize("value", [*NON_FINITE, 1.0, True, 1])
def test_sign_guards_apply_decimal_checks_first(guard: object, value: object) -> None:
    assert callable(guard)
    with pytest.raises(DomainValidationError):
        guard(value, "amount")


# --- require_utc -------------------------------------------------------------------------

# Deliberately naive: each test attaches the tzinfo it needs.
_MOMENT = datetime(2026, 1, 15, 12, 30, 45, 123456)  # noqa: DTZ001


@pytest.mark.parametrize(
    "value",
    [
        _MOMENT.replace(tzinfo=UTC),
        _MOMENT.replace(tzinfo=timezone(timedelta(0))),
        _MOMENT.replace(tzinfo=ZoneInfo("UTC")),
    ],
)
def test_require_utc_accepts_zero_offset(value: datetime) -> None:
    assert require_utc(value, "created_at") is value


class _NoOffset(tzinfo):
    """A tzinfo that cannot tell its offset: aware in form, unknown in fact."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_MOMENT, "timezone-aware"),
        (_MOMENT.replace(tzinfo=_NoOffset()), "timezone-aware"),
        (_MOMENT.replace(tzinfo=timezone(timedelta(hours=2))), "UTC"),
        (_MOMENT.replace(tzinfo=timezone(timedelta(minutes=-30))), "UTC"),
        (_MOMENT.replace(tzinfo=ZoneInfo("Europe/Kyiv")), "UTC"),
        (date(2026, 1, 15), "datetime"),
        ("2026-01-15T12:30:45Z", "datetime"),
        (1768480245123, "datetime"),
        (None, "datetime"),
    ],
)
def test_require_utc_rejects(value: object, message: str) -> None:
    with pytest.raises(DomainValidationError, match=rf"^created_at .*{message}"):
        require_utc(value, "created_at")


# --- utc_from_ms -------------------------------------------------------------------------


def test_utc_from_ms_epoch() -> None:
    assert utc_from_ms(0) == datetime(1970, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (1_768_480_245_123, datetime(2026, 1, 15, 12, 30, 45, 123000, tzinfo=UTC)),
        (1_768_480_245_001, datetime(2026, 1, 15, 12, 30, 45, 1000, tzinfo=UTC)),
        (1_768_480_245_999, datetime(2026, 1, 15, 12, 30, 45, 999000, tzinfo=UTC)),
        (253_402_300_799_999, datetime(9999, 12, 31, 23, 59, 59, 999000, tzinfo=UTC)),
    ],
)
def test_utc_from_ms_is_exact(ms: int, expected: datetime) -> None:
    result = utc_from_ms(ms)
    assert result == expected
    assert result.microsecond == (ms % 1000) * 1000
    assert result.utcoffset() == timedelta(0)


def test_utc_from_ms_result_passes_require_utc() -> None:
    value = utc_from_ms(1_768_480_245_123)
    assert require_utc(value, "exchange_ts") is value


@pytest.mark.parametrize("value", [1_768_480_245_123.0, True, "1768480245123", Decimal("1"), None])
def test_utc_from_ms_rejects_non_int(value: object) -> None:
    with pytest.raises(DomainValidationError, match=r"^ms must be an int"):
        utc_from_ms(value)  # type: ignore[arg-type]


def test_utc_from_ms_rejects_negative() -> None:
    with pytest.raises(DomainValidationError, match=r"^ms must be >= 0"):
        utc_from_ms(-1)


def test_utc_from_ms_rejects_out_of_range() -> None:
    with pytest.raises(DomainValidationError, match=r"^ms is out of the supported datetime range"):
        utc_from_ms(253_402_300_800_000)
