"""Exact Decimal and UTC datetime persistence codecs (stdlib only)."""

from __future__ import annotations

import ast
import itertools
from datetime import UTC, datetime, timedelta, timezone
from decimal import (
    ROUND_UP,
    Context,
    Decimal,
    Inexact,
    InvalidOperation,
    Rounded,
    getcontext,
    localcontext,
)
from pathlib import Path

import pytest

from app.persistence import codecs as codecs_module
from app.persistence.codecs import (
    PersistenceCodecError,
    decode_decimal,
    decode_utc_datetime,
    encode_decimal,
    encode_utc_datetime,
)

D = Decimal

# --- Decimal: lossless round-trip -----------------------------------------------------------

SPECIAL_DECIMALS = [
    "4",
    "4.0",
    "4.000",
    "0",
    "-0",
    "0.0",
    "-0.000",
    "0E+10",
    "-0E+10",
    "0E-10",
    "1E-200",
    "1E+200",
    "-1E+200",
    "123.456",
    "-0.000001",
    "100",
    "1E+2",
    "1.00E+2",
    "0.1",
    "12345678901234567890.12345678901234567890",
    "1." + "1" * 100,
    "-" + "9" * 150,
    "9" * 120 + "E-500",
    "1E+999999",
    "1E-999999",
]

SIGNS = ["", "-"]
COEFFICIENTS = ["0", "1", "10", "100", "7", "123456789", "1" * 40, "9" * 101]
EXPONENTS = [-1000, -200, -30, -7, -1, 0, 1, 5, 30, 200, 1000]
GENERATED_DECIMALS = [
    D(f"{sign}{digits}E{exponent:+d}")
    for sign, digits, exponent in itertools.product(SIGNS, COEFFICIENTS, EXPONENTS)
]
ALL_DECIMALS = [D(text) for text in SPECIAL_DECIMALS] + GENERATED_DECIMALS


@pytest.mark.parametrize("value", ALL_DECIMALS, ids=str)
def test_decimal_round_trip_preserves_sign_digits_and_exponent(value: Decimal) -> None:
    encoded = encode_decimal(value)
    decoded = decode_decimal(encoded)

    assert type(encoded) is str
    assert type(decoded) is Decimal
    assert decoded.as_tuple() == value.as_tuple()
    assert encode_decimal(decoded) == encoded  # encode -> decode -> encode is stable


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("4", "4"),
        ("4.000", "4.000"),
        ("-0", "-0"),
        ("0E+10", "0E+10"),
        ("1E-200", "1E-200"),
        ("1.00E+2", "100"),  # coefficient 100, exponent 0
        ("0.000001", "0.000001"),
        ("0.0000001", "1E-7"),
    ],
)
def test_decimal_encoding_is_the_scientific_string(value: str, text: str) -> None:
    assert encode_decimal(D(value)) == text


def test_trailing_zeros_and_signed_zero_are_kept_distinct() -> None:
    encoded = {encode_decimal(D(text)) for text in ["4", "4.0", "4.000", "0", "-0", "0E+10"]}

    assert len(encoded) == 6
    assert decode_decimal("-0").is_signed()
    assert decode_decimal("4.000").as_tuple().exponent == -3


@pytest.mark.parametrize(
    "value",
    [D("NaN"), D("-NaN"), D("sNaN"), D("Infinity"), D("-Infinity")],
    ids=str,
)
def test_non_finite_decimal_is_not_encoded(value: Decimal) -> None:
    with pytest.raises(PersistenceCodecError, match="finite"):
        encode_decimal(value)


class DecimalSubclass(Decimal):
    pass


@pytest.mark.parametrize("value", [1, 1.5, True, "1", None, DecimalSubclass("1")])
def test_only_exact_decimal_is_encoded(value: object) -> None:
    with pytest.raises(PersistenceCodecError, match="exact Decimal"):
        encode_decimal(value)  # type: ignore[arg-type]


# --- Decimal: decoding ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("+1", "1"), ("1e5", "1E+5"), (".5", "0.5"), ("5.", "5"), ("-0.0", "-0.0"), ("1E+0", "1")],
)
def test_alternative_strict_spellings_are_decoded_exactly(text: str, expected: str) -> None:
    decoded = decode_decimal(text)

    assert decoded.as_tuple() == D(text).as_tuple()
    assert encode_decimal(decoded) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\t",
        " 1",
        "1 ",
        "1\n",
        "\n1",
        "NaN",
        "nan",
        "sNaN",
        "Infinity",
        "-Infinity",
        "inf",
        "abc",
        "1,2",
        "--1",
        "+-1",
        "1..2",
        "1.2.3",
        "1E",
        "E5",
        "1E+",
        "1e5.5",
        ".",
        "-",
        "1_000",
        "\u0661",  # ARABIC-INDIC DIGIT ONE: Decimal accepts it, storage must not
        "\uff11",  # FULLWIDTH DIGIT ONE
        "0x10",
        "1/2",
    ],
)
def test_malformed_decimal_text_fails_closed(text: str) -> None:
    with pytest.raises(PersistenceCodecError):
        decode_decimal(text)


@pytest.mark.parametrize("value", [b"1", 1, None, D("1")])
def test_decoder_requires_text(value: object) -> None:
    with pytest.raises(PersistenceCodecError, match="str"):
        decode_decimal(value)  # type: ignore[arg-type]


def test_malformed_text_fails_even_when_invalid_operation_is_not_trapped() -> None:
    # Without the trap Decimal("abc") would silently return NaN.
    with localcontext() as context:
        context.traps[InvalidOperation] = False
        with pytest.raises(PersistenceCodecError):
            decode_decimal("abc")


# --- Decimal: context independence ----------------------------------------------------------


def context_state() -> tuple[object, ...]:
    context = getcontext()
    return (
        context.prec,
        context.rounding,
        context.Emin,
        context.Emax,
        context.capitals,
        context.clamp,
        dict(context.traps),
        dict(context.flags),
    )


def test_decimal_codec_ignores_an_aggressive_global_context() -> None:
    # A list, not a dict: Decimal("4") == Decimal("4.000") would collapse as keys.
    expected = [(value, encode_decimal(value)) for value in ALL_DECIMALS]

    with localcontext() as context:
        context.prec = 1
        context.rounding = ROUND_UP
        context.Emin = -1
        context.Emax = 1
        context.capitals = 0  # str() would write "1e+200" under this context
        context.clamp = 1
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        context.clear_flags()
        before = context_state()
        for _ in range(3):
            for value, text in expected:
                assert encode_decimal(value) == text
                assert decode_decimal(text).as_tuple() == value.as_tuple()
        assert context_state() == before


def test_global_context_is_unchanged_after_many_calls() -> None:
    getcontext().clear_flags()
    before = context_state()

    for value in ALL_DECIMALS * 3:
        decode_decimal(encode_decimal(value))

    assert context_state() == before


def test_encoding_uses_an_upper_case_exponent_regardless_of_context() -> None:
    with localcontext(Context(capitals=0)):
        assert encode_decimal(D("1E+200")) == "1E+200"
        assert encode_decimal(D("1E-7")) == "1E-7"


# --- UTC datetime ---------------------------------------------------------------------------

DATETIMES = [
    datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=UTC),
    datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),  # microsecond 0
    datetime(2026, 1, 2, 3, 4, 5, 1, tzinfo=UTC),
    datetime(2026, 1, 2, 3, 4, 5, 999999, tzinfo=UTC),
    datetime(2024, 2, 29, 12, 0, tzinfo=UTC),  # leap day
    datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),  # year boundary
    datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
    datetime(1, 1, 1, tzinfo=UTC),  # earliest representable
    datetime(1970, 1, 1, tzinfo=UTC),
    datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),  # latest representable
]


@pytest.mark.parametrize("value", DATETIMES, ids=str)
def test_utc_round_trip(value: datetime) -> None:
    encoded = encode_utc_datetime(value)
    decoded = decode_utc_datetime(encoded)

    assert decoded == value
    assert decoded.tzinfo is UTC
    assert decoded.utcoffset() == timedelta(0)
    assert (decoded.year, decoded.microsecond) == (value.year, value.microsecond)
    assert encode_utc_datetime(decoded) == encoded  # stable


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=UTC), "2026-01-02T03:04:05.123456+00:00"),
        (datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC), "2026-01-02T03:04:05.000000+00:00"),
        (datetime(1, 1, 1, tzinfo=UTC), "0001-01-01T00:00:00.000000+00:00"),
    ],
)
def test_utc_text_format_is_fixed(value: datetime, text: str) -> None:
    assert encode_utc_datetime(value) == text


def test_zero_offset_tzinfo_other_than_utc_is_the_same_instant() -> None:
    other_zero = timezone(timedelta(0), "Z0")
    value = datetime(2026, 1, 2, 3, 4, 5, tzinfo=other_zero)

    decoded = decode_utc_datetime(encode_utc_datetime(value))

    assert decoded == value
    assert decoded.tzinfo is UTC


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 1, 12, 0),  # noqa: DTZ001 - naive
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5))),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(microseconds=1))),
    ],
    ids=["naive", "plus-2", "minus-5", "one-microsecond"],
)
def test_non_utc_datetime_is_rejected_not_converted(value: datetime) -> None:
    with pytest.raises(PersistenceCodecError, match=r"UTC|timezone-aware"):
        encode_utc_datetime(value)


class DatetimeSubclass(datetime):
    pass


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01T00:00:00.000000+00:00",
        1_700_000_000,
        None,
        DatetimeSubclass(2026, 1, 1, tzinfo=UTC),
    ],
)
def test_only_exact_datetime_is_encoded(value: object) -> None:
    with pytest.raises(PersistenceCodecError, match="exact datetime"):
        encode_utc_datetime(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "2026-01-02T03:04:05.123456",  # naive
        "2026-01-02T03:04:05.123456Z",
        "2026-01-02T03:04:05.123456+02:00",
        "2026-01-02T03:04:05.123456-00:00",
        "2026-01-02T03:04:05.123456+0000",
        "2026-01-02T03:04:05+00:00",  # no microseconds
        "2026-01-02T03:04:05.1+00:00",
        "2026-01-02T03:04:05.12345+00:00",
        "2026-01-02T03:04:05.1234567+00:00",
        "2026-01-02 03:04:05.123456+00:00",  # space instead of T
        " 2026-01-02T03:04:05.123456+00:00",
        "2026-01-02T03:04:05.123456+00:00 ",
        "2026-01-02T03:04:05.123456+00:00\n",
        "2026-01-02",
        "2026-01-02T03:04+00:00",
        "2026-1-02T03:04:05.123456+00:00",
        "20260102T030405.123456+00:00",
        "2026-02-30T03:04:05.123456+00:00",  # impossible date
        "2026-01-02T24:00:00.000000+00:00",
        "0000-01-01T00:00:00.000000+00:00",
        "2026-01-02t03:04:05.123456+00:00",
        "\uff12026-01-02T03:04:05.123456+00:00",  # non-ASCII digit
        "",
        "garbage",
    ],
)
def test_utc_decoder_accepts_only_the_canonical_form(text: str) -> None:
    with pytest.raises(PersistenceCodecError):
        decode_utc_datetime(text)


@pytest.mark.parametrize("value", [b"2026", 1, None])
def test_utc_decoder_requires_text(value: object) -> None:
    with pytest.raises(PersistenceCodecError, match="str"):
        decode_utc_datetime(value)  # type: ignore[arg-type]


def test_error_type_is_a_value_error() -> None:
    assert issubclass(PersistenceCodecError, ValueError)


# --- dependencies ---------------------------------------------------------------------------


def test_codecs_depend_on_the_standard_library_only() -> None:
    import sys

    source = Path(codecs_module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert imports
    for name in imports:
        assert name == "__future__" or name.split(".")[0] in sys.stdlib_module_names, name
    for banned in ("normalize(", ".strip(", "astimezone", "getcontext", "float("):
        assert banned not in source, banned
