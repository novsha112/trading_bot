"""Profile schema: structure and local invariants only."""

from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from app.config.schema import AppConfig
from app.domain.enums import GridMode, GridSpacing

D = Decimal

VALID: dict[str, Any] = {
    "profile": {"name": "development"},
    "exchange": {"name": "bybit", "symbol": "BTCUSDT"},
    "strategy": {
        "type": "grid",
        "grid": {
            "lower_price": D("60000"),
            "upper_price": D("70000.5"),
            "levels": 10,
            "spacing": "arithmetic",
            "mode": "neutral",
            "order_qty": D("0.001"),
        },
    },
}


def build(**changes: Any) -> dict[str, Any]:
    """Copy of VALID with dotted-path changes, e.g. build(**{"strategy.grid.levels": 1})."""
    data = copy.deepcopy(VALID)
    for path, value in changes.items():
        *parents, leaf = path.split(".")
        node = data
        for key in parents:
            node = node[key]
        node[leaf] = value
    return data


def invalid(**changes: Any) -> ValidationError:
    with pytest.raises(ValidationError) as info:
        AppConfig.model_validate(build(**changes))
    return info.value


def test_valid_config() -> None:
    config = AppConfig.model_validate(VALID)
    grid = config.strategy.grid
    assert config.profile.name == "development"
    assert (config.exchange.name, config.exchange.symbol) == ("bybit", "BTCUSDT")
    assert config.strategy.type == "grid"
    assert (grid.lower_price, grid.upper_price, grid.levels) == (D("60000"), D("70000.5"), 10)
    assert grid.spacing is GridSpacing.ARITHMETIC
    assert grid.mode is GridMode.NEUTRAL
    assert grid.order_qty == D("0.001")


def test_config_is_frozen() -> None:
    config = AppConfig.model_validate(VALID)
    with pytest.raises(ValidationError):
        config.strategy.grid.levels = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.profile = config.profile  # type: ignore[misc]


@pytest.mark.parametrize(
    "path",
    ["extra", "profile.extra", "exchange.base_url", "strategy.extra", "strategy.grid.leverage"],
)
def test_extra_fields_rejected_at_every_level(path: str) -> None:
    error = invalid(**{path: 1})
    assert [e["type"] for e in error.errors()] == ["extra_forbidden"]


@pytest.mark.parametrize("section", ["profile", "exchange", "strategy", "strategy.grid"])
def test_sections_required(section: str) -> None:
    data = build()
    *parents, leaf = section.split(".")
    node = data
    for key in parents:
        node = node[key]
    del node[leaf]
    with pytest.raises(ValidationError, match="Field required"):
        AppConfig.model_validate(data)


def test_ints_are_accepted_for_decimal_fields() -> None:
    config = AppConfig.model_validate(build(**{"strategy.grid.lower_price": 60000}))
    assert config.strategy.grid.lower_price == D("60000")
    assert isinstance(config.strategy.grid.lower_price, Decimal)


@pytest.mark.parametrize("field", ["lower_price", "upper_price", "order_qty"])
@pytest.mark.parametrize(
    "value", [0.1, True, "0.1", None, D("NaN"), D("sNaN"), D("Infinity"), D("-Infinity")]
)
def test_decimal_fields_reject_float_bool_str_and_non_finite(field: str, value: Any) -> None:
    error = invalid(**{f"strategy.grid.{field}": value})
    assert error.errors()[0]["loc"][-1] == field


@pytest.mark.parametrize("field", ["lower_price", "order_qty"])
@pytest.mark.parametrize("value", [D("0"), D("-1"), 0])
def test_positive_fields(field: str, value: Any) -> None:
    error = invalid(**{f"strategy.grid.{field}": value})
    assert error.errors()[0]["type"] == "greater_than"


@pytest.mark.parametrize("upper", [D("60000"), D("59999.99")])
def test_upper_price_must_exceed_lower_price(upper: Decimal) -> None:
    error = invalid(**{"strategy.grid.upper_price": upper})
    assert "upper_price must be greater than lower_price" in str(error)


@pytest.mark.parametrize("levels", [1, 0, -5, True, False, 2.0, "10", D("10")])
def test_levels_must_be_int_at_least_two(levels: Any) -> None:
    invalid(**{"strategy.grid.levels": levels})


def test_two_levels_accepted() -> None:
    assert AppConfig.model_validate(build(**{"strategy.grid.levels": 2})).strategy.grid.levels == 2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("strategy.grid.spacing", "Arithmetic"),
        ("strategy.grid.spacing", "linear"),
        ("strategy.grid.mode", "LONG"),
        ("strategy.grid.mode", "both"),
        ("strategy.type", "dca"),
        ("strategy.type", "Grid"),
        ("exchange.name", "binance"),
        ("exchange.name", "Bybit"),
    ],
)
def test_exact_enum_and_literal_values(path: str, value: str) -> None:
    invalid(**{path: value})


@pytest.mark.parametrize("path", ["profile.name", "exchange.symbol"])
@pytest.mark.parametrize("value", ["", "  ", " BTCUSDT", "BTCUSDT ", True, 1])
def test_text_fields_clean_and_strict(path: str, value: Any) -> None:
    invalid(**{path: value})


def test_validation_errors_do_not_echo_values() -> None:
    error = invalid(**{"exchange.symbol": " sk-live-looking-value "})
    assert "sk-live-looking-value" not in str(error)
