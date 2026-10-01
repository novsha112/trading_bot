"""Enum values are a persistence and logging contract: changing one is a migration."""

from __future__ import annotations

from enum import StrEnum

import pytest

from app.domain.enums import (
    GridMode,
    GridSpacing,
    OrderStatus,
    OrderType,
    PositionSide,
    RoundingDirection,
    Side,
    TimeInForce,
    TradingMode,
)

EXPECTED_VALUES: dict[type[StrEnum], set[str]] = {
    TradingMode: {"backtest", "paper", "testnet", "live"},
    Side: {"buy", "sell"},
    OrderType: {"limit", "market"},
    TimeInForce: {"gtc", "ioc", "fok", "post_only"},
    OrderStatus: {
        "new",
        "submitting",
        "open",
        "partially_filled",
        "canceling",
        "filled",
        "canceled",
        "rejected",
        "expired",
        "failed",
        "unknown",
    },
    RoundingDirection: {"down", "up"},
    PositionSide: {"long", "short", "flat"},
    GridMode: {"long", "short", "neutral"},
    GridSpacing: {"arithmetic", "geometric"},
}


@pytest.mark.parametrize("enum_type", list(EXPECTED_VALUES))
def test_enum_values_are_stable(enum_type: type[StrEnum]) -> None:
    assert {member.value for member in enum_type} == EXPECTED_VALUES[enum_type]


@pytest.mark.parametrize("enum_type", list(EXPECTED_VALUES))
def test_enum_is_str_and_round_trips(enum_type: type[StrEnum]) -> None:
    for member in enum_type:
        assert isinstance(member, str)
        assert str(member) == member.value
        assert enum_type(member.value) is member


@pytest.mark.parametrize(
    ("enum_type", "raw"),
    [
        (TradingMode, "LIVE"),  # case-sensitive: no silent normalization
        (TradingMode, "production"),
        (TradingMode, ""),
        (Side, "Buy"),  # exchange-specific spelling is not a domain value
        (OrderStatus, "PartiallyFilled"),
        (TimeInForce, "PostOnly"),
    ],
)
def test_unknown_values_rejected(enum_type: type[StrEnum], raw: str) -> None:
    with pytest.raises(ValueError, match="is not a valid"):
        enum_type(raw)
