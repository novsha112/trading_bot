"""Schema of a YAML profile (configs/<profile>.yaml).

Only structure and local invariants are validated here. Exchange constraints
(tick size, quantity step, minimums) come from exchange metadata at runtime and
are never duplicated in YAML; balances, leverage, fees and risk are checked by
later layers. Secrets never appear in profiles: they live in EnvSettings only.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from app.domain.enums import GridMode, GridSpacing


def _decimal_input(value: object) -> object:
    # The YAML loader produces Decimal (from the scalar text) or int. Anything else
    # (float, bool, quoted string) is rejected instead of being converted.
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        raise ValueError("must be a YAML number (decimal or integer), not a string, float or bool")
    return Decimal(value) if isinstance(value, int) else value


def _clean_text(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("must be non-empty and without surrounding whitespace")
    return value


# Finite (NaN / Infinity are rejected by Pydantic's Decimal validation).
ConfigDecimal = Annotated[Decimal, BeforeValidator(_decimal_input)]
CleanText = Annotated[StrictStr, AfterValidator(_clean_text)]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ProfileConfig(_Section):
    name: CleanText
    """Must equal CONFIG_PROFILE; selects the allowed trading modes."""


class ExchangeConfig(_Section):
    name: Literal["bybit"]
    symbol: CleanText


class GridConfig(_Section):
    lower_price: ConfigDecimal = Field(gt=0)
    upper_price: ConfigDecimal
    levels: StrictInt = Field(ge=2)
    spacing: GridSpacing
    mode: GridMode
    order_qty: ConfigDecimal = Field(gt=0)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.upper_price <= self.lower_price:
            raise ValueError("upper_price must be greater than lower_price")
        return self


class StrategyConfig(_Section):
    type: Literal["grid"]
    grid: GridConfig


class AppConfig(_Section):
    """Validated, immutable content of one profile."""

    profile: ProfileConfig
    exchange: ExchangeConfig
    strategy: StrategyConfig
