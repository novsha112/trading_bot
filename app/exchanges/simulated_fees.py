"""Deterministic trading fees for simulated fills (opt-in).

* ``TradingFeeSchedule``: maker and taker rates and the fee asset, all explicit.
  Rates are exact finite ``Decimal`` of any sign (a negative rate is a rebate,
  zero is a known zero fee); no range is imposed. The fee asset is stated, never
  derived from a symbol.
* ``LiquidityRole``: a simulation parameter, not a domain fact. The simulator has
  no order-book model at placement time, so whether a fill is maker or taker
  cannot be proven: ``SimulatedFeePolicy`` applies one configured role to every
  fill. POST_ONLY does not make a fill maker by itself.

Formula (linear contracts): ``fee = execution price * fill qty * rate`` for the
role's rate, per fill. It is computed exactly in an explicit context (global
decimal context neither read nor modified); exchange settlement rounding to an
asset precision is not modeled, so the value is kept exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
)
from enum import StrEnum
from typing import Final

from app.domain.validation import require_text

# price * qty * rate stays exact for any realistic operands; beyond this the
# calculation fails instead of rounding.
_EXACT_PRECISION: Final = 120


class LiquidityRole(StrEnum):
    MAKER = "maker"
    TAKER = "taker"


class FeeCalculationError(ArithmeticError):
    """A fee could not be computed exactly; nothing is charged or rounded."""


def _require_rate(value: object, field: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError(f"{field} must be a finite Decimal")
    return value


def _require_positive(value: object, field: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise ValueError(f"{field} must be a finite Decimal > 0")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class TradingFeeSchedule:
    maker_rate: Decimal
    taker_rate: Decimal
    fee_asset: str

    def __post_init__(self) -> None:
        _require_rate(self.maker_rate, "maker_rate")
        _require_rate(self.taker_rate, "taker_rate")
        require_text(self.fee_asset, "fee_asset")

    def rate(self, liquidity_role: LiquidityRole) -> Decimal:
        if not isinstance(liquidity_role, LiquidityRole):
            raise ValueError("liquidity_role must be a LiquidityRole")
        return self.maker_rate if liquidity_role is LiquidityRole.MAKER else self.taker_rate

    def calculate(self, *, price: Decimal, qty: Decimal, liquidity_role: LiquidityRole) -> Decimal:
        """Exact fee of one execution of ``qty`` at ``price`` (in ``fee_asset``)."""
        rate = self.rate(liquidity_role)
        _require_positive(price, "price")
        _require_positive(qty, "qty")
        context = Context(
            prec=_EXACT_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
        )
        try:
            return context.multiply(context.multiply(price, qty), rate)
        except DecimalException:
            raise FeeCalculationError("fee cannot be computed exactly") from None


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulatedFeePolicy:
    """A fee schedule plus the liquidity role assumed for every simulated fill."""

    schedule: TradingFeeSchedule
    liquidity_role: LiquidityRole

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, TradingFeeSchedule):
            raise TypeError("schedule must be a TradingFeeSchedule")
        if not isinstance(self.liquidity_role, LiquidityRole):
            raise ValueError("liquidity_role must be a LiquidityRole")

    def fill_fee(self, *, price: Decimal, qty: Decimal) -> tuple[Decimal, str, bool]:
        """``(fee, fee_asset, is_maker)`` for one fill at its execution price/qty."""
        fee = self.schedule.calculate(price=price, qty=qty, liquidity_role=self.liquidity_role)
        return fee, self.schedule.fee_asset, self.liquidity_role is LiquidityRole.MAKER
