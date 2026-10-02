"""Immutable Risk Manager V1 models: policy, snapshot and decision.

All numbers are exact ``Decimal`` (subclasses, floats, ints, bools and strings
are rejected) and are stored as given: no arithmetic, rounding or global decimal
context is involved. Validation failures are ``DomainValidationError``; an
ordinary limit breach is never an exception but a ``RiskDecision``.

Not part of V1 (docs/ARCHITECTURE.md 9.0): equity, cash, mark prices and data
freshness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from app.domain.enums import OrderStatus, Side
from app.domain.errors import DomainValidationError
from app.domain.validation import require_bool, require_enum, require_text


class TradingState(StrEnum):
    """What new placements are allowed (docs/ARCHITECTURE.md 9.0 / 9.2)."""

    RUNNING = "running"
    REDUCE_ONLY = "reduce_only"
    PAUSED = "paused"
    HALTED = "halted"
    """Active kill switch: no strategy placements at all."""


class RiskReason(StrEnum):
    """Machine-readable rejection reasons (the contract; no free text)."""

    KILL_SWITCH_ACTIVE = "kill_switch_active"
    TRADING_PAUSED = "trading_paused"
    REDUCE_ONLY_STATE = "reduce_only_state"
    UNKNOWN_POSITION = "unknown_position"
    UNKNOWN_OPEN_ORDERS = "unknown_open_orders"
    NO_RISK_LIMITS_FOR_SYMBOL = "no_risk_limits_for_symbol"
    REDUCE_ONLY_WITHOUT_POSITION = "reduce_only_without_position"
    REDUCE_ONLY_WRONG_SIDE = "reduce_only_wrong_side"
    MAX_ORDER_QTY = "max_order_qty"
    MAX_ORDER_NOTIONAL = "max_order_notional"
    MISSING_REFERENCE_PRICE = "missing_reference_price"
    MAX_POSITION_QTY = "max_position_qty"
    MAX_OPEN_ORDERS = "max_open_orders"
    UNREPRESENTABLE_CALCULATION = "unrepresentable_calculation"


# Orders that may still execute, so Risk counts them conservatively: every
# non-terminal status, including UNKNOWN (outcome not yet known) and CANCELING.
ACTIVE_ORDER_STATUSES: Final = frozenset(
    {
        OrderStatus.NEW,
        OrderStatus.SUBMITTING,
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCELING,
        OrderStatus.UNKNOWN,
    }
)


def _exact_decimal(value: object, field: str) -> Decimal:
    if type(value) is not Decimal:
        raise DomainValidationError(f"{field} must be an exact Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise DomainValidationError(f"{field} must be finite, got {value}")
    return value


def _positive(value: object, field: str) -> Decimal:
    decimal = _exact_decimal(value, field)
    if decimal <= 0:
        raise DomainValidationError(f"{field} must be > 0, got {decimal}")
    return decimal


def _non_negative(value: object, field: str) -> Decimal:
    decimal = _exact_decimal(value, field)
    if decimal < 0:
        raise DomainValidationError(f"{field} must be >= 0, got {decimal}")
    return decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenOrderExposure:
    """A potentially active order of the snapshot's symbol, as Risk sees it."""

    side: Side
    remaining_qty: Decimal
    """Quantity that may still execute (> 0); the full remainder for UNKNOWN."""
    price: Decimal | None
    """Limit price; None when not known (e.g. a market order)."""
    reduce_only: bool
    status: OrderStatus

    def __post_init__(self) -> None:
        require_enum(self.side, Side, "side")
        _positive(self.remaining_qty, "remaining_qty")
        if self.price is not None:
            _positive(self.price, "price")
        require_bool(self.reduce_only, "reduce_only")
        status = require_enum(self.status, OrderStatus, "status")
        if status not in ACTIVE_ORDER_STATUSES:
            raise DomainValidationError(f"status {status.value} is not an active order status")


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskSnapshot:
    """Everything a V1 decision may use, supplied explicitly by the caller."""

    snapshot_id: str
    symbol: str
    trading_state: TradingState
    position_qty: Decimal | None
    """Signed net position: > 0 long, < 0 short, 0 known flat; None = unknown."""
    open_orders: tuple[OpenOrderExposure, ...] | None
    """Active orders of ``symbol``; () = known none; None = unknown."""

    def __post_init__(self) -> None:
        require_text(self.snapshot_id, "snapshot_id")
        require_text(self.symbol, "symbol")
        require_enum(self.trading_state, TradingState, "trading_state")
        if self.position_qty is not None:
            _exact_decimal(self.position_qty, "position_qty")
        if self.open_orders is not None:
            if type(self.open_orders) is not tuple:
                raise DomainValidationError("open_orders must be a tuple or None")
            for item in self.open_orders:
                if not isinstance(item, OpenOrderExposure):
                    raise DomainValidationError("open_orders must contain only OpenOrderExposure")


@dataclass(frozen=True, slots=True, kw_only=True)
class SymbolRiskLimits:
    """Per-symbol limits; every field is required and None explicitly disables it."""

    max_order_qty: Decimal | None
    """Fat-finger limit on the whole qty of an ordinary new order."""
    max_order_notional: Decimal | None
    """Fat-finger limit on price * qty of an ordinary new order (quote asset)."""
    max_position_qty: Decimal | None
    """Limit on the worst-case future position size, either side."""

    def __post_init__(self) -> None:
        for field in ("max_order_qty", "max_order_notional", "max_position_qty"):
            value = getattr(self, field)
            if value is not None:
                _positive(value, field)


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskPolicy:
    """Global and per-symbol V1 limits. An empty ``symbols`` mapping is valid and
    allows nothing: every symbol is rejected as NO_RISK_LIMITS_FOR_SYMBOL."""

    policy_id: str
    max_open_orders: int | None
    """Applies to every new placement, reduce-only included; None disables it."""
    symbols: Mapping[str, SymbolRiskLimits]

    def __post_init__(self) -> None:
        require_text(self.policy_id, "policy_id")
        value = self.max_open_orders
        if value is not None and (type(value) is not int or value <= 0):
            raise DomainValidationError(
                f"max_open_orders must be an int > 0 or None, got {value!r}"
            )
        if not isinstance(self.symbols, Mapping):
            raise DomainValidationError("symbols must be a mapping of symbol to SymbolRiskLimits")
        copied: dict[str, SymbolRiskLimits] = {}
        for symbol, limits in self.symbols.items():
            require_text(symbol, "symbol")
            if not isinstance(limits, SymbolRiskLimits):
                raise DomainValidationError(f"limits of {symbol} must be SymbolRiskLimits")
            copied[symbol] = limits
        # Defensive read-only copy: later changes to the caller's mapping are not seen.
        object.__setattr__(self, "symbols", MappingProxyType(copied))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExposureChange:
    """Decomposition of an intent against the snapshot (all magnitudes >= 0).

    ``reducing_qty`` / ``increasing_qty``: the parts that reduce the current
    position and that open or increase exposure (a reversal has both).
    ``worst_long_qty`` / ``worst_short_qty``: worst-case long and short position
    sizes after the intent, as positive magnitudes, counting all active
    non-reduce-only orders of each side and never netting opposite orders.
    """

    reducing_qty: Decimal
    increasing_qty: Decimal
    worst_long_qty: Decimal
    worst_short_qty: Decimal

    def __post_init__(self) -> None:
        for field in ("reducing_qty", "increasing_qty", "worst_long_qty", "worst_short_qty"):
            _non_negative(getattr(self, field), field)


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskDecision:
    """Approve / reject result with audit identifiers; never changes the intent.

    ``approved`` is True exactly when ``reasons`` is empty. ``exposure`` is None
    only for a rejection decided before the intent could be decomposed (e.g. an
    unknown position); an approval always carries it.
    """

    intent_id: str
    snapshot_id: str
    policy_id: str
    approved: bool
    reasons: tuple[RiskReason, ...]
    exposure: ExposureChange | None

    def __post_init__(self) -> None:
        require_text(self.intent_id, "intent_id")
        require_text(self.snapshot_id, "snapshot_id")
        require_text(self.policy_id, "policy_id")
        approved = require_bool(self.approved, "approved")
        if type(self.reasons) is not tuple or not all(
            isinstance(reason, RiskReason) for reason in self.reasons
        ):
            raise DomainValidationError("reasons must be a tuple of RiskReason")
        if len(set(self.reasons)) != len(self.reasons):
            raise DomainValidationError("reasons must not contain duplicates")
        if approved != (not self.reasons):
            raise DomainValidationError("approved must be True exactly when there are no reasons")
        if self.exposure is not None and not isinstance(self.exposure, ExposureChange):
            raise DomainValidationError("exposure must be an ExposureChange or None")
        if approved and self.exposure is None:
            raise DomainValidationError("an approved decision requires exposure")
