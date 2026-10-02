"""Risk Manager V1 evaluator (docs/ARCHITECTURE.md 9.0).

``evaluate(intent=..., snapshot=..., policy=...) -> RiskDecision`` is a pure,
deterministic function: no clock, I/O, logging or mutable state; it never changes
the intent. Programmer errors (wrong types, symbol mismatch) raise
``DomainValidationError``; every risk outcome is a ``RiskDecision``.

Order of evaluation:
1. No limits for the symbol -> NO_RISK_LIMITS_FOR_SYMBOL.
2. Trading state: HALTED -> KILL_SWITCH_ACTIVE, PAUSED -> TRADING_PAUSED,
   REDUCE_ONLY with an ordinary intent -> REDUCE_ONLY_STATE.
3. Unknown position -> UNKNOWN_POSITION.
4. Reduce-only intent: flat -> REDUCE_ONLY_WITHOUT_POSITION, same side as the
   position -> REDUCE_ONLY_WRONG_SIDE; otherwise it is a valid reduce-only.
5. Unknown symbol open orders -> UNKNOWN_OPEN_ORDERS (also for a valid
   reduce-only: an approval needs an honest exposure, decision A).
6. Exposure (and the baseline worst case without the intent); a result that
   cannot be computed exactly -> UNREPRESENTABLE_CALCULATION.
Each of these returns its single reason with ``exposure=None``. After a
successful exposure every applicable limit is checked and all violations are
returned, in ``RiskReason`` declaration order, with the exposure:
* ``max_open_orders`` (account-wide, every placement incl. reduce-only):
  unknown account count -> UNKNOWN_OPEN_ORDERS, ``count >= max`` ->
  MAX_OPEN_ORDERS.
* ``max_order_qty`` / ``max_order_notional`` (ordinary intents only, on the
  whole qty; valid reduce-only is exempt): qty above the limit, exact
  ``price * qty`` above the limit, no price -> MISSING_REFERENCE_PRICE,
  unrepresentable product -> UNREPRESENTABLE_CALCULATION.
* ``max_position_qty``: a side is rejected when the worst case after the intent
  exceeds the limit **and** exceeds the worst case before it, so an existing
  excess that the intent does not worsen is not blamed on it.
Equality with a limit is allowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from app.domain.enums import Side
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.risk.exposure import (
    ExposureCalculationError,
    calculate_exposure,
    calculate_order_notional,
    calculate_worst_case,
)
from app.risk.models import (
    ExposureChange,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    RiskSnapshot,
    SymbolRiskLimits,
    TradingState,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Evaluation:
    """Inputs of the limit checks once the intent could be decomposed."""

    intent: PlaceOrderIntent
    snapshot: RiskSnapshot
    policy: RiskPolicy
    limits: SymbolRiskLimits
    valid_reduce_only: bool
    exposure: ExposureChange
    before_long: Decimal
    before_short: Decimal


def _check_open_orders(e: _Evaluation) -> frozenset[RiskReason]:
    maximum = e.policy.max_open_orders
    if maximum is None:
        return frozenset()
    count = e.snapshot.account_open_order_count
    if count is None:
        return frozenset({RiskReason.UNKNOWN_OPEN_ORDERS})
    return frozenset({RiskReason.MAX_OPEN_ORDERS}) if count >= maximum else frozenset()


def _check_order_qty(e: _Evaluation) -> frozenset[RiskReason]:
    limit = e.limits.max_order_qty
    if e.valid_reduce_only or limit is None or e.intent.qty <= limit:
        return frozenset()
    return frozenset({RiskReason.MAX_ORDER_QTY})


def _check_order_notional(e: _Evaluation) -> frozenset[RiskReason]:
    limit = e.limits.max_order_notional
    if e.valid_reduce_only or limit is None:
        return frozenset()
    if e.intent.price is None:
        return frozenset({RiskReason.MISSING_REFERENCE_PRICE})
    try:
        notional = calculate_order_notional(price=e.intent.price, qty=e.intent.qty)
    except ExposureCalculationError:
        return frozenset({RiskReason.UNREPRESENTABLE_CALCULATION})
    return frozenset({RiskReason.MAX_ORDER_NOTIONAL}) if notional > limit else frozenset()


def _check_position(e: _Evaluation) -> frozenset[RiskReason]:
    limit = e.limits.max_position_qty
    if limit is None:
        return frozenset()
    sides = (
        (e.exposure.worst_long_qty, e.before_long),
        (e.exposure.worst_short_qty, e.before_short),
    )
    if any(after > limit and after > before for after, before in sides):
        return frozenset({RiskReason.MAX_POSITION_QTY})
    return frozenset()


# Independent checks; the result order comes from RiskReason, not from here.
_LIMIT_CHECKS: Final[tuple[Callable[[_Evaluation], frozenset[RiskReason]], ...]] = (
    _check_open_orders,
    _check_order_qty,
    _check_order_notional,
    _check_position,
)


def _decision(
    intent: PlaceOrderIntent,
    snapshot: RiskSnapshot,
    policy: RiskPolicy,
    reasons: frozenset[RiskReason],
    exposure: ExposureChange | None,
) -> RiskDecision:
    ordered = tuple(reason for reason in RiskReason if reason in reasons)
    return RiskDecision(
        intent_id=intent.intent_id,
        snapshot_id=snapshot.snapshot_id,
        policy_id=policy.policy_id,
        approved=not ordered,
        reasons=ordered,
        exposure=exposure,
    )


def _early_reason(
    intent: PlaceOrderIntent, snapshot: RiskSnapshot, policy: RiskPolicy
) -> RiskReason | None:
    """Rejections decided before (or because of) an impossible decomposition."""
    if intent.symbol not in policy.symbols:
        return RiskReason.NO_RISK_LIMITS_FOR_SYMBOL
    state = snapshot.trading_state
    if state is TradingState.HALTED:
        return RiskReason.KILL_SWITCH_ACTIVE
    if state is TradingState.PAUSED:
        return RiskReason.TRADING_PAUSED
    if state is TradingState.REDUCE_ONLY and not intent.reduce_only:
        return RiskReason.REDUCE_ONLY_STATE
    position = snapshot.position_qty
    if position is None:
        return RiskReason.UNKNOWN_POSITION
    if intent.reduce_only:
        if position == 0:
            return RiskReason.REDUCE_ONLY_WITHOUT_POSITION
        if (position > 0) == (intent.side is Side.BUY):
            return RiskReason.REDUCE_ONLY_WRONG_SIDE
    if snapshot.open_orders is None:
        return RiskReason.UNKNOWN_OPEN_ORDERS
    return None


def evaluate(
    *, intent: PlaceOrderIntent, snapshot: RiskSnapshot, policy: RiskPolicy
) -> RiskDecision:
    """Approve or reject ``intent`` against ``snapshot`` and ``policy``."""
    if type(intent) is not PlaceOrderIntent:
        raise DomainValidationError("intent must be a PlaceOrderIntent")
    if type(snapshot) is not RiskSnapshot:
        raise DomainValidationError("snapshot must be a RiskSnapshot")
    if type(policy) is not RiskPolicy:
        raise DomainValidationError("policy must be a RiskPolicy")
    if intent.symbol != snapshot.symbol:
        raise DomainValidationError(
            f"intent symbol {intent.symbol} differs from snapshot symbol {snapshot.symbol}"
        )

    early = _early_reason(intent, snapshot, policy)
    if early is not None:
        return _decision(intent, snapshot, policy, frozenset({early}), None)

    # _early_reason guarantees a known position, known symbol orders and limits.
    position = snapshot.position_qty
    open_orders = snapshot.open_orders
    if position is None or open_orders is None:  # pragma: no cover - guarded above
        raise DomainValidationError("position and open orders must be known here")
    valid_reduce_only = intent.reduce_only  # a reduce-only reaching here is valid
    try:
        exposure = calculate_exposure(
            position_qty=position,
            intent=intent,
            open_orders=open_orders,
            valid_reduce_only=valid_reduce_only,
        )
        before_long, before_short = calculate_worst_case(
            position_qty=position, open_orders=open_orders
        )
    except ExposureCalculationError:
        return _decision(
            intent, snapshot, policy, frozenset({RiskReason.UNREPRESENTABLE_CALCULATION}), None
        )

    evaluation = _Evaluation(
        intent=intent,
        snapshot=snapshot,
        policy=policy,
        limits=policy.symbols[intent.symbol],
        valid_reduce_only=valid_reduce_only,
        exposure=exposure,
        before_long=before_long,
        before_short=before_short,
    )
    reasons: frozenset[RiskReason] = frozenset()
    for check in _LIMIT_CHECKS:
        reasons |= check(evaluation)
    return _decision(intent, snapshot, policy, reasons, exposure)
