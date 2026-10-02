"""Risk Manager V1 models: immutable, strictly validated, context-independent."""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from app.domain.enums import OrderStatus, Side
from app.domain.errors import DomainValidationError
from app.domain.order_state import TERMINAL_STATUSES
from app.risk import models
from app.risk.models import (
    ACTIVE_ORDER_STATUSES,
    ExposureChange,
    OpenOrderExposure,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    RiskSnapshot,
    SymbolRiskLimits,
    TradingState,
)

D = Decimal
BAD_DECIMALS: list[object] = [
    D("NaN"),
    D("sNaN"),
    D("Infinity"),
    D("-Infinity"),
    1.5,
    1,
    True,
    "1",
    None,
]


class SubDecimal(Decimal):
    pass


@contextmanager
def low_precision() -> Iterator[None]:
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        yield


def order(**overrides: Any) -> OpenOrderExposure:
    values: dict[str, Any] = {
        "side": Side.BUY,
        "remaining_qty": D("1.5"),
        "price": D("100"),
        "reduce_only": False,
        "status": OrderStatus.OPEN,
    }
    return OpenOrderExposure(**{**values, **overrides})


def snapshot(**overrides: Any) -> RiskSnapshot:
    values: dict[str, Any] = {
        "snapshot_id": "snap-1",
        "symbol": "BTCUSDT",
        "trading_state": TradingState.RUNNING,
        "position_qty": D("0"),
        "open_orders": (),
    }
    return RiskSnapshot(**{**values, **overrides})


def limits(**overrides: Any) -> SymbolRiskLimits:
    values: dict[str, Any] = {
        "max_order_qty": D("10"),
        "max_order_notional": D("100000"),
        "max_position_qty": D("20"),
    }
    return SymbolRiskLimits(**{**values, **overrides})


def policy(**overrides: Any) -> RiskPolicy:
    values: dict[str, Any] = {
        "policy_id": "policy-1",
        "max_open_orders": 50,
        "symbols": {"BTCUSDT": limits()},
    }
    return RiskPolicy(**{**values, **overrides})


def exposure(**overrides: Any) -> ExposureChange:
    values: dict[str, Any] = {
        "reducing_qty": D("0"),
        "increasing_qty": D("2"),
        "worst_long_qty": D("2"),
        "worst_short_qty": D("0"),
    }
    return ExposureChange(**{**values, **overrides})


def decision(**overrides: Any) -> RiskDecision:
    values: dict[str, Any] = {
        "intent_id": "intent-1",
        "snapshot_id": "snap-1",
        "policy_id": "policy-1",
        "approved": True,
        "reasons": (),
        "exposure": exposure(),
    }
    return RiskDecision(**{**values, **overrides})


# --- enums ----------------------------------------------------------------------------


def test_trading_state_members_are_stable() -> None:
    assert {s.name: s.value for s in TradingState} == {
        "RUNNING": "running",
        "REDUCE_ONLY": "reduce_only",
        "PAUSED": "paused",
        "HALTED": "halted",
    }


def test_risk_reason_members_are_stable() -> None:
    assert [r.name for r in RiskReason] == [
        "KILL_SWITCH_ACTIVE",
        "TRADING_PAUSED",
        "REDUCE_ONLY_STATE",
        "UNKNOWN_POSITION",
        "UNKNOWN_OPEN_ORDERS",
        "NO_RISK_LIMITS_FOR_SYMBOL",
        "REDUCE_ONLY_WITHOUT_POSITION",
        "REDUCE_ONLY_WRONG_SIDE",
        "MAX_ORDER_QTY",
        "MAX_ORDER_NOTIONAL",
        "MISSING_REFERENCE_PRICE",
        "MAX_POSITION_QTY",
        "MAX_OPEN_ORDERS",
        "UNREPRESENTABLE_CALCULATION",
    ]
    assert all(r.value == r.name.lower() for r in RiskReason)


# --- OpenOrderExposure ----------------------------------------------------------------


def test_active_statuses_are_exactly_the_non_terminal_ones() -> None:
    assert (
        frozenset(
            {
                OrderStatus.NEW,
                OrderStatus.SUBMITTING,
                OrderStatus.OPEN,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.CANCELING,
                OrderStatus.UNKNOWN,
            }
        )
        == ACTIVE_ORDER_STATUSES
    )
    assert frozenset(OrderStatus) - TERMINAL_STATUSES == ACTIVE_ORDER_STATUSES


@pytest.mark.parametrize("status", sorted(ACTIVE_ORDER_STATUSES))
def test_open_order_accepts_active_statuses(status: OrderStatus) -> None:
    assert order(status=status).status is status


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_open_order_rejects_terminal_statuses(status: OrderStatus) -> None:
    with pytest.raises(DomainValidationError, match="status"):
        order(status=status)


@pytest.mark.parametrize("status", ["open", None, 1])
def test_open_order_status_must_be_the_enum(status: object) -> None:
    with pytest.raises(DomainValidationError, match="status"):
        order(status=status)


@pytest.mark.parametrize("side", ["buy", None])
def test_open_order_side_must_be_the_enum(side: object) -> None:
    with pytest.raises(DomainValidationError, match="side"):
        order(side=side)


@pytest.mark.parametrize("qty", [D("0"), D("-0"), D("-1"), *BAD_DECIMALS, SubDecimal("1")])
def test_open_order_remaining_qty_must_be_positive_exact_decimal(qty: object) -> None:
    with pytest.raises(DomainValidationError, match="remaining_qty"):
        order(remaining_qty=qty)


def test_open_order_price_is_optional() -> None:
    assert order(price=None).price is None


@pytest.mark.parametrize(
    "price", [D("0"), D("-1"), D("NaN"), D("Infinity"), 1.5, 1, True, "1", SubDecimal("1")]
)
def test_open_order_price_invalid(price: object) -> None:
    with pytest.raises(DomainValidationError, match="price"):
        order(price=price)


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_open_order_reduce_only_must_be_bool(value: object) -> None:
    with pytest.raises(DomainValidationError, match="reduce_only"):
        order(reduce_only=value)


# --- RiskSnapshot ---------------------------------------------------------------------


def test_unknown_position_is_distinct_from_known_flat() -> None:
    unknown = snapshot(position_qty=None)
    flat = snapshot(position_qty=D("0"))

    assert unknown.position_qty is None
    assert flat.position_qty == 0
    assert unknown != flat


def test_unknown_open_orders_is_distinct_from_known_none() -> None:
    unknown = snapshot(open_orders=None)
    empty = snapshot(open_orders=())

    assert unknown.open_orders is None
    assert empty.open_orders == ()
    assert unknown != empty


@pytest.mark.parametrize("qty", [D("5"), D("-3.333"), D("0")])
def test_snapshot_position_any_sign(qty: Decimal) -> None:
    assert snapshot(position_qty=qty).position_qty == qty


@pytest.mark.parametrize("qty", [*[v for v in BAD_DECIMALS if v is not None], SubDecimal("1")])
def test_snapshot_position_invalid(qty: object) -> None:
    with pytest.raises(DomainValidationError, match="position_qty"):
        snapshot(position_qty=qty)


def test_snapshot_open_orders_must_be_a_tuple_of_exposures() -> None:
    orders = (order(), order(side=Side.SELL, reduce_only=True))

    assert snapshot(open_orders=orders).open_orders == orders
    with pytest.raises(DomainValidationError, match="open_orders"):
        snapshot(open_orders=list(orders))
    with pytest.raises(DomainValidationError, match="open_orders"):
        snapshot(open_orders=(order(), "order"))


@pytest.mark.parametrize("field", ["snapshot_id", "symbol"])
@pytest.mark.parametrize("value", ["", " x", None, 1])
def test_snapshot_text_fields(field: str, value: object) -> None:
    with pytest.raises(DomainValidationError, match=field):
        snapshot(**{field: value})


@pytest.mark.parametrize("value", ["running", None])
def test_snapshot_trading_state_must_be_the_enum(value: object) -> None:
    with pytest.raises(DomainValidationError, match="trading_state"):
        snapshot(trading_state=value)


def test_snapshot_has_no_cash_equity_mark_or_time() -> None:
    names = {f.name for f in dataclasses.fields(RiskSnapshot)}

    assert names == {"snapshot_id", "symbol", "trading_state", "position_qty", "open_orders"}


# --- SymbolRiskLimits -----------------------------------------------------------------


def test_limits_may_all_be_disabled_explicitly() -> None:
    disabled = SymbolRiskLimits(max_order_qty=None, max_order_notional=None, max_position_qty=None)

    assert (disabled.max_order_qty, disabled.max_order_notional, disabled.max_position_qty) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("field", ["max_order_qty", "max_order_notional", "max_position_qty"])
def test_limit_fields_have_no_defaults(field: str) -> None:
    values = {"max_order_qty": None, "max_order_notional": None, "max_position_qty": None}
    del values[field]

    with pytest.raises(TypeError):
        SymbolRiskLimits(**values)


@pytest.mark.parametrize("field", ["max_order_qty", "max_order_notional", "max_position_qty"])
def test_limit_accepts_positive_decimal(field: str) -> None:
    assert getattr(limits(**{field: D("0.001")}), field) == D("0.001")


@pytest.mark.parametrize("field", ["max_order_qty", "max_order_notional", "max_position_qty"])
@pytest.mark.parametrize(
    "value",
    [D("0"), D("-1"), *[v for v in BAD_DECIMALS if v is not None], SubDecimal("1")],
)
def test_limit_rejects_invalid(field: str, value: object) -> None:
    with pytest.raises(DomainValidationError, match=field):
        limits(**{field: value})


# --- RiskPolicy -----------------------------------------------------------------------


def test_policy_with_no_symbols_is_valid() -> None:
    # A valid policy that allows nothing: every symbol later fails closed with
    # NO_RISK_LIMITS_FOR_SYMBOL.
    empty = policy(symbols={})

    assert dict(empty.symbols) == {}


def test_policy_symbols_are_a_defensive_read_only_copy() -> None:
    source = {"BTCUSDT": limits()}
    p = policy(symbols=source)
    source["ETHUSDT"] = limits()
    del source["BTCUSDT"]

    assert set(p.symbols) == {"BTCUSDT"}
    assert isinstance(p.symbols, MappingProxyType)
    with pytest.raises(TypeError):
        p.symbols["ETHUSDT"] = limits()  # type: ignore[index]


@pytest.mark.parametrize("value", [None, 1, 50])
def test_policy_max_open_orders(value: int | None) -> None:
    assert policy(max_open_orders=value).max_open_orders == value


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, D("1"), "1"])
def test_policy_max_open_orders_invalid(value: object) -> None:
    with pytest.raises(DomainValidationError, match="max_open_orders"):
        policy(max_open_orders=value)


@pytest.mark.parametrize("symbols", [{"": limits()}, {" BTC": limits()}, {1: limits()}])
def test_policy_symbol_keys_validated(symbols: dict[object, object]) -> None:
    with pytest.raises(DomainValidationError, match="symbol"):
        policy(symbols=symbols)


@pytest.mark.parametrize("value", [None, {"max_order_qty": D("1")}, "limits"])
def test_policy_symbol_values_must_be_limits(value: object) -> None:
    with pytest.raises(DomainValidationError, match="SymbolRiskLimits"):
        policy(symbols={"BTCUSDT": value})


@pytest.mark.parametrize("symbols", [None, [("BTCUSDT", limits())], "BTCUSDT"])
def test_policy_symbols_must_be_a_mapping(symbols: object) -> None:
    with pytest.raises(DomainValidationError, match="symbols"):
        policy(symbols=symbols)


@pytest.mark.parametrize("value", ["", " p", None])
def test_policy_id_validated(value: object) -> None:
    with pytest.raises(DomainValidationError, match="policy_id"):
        policy(policy_id=value)


def test_policy_has_no_equity_fields() -> None:
    assert {f.name for f in dataclasses.fields(RiskPolicy)} == {
        "policy_id",
        "max_open_orders",
        "symbols",
    }


# --- ExposureChange -------------------------------------------------------------------


def test_exposure_valid_values() -> None:
    zeros = exposure(
        reducing_qty=D("0"), increasing_qty=D("0"), worst_long_qty=D("0"), worst_short_qty=D("0")
    )
    reversal = exposure(
        reducing_qty=D("5"), increasing_qty=D("3"), worst_long_qty=D("0"), worst_short_qty=D("3")
    )

    assert zeros.increasing_qty == 0
    assert reversal.worst_short_qty == D("3")


@pytest.mark.parametrize(
    "field", ["reducing_qty", "increasing_qty", "worst_long_qty", "worst_short_qty"]
)
@pytest.mark.parametrize("value", [D("-1"), D("-0.001"), *BAD_DECIMALS, SubDecimal("1")])
def test_exposure_invalid(field: str, value: object) -> None:
    with pytest.raises(DomainValidationError, match=field):
        exposure(**{field: value})


# --- RiskDecision ---------------------------------------------------------------------


def test_approved_decision_has_no_reasons() -> None:
    d = decision()

    assert d.approved is True
    assert d.reasons == ()


def test_rejected_decision_has_reasons() -> None:
    d = decision(approved=False, reasons=(RiskReason.MAX_ORDER_QTY, RiskReason.MAX_OPEN_ORDERS))

    assert d.reasons == (RiskReason.MAX_ORDER_QTY, RiskReason.MAX_OPEN_ORDERS)


def test_approved_with_reasons_is_invalid() -> None:
    with pytest.raises(DomainValidationError, match="approved"):
        decision(approved=True, reasons=(RiskReason.MAX_ORDER_QTY,))


def test_rejected_without_reasons_is_invalid() -> None:
    with pytest.raises(DomainValidationError, match="approved"):
        decision(approved=False, reasons=())


def test_duplicate_reasons_rejected() -> None:
    with pytest.raises(DomainValidationError, match="duplicate"):
        decision(approved=False, reasons=(RiskReason.MAX_ORDER_QTY, RiskReason.MAX_ORDER_QTY))


@pytest.mark.parametrize(
    "reasons", [[RiskReason.MAX_ORDER_QTY], ("max_order_qty",), (None,), "max_order_qty"]
)
def test_reasons_must_be_a_tuple_of_reason_members(reasons: object) -> None:
    with pytest.raises(DomainValidationError, match="reasons"):
        decision(approved=False, reasons=reasons)


def test_rejection_before_decomposition_has_no_exposure() -> None:
    d = decision(approved=False, reasons=(RiskReason.UNKNOWN_POSITION,), exposure=None)

    assert d.exposure is None


def test_approval_requires_exposure() -> None:
    with pytest.raises(DomainValidationError, match="exposure"):
        decision(exposure=None)


def test_exposure_type_checked() -> None:
    with pytest.raises(DomainValidationError, match="exposure"):
        decision(exposure={"reducing_qty": D("0")})


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_approved_must_be_bool(value: object) -> None:
    with pytest.raises(DomainValidationError, match="approved"):
        decision(approved=value)


@pytest.mark.parametrize("field", ["intent_id", "snapshot_id", "policy_id"])
@pytest.mark.parametrize("value", ["", " x", None, 1])
def test_decision_ids_validated(field: str, value: object) -> None:
    with pytest.raises(DomainValidationError, match=field):
        decision(**{field: value})


def test_models_are_frozen_slotted_keyword_only() -> None:
    for instance in (order(), snapshot(), limits(), policy(), exposure(), decision()):
        assert dataclasses.is_dataclass(instance)
        assert not hasattr(instance, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, dataclasses.fields(instance)[0].name, "x")
    with pytest.raises(TypeError):
        RiskDecision("i", "s", "p", True, (), exposure())  # type: ignore[call-arg]


# --- decimal context ------------------------------------------------------------------


def test_models_keep_exact_values_under_a_low_precision_context() -> None:
    def build() -> tuple[object, ...]:
        return (
            order(remaining_qty=D("3.333"), price=D("101.37")),
            snapshot(position_qty=D("-7.777"), open_orders=(order(remaining_qty=D("5.555")),)),
            limits(max_order_qty=D("3.333"), max_order_notional=D("123456.789")),
            policy(symbols={"BTCUSDT": limits(max_position_qty=D("7.777"))}),
            exposure(reducing_qty=D("3.333"), increasing_qty=D("4.444"), worst_long_qty=D("7.777")),
        )

    before = (getcontext().prec, getcontext().rounding, dict(getcontext().traps))
    baseline = build()
    with low_precision():
        low = build()
    after = (getcontext().prec, getcontext().rounding, dict(getcontext().traps))

    assert low == baseline
    assert before == after
    built_order = low[0]
    assert isinstance(built_order, OpenOrderExposure)
    assert str(built_order.remaining_qty) == "3.333"


def test_models_module_has_no_forbidden_dependencies() -> None:
    source = Path(models.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith("app.domain"), name
    for banned in ("float(", "getcontext", "setcontext", "localcontext", "abs(", "quantize"):
        assert banned not in source, banned
