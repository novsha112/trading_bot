"""Risk V1 evaluator: pure approve / reject decisions."""

from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.risk import exposure as exposure_module
from app.risk import manager
from app.risk.exposure import calculate_worst_case
from app.risk.manager import evaluate
from app.risk.models import (
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
R = RiskReason
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
NO_LIMITS: dict[str, Any] = {
    "max_order_qty": None,
    "max_order_notional": None,
    "max_position_qty": None,
}


def intent(
    side: Side,
    qty: str,
    *,
    price: str | None = "100",
    reduce_only: bool = False,
    symbol: str = "BTCUSDT",
) -> PlaceOrderIntent:
    market = price is None
    return PlaceOrderIntent(
        intent_id="i-1",
        strategy_id="grid",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET if market else OrderType.LIMIT,
        price=None if price is None else D(price),
        qty=D(qty),
        time_in_force=TimeInForce.IOC if market else TimeInForce.GTC,
        reduce_only=reduce_only,
        tag=None,
        created_at=T0,
    )


def pending(
    side: Side, qty: str, *, reduce_only: bool = False, status: OrderStatus = OrderStatus.OPEN
) -> OpenOrderExposure:
    return OpenOrderExposure(
        side=side, remaining_qty=D(qty), price=D("100"), reduce_only=reduce_only, status=status
    )


def snapshot(
    position: str | None = "0",
    orders: tuple[OpenOrderExposure, ...] | None = (),
    *,
    count: int | None = None,
    state: TradingState = TradingState.RUNNING,
) -> RiskSnapshot:
    if count is None and orders is not None:
        count = len(orders)
    return RiskSnapshot(
        snapshot_id="snap-1",
        symbol="BTCUSDT",
        trading_state=state,
        position_qty=None if position is None else D(position),
        open_orders=orders,
        account_open_order_count=count,
    )


def unknown_count_snapshot(**kwargs: Any) -> RiskSnapshot:
    s = snapshot(**kwargs)
    return RiskSnapshot(
        snapshot_id=s.snapshot_id,
        symbol=s.symbol,
        trading_state=s.trading_state,
        position_qty=s.position_qty,
        open_orders=s.open_orders,
        account_open_order_count=None,
    )


def policy(*, max_open_orders: int | None = None, **limits: Any) -> RiskPolicy:
    values = {**NO_LIMITS, **{k: None if v is None else D(v) for k, v in limits.items()}}
    return RiskPolicy(
        policy_id="policy-1",
        max_open_orders=max_open_orders,
        symbols={"BTCUSDT": SymbolRiskLimits(**values)},
    )


def run(new: PlaceOrderIntent, snap: RiskSnapshot, pol: RiskPolicy | None = None) -> RiskDecision:
    return evaluate(intent=new, snapshot=snap, policy=pol or policy())


def approved(d: RiskDecision) -> None:
    assert d.approved, d.reasons
    assert d.reasons == ()
    assert isinstance(d.exposure, ExposureChange)


def rejected_early(d: RiskDecision, reason: RiskReason) -> None:
    assert (d.approved, d.reasons, d.exposure) == (False, (reason,), None)


@contextmanager
def low_precision() -> Iterator[None]:
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        yield


# --- contract ----------------------------------------------------------------------


def test_decision_carries_audit_identifiers() -> None:
    d = run(intent(Side.BUY, "1"), snapshot())

    approved(d)
    assert (d.intent_id, d.snapshot_id, d.policy_id) == ("i-1", "snap-1", "policy-1")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intent": "buy"},
        {"snapshot": None},
        {"policy": {"BTCUSDT": None}},
    ],
)
def test_wrong_input_types_are_programmer_errors(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "intent": intent(Side.BUY, "1"),
        "snapshot": snapshot(),
        "policy": policy(),
        **kwargs,
    }
    with pytest.raises(DomainValidationError):
        evaluate(**values)


def test_symbol_mismatch_is_a_programmer_error() -> None:
    with pytest.raises(DomainValidationError, match="symbol"):
        run(intent(Side.BUY, "1", symbol="ETHUSDT"), snapshot())


def test_symbol_without_limits_is_rejected_early() -> None:
    empty = RiskPolicy(policy_id="p", max_open_orders=None, symbols={})

    rejected_early(run(intent(Side.BUY, "1"), snapshot(), empty), R.NO_RISK_LIMITS_FOR_SYMBOL)


# --- trading state -------------------------------------------------------------------


@pytest.mark.parametrize("reduce_only", [False, True])
def test_halted_rejects_every_placement(reduce_only: bool) -> None:
    d = run(
        intent(Side.SELL, "1", reduce_only=reduce_only), snapshot("5", state=TradingState.HALTED)
    )

    rejected_early(d, R.KILL_SWITCH_ACTIVE)


@pytest.mark.parametrize("reduce_only", [False, True])
def test_paused_rejects_every_placement(reduce_only: bool) -> None:
    d = run(
        intent(Side.SELL, "1", reduce_only=reduce_only), snapshot("5", state=TradingState.PAUSED)
    )

    rejected_early(d, R.TRADING_PAUSED)


def test_reduce_only_state_rejects_ordinary_intents_even_if_closing() -> None:
    d = run(intent(Side.SELL, "1"), snapshot("5", state=TradingState.REDUCE_ONLY))

    rejected_early(d, R.REDUCE_ONLY_STATE)


def test_reduce_only_state_allows_valid_reduce_only() -> None:
    d = run(intent(Side.SELL, "1", reduce_only=True), snapshot("5", state=TradingState.REDUCE_ONLY))

    approved(d)


def test_running_allows_ordinary_intents() -> None:
    approved(run(intent(Side.BUY, "1"), snapshot()))


def test_state_gates_do_not_need_position_or_orders() -> None:
    d = run(intent(Side.BUY, "1"), snapshot(None, None, state=TradingState.HALTED))

    rejected_early(d, R.KILL_SWITCH_ACTIVE)


# --- reduce-only ----------------------------------------------------------------------


def test_reduce_only_against_flat() -> None:
    rejected_early(
        run(intent(Side.SELL, "1", reduce_only=True), snapshot("0")),
        R.REDUCE_ONLY_WITHOUT_POSITION,
    )


@pytest.mark.parametrize(("position", "side"), [("5", Side.BUY), ("-5", Side.SELL)])
def test_reduce_only_wrong_side(position: str, side: Side) -> None:
    rejected_early(
        run(intent(side, "1", reduce_only=True), snapshot(position)), R.REDUCE_ONLY_WRONG_SIDE
    )


@pytest.mark.parametrize(
    ("position", "side", "qty", "reducing"),
    [
        ("5", Side.SELL, "2", "2"),
        ("-5", Side.BUY, "2", "2"),
        ("5", Side.SELL, "100", "5"),  # oversized: execution caps the fill
    ],
)
def test_valid_reduce_only_is_approved(position: str, side: Side, qty: str, reducing: str) -> None:
    d = run(
        intent(side, qty, reduce_only=True),
        snapshot(position),
        policy(max_order_qty="1", max_order_notional="1", max_position_qty="1"),
    )

    approved(d)
    assert d.exposure is not None
    assert (d.exposure.reducing_qty, d.exposure.increasing_qty) == (D(reducing), D("0"))


def test_valid_reduce_only_ignores_existing_excess_position() -> None:
    d = run(
        intent(Side.SELL, "1", reduce_only=True),
        snapshot("50", (pending(Side.BUY, "10"),)),
        policy(max_position_qty="10"),
    )

    approved(d)


def test_valid_reduce_only_without_price_needs_no_reference() -> None:
    d = run(
        intent(Side.SELL, "1", price=None, reduce_only=True),
        snapshot("5"),
        policy(max_order_notional="10"),
    )

    approved(d)


def test_valid_reduce_only_still_obeys_max_open_orders() -> None:
    d = run(
        intent(Side.SELL, "1", reduce_only=True),
        snapshot("5", count=3),
        policy(max_open_orders=3),
    )

    assert (d.approved, d.reasons) == (False, (R.MAX_OPEN_ORDERS,))
    assert d.exposure is not None


# --- unknown state ---------------------------------------------------------------------


def test_unknown_position() -> None:
    rejected_early(run(intent(Side.BUY, "1"), snapshot(None)), R.UNKNOWN_POSITION)


@pytest.mark.parametrize("reduce_only", [False, True])
def test_unknown_symbol_orders(reduce_only: bool) -> None:
    # Decision A: even a valid reduce-only fails closed (no honest worst case).
    d = run(
        intent(Side.SELL, "1", reduce_only=reduce_only),
        snapshot("5", None, count=0),
    )

    rejected_early(d, R.UNKNOWN_OPEN_ORDERS)


def test_unknown_account_count_with_limit_enabled() -> None:
    d = run(intent(Side.BUY, "1"), unknown_count_snapshot(), policy(max_open_orders=5))

    assert (d.approved, d.reasons) == (False, (R.UNKNOWN_OPEN_ORDERS,))
    assert d.exposure is not None


def test_unknown_account_count_with_limit_disabled() -> None:
    approved(run(intent(Side.BUY, "1"), unknown_count_snapshot(), policy(max_open_orders=None)))


# --- order qty -------------------------------------------------------------------------


@pytest.mark.parametrize(("qty", "ok"), [("9.999", True), ("10", True), ("10.001", False)])
def test_max_order_qty(qty: str, ok: bool) -> None:
    d = run(intent(Side.BUY, qty), snapshot(), policy(max_order_qty="10"))

    assert d.reasons == (() if ok else (R.MAX_ORDER_QTY,))


def test_max_order_qty_applies_to_the_whole_ordinary_closing_order() -> None:
    d = run(intent(Side.SELL, "100"), snapshot("5"), policy(max_order_qty="10"))

    assert d.reasons == (R.MAX_ORDER_QTY,)


# --- order notional ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qty", "price", "ok"),
    [("9", "100", True), ("10", "100", True), ("10.001", "100", False), ("3.333", "300.03", True)],
)
def test_max_order_notional(qty: str, price: str, ok: bool) -> None:
    d = run(intent(Side.BUY, qty, price=price), snapshot(), policy(max_order_notional="1000"))

    assert d.reasons == (() if ok else (R.MAX_ORDER_NOTIONAL,))


def test_awkward_notional_boundary_is_exact() -> None:
    # 3.333 * 300.03 = 999.99999 exactly: just below 1000, no rounding up.
    limit = policy(max_order_notional="999.99999")
    assert run(intent(Side.BUY, "3.333", price="300.03"), snapshot(), limit).approved
    tighter = policy(max_order_notional="999.99998")
    d = run(intent(Side.BUY, "3.333", price="300.03"), snapshot(), tighter)
    assert d.reasons == (R.MAX_ORDER_NOTIONAL,)


def test_missing_price_only_when_notional_limit_enabled() -> None:
    market = intent(Side.BUY, "1", price=None)

    assert run(market, snapshot(), policy(max_order_notional="10")).reasons == (
        R.MISSING_REFERENCE_PRICE,
    )
    approved(run(market, snapshot(), policy(max_order_notional=None)))


def test_unrepresentable_notional() -> None:
    d = run(
        intent(Side.BUY, "1." + "1" * 60, price="1." + "3" * 60),
        snapshot(),
        policy(max_order_notional="10"),
    )

    assert d.reasons == (R.UNREPRESENTABLE_CALCULATION,)
    assert d.exposure is not None


def test_unrepresentable_exposure_is_rejected_early() -> None:
    d = run(intent(Side.BUY, "1E-200"), snapshot("1E+200"))

    rejected_early(d, R.UNREPRESENTABLE_CALCULATION)


# --- position worst case -------------------------------------------------------------------


def position_reasons(
    position: str, new: PlaceOrderIntent, *orders: OpenOrderExposure
) -> tuple[RiskReason, ...]:
    return run(new, snapshot(position, tuple(orders)), policy(max_position_qty="10")).reasons


@pytest.mark.parametrize(
    ("position", "side", "qty", "orders", "rejected"),
    [
        ("0", Side.BUY, "10", (), False),  # flat open within
        ("0", Side.BUY, "10.001", (), True),  # flat open over
        ("0", Side.SELL, "10.001", (), True),
        ("8", Side.BUY, "2", (), False),  # long increase to the limit
        ("8", Side.BUY, "3", (), True),  # long increase over
        ("-8", Side.SELL, "3", (), True),  # short increase over
        ("5", Side.SELL, "15", (), False),  # reversal to short 10
        ("5", Side.SELL, "16", (), True),  # reversal to short 11
        ("0", Side.BUY, "4", (("buy", "3"), ("buy", "3")), False),  # 10
        ("0", Side.BUY, "4", (("buy", "3"), ("buy", "3.001")), True),  # multiple pending
        ("0", Side.BUY, "4", (("sell", "50"), ("buy", "6")), False),  # opposite not netted
        ("0", Side.SELL, "4", (("buy", "50"), ("sell", "7")), True),  # 11 short
    ],
)
def test_max_position_qty(
    position: str, side: Side, qty: str, orders: tuple[tuple[str, str], ...], rejected: bool
) -> None:
    built = [pending(Side(s), q) for s, q in orders]

    reasons = position_reasons(position, intent(side, qty), *built)

    assert reasons == ((R.MAX_POSITION_QTY,) if rejected else ())


def test_unknown_pending_counts_in_full() -> None:
    reasons = position_reasons(
        "0", intent(Side.BUY, "4"), pending(Side.BUY, "7", status=OrderStatus.UNKNOWN)
    )

    assert reasons == (R.MAX_POSITION_QTY,)


def test_reduce_only_pending_is_ignored() -> None:
    reasons = position_reasons(
        "0", intent(Side.BUY, "4"), pending(Side.BUY, "50", reduce_only=True)
    )

    assert reasons == ()


@pytest.mark.parametrize(
    ("position", "orders", "new", "rejected"),
    [
        # long: before 12 > 10
        ("12", (), intent(Side.SELL, "1"), False),  # after long == before
        ("12", (), intent(Side.SELL, "11"), False),  # long unchanged, short path 1 -> 0
        ("6", (("buy", "6"),), intent(Side.SELL, "3"), False),  # long 12 unchanged, short 0
        ("12", (), intent(Side.BUY, "1"), True),  # after 13 > before 12
        ("9", (), intent(Side.BUY, "2"), True),  # before 9 <= 10, after 11
        # short: before 12 > 10
        ("-12", (), intent(Side.BUY, "1"), False),
        ("-6", (("sell", "6"),), intent(Side.BUY, "3"), False),
        ("-12", (), intent(Side.SELL, "1"), True),
        ("-9", (), intent(Side.SELL, "2"), True),
    ],
)
def test_existing_excess_is_not_blamed_on_a_non_worsening_intent(
    position: str, orders: tuple[tuple[str, str], ...], new: PlaceOrderIntent, rejected: bool
) -> None:
    built = [pending(Side(s), q) for s, q in orders]

    reasons = position_reasons(position, new, *built)

    assert reasons == ((R.MAX_POSITION_QTY,) if rejected else ())


def test_after_not_above_before_but_above_limit_is_not_rejected() -> None:
    # A candidate only ever adds to its own worst-case side (opposite pending
    # orders are never netted), so "after < before" cannot occur here: an
    # existing excess (long 12 > 10) that the intent does not worsen is allowed.
    before = calculate_worst_case(position_qty=D("12"), open_orders=())
    assert before == (D("12"), D("0"))
    reasons = position_reasons("12", intent(Side.SELL, "2"))
    assert reasons == ()


def test_ordinary_closing_order_counts_its_whole_side() -> None:
    # LONG 5, ordinary SELL 16 looks like a close but may end SHORT 11.
    assert position_reasons("5", intent(Side.SELL, "16")) == (R.MAX_POSITION_QTY,)


# --- max open orders ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "maximum", "rejected"),
    [(0, 5, False), (4, 5, False), (5, 5, True), (9, 5, True), (9, None, False)],
)
def test_max_open_orders(count: int, maximum: int | None, rejected: bool) -> None:
    d = run(intent(Side.BUY, "1"), snapshot(count=count), policy(max_open_orders=maximum))

    assert d.reasons == ((R.MAX_OPEN_ORDERS,) if rejected else ())


def test_account_count_not_symbol_count_is_used() -> None:
    d = run(
        intent(Side.BUY, "1"),
        snapshot("0", (pending(Side.BUY, "1"),), count=5),
        policy(max_open_orders=5),
    )

    assert d.reasons == (R.MAX_OPEN_ORDERS,)


# --- aggregation ------------------------------------------------------------------------------


def test_all_limit_reasons_in_enum_order_with_exposure() -> None:
    d = run(
        intent(Side.BUY, "20", price="1000"),
        snapshot("0", count=3),
        policy(
            max_open_orders=3, max_order_qty="10", max_order_notional="1000", max_position_qty="10"
        ),
    )

    assert d.approved is False
    assert d.reasons == (
        R.MAX_ORDER_QTY,
        R.MAX_ORDER_NOTIONAL,
        R.MAX_POSITION_QTY,
        R.MAX_OPEN_ORDERS,
    )
    assert d.exposure is not None
    assert d.exposure.worst_long_qty == D("20")


def test_reason_order_does_not_depend_on_check_order(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = list(manager._LIMIT_CHECKS)
    results = set()
    for permutation in itertools.permutations(checks):
        monkeypatch.setattr(manager, "_LIMIT_CHECKS", tuple(permutation))
        d = run(
            intent(Side.BUY, "20", price=None),
            unknown_count_snapshot(),
            policy(
                max_open_orders=3,
                max_order_qty="10",
                max_order_notional="1000",
                max_position_qty="10",
            ),
        )
        results.add(d.reasons)

    assert results == {
        (R.UNKNOWN_OPEN_ORDERS, R.MAX_ORDER_QTY, R.MISSING_REFERENCE_PRICE, R.MAX_POSITION_QTY)
    }


# --- decimal context and purity ---------------------------------------------------------------


def test_low_precision_context_gives_identical_decisions() -> None:
    def run_all() -> list[RiskDecision]:
        orders = (
            pending(Side.BUY, "3.333"),
            pending(Side.SELL, "7.777", status=OrderStatus.UNKNOWN),
        )
        pol = policy(
            max_open_orders=10,
            max_order_qty="5.555",
            max_order_notional="1666.665",
            max_position_qty="12.221",
        )
        return [
            run(intent(Side.BUY, "5.555", price="300.03"), snapshot("3.333", orders), pol),
            run(intent(Side.SELL, "5.555", price="300"), snapshot("3.333", orders), pol),
            run(intent(Side.SELL, "7.777", reduce_only=True), snapshot("3.333", orders), pol),
            run(intent(Side.BUY, "5.556"), snapshot("-3.333", orders), pol),
        ]

    baseline = run_all()
    with low_precision():
        low = run_all()

    assert low == baseline
    first = baseline[0]
    assert first.exposure is not None
    assert first.exposure.worst_long_qty == D("12.221")  # 3.333 + 3.333 + 5.555
    assert first.reasons == (R.MAX_ORDER_NOTIONAL,)  # 5.555 * 300.03 = 1666.66665


def test_global_context_is_untouched() -> None:
    def state() -> tuple[object, ...]:
        c = getcontext()
        return c.prec, c.rounding, dict(c.traps), dict(c.flags), c.Emin, c.Emax

    getcontext().clear_flags()
    before = state()
    run(
        intent(Side.BUY, "3.333", price="300.03"), snapshot("1.111"), policy(max_order_notional="1")
    )
    run(intent(Side.BUY, "1E-200"), snapshot("1E+200"))

    assert state() == before


def test_evaluation_is_pure_and_repeatable() -> None:
    new = intent(Side.SELL, "7.777")
    snap = snapshot("3.333", (pending(Side.BUY, "1.111"),))
    pol = policy(max_open_orders=5, max_order_qty="10", max_position_qty="10")
    before = (new, snap, pol, dict(pol.symbols), snap.open_orders)

    first = run(new, snap, pol)
    second = run(new, snap, pol)

    assert first == second
    assert (new, snap, pol, dict(pol.symbols), snap.open_orders) == before


def test_manager_dependencies() -> None:
    source = Path(manager.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    allowed = ("app.domain", "app.risk.models", "app.risk.exposure")
    for name in imports:
        assert not name.startswith("app.") or name.startswith(allowed), name
    assert not {"logging", "structlog", "app.domain.clock", "time", "datetime"} & imports
    for banned in ("float(", "getcontext", "setcontext", "localcontext"):
        assert banned not in source, banned
    assert exposure_module.calculate_worst_case is calculate_worst_case
