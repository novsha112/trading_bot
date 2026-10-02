"""Mapping of local domain orders into Risk V1 snapshots (pure boundary)."""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.order_state import TERMINAL_STATUSES
from app.domain.orders import Order
from app.risk import snapshots as snapshots_module
from app.risk.exposure import ExposureCalculationError, calculate_remaining_qty
from app.risk.models import ACTIVE_ORDER_STATUSES, OpenOrderExposure, RiskSnapshot, TradingState
from app.risk.snapshots import build_risk_snapshot, open_order_exposure

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
PRICE = D("100.5")
FLAT = D("0")

# A domain-valid cumulative fill (qty = 1.0) for every status.
FILL_FOR_STATUS = {
    S.NEW: "0",
    S.SUBMITTING: "0",
    S.OPEN: "0",
    S.PARTIALLY_FILLED: "0.3",
    S.CANCELING: "0.3",
    S.UNKNOWN: "0.3",
    S.FILLED: "1.0",
    S.CANCELED: "0.3",
    S.EXPIRED: "0",
    S.REJECTED: "0",
    S.FAILED: "0",
}


def order(
    status: OrderStatus = S.OPEN,
    filled_qty: str | None = None,
    **overrides: Any,
) -> Order:
    filled = D(FILL_FOR_STATUS[status] if filled_qty is None else filled_qty)
    values: dict[str, Any] = {
        "client_order_id": "grid1-buy-0001",
        "exchange_order_id": None,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": PRICE,
        "qty": D("1.0"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "status": status,
        "filled_qty": filled,
        "avg_fill_price": PRICE if filled > 0 else None,
        "created_at": T0,
        "updated_at": T0,
        "last_exchange_update_ts": None,
        "version": 0,
    }
    return Order(**{**values, **overrides})


def snapshot(
    orders: tuple[Order, ...] | None = (),
    *,
    account_open_order_count: int | None = 0,
    position_qty: Decimal | None = FLAT,
) -> RiskSnapshot:
    return build_risk_snapshot(
        snapshot_id="acct:7",
        symbol="BTCUSDT",
        trading_state=TradingState.RUNNING,
        position_qty=position_qty,
        orders=orders,
        account_open_order_count=account_open_order_count,
    )


def context_state() -> tuple[object, ...]:
    context = getcontext()
    return (
        context.prec,
        context.rounding,
        dict(context.traps),
        dict(context.flags),
        context.Emin,
        context.Emax,
    )


# --- open_order_exposure: statuses ----------------------------------------------------------


def test_status_partition_matches_the_domain() -> None:
    assert set(OrderStatus) == ACTIVE_ORDER_STATUSES | TERMINAL_STATUSES
    assert not ACTIVE_ORDER_STATUSES & TERMINAL_STATUSES


@pytest.mark.parametrize("status", sorted(ACTIVE_ORDER_STATUSES))
def test_every_active_status_is_mapped(status: OrderStatus) -> None:
    source = order(status)

    exposure = open_order_exposure(source)

    assert exposure == OpenOrderExposure(
        side=Side.BUY,
        remaining_qty=D("1.0") - D(FILL_FOR_STATUS[status]),
        price=PRICE,
        reduce_only=False,
        status=status,
    )


def test_all_six_active_statuses_are_covered() -> None:
    assert {
        S.NEW,
        S.SUBMITTING,
        S.OPEN,
        S.PARTIALLY_FILLED,
        S.CANCELING,
        S.UNKNOWN,
    } == ACTIVE_ORDER_STATUSES


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_terminal_status_is_rejected(status: OrderStatus) -> None:
    with pytest.raises(DomainValidationError, match="not an active order status"):
        open_order_exposure(order(status))


def test_domain_does_not_allow_an_active_order_without_remainder() -> None:
    with pytest.raises(DomainValidationError, match="unknown: must be < qty"):
        order(S.UNKNOWN, filled_qty="1.0")


@pytest.mark.parametrize(
    ("status", "filled_qty", "remaining"),
    [
        (S.NEW, "0", "1.0"),
        (S.SUBMITTING, "0", "1.0"),
        (S.OPEN, "0", "1.0"),
        (S.PARTIALLY_FILLED, "0.3", "0.7"),
        (S.PARTIALLY_FILLED, "0.999", "0.001"),
        (S.CANCELING, "0", "1.0"),
        (S.CANCELING, "0.999", "0.001"),
        (S.UNKNOWN, "0", "1.0"),
        (S.UNKNOWN, "0.999", "0.001"),
    ],
)
def test_every_valid_active_order_maps_to_a_positive_remainder(
    status: OrderStatus, filled_qty: str, remaining: str
) -> None:
    exposure = open_order_exposure(order(status, filled_qty=filled_qty))

    assert exposure.remaining_qty == D(remaining)
    assert exposure.remaining_qty > 0


def test_defensive_remaining_check_is_kept() -> None:
    # Bypass the frozen domain invariant on purpose: the mapping must still refuse it.
    source = order(S.UNKNOWN, filled_qty="0.3")
    object.__setattr__(source, "filled_qty", source.qty)

    with pytest.raises(DomainValidationError, match="remaining_qty"):
        open_order_exposure(source)


@pytest.mark.parametrize("value", [None, "order", object()])
def test_non_order_is_rejected(value: object) -> None:
    with pytest.raises(DomainValidationError, match="Order"):
        open_order_exposure(value)  # type: ignore[arg-type]


def test_order_subclass_is_rejected() -> None:
    class LocalOrder(Order):
        __slots__ = ()

    source = order()
    subclass = LocalOrder(**{name: getattr(source, name) for name in Order.__dataclass_fields__})

    with pytest.raises(DomainValidationError, match="Order"):
        open_order_exposure(subclass)


# --- open_order_exposure: fields ------------------------------------------------------------


def test_partial_remaining_is_exact() -> None:
    source = order(
        S.PARTIALLY_FILLED,
        qty=D("0.30000000000000000000000000000000000000000001"),
        filled_qty="0.1",
    )

    exposure = open_order_exposure(source)

    assert exposure.remaining_qty == D("0.20000000000000000000000000000000000000000001")
    assert str(exposure.remaining_qty) == "0.20000000000000000000000000000000000000000001"


def test_market_order_price_none_is_preserved() -> None:
    source = order(order_type=OrderType.MARKET, price=None, time_in_force=TimeInForce.IOC)

    exposure = open_order_exposure(source)

    assert exposure.price is None


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
@pytest.mark.parametrize("reduce_only", [True, False])
def test_side_and_reduce_only_are_preserved(side: Side, reduce_only: bool) -> None:
    exposure = open_order_exposure(order(side=side, reduce_only=reduce_only))

    assert exposure.side is side
    assert exposure.reduce_only is reduce_only


def test_exchange_metadata_is_not_used() -> None:
    plain = open_order_exposure(order(S.OPEN))
    acked = open_order_exposure(
        order(S.OPEN, exchange_order_id="ex-1", last_exchange_update_ts=T0, version=5)
    )

    assert plain == acked


# --- Decimal context ------------------------------------------------------------------------


def test_awkward_remaining_is_exact_under_a_low_precision_context() -> None:
    source = order(S.PARTIALLY_FILLED, qty=D("3.4"), filled_qty="0.067")

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        exposure = open_order_exposure(source)

    assert exposure.remaining_qty == D("3.333")


def test_snapshot_under_a_low_precision_context_is_exact() -> None:
    source = order(S.CANCELING, qty=D("7.777"), filled_qty="1.111", client_order_id="c-1")

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        result = snapshot((source,), account_open_order_count=1, position_qty=D("-3.333"))

    assert result.open_orders is not None
    assert result.open_orders[0].remaining_qty == D("6.666")
    assert result.position_qty == D("-3.333")


def test_global_decimal_context_is_unchanged() -> None:
    getcontext().clear_flags()
    before = context_state()

    open_order_exposure(order(S.PARTIALLY_FILLED, qty=D("3.4"), filled_qty="0.067"))
    snapshot((order(S.UNKNOWN),), account_open_order_count=3)

    assert context_state() == before


def test_unrepresentable_remaining_raises_calculation_error() -> None:
    source = order(S.PARTIALLY_FILLED, qty=D("1E+200"), filled_qty="1E-200")

    with pytest.raises(ExposureCalculationError):
        open_order_exposure(source)


@pytest.mark.parametrize(
    ("qty", "filled", "remaining"),
    [("1.0", "0", "1.0"), ("5", "4.999", "0.001"), ("2.50", "0.25", "2.25")],
)
def test_calculate_remaining_qty(qty: str, filled: str, remaining: str) -> None:
    result = calculate_remaining_qty(qty=D(qty), filled_qty=D(filled))

    assert result == D(remaining)
    assert type(result) is Decimal


@pytest.mark.parametrize(
    ("qty", "filled"),
    [
        (D("0"), D("0")),
        (D("1"), D("-0.1")),
        (D("1"), D("1.1")),
        (D("NaN"), D("0")),
        (D("1"), D("Infinity")),
        (1, D("0")),
        (D("1"), 0.5),
    ],
)
def test_calculate_remaining_qty_rejects_invalid_inputs(qty: object, filled: object) -> None:
    with pytest.raises(DomainValidationError):
        calculate_remaining_qty(qty=qty, filled_qty=filled)  # type: ignore[arg-type]


# --- build_risk_snapshot --------------------------------------------------------------------


def test_known_empty_orders() -> None:
    result = snapshot((), account_open_order_count=0)

    assert result == RiskSnapshot(
        snapshot_id="acct:7",
        symbol="BTCUSDT",
        trading_state=TradingState.RUNNING,
        position_qty=D("0"),
        open_orders=(),
        account_open_order_count=0,
    )


def test_unknown_orders_stay_unknown() -> None:
    result = snapshot(None, account_open_order_count=None, position_qty=None)

    assert result.open_orders is None
    assert result.account_open_order_count is None
    assert result.position_qty is None


def test_unknown_symbol_orders_with_known_account_count() -> None:
    result = snapshot(None, account_open_order_count=4)

    assert result.open_orders is None
    assert result.account_open_order_count == 4


def test_orders_are_mapped_in_input_order() -> None:
    first = order(S.OPEN, client_order_id="c-1", side=Side.SELL, price=D("101"))
    second = order(S.NEW, client_order_id="c-2", qty=D("2"))
    third = order(S.PARTIALLY_FILLED, client_order_id="c-3", reduce_only=True)

    result = snapshot((first, second, third), account_open_order_count=3)

    assert result.open_orders == tuple(open_order_exposure(o) for o in (first, second, third))
    assert result.open_orders is not None
    assert [o.status for o in result.open_orders] == [S.OPEN, S.NEW, S.PARTIALLY_FILLED]


def test_snapshot_fields_are_passed_through() -> None:
    result = build_risk_snapshot(
        snapshot_id="acct:42",
        symbol="BTCUSDT",
        trading_state=TradingState.REDUCE_ONLY,
        position_qty=D("-1.5"),
        orders=(),
        account_open_order_count=9,
    )

    assert (result.snapshot_id, result.trading_state, result.position_qty) == (
        "acct:42",
        TradingState.REDUCE_ONLY,
        D("-1.5"),
    )


def test_account_count_is_not_derived_from_symbol_orders() -> None:
    orders = (order(client_order_id="c-1"), order(client_order_id="c-2"))

    result = snapshot(orders, account_open_order_count=17)

    assert result.account_open_order_count == 17
    assert result.open_orders is not None
    assert len(result.open_orders) == 2


def test_unknown_account_count_is_not_filled_in_from_orders() -> None:
    result = snapshot((order(),), account_open_order_count=None)

    assert result.account_open_order_count is None


def test_account_count_below_symbol_orders_is_rejected_by_the_snapshot() -> None:
    orders = (order(client_order_id="c-1"), order(client_order_id="c-2"))

    with pytest.raises(DomainValidationError, match="account_open_order_count"):
        snapshot(orders, account_open_order_count=1)


def test_foreign_symbol_order_is_rejected() -> None:
    foreign = order(symbol="ETHUSDT")

    with pytest.raises(DomainValidationError, match="ETHUSDT"):
        snapshot((order(client_order_id="c-1"), foreign), account_open_order_count=5)


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_terminal_order_in_view_is_rejected(status: OrderStatus) -> None:
    with pytest.raises(DomainValidationError, match="not an active order status"):
        snapshot((order(status),), account_open_order_count=5)


def test_duplicate_client_order_id_is_rejected() -> None:
    # The same local order twice would double its exposure.
    duplicated = (order(client_order_id="c-1"), order(S.NEW, client_order_id="c-1"))

    with pytest.raises(DomainValidationError, match="c-1"):
        snapshot(duplicated, account_open_order_count=5)


@pytest.mark.parametrize(
    "orders",
    [[], (open_order_exposure(order()),), ("order",), (None,)],
)
def test_invalid_orders_container_is_rejected(orders: object) -> None:
    with pytest.raises(DomainValidationError, match="orders"):
        snapshot(orders, account_open_order_count=5)  # type: ignore[arg-type]


def test_order_subclass_in_view_is_rejected() -> None:
    class LocalOrder(Order):
        __slots__ = ()

    source = order()
    subclass = LocalOrder(**{name: getattr(source, name) for name in Order.__dataclass_fields__})

    with pytest.raises(DomainValidationError, match="Order"):
        snapshot((subclass,), account_open_order_count=5)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"snapshot_id": ""}, "snapshot_id"),
        ({"symbol": " BTCUSDT"}, "symbol"),
        ({"trading_state": "running"}, "trading_state"),
        ({"position_qty": 1}, "position_qty"),
        ({"position_qty": D("NaN")}, "position_qty"),
        ({"account_open_order_count": -1}, "account_open_order_count"),
        ({"account_open_order_count": True}, "account_open_order_count"),
    ],
)
def test_invalid_scalar_inputs_are_rejected_by_the_snapshot(
    overrides: dict[str, Any], match: str
) -> None:
    values: dict[str, Any] = {
        "snapshot_id": "acct:7",
        "symbol": "BTCUSDT",
        "trading_state": TradingState.RUNNING,
        "position_qty": D("0"),
        "orders": (),
        "account_open_order_count": 0,
    }

    with pytest.raises(DomainValidationError, match=match):
        build_risk_snapshot(**{**values, **overrides})


# --- purity ---------------------------------------------------------------------------------


def test_mapping_is_deterministic_and_does_not_mutate_inputs() -> None:
    orders = (
        order(S.PARTIALLY_FILLED, client_order_id="c-1", qty=D("3.4"), filled_qty="0.067"),
        order(S.UNKNOWN, client_order_id="c-2", side=Side.SELL),
    )
    copies = tuple(replace(o) for o in orders)

    first = snapshot(orders, account_open_order_count=2)
    second = snapshot(orders, account_open_order_count=2)

    assert first == second
    assert orders == copies
    assert [o.filled_qty for o in orders] == [D("0.067"), D("0.3")]


def test_snapshot_module_has_no_side_effect_dependencies() -> None:
    source = Path(snapshots_module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith(("app.domain", "app.risk")), name
    banned_modules = {"asyncio", "logging", "time", "random", "uuid", "app.domain.clock"}
    assert not imports & banned_modules
    for banned in ("float(", "getcontext", "setcontext", "localcontext", "datetime"):
        assert banned not in source, banned
    assert not re.search(r"(?<!copy_)\babs\(", source)
