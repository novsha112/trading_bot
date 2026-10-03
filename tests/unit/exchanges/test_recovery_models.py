"""Recovery read DTOs: strict validation, exact values kept as given."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone
from decimal import Context, Decimal, Inexact, localcontext
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.order_state import EXCHANGE_REPORTED_STATUSES
from app.exchanges.recovery import (
    ExchangeExecution,
    ExchangeOrder,
    ExchangePosition,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
    OpenOrdersSnapshot,
    PositionSnapshot,
)

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
NAIVE = datetime(2026, 1, 15, 12, 0)  # noqa: DTZ001 - naive on purpose
PLUS_TWO = datetime(2026, 1, 15, 14, 0, tzinfo=timezone(timedelta(hours=2)))
LONG = D("1." + "7" * 150)


class DecimalSub(Decimal):
    pass


def order(**overrides: Any) -> ExchangeOrder:
    values: dict[str, Any] = {
        "exchange_order_id": "ex-1",
        "client_order_id": "c-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "status": S.OPEN,
        "cum_filled_qty": D("0"),
        "cum_filled_notional": None,
        "avg_fill_price": None,
        "created_ts": T0,
        "updated_ts": T1,
    }
    return ExchangeOrder(**{**values, **overrides})


def partial(**overrides: Any) -> ExchangeOrder:
    values: dict[str, Any] = {
        "status": S.PARTIALLY_FILLED,
        "cum_filled_qty": D("4"),
        "cum_filled_notional": D("400"),
        "avg_fill_price": D("100"),
    }
    return order(**{**values, **overrides})


def execution(exec_id: str = "e-1", **overrides: Any) -> ExchangeExecution:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": "ex-1",
        "client_order_id": "c-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D("100"),
        "qty": D("1"),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "kind": ExecutionKind.TRADE,
        "exchange_ts": T1,
    }
    return ExchangeExecution(**{**values, **overrides})


def query(**overrides: Any) -> ExecutionQuery:
    values: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "exchange_order_id": None,
        "start": T0,
        "end": T2,
    }
    return ExecutionQuery(**{**values, **overrides})


# --- ExchangeOrder --------------------------------------------------------------------------


def test_a_valid_order_keeps_every_value_as_given() -> None:
    hostile = Context(prec=1, traps=[Inexact])
    with localcontext(hostile):
        built = partial(
            qty=D("10.000"),
            cum_filled_qty=D("4.000"),
            cum_filled_notional=LONG,
            avg_fill_price=D("1E-200"),
            price=D("100.10"),
        )
    assert repr(built.qty) == "Decimal('10.000')"
    assert repr(built.cum_filled_qty) == "Decimal('4.000')"
    assert built.cum_filled_notional is LONG
    assert repr(built.avg_fill_price) == "Decimal('1E-200')"
    assert repr(built.price) == "Decimal('100.10')"
    with pytest.raises(dataclasses.FrozenInstanceError):
        built.qty = D("1")  # type: ignore[misc]


def test_client_order_id_is_optional() -> None:
    assert order(client_order_id=None).client_order_id is None
    with pytest.raises(DomainValidationError, match="client_order_id"):
        order(client_order_id="")


@pytest.mark.parametrize("status", sorted(set(S) - EXCHANGE_REPORTED_STATUSES, key=str))
def test_local_only_statuses_are_rejected(status: OrderStatus) -> None:
    with pytest.raises(DomainValidationError, match="not an exchange-reported status"):
        order(status=status)


@pytest.mark.parametrize(
    ("status", "filled", "ok"),
    [
        (S.OPEN, "0", True),
        (S.OPEN, "1", False),
        (S.REJECTED, "0", True),
        (S.REJECTED, "1", False),
        (S.PARTIALLY_FILLED, "4", True),
        (S.PARTIALLY_FILLED, "0", False),
        (S.PARTIALLY_FILLED, "10", False),
        (S.FILLED, "10", True),
        (S.FILLED, "9", False),
        (S.CANCELED, "0", True),
        (S.CANCELED, "9", True),
        (S.CANCELED, "10", False),
        (S.EXPIRED, "3", True),
        (S.EXPIRED, "10", False),
    ],
)
def test_cumulative_quantity_must_fit_the_status(
    status: OrderStatus, filled: str, ok: bool
) -> None:
    values: dict[str, Any] = {"status": status, "cum_filled_qty": D(filled)}
    if D(filled) > 0:
        values["avg_fill_price"] = D("100")
    if ok:
        assert order(**values).status is status
    else:
        with pytest.raises(DomainValidationError):
            order(**values)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"exchange_order_id": ""}, "exchange_order_id"),
        ({"exchange_order_id": None}, "exchange_order_id"),
        ({"symbol": " BTCUSDT"}, "symbol"),
        ({"qty": D("0")}, "qty"),
        ({"qty": D("-1")}, "qty"),
        ({"qty": 10}, "Decimal"),
        ({"qty": 10.0}, "Decimal"),
        ({"qty": DecimalSub("10")}, "Decimal"),
        ({"qty": D("NaN")}, "finite"),
        ({"cum_filled_qty": D("-0.1")}, "cum_filled_qty"),
        ({"status": S.CANCELED, "cum_filled_qty": D("11"), "avg_fill_price": D("1")}, "exceeds"),
        ({"cum_filled_notional": D("-1")}, "cum_filled_notional"),
        ({"cum_filled_notional": D("5")}, "inconsistent"),
        ({"avg_fill_price": D("0")}, "avg_fill_price"),
        ({"avg_fill_price": D("100")}, "requires an execution"),
        ({"price": None}, "price"),
        ({"price": D("0")}, "price"),
        ({"order_type": OrderType.MARKET}, "market order has no price"),
        (
            {"order_type": OrderType.MARKET, "price": None, "time_in_force": TimeInForce.POST_ONLY},
            "post_only",
        ),
        ({"reduce_only": 1}, "reduce_only"),
        ({"side": "buy"}, "side"),
        ({"status": "open"}, "status"),
        ({"created_ts": NAIVE}, "created_ts"),
        ({"updated_ts": PLUS_TWO}, "updated_ts"),
        ({"created_ts": "2026-01-15"}, "created_ts"),
        ({"created_ts": T2, "updated_ts": T1}, "before created_ts"),
    ],
)
def test_order_invariants(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(DomainValidationError, match=match):
        order(**overrides)


def test_market_order_has_no_price_and_filled_needs_no_notional() -> None:
    market = order(order_type=OrderType.MARKET, price=None, time_in_force=TimeInForce.IOC)
    assert market.price is None
    assert partial(cum_filled_notional=None, avg_fill_price=None).cum_filled_notional is None
    assert order(cum_filled_notional=D("0")).cum_filled_notional == 0


# --- OpenOrdersSnapshot ---------------------------------------------------------------------


def test_open_snapshot_accepts_open_orders_with_optional_client_ids() -> None:
    snapshot = OpenOrdersSnapshot(
        orders=(
            order(),
            partial(exchange_order_id="ex-2", client_order_id="c-2"),
            order(exchange_order_id="ex-3", client_order_id=None),
            order(exchange_order_id="ex-4", client_order_id=None),
        ),
        server_ts=T1,
    )
    assert len(snapshot.orders) == 4
    assert OpenOrdersSnapshot(orders=(), server_ts=T0).orders == ()


@pytest.mark.parametrize(
    ("orders", "match"),
    [
        ((order(), order(client_order_id="c-2")), "exchange_order_id ex-1"),
        ((order(), order(exchange_order_id="ex-2")), "client_order_id c-1"),
        ((order(status=S.FILLED, cum_filled_qty=D("10"), avg_fill_price=D("1")),), "not open"),
        ((order(status=S.CANCELED),), "not open"),
        ([order()], "tuple"),
        ((object(),), "ExchangeOrder"),
    ],
)
def test_open_snapshot_invariants(orders: Any, match: str) -> None:
    with pytest.raises(DomainValidationError, match=match):
        OpenOrdersSnapshot(orders=orders, server_ts=T2)


@pytest.mark.parametrize("server_ts", [T0, NAIVE, PLUS_TWO, None])
def test_open_snapshot_time_is_utc_and_not_before_its_orders(server_ts: Any) -> None:
    with pytest.raises(DomainValidationError):
        OpenOrdersSnapshot(orders=(order(),), server_ts=server_ts)  # order updated at T1


# --- positions ------------------------------------------------------------------------------


@pytest.mark.parametrize("qty", [D("0"), D("-0"), D("4.000"), D("-3"), D("1E-200"), LONG])
def test_position_quantity_is_signed_and_kept_exactly(qty: Decimal) -> None:
    position = ExchangePosition(symbol="BTCUSDT", qty=qty)
    assert repr(position.qty) == repr(qty)


@pytest.mark.parametrize("qty", [0, 1.5, "1", DecimalSub("1"), D("Infinity"), None])
def test_position_quantity_must_be_an_exact_decimal(qty: Any) -> None:
    with pytest.raises(DomainValidationError, match="qty"):
        ExchangePosition(symbol="BTCUSDT", qty=qty)


@pytest.mark.parametrize("complete", [True, False])
def test_position_snapshot_completeness_flag(complete: bool) -> None:
    snapshot = PositionSnapshot(
        positions=(ExchangePosition(symbol="BTCUSDT", qty=D("1")),),
        complete=complete,
        server_ts=T0,
    )
    assert snapshot.complete is complete


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "positions": (
                    ExchangePosition(symbol="BTCUSDT", qty=D("1")),
                    ExchangePosition(symbol="BTCUSDT", qty=D("0")),
                )
            },
            "more than once",
        ),
        ({"complete": 1}, "complete"),
        ({"complete": None}, "complete"),
        ({"positions": [ExchangePosition(symbol="BTCUSDT", qty=D("1"))]}, "tuple"),
        ({"server_ts": NAIVE}, "server_ts"),
    ],
)
def test_position_snapshot_invariants(kwargs: dict[str, Any], match: str) -> None:
    values: dict[str, Any] = {"positions": (), "complete": True, "server_ts": T0}
    with pytest.raises(DomainValidationError, match=match):
        PositionSnapshot(**{**values, **kwargs})


# --- executions -----------------------------------------------------------------------------


def test_execution_keeps_values_and_allows_signed_fees() -> None:
    rebate = execution(fee=D("-0.0100"), fee_asset="USDT", is_maker=True, qty=D("4.000"))
    assert repr(rebate.fee) == "Decimal('-0.0100')"
    assert repr(rebate.qty) == "Decimal('4.000')"
    assert execution(client_order_id=None).client_order_id is None
    assert execution(kind=ExecutionKind.ADL).kind is ExecutionKind.ADL


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"exec_id": ""}, "exec_id"),
        ({"exchange_order_id": None}, "exchange_order_id"),
        ({"client_order_id": ""}, "client_order_id"),
        ({"price": D("0")}, "price"),
        ({"price": D("-0")}, "price"),
        ({"qty": D("0")}, "qty"),
        ({"qty": 1}, "Decimal"),
        ({"fee": 0.1, "fee_asset": "USDT"}, "fee"),
        ({"fee": D("1")}, "both known"),
        ({"fee_asset": "USDT"}, "both known"),
        ({"is_maker": 1}, "is_maker"),
        ({"kind": "trade"}, "kind"),
        ({"exchange_ts": NAIVE}, "exchange_ts"),
        ({"exchange_ts": PLUS_TWO}, "exchange_ts"),
    ],
)
def test_execution_invariants(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(DomainValidationError, match=match):
        execution(**overrides)


def test_execution_kinds_are_exchange_neutral() -> None:
    assert {kind.value for kind in ExecutionKind} == {
        "trade",
        "liquidation",
        "adl",
        "bust",
        "other",
    }


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"symbol": ""}, "symbol"),
        ({"exchange_order_id": ""}, "exchange_order_id"),
        ({"start": T2, "end": T1}, "before its start"),
        ({"start": NAIVE}, "start"),
        ({"end": PLUS_TWO}, "end"),
    ],
)
def test_query_invariants(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(DomainValidationError, match=match):
        query(**overrides)


def test_query_window_is_inclusive_and_filters_symbol_and_order() -> None:
    q = query(exchange_order_id="ex-1")
    assert q.matches(execution(exchange_ts=T0))
    assert q.matches(execution(exchange_ts=T2))
    assert not q.matches(execution(exchange_ts=T0 - timedelta(microseconds=1)))
    assert not q.matches(execution(exchange_ts=T2 + timedelta(microseconds=1)))
    assert not q.matches(execution(exchange_order_id="ex-2"))
    assert not q.matches(execution(symbol="ETHUSDT"))
    assert query(start=T1, end=T1).matches(execution(exchange_ts=T1))


def test_page_accepts_matching_executions_and_a_cursor() -> None:
    page = ExecutionPage(query=query(), executions=(execution(), execution("e-2")), next_cursor="x")
    assert page.next_cursor == "x"
    assert ExecutionPage(query=query(), executions=(), next_cursor=None).executions == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"executions": (execution(), execution())}, "more than once"),
        ({"executions": (execution(), execution(fee=D("1"), fee_asset="USDT"))}, "more than once"),
        ({"next_cursor": ""}, "next_cursor"),
        ({"next_cursor": 5}, "next_cursor"),
        ({"executions": [execution()]}, "tuple"),
        ({"executions": (execution(symbol="ETHUSDT"),)}, "does not match"),
        ({"executions": (execution(exchange_ts=T2 + timedelta(seconds=1)),)}, "does not match"),
        ({"query": object()}, "ExecutionQuery"),
    ],
)
def test_page_invariants(kwargs: dict[str, Any], match: str) -> None:
    values: dict[str, Any] = {"query": query(), "executions": (), "next_cursor": None}
    with pytest.raises(DomainValidationError, match=match):
        ExecutionPage(**{**values, **kwargs})
