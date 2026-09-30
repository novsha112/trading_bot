"""Order state machine: explicit transition table and transition()."""

from __future__ import annotations

import dataclasses
import itertools
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainError, DomainValidationError, InvalidOrderTransition
from app.domain.order_state import ALLOWED_TRANSITIONS, TERMINAL_STATUSES, transition
from app.domain.orders import Order

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
PRICE = D("65000")

# The contract, written out independently of the implementation (docs/ARCHITECTURE.md, s. 8).
EXPECTED: dict[OrderStatus, set[OrderStatus]] = {
    S.NEW: {S.SUBMITTING, S.FAILED},
    S.SUBMITTING: {
        S.OPEN,
        S.PARTIALLY_FILLED,
        S.FILLED,
        S.CANCELED,
        S.REJECTED,
        S.FAILED,
        S.UNKNOWN,
    },
    S.OPEN: {S.PARTIALLY_FILLED, S.FILLED, S.CANCELING, S.CANCELED, S.EXPIRED, S.UNKNOWN},
    S.PARTIALLY_FILLED: {
        S.PARTIALLY_FILLED,
        S.FILLED,
        S.CANCELING,
        S.CANCELED,
        S.EXPIRED,
        S.UNKNOWN,
    },
    S.CANCELING: {S.CANCELING, S.FILLED, S.CANCELED, S.EXPIRED, S.UNKNOWN},
    S.UNKNOWN: {
        S.OPEN,
        S.PARTIALLY_FILLED,
        S.FILLED,
        S.CANCELED,
        S.REJECTED,
        S.EXPIRED,
        S.FAILED,
    },
    S.FILLED: set(),
    S.CANCELED: set(),
    S.REJECTED: set(),
    S.EXPIRED: set(),
    S.FAILED: set(),
}

# Filled quantity of the source order for each status (qty = 1.0).
SOURCE_FILL = {
    S.NEW: "0",
    S.SUBMITTING: "0",
    S.OPEN: "0",
    S.PARTIALLY_FILLED: "0.3",
    S.CANCELING: "0.3",
    S.UNKNOWN: "0",
    S.FILLED: "1.0",
    S.CANCELED: "0.3",
    S.EXPIRED: "0",
    S.REJECTED: "0",
    S.FAILED: "0",
}


def make_order(status: OrderStatus = S.OPEN, filled_qty: str = "0", **overrides: Any) -> Order:
    qty = D(filled_qty)
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
        "filled_qty": qty,
        "avg_fill_price": PRICE if qty > 0 else None,
        "created_at": T0,
        "updated_at": T0,
        "last_exchange_update_ts": None,
        "version": 3,
    }
    return Order(**{**values, **overrides})


def step(o: Order, status: OrderStatus, fill: str | None = None, seconds: int = 1) -> Order:
    """Move to ``status``; a cumulative fill (with avg price) is passed only if given."""
    kwargs: dict[str, Any] = {}
    if fill is not None:
        kwargs = {"filled_qty": D(fill), "avg_fill_price": PRICE if D(fill) > 0 else None}
    return transition(o, status, at=o.updated_at + timedelta(seconds=seconds), **kwargs)


def target_fill(source: OrderStatus, target: OrderStatus) -> str:
    """A cumulative fill that satisfies the target status invariants."""
    src = D(SOURCE_FILL[source])
    if target is S.FILLED:
        return "1.0"
    if target in (S.PARTIALLY_FILLED, S.CANCELING) and target is source:
        return str(src + D("0.3"))  # special self-transition: must increase
    if target is S.PARTIALLY_FILLED and src == 0:
        return "0.3"
    return str(src)


# --- Table -------------------------------------------------------------------------------


def test_table_matches_contract_exactly() -> None:
    assert {k: set(v) for k, v in ALLOWED_TRANSITIONS.items()} == EXPECTED


def test_table_covers_every_status_and_is_read_only() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(OrderStatus)
    with pytest.raises(TypeError):
        ALLOWED_TRANSITIONS[S.NEW] = frozenset()  # type: ignore[index]


def test_terminal_statuses() -> None:
    assert {S.FILLED, S.CANCELED, S.REJECTED, S.EXPIRED, S.FAILED} == TERMINAL_STATUSES
    for status in TERMINAL_STATUSES:
        assert ALLOWED_TRANSITIONS[status] == frozenset()
    for status in set(OrderStatus) - TERMINAL_STATUSES:
        assert ALLOWED_TRANSITIONS[status], f"{status} is not terminal but has no exits"
    assert S.UNKNOWN not in TERMINAL_STATUSES
    assert S.CANCELING not in TERMINAL_STATUSES


def test_only_two_self_transitions() -> None:
    self_loops = {s for s, targets in ALLOWED_TRANSITIONS.items() if s in targets}
    assert self_loops == {S.PARTIALLY_FILLED, S.CANCELING}


@pytest.mark.parametrize(("source", "target"), list(itertools.product(OrderStatus, OrderStatus)))
def test_full_transition_matrix(source: OrderStatus, target: OrderStatus) -> None:
    o = make_order(source, SOURCE_FILL[source])
    if target in EXPECTED[source]:
        result = step(o, target, target_fill(source, target))
        assert result.status is target
        assert result.version == o.version + 1
    else:
        with pytest.raises(InvalidOrderTransition, match=f"{source.value} -> {target.value}"):
            step(o, target, target_fill(source, target))


# --- Regression scenarios ----------------------------------------------------------------


def test_a_multiple_partial_fills() -> None:
    o = make_order(S.OPEN, "0", version=0)
    o = step(o, S.PARTIALLY_FILLED, "0.3")
    o = step(o, S.PARTIALLY_FILLED, "0.6")
    o = step(o, S.FILLED, "1.0")
    assert (o.status, o.filled_qty, o.version) == (S.FILLED, D("1.0"), 3)


def test_b_cancel_race_ends_filled() -> None:
    o = make_order(S.PARTIALLY_FILLED, "0.3")
    o = step(o, S.CANCELING)
    assert (o.status, o.filled_qty) == (S.CANCELING, D("0.3"))
    o = step(o, S.CANCELING, "0.6")
    assert (o.status, o.filled_qty) == (S.CANCELING, D("0.6"))
    o = step(o, S.FILLED, "1.0")
    assert (o.status, o.filled_qty) == (S.FILLED, D("1.0"))


def test_c_cancel_success_after_partial_fill() -> None:
    o = make_order(S.PARTIALLY_FILLED, "0.3")
    o = step(o, S.CANCELING)
    o = step(o, S.CANCELING, "0.6")
    o = step(o, S.CANCELED)
    assert (o.status, o.filled_qty, o.avg_fill_price) == (S.CANCELED, D("0.6"), PRICE)


def test_d_unknown_resolution() -> None:
    assert step(step(make_order(S.OPEN), S.UNKNOWN), S.OPEN).status is S.OPEN

    pf = step(step(make_order(S.PARTIALLY_FILLED, "0.3"), S.UNKNOWN), S.PARTIALLY_FILLED, "0.5")
    assert (pf.status, pf.filled_qty) == (S.PARTIALLY_FILLED, D("0.5"))

    assert step(make_order(S.UNKNOWN), S.FILLED, "1.0").status is S.FILLED
    assert step(make_order(S.UNKNOWN, "0.3"), S.CANCELED).status is S.CANCELED


@pytest.mark.parametrize("fill", ["0", "1.0"])
def test_d_unknown_to_partially_filled_keeps_fill_invariants(fill: str) -> None:
    with pytest.raises(DomainValidationError, match="partially_filled"):
        step(make_order(S.UNKNOWN, "0"), S.PARTIALLY_FILLED, fill)


def test_unknown_with_fills_cannot_resolve_to_failed() -> None:
    # The arc exists, but FAILED requires zero fill: invariants are the second guard.
    with pytest.raises(DomainValidationError, match="failed"):
        step(make_order(S.UNKNOWN, "0.3"), S.FAILED)


@pytest.mark.parametrize("status", [S.OPEN, S.SUBMITTING, S.UNKNOWN, *TERMINAL_STATUSES])
def test_e_no_op_self_transitions_rejected(status: OrderStatus) -> None:
    o = make_order(status, SOURCE_FILL[status])
    with pytest.raises(InvalidOrderTransition, match=f"{status.value} -> {status.value}"):
        step(o, status)


@pytest.mark.parametrize("status", [S.PARTIALLY_FILLED, S.CANCELING])
@pytest.mark.parametrize("fill", [None, "0.3"])
def test_f_special_self_transition_requires_fill_increase(
    status: OrderStatus, fill: str | None
) -> None:
    with pytest.raises(InvalidOrderTransition, match="must increase filled_qty"):
        step(make_order(status, "0.3"), status, fill)


# --- transition() rules ------------------------------------------------------------------


def test_filled_qty_cannot_decrease() -> None:
    with pytest.raises(InvalidOrderTransition, match=r"filled_qty cannot decrease"):
        step(make_order(S.PARTIALLY_FILLED, "0.3"), S.CANCELED, "0.2")


def test_filled_qty_cannot_exceed_qty() -> None:
    with pytest.raises(DomainValidationError, match=r"must be <= qty"):
        step(make_order(S.PARTIALLY_FILLED, "0.3"), S.CANCELING, "1.1")


def test_ordinary_transition_may_keep_filled_qty() -> None:
    o = step(make_order(S.PARTIALLY_FILLED, "0.3"), S.CANCELED, "0.3")
    assert o.filled_qty == D("0.3")


def test_at_cannot_move_backwards_but_may_be_equal() -> None:
    o = make_order(S.OPEN)
    assert transition(o, S.CANCELING, at=o.updated_at).updated_at == o.updated_at
    with pytest.raises(InvalidOrderTransition, match="cannot move backwards"):
        transition(o, S.CANCELING, at=o.updated_at - timedelta(microseconds=1))


def test_at_must_be_utc() -> None:
    with pytest.raises(DomainValidationError, match=r"^at "):
        transition(make_order(), S.CANCELING, at=datetime(2026, 1, 16))  # noqa: DTZ001


@pytest.mark.parametrize("status", ["canceling", None, OrderType.LIMIT])
def test_new_status_must_be_enum(status: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^new_status must be a OrderStatus"):
        transition(make_order(), status, at=T0)


def test_avg_price_without_filled_qty_rejected() -> None:
    with pytest.raises(DomainValidationError, match="avg_fill_price requires filled_qty"):
        transition(make_order(S.OPEN), S.CANCELING, at=T0, avg_fill_price=PRICE)


def test_positive_fill_requires_avg_price() -> None:
    with pytest.raises(DomainValidationError, match=r"^avg_fill_price must be a Decimal"):
        transition(make_order(S.OPEN), S.PARTIALLY_FILLED, at=T0, filled_qty=D("0.3"))


def test_original_order_unchanged_and_version_increments_once() -> None:
    original = make_order(S.OPEN)
    snapshot = dataclasses.asdict(original)
    at = T0 + timedelta(seconds=5)

    result = transition(
        original, S.PARTIALLY_FILLED, at=at, filled_qty=D("0.3"), avg_fill_price=PRICE
    )

    assert dataclasses.asdict(original) == snapshot
    assert result is not original
    assert result.version == original.version + 1
    assert result.updated_at == at
    assert result.created_at == original.created_at
    unchanged = {"client_order_id", "strategy_id", "symbol", "side", "order_type", "price", "qty"}
    for name in unchanged:
        assert getattr(result, name) == getattr(original, name)


def test_exchange_fields_set_on_transition() -> None:
    ts = T0 + timedelta(milliseconds=500)
    o = transition(
        make_order(S.SUBMITTING), S.OPEN, at=T0, exchange_order_id="o-1", last_exchange_update_ts=ts
    )
    assert (o.exchange_order_id, o.last_exchange_update_ts) == ("o-1", ts)
    # Kept when not given; the same id may be repeated.
    o = transition(o, S.CANCELING, at=T0, exchange_order_id="o-1")
    assert (o.exchange_order_id, o.last_exchange_update_ts) == ("o-1", ts)


def test_exchange_order_id_cannot_change() -> None:
    o = make_order(S.OPEN, exchange_order_id="o-1")
    with pytest.raises(InvalidOrderTransition, match="exchange_order_id cannot change"):
        transition(o, S.CANCELING, at=T0, exchange_order_id="o-2")


def test_invalid_transition_is_domain_error() -> None:
    assert issubclass(InvalidOrderTransition, DomainError)
