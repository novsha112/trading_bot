"""Pure recovery order matching: identity first, then terms; mutually exclusive,
deterministic buckets; never an exception for a classification outcome."""

from __future__ import annotations

import ast
import itertools
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.orders import Order
from app.exchanges.recovery import ExchangeOrder
from app.execution import recovery_matching as module
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.recovery_matching import (
    RELEVANT_LOCAL_STATUSES,
    IdentityConflict,
    IdentityConflictReason,
    ManagedOrderMatch,
    OpenOrderClassification,
    OrderTerm,
    classify_open_orders,
    order_terms_mismatch,
)

D = Decimal
S = OrderStatus
R = IdentityConflictReason
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
NS = ClientOrderNamespace("bot01")
OTHER_NS = ClientOrderNamespace("bot02")
A, B, C = NS.build("aaa"), NS.build("bbb"), NS.build("ccc")

TERMS: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "side": Side.BUY,
    "order_type": OrderType.LIMIT,
    "price": D("100"),
    "qty": D("10"),
    "time_in_force": TimeInForce.GTC,
    "reduce_only": False,
}


def local(cid: str, status: OrderStatus = S.OPEN, eid: str | None = "X-1", **terms: Any) -> Order:
    filled, avg = D("0"), None
    if status in (S.PARTIALLY_FILLED, S.CANCELED, S.EXPIRED):
        filled, avg = D("4"), D("100")
    if status is S.FILLED:
        filled, avg = D("10"), D("100")
    if status in (S.NEW, S.SUBMITTING, S.FAILED, S.REJECTED):
        eid = None if status is not S.REJECTED else eid
    return Order(
        client_order_id=cid,
        exchange_order_id=eid,
        strategy_id="grid-1",
        **{**TERMS, **terms},
        status=status,
        filled_qty=filled,
        avg_fill_price=avg,
        created_at=T0,
        updated_at=T1,
        last_exchange_update_ts=None,
        version=2,
    )


def ex(
    cid: str | None, eid: str = "X-1", status: OrderStatus = S.OPEN, **terms: Any
) -> ExchangeOrder:
    filled, avg, notional = D("0"), None, None
    if status is S.PARTIALLY_FILLED:
        filled, avg, notional = D("4"), D("100"), D("400")
    return ExchangeOrder(
        exchange_order_id=eid,
        client_order_id=cid,
        **{**TERMS, **terms},
        status=status,
        cum_filled_qty=filled,
        cum_filled_notional=notional,
        avg_fill_price=avg,
        created_ts=T0,
        updated_ts=T1,
    )


def classify(
    locals_: tuple[Order, ...],
    exchanges: tuple[ExchangeOrder, ...],
    namespace: ClientOrderNamespace = NS,
) -> OpenOrderClassification:
    result = classify_open_orders(
        local_orders=locals_, exchange_orders=exchanges, namespace=namespace
    )
    assert_exclusive(result, locals_, exchanges)
    return result


def conflict_reasons(result: OpenOrderClassification) -> list[IdentityConflictReason]:
    return [c.reason for c in result.identity_conflicts]


def assert_exclusive(
    result: OpenOrderClassification,
    locals_: tuple[Order, ...],
    exchanges: tuple[ExchangeOrder, ...],
) -> None:
    """Every exchange order in exactly one bucket; every relevant local order in
    at most one of managed / missing / conflict, and in one if it is relevant."""
    conflict_eids = {e for c in result.identity_conflicts for e in c.exchange_order_ids}
    conflict_cids = {c_id for c in result.identity_conflicts for c_id in c.client_order_ids}
    managed_ex = [id(m.exchange_order) for m in result.managed]
    for order in exchanges:
        places = (
            managed_ex.count(id(order))
            + sum(1 for f in result.foreign if f is order)
            + sum(1 for lost in result.lost_managed if lost.exchange_order is order)
            + (1 if order.exchange_order_id in conflict_eids else 0)
        )
        assert places == 1, (order, result)
    managed_cids = [m.local_order.client_order_id for m in result.managed]
    missing_cids = [o.client_order_id for o in result.missing_local_active]
    for mine in locals_:
        if mine.status not in RELEVANT_LOCAL_STATUSES:
            assert mine.client_order_id not in managed_cids + missing_cids
            continue
        places = (
            managed_cids.count(mine.client_order_id)
            + missing_cids.count(mine.client_order_id)
            + (1 if mine.client_order_id in conflict_cids else 0)
        )
        assert places == 1, (mine, result)


# --- managed ------------------------------------------------------------------------------


def test_exact_identity_and_terms_is_managed() -> None:
    mine, theirs = local(A, eid="X-1"), ex(A, "X-1")

    result = classify((mine,), (theirs,))

    assert result.managed == (
        ManagedOrderMatch(
            local_order=mine, exchange_order=theirs, exchange_id_completion_required=False
        ),
    )
    assert result.is_clean


def test_unknown_local_exchange_id_is_a_completion_candidate_not_a_conflict() -> None:
    mine, theirs = local(A, S.UNKNOWN, eid=None), ex(A, "X-9")

    result = classify((mine,), (theirs,))

    (match,) = result.managed
    assert match.exchange_id_completion_required is True
    assert match.exchange_order.exchange_order_id == "X-9"
    assert mine.exchange_order_id is None  # nothing was written


@pytest.mark.parametrize(
    ("local_status", "exchange_status"),
    [
        (S.UNKNOWN, S.OPEN),
        (S.CANCELING, S.OPEN),
        (S.OPEN, S.PARTIALLY_FILLED),
        (S.PARTIALLY_FILLED, S.OPEN),
        (S.CANCELING, S.PARTIALLY_FILLED),
    ],
)
def test_status_and_execution_differences_are_not_conflicts(
    local_status: OrderStatus, exchange_status: OrderStatus
) -> None:
    result = classify((local(A, local_status),), (ex(A, "X-1", exchange_status),))
    assert len(result.managed) == 1
    assert result.identity_conflicts == ()


# --- terms --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "term"),
    [
        ({"symbol": "ETHUSDT"}, OrderTerm.SYMBOL),
        ({"side": Side.SELL}, OrderTerm.SIDE),
        ({"qty": D("10.5")}, OrderTerm.QTY),
        ({"price": D("100.1")}, OrderTerm.PRICE),
        ({"time_in_force": TimeInForce.POST_ONLY}, OrderTerm.TIME_IN_FORCE),
        ({"reduce_only": True}, OrderTerm.REDUCE_ONLY),
    ],
)
def test_each_term_mismatch_is_reported_exactly(overrides: dict[str, Any], term: OrderTerm) -> None:
    result = classify((local(A),), (ex(A, "X-1", **overrides),))

    assert result.managed == ()
    assert result.identity_conflicts == (
        IdentityConflict(
            reason=R.TERMS_MISMATCH,
            client_order_ids=(A,),
            exchange_order_ids=("X-1",),
            mismatched_terms=(term,),
        ),
    )


def test_order_type_mismatch_reports_type_and_price() -> None:
    theirs = ex(A, "X-1", order_type=OrderType.MARKET, price=None, time_in_force=TimeInForce.IOC)
    assert order_terms_mismatch(local(A), theirs) == (
        OrderTerm.ORDER_TYPE,
        OrderTerm.PRICE,
        OrderTerm.TIME_IN_FORCE,
    )


def test_several_mismatches_come_in_fixed_order() -> None:
    theirs = ex(A, "X-1", reduce_only=True, symbol="ETHUSDT", qty=D("1"))
    assert order_terms_mismatch(local(A), theirs) == (
        OrderTerm.SYMBOL,
        OrderTerm.QTY,
        OrderTerm.REDUCE_ONLY,
    )


def test_market_orders_match_with_no_price() -> None:
    market: dict[str, Any] = {
        "order_type": OrderType.MARKET,
        "price": None,
        "time_in_force": TimeInForce.IOC,
    }
    assert order_terms_mismatch(local(A, **market), ex(A, "X-1", **market)) == ()


@pytest.mark.parametrize(
    ("local_value", "exchange_value"),
    [
        (D("10"), D("10.000")),
        (D("1E-200"), D("1.0E-200")),
        (D("1" + "0" * 40), D("1E+40")),
    ],
)
def test_terms_compare_exact_values_not_representations(
    local_value: Decimal, exchange_value: Decimal
) -> None:
    mine = local(A, qty=local_value, price=local_value)
    theirs = ex(A, "X-1", qty=exchange_value, price=exchange_value)
    assert order_terms_mismatch(mine, theirs) == ()
    with localcontext() as context:
        context.prec = 400  # exact: one unit 30 orders of magnitude smaller
        tiny_more = exchange_value + exchange_value.scaleb(-30)
    assert order_terms_mismatch(mine, ex(A, "X-1", qty=exchange_value, price=tiny_more)) == (
        OrderTerm.PRICE,
    )


def test_execution_state_is_not_a_term() -> None:
    mine = local(A, S.OPEN)
    theirs = ex(A, "X-1", S.PARTIALLY_FILLED)
    assert order_terms_mismatch(mine, theirs) == ()


# --- foreign / lost -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cid",
    [None, "manual-1", "external-123", OTHER_NS.build("aaa")],
    ids=["absent", "plain", "dash", "other-ns"],
)
def test_foreign_orders_are_never_matched(cid: str | None) -> None:
    # Identical terms to a local order: similarity means nothing.
    mine, theirs = local(A, eid="X-1"), ex(cid, "X-2")

    result = classify((mine,), (theirs,))

    assert result.foreign == (theirs,)
    assert result.managed == ()
    assert result.missing_local_active == (mine,)


def test_ours_without_local_is_lost_managed() -> None:
    theirs = ex(B, "X-2")

    result = classify((), (theirs,))

    (lost,) = result.lost_managed
    assert lost.exchange_order is theirs
    assert (lost.identity.namespace, lost.identity.order_token) == (NS, "bbb")
    assert lost.local_order is None


@pytest.mark.parametrize("status", [S.FILLED, S.CANCELED, S.FAILED, S.REJECTED, S.NEW])
def test_ours_with_only_a_non_relevant_local_is_lost_and_reports_it(status: OrderStatus) -> None:
    mine = local(B, status, eid="X-2" if status in (S.FILLED, S.CANCELED, S.REJECTED) else None)
    theirs = ex(B, "X-2")

    result = classify((mine,), (theirs,))

    (lost,) = result.lost_managed
    assert lost.local_order is mine
    assert result.missing_local_active == ()


# --- identity conflicts ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cid", "reason"),
    [
        ("tb1_bot01_AAA", R.MALFORMED_MANAGED_ID),
        ("tb1_bot01", R.MALFORMED_MANAGED_ID),
        ("TB1_bot01_aaa", R.MALFORMED_MANAGED_ID),
        ("tb2_bot01_aaa", R.UNSUPPORTED_MANAGED_VERSION),
    ],
)
def test_damaged_exchange_ids_are_conflicts_not_foreign(
    cid: str, reason: IdentityConflictReason
) -> None:
    result = classify((), (ex(cid, "X-5"),))

    assert result.foreign == ()
    assert result.lost_managed == ()
    assert conflict_reasons(result) == [reason]
    assert result.identity_conflicts[0].client_order_ids == (cid,)


def test_exchange_id_mismatch_is_a_conflict() -> None:
    result = classify((local(A, eid="X-1"),), (ex(A, "X-2"),))

    assert result.managed == ()
    assert result.identity_conflicts == (
        IdentityConflict(
            reason=R.EXCHANGE_ID_MISMATCH, client_order_ids=(A,), exchange_order_ids=("X-1", "X-2")
        ),
    )


def test_reverse_exchange_id_collision_even_with_matching_terms() -> None:
    a, b = local(A, eid="X-1"), local(B, eid="X-2")

    result = classify((a, b), (ex(A, "X-2"),))

    assert result.managed == ()
    assert result.missing_local_active == ()
    assert result.identity_conflicts == (
        IdentityConflict(
            reason=R.REVERSE_EXCHANGE_ID_COLLISION,
            client_order_ids=(A, B),
            exchange_order_ids=("X-1", "X-2"),
        ),
    )


@pytest.mark.parametrize("cid", [None, "manual-1", OTHER_NS.build("zzz")])
def test_a_foreign_looking_order_with_a_known_exchange_id_is_a_collision(cid: str | None) -> None:
    result = classify((local(A, eid="X-1"),), (ex(cid, "X-1"),))

    assert result.foreign == ()
    assert conflict_reasons(result) == [R.REVERSE_EXCHANGE_ID_COLLISION]


def test_collision_with_a_terminal_local_order_is_detected() -> None:
    done = local(B, S.FILLED, eid="X-7")
    result = classify((done,), (ex(A, "X-7"),))
    assert conflict_reasons(result) == [R.REVERSE_EXCHANGE_ID_COLLISION]


@pytest.mark.parametrize(
    ("cid", "reason"),
    [
        ("legacy-0001", R.LOCAL_UNMANAGED_ID),
        (OTHER_NS.build("aaa"), R.LOCAL_UNMANAGED_ID),
        ("tb1_bot01_AAA", R.LOCAL_MALFORMED_MANAGED_ID),
        ("tb3_bot01_aaa", R.LOCAL_UNSUPPORTED_MANAGED_VERSION),
    ],
)
def test_local_orders_without_our_managed_id_fail_closed(
    cid: str, reason: IdentityConflictReason
) -> None:
    mine = local(cid, eid=None)
    # The exchange counterpart has identical terms and the same id: never matched,
    # never foreign, never lost.
    theirs = ex(cid, "X-3")

    result = classify((mine,), (theirs,))

    assert result.managed == ()
    assert result.foreign == ()
    assert result.lost_managed == ()
    assert result.missing_local_active == ()
    assert conflict_reasons(result) == [reason]
    assert result.identity_conflicts[0].exchange_order_ids == ("X-3",)


def test_local_legacy_order_without_exchange_counterpart_is_still_a_conflict() -> None:
    result = classify((local("legacy-1", eid=None),), ())
    assert conflict_reasons(result) == [R.LOCAL_UNMANAGED_ID]
    assert result.missing_local_active == ()


def test_legacy_id_of_a_terminal_local_order_is_not_foreign() -> None:
    done = local("legacy-1", S.FILLED, eid="X-4")
    result = classify((done,), (ex("legacy-1", "X-8"),))
    assert result.foreign == ()
    assert conflict_reasons(result) == [R.LOCAL_UNMANAGED_ID]


# --- duplicates -----------------------------------------------------------------------------


def test_duplicate_local_client_ids_fail_closed() -> None:
    first, second = local(A, eid="X-1"), local(A, S.UNKNOWN, eid=None)
    result = classify((first, second), (ex(A, "X-1"),))
    assert result.managed == ()
    assert conflict_reasons(result) == [R.DUPLICATE_LOCAL_CLIENT_ID]
    assert result.identity_conflicts[0].exchange_order_ids == ("X-1",)


def test_duplicate_local_exchange_ids_fail_closed() -> None:
    result = classify((local(A, eid="X-1"), local(B, eid="X-1")), (ex(A, "X-1"),))
    assert result.managed == ()
    assert conflict_reasons(result) == [R.DUPLICATE_LOCAL_EXCHANGE_ID]


def test_duplicate_exchange_client_ids_fail_closed() -> None:
    result = classify((local(A, eid=None),), (ex(A, "X-1"), ex(A, "X-2")))
    assert result.managed == ()
    assert result.missing_local_active == ()
    assert conflict_reasons(result) == [R.DUPLICATE_EXCHANGE_CLIENT_ID]
    assert result.identity_conflicts[0].exchange_order_ids == ("X-1", "X-2")


def test_duplicate_exchange_order_ids_fail_closed() -> None:
    result = classify((), (ex(A, "X-1"), ex("manual", "X-1")))
    assert result.foreign == ()
    assert result.lost_managed == ()
    assert conflict_reasons(result) == [R.DUPLICATE_EXCHANGE_ORDER_ID]


# --- missing --------------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(RELEVANT_LOCAL_STATUSES, key=str))
def test_relevant_local_orders_absent_from_the_snapshot_are_missing(status: OrderStatus) -> None:
    mine = local(A, status, eid=None if status is S.UNKNOWN else "X-1")

    result = classify((mine,), ())

    assert result.missing_local_active == (mine,)
    assert mine.status is status  # nothing inferred, nothing changed
    assert not result.is_clean


@pytest.mark.parametrize(
    "status", [s for s in S if s not in RELEVANT_LOCAL_STATUSES], ids=lambda s: s.value
)
def test_other_local_statuses_are_never_missing(status: OrderStatus) -> None:
    eid = "X-1" if status in (S.FILLED, S.CANCELED, S.EXPIRED, S.REJECTED) else None
    result = classify((local(A, status, eid=eid),), ())
    assert result.missing_local_active == ()
    assert result.is_clean


def test_relevant_statuses_are_exactly_the_documented_ones() -> None:
    assert (
        frozenset({S.UNKNOWN, S.OPEN, S.PARTIALLY_FILLED, S.CANCELING}) == RELEVANT_LOCAL_STATUSES
    )


# --- a mixed scenario and determinism -------------------------------------------------------


def mixed() -> tuple[tuple[Order, ...], tuple[ExchangeOrder, ...]]:
    locals_ = (
        local(A, eid="X-1"),  # managed
        local(B, S.UNKNOWN, eid=None),  # managed, completion
        local(C, eid="X-3"),  # missing
        local(NS.build("ddd"), eid="X-4"),  # exchange id mismatch
        local("legacy-9", eid=None),  # local unmanaged
    )
    exchanges = (
        ex(A, "X-1"),
        ex(B, "X-2"),
        ex(NS.build("ddd"), "X-40"),
        ex(NS.build("eee"), "X-5"),  # lost
        ex(None, "X-6"),  # foreign
        ex("manual", "X-7"),  # foreign
        ex("tb1_bot01_E", "X-8"),  # malformed
        ex(NS.build("fff"), "X-9", qty=D("1")),  # lost (no local, terms irrelevant)
    )
    return locals_, exchanges


def test_mixed_snapshot_buckets() -> None:
    locals_, exchanges = mixed()

    result = classify(locals_, exchanges)

    assert [
        (m.local_order.client_order_id, m.exchange_id_completion_required) for m in result.managed
    ] == [
        (A, False),
        (B, True),
    ]
    assert [o.exchange_order_id for o in result.foreign] == ["X-6", "X-7"]
    assert [lost.exchange_order.exchange_order_id for lost in result.lost_managed] == ["X-5", "X-9"]
    assert conflict_reasons(result) == [
        R.LOCAL_UNMANAGED_ID,
        R.MALFORMED_MANAGED_ID,
        R.EXCHANGE_ID_MISMATCH,
    ]
    assert [o.client_order_id for o in result.missing_local_active] == [C]


def test_result_does_not_depend_on_input_order() -> None:
    locals_, exchanges = mixed()
    expected = classify(locals_, exchanges)
    for perm in itertools.islice(itertools.permutations(exchanges), 0, 200, 7):
        assert classify(tuple(reversed(locals_)), tuple(perm)) == expected
    for local_perm in itertools.permutations(locals_):
        assert classify(tuple(local_perm), tuple(reversed(exchanges))) == expected


def test_inputs_are_not_mutated() -> None:
    locals_, exchanges = mixed()
    before = (tuple(locals_), tuple(exchanges), repr(locals_), repr(exchanges))
    classify(locals_, exchanges)
    assert before == (tuple(locals_), tuple(exchanges), repr(locals_), repr(exchanges))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_orders": [local(A)]},
        {"local_orders": (object(),)},
        {"exchange_orders": [ex(A)]},
        {"exchange_orders": (local(A),)},
        {"namespace": "bot01"},
    ],
)
def test_wrong_input_types_raise(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {"local_orders": (), "exchange_orders": (), "namespace": NS}
    with pytest.raises(DomainValidationError):
        classify_open_orders(**{**values, **kwargs})


def test_terms_helper_rejects_wrong_types() -> None:
    with pytest.raises(DomainValidationError):
        order_terms_mismatch(ex(A), local(A))  # type: ignore[arg-type]


# --- boundaries -----------------------------------------------------------------------------


def test_module_is_pure() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    app_imports = {name for name in imported if name.startswith("app.")}
    assert app_imports <= {
        "app.domain.enums",
        "app.domain.errors",
        "app.domain.orders",
        "app.exchanges.recovery",
        "app.execution.client_order_id",
    }
    for banned in ("asyncio", "random", "secrets", "time", "datetime", "logging"):
        assert banned not in imported
    for word in ("Clock", "now(", "simulated", "bybit", "ExchangeStateReader", "persistence"):
        assert word not in source
