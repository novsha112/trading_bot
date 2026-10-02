"""AccountStateStore contract on the in-memory reference implementation."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.fills import Fill
from app.domain.intents import PlaceOrderIntent
from app.domain.order_state import transition
from app.domain.orders import Order
from app.execution.models import PlacementRecord
from app.persistence import memory as memory_module
from app.persistence import models as models_module
from app.persistence import protocols as protocols_module
from app.persistence.errors import (
    PersistenceStoreError,
    StoreCommitError,
    StoreConflictError,
    StoreUncertainError,
    StoreValidationError,
)
from app.persistence.memory import CommitFailure, InMemoryAccountStateStore
from app.persistence.models import (
    AccountStateChange,
    PersistedAccountState,
    PersistedOrderNotional,
    PersistedPosition,
)
from app.persistence.protocols import AccountStateStore
from app.risk.models import ExposureChange, RiskDecision, RiskReason

D = Decimal
S = OrderStatus
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
SCOPE = "acct-1"


# --- builders -------------------------------------------------------------------------------


def intent(intent_id: str = "i-1", **overrides: Any) -> PlaceOrderIntent:
    values: dict[str, Any] = {
        "intent_id": intent_id,
        "strategy_id": "grid-1",
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "price": D("100.5"),
        "qty": D("10"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "tag": None,
        "created_at": T0,
    }
    return PlaceOrderIntent(**{**values, **overrides})


def approved(source: PlaceOrderIntent, cid: str) -> PlacementRecord:
    return PlacementRecord(
        intent=source,
        decision=RiskDecision(
            intent_id=source.intent_id,
            snapshot_id=f"{SCOPE}:0",
            policy_id="policy-1",
            approved=True,
            reasons=(),
            exposure=ExposureChange(
                reducing_qty=D("0"),
                increasing_qty=source.qty,
                worst_long_qty=source.qty,
                worst_short_qty=D("0"),
            ),
        ),
        client_order_id=cid,
    )


def rejected(source: PlaceOrderIntent) -> PlacementRecord:
    return PlacementRecord(
        intent=source,
        decision=RiskDecision(
            intent_id=source.intent_id,
            snapshot_id=f"{SCOPE}:0",
            policy_id="policy-1",
            approved=False,
            reasons=(RiskReason.MAX_OPEN_ORDERS,),
            exposure=None,
        ),
        client_order_id=None,
    )


def new_order(source: PlaceOrderIntent, cid: str) -> Order:
    return Order(
        client_order_id=cid,
        exchange_order_id=None,
        strategy_id=source.strategy_id,
        symbol=source.symbol,
        side=source.side,
        order_type=source.order_type,
        price=source.price,
        qty=source.qty,
        time_in_force=source.time_in_force,
        reduce_only=source.reduce_only,
        status=S.NEW,
        filled_qty=D("0"),
        avg_fill_price=None,
        created_at=T0,
        updated_at=T0,
        last_exchange_update_ts=None,
        version=0,
    )


def fill(exec_id: str = "e-1", cid: str | None = "c-1", **overrides: Any) -> Fill:
    values: dict[str, Any] = {
        "exec_id": exec_id,
        "exchange_order_id": "ex-1",
        "client_order_id": cid,
        "symbol": "BTCUSDT",
        "side": Side.BUY,
        "price": D("100"),
        "qty": D("4"),
        "fee": None,
        "fee_asset": None,
        "is_maker": None,
        "exchange_ts": T1,
    }
    return Fill(**{**values, **overrides})


def notional(cid: str, value: str) -> PersistedOrderNotional:
    return PersistedOrderNotional(client_order_id=cid, filled_notional=D(value))


def known(symbol: str, qty: str) -> PersistedPosition:
    return PersistedPosition(symbol=symbol, known=True, qty=D(qty))


def unknown(symbol: str) -> PersistedPosition:
    return PersistedPosition(symbol=symbol, known=False, qty=None)


def change(
    expected: int, new: int | None = None, *, scope: str = SCOPE, **writes: Any
) -> AccountStateChange:
    return AccountStateChange(
        account_scope_id=scope,
        expected_revision=expected,
        new_revision=expected + 1 if new is None else new,
        **writes,
    )


def reservation(intent_id: str, cid: str, **overrides: Any) -> dict[str, Any]:
    """Writes of an approved reservation: placement + Order(NEW) + zero notional."""
    source = intent(intent_id, **overrides)
    return {
        "placement_writes": (approved(source, cid),),
        "order_writes": (new_order(source, cid),),
        "notional_writes": (notional(cid, "0"),),
    }


def filled(order: Order, qty: str, *, at: datetime = T1) -> Order:
    """SUBMITTING -> PARTIALLY_FILLED / FILLED with an exchange id (domain path)."""
    submitting = transition(order, S.SUBMITTING, at=at)
    target = S.FILLED if D(qty) == order.qty else S.PARTIALLY_FILLED
    return transition(
        submitting,
        target,
        at=at,
        filled_qty=D(qty),
        avg_fill_price=D("100"),
        exchange_order_id="ex-1",
    )


async def seeded(store: InMemoryAccountStateStore | None = None) -> InMemoryAccountStateStore:
    """Revision 1: one reserved order c-1 and a known flat BTCUSDT."""
    store = InMemoryAccountStateStore() if store is None else store
    await store.commit(
        change(0, position_writes=(known("BTCUSDT", "0"),), **reservation("i-1", "c-1"))
    )
    return store


async def loaded(store: InMemoryAccountStateStore, scope: str = SCOPE) -> PersistedAccountState:
    state = await store.load(account_scope_id=scope)
    assert state is not None
    return state


def as_tuple(state: PersistedAccountState) -> tuple[Any, ...]:
    return (
        state.revision,
        dict(state.placements),
        dict(state.orders),
        dict(state.fills),
        dict(state.positions),
        dict(state.notionals),
    )


# --- protocol / models ----------------------------------------------------------------------


def test_memory_store_satisfies_the_protocol() -> None:
    store: AccountStateStore = InMemoryAccountStateStore()

    assert callable(store.load)
    assert callable(store.commit)


def test_error_hierarchy() -> None:
    for error in (StoreValidationError, StoreConflictError, StoreCommitError, StoreUncertainError):
        assert issubclass(error, PersistenceStoreError)
    assert not issubclass(StoreConflictError, StoreValidationError)
    assert not issubclass(StoreUncertainError, StoreCommitError)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"known": False, "qty": D("0")}, "unknown position has no qty"),
        ({"known": True, "qty": None}, "known position"),
        ({"known": True, "qty": 1}, "known position"),
        ({"known": True, "qty": D("NaN")}, "known position"),
        ({"known": 1, "qty": None}, "known must be a bool"),
        ({"symbol": "", "known": False, "qty": None}, "symbol"),
    ],
)
def test_position_marker_invariant(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(StoreValidationError, match=match):
        PersistedPosition(**{"symbol": "BTCUSDT", **kwargs})


@pytest.mark.parametrize("qty", ["0", "-0", "2.5", "-2.5", "4.000"])
def test_known_positions_keep_the_exact_decimal(qty: str) -> None:
    position = known("BTCUSDT", qty)

    assert position.qty is not None
    assert position.qty.as_tuple() == D(qty).as_tuple()


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (D("-1"), ">= 0"),
        (D("-0.000001"), ">= 0"),
        (D("NaN"), "finite"),
        (D("Infinity"), "finite"),
        (1, "exact"),
        (1.5, "exact"),
    ],
)
def test_notional_must_be_exact_finite_and_non_negative(value: object, match: str) -> None:
    with pytest.raises(StoreValidationError, match=match):
        PersistedOrderNotional(client_order_id="c-1", filled_notional=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expected", "new", "match"),
    [
        (0, 2, "expected_revision \\+ 1"),
        (3, 2, "expected_revision"),
        (-1, 0, "expected_revision"),
        (True, 1, "expected_revision"),
    ],
)
def test_revision_step_is_validated_before_anything(expected: Any, new: Any, match: str) -> None:
    with pytest.raises(StoreValidationError, match=match):
        AccountStateChange(account_scope_id=SCOPE, expected_revision=expected, new_revision=new)


@pytest.mark.parametrize(
    "writes",
    [
        {"order_writes": (new_order(intent(), "c-1"),)},
        {"fill_writes": (fill(),)},
        {"position_writes": (known("BTCUSDT", "0"),)},
        {"notional_writes": (notional("c-1", "0"),)},
        {"placement_writes": (approved(intent(), "c-1"),)},
    ],
)
def test_same_revision_change_may_only_write_rejected_placements(writes: dict[str, Any]) -> None:
    with pytest.raises(StoreValidationError, match="rejected placements"):
        change(4, 4, **writes)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("placement_writes", [rejected(intent())]),
        ("order_writes", ("order",)),
        ("fill_writes", (None,)),
        ("position_writes", (notional("c-1", "0"),)),
        ("notional_writes", (known("BTCUSDT", "0"),)),
    ],
)
def test_change_write_types_are_validated(field_name: str, value: object) -> None:
    writes: dict[str, Any] = {field_name: value}
    with pytest.raises(StoreValidationError, match=field_name):
        change(0, **writes)


# --- load / revision ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_account_loads_as_none() -> None:
    assert await InMemoryAccountStateStore().load(account_scope_id=SCOPE) is None


@pytest.mark.asyncio
async def test_committed_account_loads_completely() -> None:
    store = await seeded()

    state = await loaded(store)

    source = intent("i-1")
    assert state.account_scope_id == SCOPE
    assert state.revision == 1
    assert dict(state.placements) == {"i-1": approved(source, "c-1")}
    assert dict(state.orders) == {"c-1": new_order(source, "c-1")}
    assert dict(state.fills) == {}
    assert dict(state.positions) == {"BTCUSDT": known("BTCUSDT", "0")}
    assert dict(state.notionals) == {"c-1": D("0")}


@pytest.mark.asyncio
async def test_first_commit_expects_revision_zero() -> None:
    store = InMemoryAccountStateStore()

    with pytest.raises(StoreConflictError, match="expected revision 1, stored revision 0"):
        await store.commit(change(1, **reservation("i-1", "c-1")))

    assert await store.load(account_scope_id=SCOPE) is None


@pytest.mark.asyncio
async def test_first_commit_may_keep_revision_zero() -> None:
    store = InMemoryAccountStateStore()

    await store.commit(change(0, 0, placement_writes=(rejected(intent()),)))

    state = await loaded(store)
    assert state.revision == 0
    assert state.placements["i-1"].approved is False


@pytest.mark.parametrize("expected", [0, 2, 5])
@pytest.mark.asyncio
async def test_stale_expected_revision_is_a_conflict(expected: int) -> None:
    store = await seeded()
    before = as_tuple(await loaded(store))

    with pytest.raises(StoreConflictError, match="stored revision 1"):
        await store.commit(change(expected, position_writes=(known("BTCUSDT", "1"),)))

    assert as_tuple(await loaded(store)) == before


@pytest.mark.asyncio
async def test_revision_plus_one_and_same_revision_are_accepted() -> None:
    store = await seeded()

    await store.commit(change(1, 2, position_writes=(known("BTCUSDT", "1"),)))
    await store.commit(change(2, 2, placement_writes=(rejected(intent("i-2")),)))

    assert (await loaded(store)).revision == 2


# --- load isolation / multiple accounts -----------------------------------------------------


@pytest.mark.asyncio
async def test_loaded_state_is_immutable_and_isolated() -> None:
    store = await seeded()
    state = await loaded(store)

    with pytest.raises(TypeError):
        state.orders["c-9"] = new_order(intent(), "c-9")  # type: ignore[index]
    with pytest.raises(TypeError):
        state.notionals["c-1"] = D("1")  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.revision = 9  # type: ignore[misc]

    await store.commit(change(1, position_writes=(known("BTCUSDT", "3"),)))

    assert state.revision == 1  # an earlier snapshot never changes
    assert state.positions["BTCUSDT"] == known("BTCUSDT", "0")
    assert (await loaded(store)).positions["BTCUSDT"] == known("BTCUSDT", "3")


def test_persisted_state_copies_the_given_mappings() -> None:
    positions = {"BTCUSDT": known("BTCUSDT", "0")}
    state = PersistedAccountState(
        account_scope_id=SCOPE,
        revision=1,
        placements={},
        orders={},
        fills={},
        positions=positions,
        notionals={},
    )

    positions["ETHUSDT"] = known("ETHUSDT", "1")

    assert "ETHUSDT" not in state.positions


def test_persisted_state_rejects_entries_that_do_not_match_their_key() -> None:
    with pytest.raises(StoreValidationError, match="positions"):
        PersistedAccountState(
            account_scope_id=SCOPE,
            revision=0,
            placements={},
            orders={},
            fills={},
            positions={"ETHUSDT": known("BTCUSDT", "0")},
            notionals={},
        )


@pytest.mark.asyncio
async def test_accounts_are_isolated() -> None:
    store = InMemoryAccountStateStore()

    await store.commit(change(0, scope="a", **reservation("i-1", "c-1")))
    await store.commit(change(0, scope="b", **reservation("i-1", "c-1", qty=D("3"))))
    await store.commit(change(1, scope="b", position_writes=(known("BTCUSDT", "0"),)))

    a, b = await loaded(store, "a"), await loaded(store, "b")
    assert (a.revision, b.revision) == (1, 2)
    assert (a.orders["c-1"].qty, b.orders["c-1"].qty) == (D("10"), D("3"))
    assert dict(a.positions) == {}


# --- placements -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_placement_is_a_no_op() -> None:
    store = await seeded()

    await store.commit(change(1, 1, placement_writes=(rejected(intent("i-2")),)))
    await store.commit(change(1, 1, placement_writes=(rejected(intent("i-2")),)))

    assert len((await loaded(store)).placements) == 2


@pytest.mark.parametrize(
    "other",
    [
        rejected(intent("i-1", qty=D("11"))),
        rejected(intent("i-1", price=D("100.50"))),  # same value, other representation
    ],
)
@pytest.mark.asyncio
async def test_same_intent_with_different_data_is_a_conflict(other: PlacementRecord) -> None:
    store = InMemoryAccountStateStore()
    await store.commit(change(0, 0, placement_writes=(rejected(intent("i-1")),)))

    with pytest.raises(StoreConflictError, match="intent_id 'i-1'"):
        await store.commit(change(0, 0, placement_writes=(other,)))


@pytest.mark.asyncio
async def test_client_order_id_cannot_serve_two_placements() -> None:
    store = await seeded()
    before = as_tuple(await loaded(store))
    source = intent("i-2")

    with pytest.raises(StoreConflictError, match="client_order_id c-1"):
        await store.commit(change(1, placement_writes=(approved(source, "c-1"),)))

    assert as_tuple(await loaded(store)) == before


# --- orders ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_order_versions() -> None:
    store = await seeded()
    original = (await loaded(store)).orders["c-1"]
    submitting = transition(original, S.SUBMITTING, at=T1)  # version 1

    await store.commit(change(1, order_writes=(submitting,)))  # higher version: update
    await store.commit(change(2, order_writes=(submitting,)))  # identical: no-op
    assert (await loaded(store)).orders["c-1"] == submitting

    with pytest.raises(StoreConflictError, match="older than the stored version 1"):
        await store.commit(change(3, order_writes=(original,)))
    other = dataclasses.replace(submitting, updated_at=T1 + timedelta(seconds=1))
    with pytest.raises(StoreConflictError, match="version 1 is already stored"):
        await store.commit(change(3, order_writes=(other,)))

    state = await loaded(store)
    assert (state.revision, state.orders["c-1"]) == (3, submitting)


@pytest.mark.asyncio
async def test_exchange_order_id_is_unique_among_orders() -> None:
    store = await seeded()
    await store.commit(change(1, **reservation("i-2", "c-2")))
    state = await loaded(store)
    first = transition(state.orders["c-1"], S.SUBMITTING, at=T1, exchange_order_id="ex-1")
    second = transition(state.orders["c-2"], S.SUBMITTING, at=T1, exchange_order_id="ex-1")
    before = as_tuple(state)

    with pytest.raises(StoreConflictError, match="exchange_order_id ex-1"):
        await store.commit(change(2, order_writes=(first, second)))

    assert as_tuple(await loaded(store)) == before


@pytest.mark.asyncio
async def test_order_must_carry_its_placement_terms() -> None:
    store = InMemoryAccountStateStore()
    source = intent()
    mismatched = dataclasses.replace(new_order(source, "c-1"), qty=D("11"))

    with pytest.raises(StoreValidationError, match="qty does not match"):
        await store.commit(
            change(
                0,
                placement_writes=(approved(source, "c-1"),),
                order_writes=(mismatched,),
                notional_writes=(notional("c-1", "0"),),
            )
        )


# --- fills ----------------------------------------------------------------------------------


async def with_fill(store: InMemoryAccountStateStore) -> InMemoryAccountStateStore:
    state = await loaded(store)
    order = filled(state.orders["c-1"], "4")
    await store.commit(
        change(
            state.revision,
            fill_writes=(fill(),),
            order_writes=(order,),
            notional_writes=(notional("c-1", "400"),),
            position_writes=(known("BTCUSDT", "4"),),
        )
    )
    return store


@pytest.mark.asyncio
async def test_fill_order_notional_and_position_in_one_commit() -> None:
    store = await with_fill(await seeded())

    state = await loaded(store)
    assert state.revision == 2
    assert state.fills["e-1"] == fill()
    assert state.orders["c-1"].filled_qty == D("4")
    assert state.notionals["c-1"] == D("400")
    assert state.positions["BTCUSDT"] == known("BTCUSDT", "4")


@pytest.mark.asyncio
async def test_identical_fill_is_a_no_op_and_different_fill_a_conflict() -> None:
    store = await with_fill(await seeded())

    await store.commit(change(2, fill_writes=(fill(),)))
    assert (await loaded(store)).revision == 3
    before = as_tuple(await loaded(store))

    for other in (fill(qty=D("5")), fill(qty=D("4.0")), fill(price=D("101"))):
        with pytest.raises(StoreConflictError, match="exec_id 'e-1'"):
            await store.commit(change(3, fill_writes=(other,)))
    assert as_tuple(await loaded(store)) == before


@pytest.mark.parametrize(
    ("bad_fill", "match"),
    [
        (fill(cid="c-9"), "no order c-9"),
        (fill(cid=None), "no client_order_id"),
        (fill(symbol="ETHUSDT"), "does not match"),
        (fill(side=Side.SELL), "does not match"),
        (fill(exchange_order_id="ex-2"), "differs"),
    ],
)
@pytest.mark.asyncio
async def test_fill_must_reference_its_order(bad_fill: Fill, match: str) -> None:
    store = await seeded()
    state = await loaded(store)
    order = filled(state.orders["c-1"], "4")
    before = as_tuple(state)

    with pytest.raises(StoreValidationError, match=match):
        await store.commit(
            change(
                1,
                fill_writes=(bad_fill,),
                order_writes=(order,),
                notional_writes=(notional("c-1", "400"),),
            )
        )

    assert as_tuple(await loaded(store)) == before


# --- positions / notionals ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positions_unknown_flat_long_short_and_replacement() -> None:
    store = InMemoryAccountStateStore()

    await store.commit(
        change(
            0,
            position_writes=(
                unknown("BTCUSDT"),
                known("ETHUSDT", "0"),
                known("SOLUSDT", "2.5"),
                known("XRPUSDT", "-7"),
            ),
        )
    )
    await store.commit(change(1, position_writes=(known("BTCUSDT", "1"), unknown("ETHUSDT"))))
    await store.commit(change(2, position_writes=(known("BTCUSDT", "1"),)))  # same value

    positions = (await loaded(store)).positions
    assert positions["BTCUSDT"] == known("BTCUSDT", "1")
    assert positions["ETHUSDT"] == unknown("ETHUSDT")  # unknown is a stored marker
    assert positions["SOLUSDT"] == known("SOLUSDT", "2.5")
    assert positions["XRPUSDT"] == known("XRPUSDT", "-7")


@pytest.mark.parametrize("value", ["400", "0.000001", "1." + "2" * 100, "4.000E+2"])
@pytest.mark.asyncio
async def test_notional_replacement_keeps_the_exact_value(value: str) -> None:
    store = await seeded()
    order = filled((await loaded(store)).orders["c-1"], "4")

    await store.commit(change(1, order_writes=(order,), notional_writes=(notional("c-1", value),)))

    stored = (await loaded(store)).notionals["c-1"]
    assert type(stored) is Decimal
    assert stored.as_tuple() == D(value).as_tuple()


@pytest.mark.parametrize(
    ("writes", "match"),
    [
        ({"notional_writes": (notional("c-9", "0"),)}, "unknown order c-9"),
        (
            {
                "placement_writes": (approved(intent("i-2"), "c-2"),),
                "notional_writes": (notional("c-2", "0"),),
            },
            "no order c-2",
        ),
        (
            {
                "placement_writes": (approved(intent("i-2"), "c-2"),),
                "order_writes": (new_order(intent("i-2"), "c-2"),),
            },
            "no filled notional",
        ),
        ({"notional_writes": (notional("c-1", "1"),)}, "inconsistent with filled_qty"),
    ],
)
@pytest.mark.asyncio
async def test_dangling_or_inconsistent_references_are_rejected(
    writes: dict[str, Any], match: str
) -> None:
    store = await seeded()
    before = as_tuple(await loaded(store))

    with pytest.raises(StoreValidationError, match=match):
        await store.commit(change(1, **writes))

    assert as_tuple(await loaded(store)) == before


# --- atomicity ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conflict_in_the_last_write_applies_nothing() -> None:
    store = await with_fill(await seeded())
    before = as_tuple(await loaded(store))

    with pytest.raises(StoreConflictError, match="exec_id 'e-1'"):
        await store.commit(
            change(
                2,
                placement_writes=(rejected(intent("i-9")),),
                position_writes=(known("BTCUSDT", "99"), known("ETHUSDT", "1")),
                fill_writes=(fill("e-2", qty=D("1")), fill("e-1", qty=D("9"))),  # last conflicts
                order_writes=((await loaded(store)).orders["c-1"],),  # identical rewrite
            )
        )

    assert as_tuple(await loaded(store)) == before


@pytest.mark.parametrize(
    "writes",
    [
        {"position_writes": (known("BTCUSDT", "1"), known("BTCUSDT", "2"))},
        {"fill_writes": (fill("e-1"), fill("e-1"))},
        {"placement_writes": (rejected(intent("i-7")), rejected(intent("i-7")))},
    ],
)
@pytest.mark.asyncio
async def test_duplicate_identity_within_one_change_is_rejected(writes: dict[str, Any]) -> None:
    store = await seeded()
    before = as_tuple(await loaded(store))

    with pytest.raises(StoreValidationError, match="more than once"):
        await store.commit(change(1, **writes))

    assert as_tuple(await loaded(store)) == before


@pytest.mark.asyncio
async def test_non_change_is_rejected() -> None:
    with pytest.raises(StoreValidationError, match="AccountStateChange"):
        await InMemoryAccountStateStore().commit("change")  # type: ignore[arg-type]


# --- failure injection ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_definite_failure_leaves_the_old_state() -> None:
    store = await seeded()
    before = as_tuple(await loaded(store))
    store.inject_commit_failure(CommitFailure.DEFINITE)

    with pytest.raises(StoreCommitError):
        await store.commit(change(1, position_writes=(known("BTCUSDT", "3"),)))

    assert as_tuple(await loaded(store)) == before
    await store.commit(change(1, position_writes=(known("BTCUSDT", "3"),)))  # one-shot
    assert (await loaded(store)).revision == 2


@pytest.mark.asyncio
async def test_uncertain_failure_applies_the_whole_change_then_raises() -> None:
    store = await seeded()
    store.inject_commit_failure(CommitFailure.UNCERTAIN)
    writes = change(1, position_writes=(known("BTCUSDT", "3"),), **reservation("i-2", "c-2"))

    with pytest.raises(StoreUncertainError):
        await store.commit(writes)

    state = await loaded(store)
    assert state.revision == 2
    assert state.positions["BTCUSDT"] == known("BTCUSDT", "3")
    assert set(state.orders) == {"c-1", "c-2"}

    # A blind retry of a risk-relevant change cannot apply twice: the CAS fails.
    with pytest.raises(StoreConflictError, match="stored revision 2"):
        await store.commit(writes)


@pytest.mark.asyncio
async def test_retry_of_a_same_revision_write_after_uncertain_is_idempotent() -> None:
    store = await seeded()
    store.inject_commit_failure(CommitFailure.UNCERTAIN)
    writes = change(1, 1, placement_writes=(rejected(intent("i-2")),))

    with pytest.raises(StoreUncertainError):
        await store.commit(writes)
    await store.commit(writes)  # identical rewrite: no-op

    state = await loaded(store)
    assert state.revision == 1
    assert set(state.placements) == {"i-1", "i-2"}


@pytest.mark.parametrize("failure", list(CommitFailure))
@pytest.mark.asyncio
async def test_validation_and_conflicts_take_precedence_over_injected_failures(
    failure: CommitFailure,
) -> None:
    store = await seeded()
    before = as_tuple(await loaded(store))
    store.inject_commit_failure(failure)

    with pytest.raises(StoreValidationError):
        await store.commit(change(1, fill_writes=(fill(cid="c-9"),)))
    with pytest.raises(StoreConflictError):
        await store.commit(change(7, position_writes=(known("BTCUSDT", "1"),)))
    assert as_tuple(await loaded(store)) == before

    # The injection is still armed for the next valid commit.
    error = StoreCommitError if failure is CommitFailure.DEFINITE else StoreUncertainError
    with pytest.raises(error):
        await store.commit(change(1, position_writes=(known("BTCUSDT", "1"),)))


def test_injection_requires_a_commit_failure() -> None:
    with pytest.raises(StoreValidationError):
        InMemoryAccountStateStore().inject_commit_failure("definite")  # type: ignore[arg-type]


# --- concurrency ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_same_revision_independent_writes_both_survive() -> None:
    store = await seeded()

    await asyncio.gather(
        store.commit(change(1, 1, placement_writes=(rejected(intent("i-a")),))),
        store.commit(change(1, 1, placement_writes=(rejected(intent("i-b")),))),
    )

    state = await loaded(store)
    assert state.revision == 1
    assert {"i-a", "i-b"} <= set(state.placements)


@pytest.mark.asyncio
async def test_concurrent_same_revision_identity_conflict_is_deterministic() -> None:
    store = await seeded()

    results = await asyncio.gather(
        store.commit(change(1, 1, placement_writes=(rejected(intent("i-x")),))),
        store.commit(change(1, 1, placement_writes=(rejected(intent("i-x", qty=D("2"))),))),
        return_exceptions=True,
    )

    assert results[0] is None  # the lock serializes in call order
    assert isinstance(results[1], StoreConflictError)
    assert (await loaded(store)).placements["i-x"].intent.qty == D("10")


@pytest.mark.asyncio
async def test_concurrent_risk_relevant_commits_on_one_revision_cannot_both_apply() -> None:
    store = await seeded()

    results = await asyncio.gather(
        store.commit(change(1, position_writes=(known("BTCUSDT", "1"),))),
        store.commit(change(1, position_writes=(known("BTCUSDT", "2"),))),
        return_exceptions=True,
    )

    assert results[0] is None
    assert isinstance(results[1], StoreConflictError)
    state = await loaded(store)
    assert (state.revision, state.positions["BTCUSDT"]) == (2, known("BTCUSDT", "1"))


@pytest.mark.asyncio
async def test_accounts_commit_independently_and_concurrently() -> None:
    store = InMemoryAccountStateStore()

    await asyncio.gather(
        *(store.commit(change(0, scope=f"a-{n}", **reservation("i-1", "c-1"))) for n in range(5))
    )

    for n in range(5):
        assert (await loaded(store, f"a-{n}")).revision == 1


# --- Decimal preservation -------------------------------------------------------------------


@pytest.mark.parametrize("value", ["4.000", "-0", "1E-200", "-1." + "7" * 100])
@pytest.mark.asyncio
async def test_exact_decimal_objects_are_preserved(value: str) -> None:
    store = InMemoryAccountStateStore()

    await store.commit(change(0, position_writes=(known("BTCUSDT", value),)))

    qty = (await loaded(store)).positions["BTCUSDT"].qty
    assert type(qty) is Decimal
    assert qty is not None
    assert qty.as_tuple() == D(value).as_tuple()


# --- dependencies ---------------------------------------------------------------------------


def _imports(module: Any) -> set[str]:
    source = Path(module.__file__).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_store_modules_have_no_driver_codec_or_runtime_dependencies() -> None:
    for module in (models_module, protocols_module, memory_module):
        for name in _imports(module):
            assert not name.startswith(
                ("app.exchanges", "app.services", "app.risk.manager", "app.config")
            ), name
            assert name.split(".")[0] not in {"sqlalchemy", "sqlite3", "random", "time"}, name
        assert "app.persistence.codecs" not in _imports(module)  # codecs belong to a DB adapter
    assert "app.persistence.memory" not in _imports(protocols_module)
