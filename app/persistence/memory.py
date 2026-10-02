"""Deterministic in-memory ``AccountStateStore``: the reference adapter and test
fake of the execution-owned persistence port (``app.execution.persistence``).
NOT durable storage (lost with the process).

It models the future database constraints within each ``account_scope_id``:
unique ``intent_id`` (placements), ``client_order_id`` (orders, notionals,
approved placements and safety blocks), ``exec_id`` (fills),
``exchange_order_id`` (when known), ``symbol`` (positions). A commit is
validated completely first and then applied as a whole, or not at all:

1. revision CAS: ``expected_revision`` must equal the stored revision (0 for an
   account never committed), else ``StoreConflictError``;
2. identities: an identical rewrite is a no-op; the same identity with different
   data is a ``StoreConflictError`` (placements, fills, safety blocks); an
   ``Order`` is replaced only by a higher ``version`` (same version: identical ->
   no-op, different -> conflict; lower -> conflict); positions and notionals
   are projections replaced inside the successful CAS commit. "Identical" means equal AND equally
   represented (``Decimal("4")`` and ``Decimal("4.0")`` are different payloads);
3. references of the resulting state: an approved placement's order exists and
   carries the intent's terms; a fill belongs to an existing order (same symbol
   and side, compatible exchange id); every order has exactly one notional, zero
   exactly when nothing is filled; a dangling reference is a
   ``StoreValidationError``. These reference invariants are execution-owned
   (``app.execution.state_invariants``): this adapter reuses them, startup
   hydration applies the same checks to a loaded snapshot.

Transition legality stays with the domain / account layer: the store never
replays the state machine and never computes values.

Commits are serialized by an internal lock. That only makes this implementation
atomic; it is not a multi-writer coordination mechanism (the account aggregate
serializes its own mutations).

Failure injection (deterministic, for tests): ``inject_commit_failure`` arms the
NEXT commit that passes validation. ``DEFINITE`` raises ``StoreCommitError`` before
anything is written; ``UNCERTAIN`` applies the whole change and then raises
``StoreUncertainError`` (a later ``load`` shows the new state). Validation and
conflict errors always take precedence and do not consume the injection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TypeVar

from app.domain.fills import Fill
from app.domain.orders import Order
from app.execution.models import PlacementRecord, SafetyBlockRecord
from app.execution.persistence import (
    AccountStateChange,
    PersistedAccountState,
    PersistedPosition,
    StoreCommitError,
    StoreConflictError,
    StoreUncertainError,
    StoreValidationError,
)
from app.execution.state_invariants import check_account_references, identical

_T = TypeVar("_T")


class CommitFailure(Enum):
    """A commit failure to inject into the next commit (tests only)."""

    DEFINITE = "definite"
    UNCERTAIN = "uncertain"


@dataclass(slots=True, kw_only=True)
class _Account:
    revision: int = 0
    placements: dict[str, PlacementRecord] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    fills: dict[str, Fill] = field(default_factory=dict)
    positions: dict[str, PersistedPosition] = field(default_factory=dict)
    notionals: dict[str, Decimal] = field(default_factory=dict)
    safety_blocks: dict[str, SafetyBlockRecord] = field(default_factory=dict)

    def copy(self) -> _Account:
        return _Account(
            revision=self.revision,
            placements=dict(self.placements),
            orders=dict(self.orders),
            fills=dict(self.fills),
            positions=dict(self.positions),
            notionals=dict(self.notionals),
            safety_blocks=dict(self.safety_blocks),
        )


def _no_duplicates(items: tuple[_T, ...], identity: str, label: str) -> None:
    seen: set[object] = set()
    for item in items:
        key = getattr(item, identity)
        if key in seen:
            raise StoreValidationError(f"{label} {key!r} appears more than once in the change")
        seen.add(key)


def _write_immutable(store: dict[str, _T], key: str, item: _T, label: str) -> None:
    existing = store.get(key)
    if existing is None:
        store[key] = item
    elif not identical(existing, item):
        raise StoreConflictError(f"{label} {key!r} is already stored with different data")


def _write_order(orders: dict[str, Order], order: Order) -> None:
    existing = orders.get(order.client_order_id)
    if existing is None or order.version > existing.version:
        orders[order.client_order_id] = order
    elif order.version < existing.version:
        raise StoreConflictError(
            f"order {order.client_order_id} version {order.version} is older than the "
            f"stored version {existing.version}"
        )
    elif not identical(existing, order):
        raise StoreConflictError(
            f"order {order.client_order_id} version {order.version} is already stored "
            "with different data"
        )


def _prepare(current: _Account | None, change: AccountStateChange) -> _Account:
    """The account after ``change``, fully validated; ``current`` is not modified."""
    stored_revision = 0 if current is None else current.revision
    if change.expected_revision != stored_revision:
        raise StoreConflictError(
            f"account {change.account_scope_id}: expected revision {change.expected_revision}, "
            f"stored revision {stored_revision}"
        )
    _no_duplicates(change.placement_writes, "intent_id", "intent_id")
    _no_duplicates(change.order_writes, "client_order_id", "order")
    _no_duplicates(change.fill_writes, "exec_id", "exec_id")
    _no_duplicates(change.position_writes, "symbol", "position")
    _no_duplicates(change.notional_writes, "client_order_id", "notional")
    _no_duplicates(change.safety_block_writes, "client_order_id", "safety block")
    account = _Account() if current is None else current.copy()
    for record in change.placement_writes:
        _write_immutable(account.placements, record.intent_id, record, "intent_id")
    for order in change.order_writes:
        _write_order(account.orders, order)
    for fill in change.fill_writes:
        _write_immutable(account.fills, fill.exec_id, fill, "exec_id")
    for position in change.position_writes:
        account.positions[position.symbol] = position
    for notional in change.notional_writes:
        account.notionals[notional.client_order_id] = notional.filled_notional
    for block in change.safety_block_writes:
        _write_immutable(account.safety_blocks, block.client_order_id, block, "safety block")
    check_account_references(
        placements=account.placements,
        orders=account.orders,
        fills=account.fills,
        notionals=account.notionals,
        safety_blocks=account.safety_blocks,
    )
    account.revision = change.new_revision
    return account


class InMemoryAccountStateStore:
    """In-memory reference ``AccountStateStore`` for several account scopes."""

    __slots__ = ("_accounts", "_failure", "_lock")

    def __init__(self) -> None:
        self._accounts: dict[str, _Account] = {}
        self._failure: CommitFailure | None = None
        self._lock = asyncio.Lock()

    def inject_commit_failure(self, failure: CommitFailure) -> None:
        """Make the next commit that passes validation fail as ``failure`` (tests)."""
        if type(failure) is not CommitFailure:
            raise StoreValidationError("failure must be a CommitFailure")
        self._failure = failure

    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        async with self._lock:
            account = self._accounts.get(account_scope_id)
            if account is None:
                return None
            return PersistedAccountState(
                account_scope_id=account_scope_id,
                revision=account.revision,
                placements=account.placements,
                orders=account.orders,
                fills=account.fills,
                positions=account.positions,
                notionals=account.notionals,
                safety_blocks=account.safety_blocks,
            )

    async def commit(self, change: AccountStateChange) -> None:
        if type(change) is not AccountStateChange:
            raise StoreValidationError("change must be an AccountStateChange")
        async with self._lock:
            prepared = _prepare(self._accounts.get(change.account_scope_id), change)
            await asyncio.sleep(0)  # the commit I/O point of a real database
            failure, self._failure = self._failure, None
            if failure is CommitFailure.DEFINITE:
                raise StoreCommitError(
                    f"account {change.account_scope_id}: commit failed, nothing written"
                )
            self._accounts[change.account_scope_id] = prepared
            if failure is CommitFailure.UNCERTAIN:
                raise StoreUncertainError(
                    f"account {change.account_scope_id}: commit outcome unknown"
                )
