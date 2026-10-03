"""The account-state persistence port, owned by the execution layer.

The execution / account-state layer defines what it needs from durable storage;
storage adapters implement it (dependency inversion, docs/ARCHITECTURE.md 11.0)::

    domain <- execution (this port) <- app.persistence adapters (memory, later SQL)

``app.execution`` never imports ``app.persistence``. The port uses the standard
library, the domain and execution's own models only.

* ``AccountStateStore``: one transactional boundary for the whole risk-relevant
  account aggregate (not per-table repositories). NOT a multi-writer
  coordination mechanism: the account aggregate serializes its mutations (one
  writer per account scope in V1); ``expected_revision`` is a compare-and-set
  guard against stale state.
* ``PersistedPosition``: an explicit position marker. ``known=False`` -> ``qty``
  is None; ``known=True`` -> ``qty`` is an exact finite ``Decimal`` (0 included).
  An unknown position is never encoded by absence.
* ``PersistedOrderNotional``: the exact accumulated fill notional of an order
  (``sum(price * qty)`` of its fills): an exact, finite ``Decimal`` >= 0.
* ``AccountStateChange``: one atomic change set, guarded by ``expected_revision``.
* ``SafetyBlockRecord`` (execution model): the durable reason of an order FAILED
  by the local safety gate (definitely not sent), immutable per order.
* ``PositionBaselineRecord`` (execution model): the append-only audit record of
  an explicit position baseline acceptance, immutable per ``baseline_id``. A
  change writing one must, in the same change, write the known durable position
  it accepted and advance the revision to the record's ``account_revision``.
* ``PersistedAccountState``: an immutable snapshot of one account's durable state.
* Errors of the port: ``PersistenceStoreError`` and its four outcomes, so callers
  handle them without importing an adapter.

Values are kept as the exact objects given: no arithmetic, rounding or encoding
(codecs belong to a physical database adapter). ``PlacementRecord``, ``Order``
and ``Fill`` are reused as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol, TypeVar

from app.domain.fills import Fill
from app.domain.orders import Order
from app.execution.models import PlacementRecord, PositionBaselineRecord, SafetyBlockRecord


class PersistenceStoreError(Exception):
    """Base class of account state store errors."""


class StoreValidationError(PersistenceStoreError):
    """The change set is malformed (types, revisions, references); nothing was
    written. A caller bug, never a storage condition."""


class StoreConflictError(PersistenceStoreError):
    """The change contradicts the durable state (stale ``expected_revision``, or
    an identity already stored with different data); nothing was written."""


class StoreCommitError(PersistenceStoreError):
    """The commit definitely did not happen: the durable state is unchanged."""


class StoreUncertainError(PersistenceStoreError):
    """The outcome of the commit is unknown (it may have been applied in full).
    The caller must treat its in-memory state as unusable until it reloads."""


_T = TypeVar("_T")


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise StoreValidationError(f"{field} must be non-empty text without surrounding spaces")
    return value


def _revision(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise StoreValidationError(f"{field} must be an int >= 0, got {value!r}")
    return value


def _exact_decimal(value: object, field: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise StoreValidationError(f"{field} must be an exact, finite Decimal, got {value!r}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PersistedPosition:
    """Signed net position of one symbol, with an explicit known / unknown marker."""

    symbol: str
    known: bool
    qty: Decimal | None

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        if type(self.known) is not bool:
            raise StoreValidationError("known must be a bool")
        if self.known:
            _exact_decimal(self.qty, "qty of a known position")
        elif self.qty is not None:
            raise StoreValidationError("an unknown position has no qty")


@dataclass(frozen=True, slots=True, kw_only=True)
class PersistedOrderNotional:
    """Exact accumulated fill notional of one order (>= 0)."""

    client_order_id: str
    filled_notional: Decimal

    def __post_init__(self) -> None:
        _text(self.client_order_id, "client_order_id")
        if _exact_decimal(self.filled_notional, "filled_notional") < 0:
            raise StoreValidationError("filled_notional must be >= 0")


def _typed_tuple(value: object, kind: type[_T], field: str) -> tuple[_T, ...]:
    if type(value) is not tuple or not all(type(item) is kind for item in value):
        raise StoreValidationError(f"{field} must be a tuple of {kind.__name__}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountStateChange:
    """One atomic change of an account; applied only if the durable revision is
    ``expected_revision``. ``new_revision`` is ``expected_revision + 1`` for a
    risk-relevant change, or ``expected_revision`` for a change that does not
    touch risk-relevant state (only rejected placements)."""

    account_scope_id: str
    expected_revision: int
    new_revision: int
    placement_writes: tuple[PlacementRecord, ...] = ()
    order_writes: tuple[Order, ...] = ()
    fill_writes: tuple[Fill, ...] = ()
    position_writes: tuple[PersistedPosition, ...] = ()
    notional_writes: tuple[PersistedOrderNotional, ...] = ()
    safety_block_writes: tuple[SafetyBlockRecord, ...] = ()
    """Immutable reasons of safety-blocked (definitely unsent) FAILED orders."""
    position_baseline_writes: tuple[PositionBaselineRecord, ...] = ()
    """Append-only audit records of explicitly accepted position baselines."""

    def __post_init__(self) -> None:
        _text(self.account_scope_id, "account_scope_id")
        expected = _revision(self.expected_revision, "expected_revision")
        new = _revision(self.new_revision, "new_revision")
        if new not in (expected, expected + 1):
            raise StoreValidationError(
                f"new_revision must be expected_revision or expected_revision + 1, "
                f"got {expected} -> {new}"
            )
        placements = _typed_tuple(self.placement_writes, PlacementRecord, "placement_writes")
        _typed_tuple(self.order_writes, Order, "order_writes")
        _typed_tuple(self.fill_writes, Fill, "fill_writes")
        _typed_tuple(self.position_writes, PersistedPosition, "position_writes")
        _typed_tuple(self.notional_writes, PersistedOrderNotional, "notional_writes")
        _typed_tuple(self.safety_block_writes, SafetyBlockRecord, "safety_block_writes")
        baselines = _typed_tuple(
            self.position_baseline_writes, PositionBaselineRecord, "position_baseline_writes"
        )
        if new == expected and (
            self.order_writes
            or self.safety_block_writes
            or baselines
            or self.fill_writes
            or self.position_writes
            or self.notional_writes
            or any(record.approved for record in placements)
        ):
            raise StoreValidationError(
                "a change that keeps the revision may only write rejected placements"
            )
        positions = {position.symbol: position for position in self.position_writes}
        for record in baselines:
            if record.account_revision != new:
                raise StoreValidationError(
                    f"position baseline {record.baseline_id}: account_revision "
                    f"{record.account_revision} is not the new revision {new}"
                )
            accepted = PersistedPosition(symbol=record.symbol, known=True, qty=record.qty)
            written = positions.get(record.symbol)
            if written != accepted or repr(written) != repr(accepted):
                raise StoreValidationError(
                    f"position baseline {record.baseline_id} without its known position "
                    f"{record.symbol} = {record.qty} in the same change"
                )


def _keyed(item: object, kind: type, key: str, identity: object, name: str) -> None:
    if type(item) is not kind or identity != key:
        raise StoreValidationError(f"{name}[{key!r}] does not match its key")


def _frozen(items: Mapping[str, _T]) -> Mapping[str, _T]:
    return MappingProxyType(dict(items))


@dataclass(frozen=True, slots=True, kw_only=True)
class PersistedAccountState:
    """Immutable snapshot of one account's durable state (read-only mappings)."""

    account_scope_id: str
    revision: int
    placements: Mapping[str, PlacementRecord]
    """By intent_id."""
    orders: Mapping[str, Order]
    """By client_order_id."""
    fills: Mapping[str, Fill]
    """By exec_id."""
    positions: Mapping[str, PersistedPosition]
    """By symbol."""
    notionals: Mapping[str, Decimal]
    """Exact accumulated fill notional by client_order_id."""
    safety_blocks: Mapping[str, SafetyBlockRecord] = field(default_factory=dict)
    """Why an order was FAILED by the safety gate, by client_order_id."""
    position_baselines: Mapping[str, PositionBaselineRecord] = field(default_factory=dict)
    """Audit records of accepted position baselines, by baseline_id."""

    def __post_init__(self) -> None:
        _text(self.account_scope_id, "account_scope_id")
        _revision(self.revision, "revision")
        for key, record in self.placements.items():
            _keyed(record, PlacementRecord, key, getattr(record, "intent_id", None), "placements")
        for key, order in self.orders.items():
            _keyed(order, Order, key, getattr(order, "client_order_id", None), "orders")
        for key, fill in self.fills.items():
            _keyed(fill, Fill, key, getattr(fill, "exec_id", None), "fills")
        for key, position in self.positions.items():
            _keyed(position, PersistedPosition, key, getattr(position, "symbol", None), "positions")
        for key, notional in self.notionals.items():
            _exact_decimal(notional, f"notionals[{key!r}]")
        for key, block in self.safety_blocks.items():
            _keyed(
                block,
                SafetyBlockRecord,
                key,
                getattr(block, "client_order_id", None),
                "safety_blocks",
            )
        for key, baseline in self.position_baselines.items():
            _keyed(
                baseline,
                PositionBaselineRecord,
                key,
                getattr(baseline, "baseline_id", None),
                "position_baselines",
            )
            if baseline.account_revision > self.revision:
                raise StoreValidationError(
                    f"position baseline {key} is from revision {baseline.account_revision}, "
                    f"after the snapshot revision {self.revision}"
                )
        # Defensive read-only copies: later changes to the given mappings are not seen.
        for name in (
            "placements",
            "orders",
            "fills",
            "positions",
            "notionals",
            "safety_blocks",
            "position_baselines",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


class AccountStateStore(Protocol):
    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        """The committed state of the account, or None if nothing was ever committed."""
        ...

    async def commit(self, change: AccountStateChange) -> None:
        """Apply the whole change atomically, or nothing.

        Raises:
            StoreValidationError: malformed change (checked before anything else).
            StoreConflictError: stale ``expected_revision`` or an identity already
                stored with different data.
            StoreCommitError: the commit definitely did not happen.
            StoreUncertainError: the outcome is unknown; it may be fully applied.
        """
        ...
