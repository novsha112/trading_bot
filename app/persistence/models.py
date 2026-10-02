"""Persisted representation of the risk-relevant account aggregate.

Reuses the domain and execution models as they are (``PlacementRecord``,
``Order``, ``Fill``); only what has no model yet is defined here:

* ``PersistedPosition``: an explicit position marker. ``known=False`` -> ``qty``
  is None; ``known=True`` -> ``qty`` is an exact finite ``Decimal`` (0 included).
  An unknown position is never encoded by absence.
* ``PersistedOrderNotional``: the exact accumulated fill notional of an order
  (``sum(price * qty)`` of its fills): an exact, finite ``Decimal`` >= 0.
* ``AccountStateChange``: one atomic change set, guarded by ``expected_revision``.
* ``PersistedAccountState``: an immutable snapshot of one account's durable state.

Values are kept as the exact objects given: no arithmetic, rounding or encoding
(codecs belong to a physical database adapter).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import TypeVar

from app.domain.fills import Fill
from app.domain.orders import Order
from app.execution.models import PlacementRecord
from app.persistence.errors import StoreValidationError

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
        if new == expected and (
            self.order_writes
            or self.fill_writes
            or self.position_writes
            or self.notional_writes
            or any(record.approved for record in placements)
        ):
            raise StoreValidationError(
                "a change that keeps the revision may only write rejected placements"
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
        # Defensive read-only copies: later changes to the given mappings are not seen.
        for name in ("placements", "orders", "fills", "positions", "notionals"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
