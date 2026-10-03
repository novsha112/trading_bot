"""Pure classification of an exchange open-order set against the local orders
(docs/ARCHITECTURE.md 13.7). Snapshot classification only: no exchange read, no
state change, no reconciliation of statuses, fills or positions.

``classify_open_orders(local_orders=..., exchange_orders=..., namespace=...)``
puts every exchange order into exactly ONE bucket and every relevant local
order into at most one (managed, missing, or a conflict):

* ``managed``: the exchange client id is ours (``ClientOrderNamespace``), a
  relevant local order has the same client id, a known local exchange id is
  equal, no other order claims either identity, and the order terms are equal.
  ``exchange_id_completion_required`` marks a local order whose exchange id is
  not known yet (identity completion, not a conflict).
* ``foreign``: no client id, or an id outside our namespace (unmanaged, or a
  valid id of another namespace), with no local claim on either identity.
  Foreign orders are never matched by terms, symbol or exchange id.
* ``lost_managed``: our valid client id with no relevant local order (nothing
  imported, nothing invented; a non-relevant local order of that id is reported
  alongside).
* ``identity_conflicts``: everything that must fail closed, with a structured
  reason (``IdentityConflictReason``): malformed / unsupported managed ids
  (exchange or local), local orders without our managed id (legacy, other
  namespace), duplicates on either side, exchange id mismatch, an exchange id
  claimed by another local order (checked in both directions, against every
  local order), and terms mismatches (with the exact fields).
* ``missing_local_active``: relevant local orders absent from the snapshot. This
  says nothing about their state (not FAILED, not CANCELED, not "not found"); a
  targeted lookup decides later.

Relevant local statuses: UNKNOWN, OPEN, PARTIALLY_FILLED, CANCELING (after
hydration NEW and SUBMITTING no longer exist; terminal orders are final). Status
differences (e.g. local UNKNOWN or CANCELING vs exchange OPEN) are NOT
conflicts: status, cumulative execution, average price and timestamps belong to
state reconciliation. Terms are compared exactly (``Decimal`` equality, no
epsilon, no quantization): symbol, side, order type, qty, price,
time-in-force, reduce-only.

The result does not depend on the input order (canonical sorting). Inputs are
validated cheaply on their own (the API does not rely on an
``OpenOrdersSnapshot`` having done it); only wrong types raise.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError
from app.domain.orders import Order
from app.exchanges.recovery import ExchangeOrder
from app.execution.client_order_id import (
    ClientOrderIdOwnership,
    ClientOrderNamespace,
    ParsedClientOrderId,
    classify_client_order_id,
    parse_client_order_id,
)

RELEVANT_LOCAL_STATUSES: Final = frozenset(
    {
        OrderStatus.UNKNOWN,
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCELING,
    }
)


class OrderTerm(StrEnum):
    """Immutable order terms, in the fixed comparison order."""

    SYMBOL = "symbol"
    SIDE = "side"
    ORDER_TYPE = "order_type"
    QTY = "qty"
    PRICE = "price"
    TIME_IN_FORCE = "time_in_force"
    REDUCE_ONLY = "reduce_only"


class IdentityConflictReason(StrEnum):
    """Why an identity cannot be established; always fail closed. The order of
    the members is the canonical order of conflicts in a result."""

    DUPLICATE_LOCAL_CLIENT_ID = "duplicate_local_client_id"
    DUPLICATE_LOCAL_EXCHANGE_ID = "duplicate_local_exchange_id"
    DUPLICATE_EXCHANGE_CLIENT_ID = "duplicate_exchange_client_id"
    DUPLICATE_EXCHANGE_ORDER_ID = "duplicate_exchange_order_id"
    LOCAL_UNMANAGED_ID = "local_unmanaged_id"
    """A relevant local order whose id is not ours (legacy, other namespace)."""
    LOCAL_MALFORMED_MANAGED_ID = "local_malformed_managed_id"
    LOCAL_UNSUPPORTED_MANAGED_VERSION = "local_unsupported_managed_version"
    MALFORMED_MANAGED_ID = "malformed_managed_id"
    UNSUPPORTED_MANAGED_VERSION = "unsupported_managed_version"
    EXCHANGE_ID_MISMATCH = "exchange_id_mismatch"
    REVERSE_EXCHANGE_ID_COLLISION = "reverse_exchange_id_collision"
    TERMS_MISMATCH = "terms_mismatch"


_REASON_RANK: Final = {reason: rank for rank, reason in enumerate(IdentityConflictReason)}
_LOCAL_ID_REASON: Final = {
    ClientOrderIdOwnership.OTHER: IdentityConflictReason.LOCAL_UNMANAGED_ID,
    ClientOrderIdOwnership.ABSENT: IdentityConflictReason.LOCAL_UNMANAGED_ID,
    ClientOrderIdOwnership.MALFORMED_MANAGED: IdentityConflictReason.LOCAL_MALFORMED_MANAGED_ID,
    ClientOrderIdOwnership.UNSUPPORTED_MANAGED_VERSION: (
        IdentityConflictReason.LOCAL_UNSUPPORTED_MANAGED_VERSION
    ),
}
_EXCHANGE_ID_REASON: Final = {
    ClientOrderIdOwnership.MALFORMED_MANAGED: IdentityConflictReason.MALFORMED_MANAGED_ID,
    ClientOrderIdOwnership.UNSUPPORTED_MANAGED_VERSION: (
        IdentityConflictReason.UNSUPPORTED_MANAGED_VERSION
    ),
}


def order_terms_mismatch(local: Order, exchange: ExchangeOrder) -> tuple[OrderTerm, ...]:
    """The terms that differ, in ``OrderTerm`` order (empty: equal). Exact
    equality; status, execution and timestamps are not terms."""
    if type(local) is not Order or type(exchange) is not ExchangeOrder:
        raise DomainValidationError("expected a domain Order and an ExchangeOrder")
    return tuple(term for term in OrderTerm if getattr(local, term) != getattr(exchange, term))


@dataclass(frozen=True, slots=True, kw_only=True)
class ManagedOrderMatch:
    local_order: Order
    exchange_order: ExchangeOrder
    exchange_id_completion_required: bool
    """The local order does not know its exchange id yet (to be recorded later)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LostManagedOrder:
    exchange_order: ExchangeOrder
    identity: ParsedClientOrderId
    local_order: Order | None
    """A NON-relevant local order with this client id, if any (e.g. locally
    terminal while open on the exchange); never a relevant one."""


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityConflict:
    reason: IdentityConflictReason
    client_order_ids: tuple[str, ...]
    """Every client id involved (local and exchange), sorted."""
    exchange_order_ids: tuple[str, ...]
    """Every exchange id involved (local and exchange), sorted."""
    mismatched_terms: tuple[OrderTerm, ...] = ()
    """For TERMS_MISMATCH: the differing terms in ``OrderTerm`` order."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenOrderClassification:
    """Mutually exclusive buckets, each in canonical order."""

    managed: tuple[ManagedOrderMatch, ...]
    foreign: tuple[ExchangeOrder, ...]
    lost_managed: tuple[LostManagedOrder, ...]
    identity_conflicts: tuple[IdentityConflict, ...]
    missing_local_active: tuple[Order, ...]

    @property
    def is_clean(self) -> bool:
        """Only managed orders and nothing missing: no recovery block from here."""
        return not (
            self.foreign
            or self.lost_managed
            or self.identity_conflicts
            or self.missing_local_active
        )


@dataclass(slots=True)
class _Draft:
    reason: IdentityConflictReason
    client_ids: set[str] = field(default_factory=set)
    exchange_ids: set[str] = field(default_factory=set)
    terms: tuple[OrderTerm, ...] = ()

    def add_local(self, order: Order) -> None:
        self.client_ids.add(order.client_order_id)
        if order.exchange_order_id is not None:
            self.exchange_ids.add(order.exchange_order_id)

    def add_exchange(self, order: ExchangeOrder) -> None:
        if order.client_order_id is not None:
            self.client_ids.add(order.client_order_id)
        self.exchange_ids.add(order.exchange_order_id)

    def freeze(self) -> IdentityConflict:
        return IdentityConflict(
            reason=self.reason,
            client_order_ids=tuple(sorted(self.client_ids)),
            exchange_order_ids=tuple(sorted(self.exchange_ids)),
            mismatched_terms=self.terms,
        )


def _typed(value: object, kind: type, field_name: str) -> tuple[object, ...]:
    if type(value) is not tuple or not all(type(item) is kind for item in value):
        raise DomainValidationError(f"{field_name} must be a tuple of {kind.__name__}")
    return value


def _local_key(order: Order) -> tuple[str, str]:
    return (order.client_order_id, order.exchange_order_id or "")


def _exchange_key(order: ExchangeOrder) -> tuple[str, str]:
    return (order.client_order_id or "", order.exchange_order_id)


def _duplicates(keys: Iterable[str | None]) -> set[str]:
    counts = Counter(key for key in keys if key is not None)
    return {key for key, count in counts.items() if count > 1}


def classify_open_orders(
    *,
    local_orders: tuple[Order, ...],
    exchange_orders: tuple[ExchangeOrder, ...],
    namespace: ClientOrderNamespace,
) -> OpenOrderClassification:
    """Classify the exchange open-order set against the local orders (all
    statuses may be given; only the relevant ones are matched)."""
    _typed(local_orders, Order, "local_orders")
    _typed(exchange_orders, ExchangeOrder, "exchange_orders")
    if type(namespace) is not ClientOrderNamespace:
        raise DomainValidationError("namespace must be a ClientOrderNamespace")
    locals_sorted = sorted(local_orders, key=_local_key)
    exchanges_sorted = sorted(exchange_orders, key=_exchange_key)

    drafts: list[_Draft] = []
    conflicted_locals: dict[str, _Draft] = {}  # client id -> its conflict
    conflicted_exchange: set[str] = set()  # exchange order ids placed in a conflict

    def local_conflict(reason: IdentityConflictReason, orders: Iterable[Order]) -> _Draft:
        draft = _Draft(reason)
        drafts.append(draft)
        for order in orders:
            draft.add_local(order)
            conflicted_locals.setdefault(order.client_order_id, draft)
        return draft

    def exchange_conflict(
        reason: IdentityConflictReason, order: ExchangeOrder, terms: tuple[OrderTerm, ...] = ()
    ) -> _Draft:
        draft = _Draft(reason, terms=terms)
        drafts.append(draft)
        draft.add_exchange(order)
        conflicted_exchange.add(order.exchange_order_id)
        return draft

    # 1. Local identities: duplicates (all statuses), then the relevant ids' format.
    for cid in sorted(_duplicates(o.client_order_id for o in locals_sorted)):
        local_conflict(
            IdentityConflictReason.DUPLICATE_LOCAL_CLIENT_ID,
            (o for o in locals_sorted if o.client_order_id == cid),
        )
    for eid in sorted(_duplicates(o.exchange_order_id for o in locals_sorted)):
        local_conflict(
            IdentityConflictReason.DUPLICATE_LOCAL_EXCHANGE_ID,
            (o for o in locals_sorted if o.exchange_order_id == eid),
        )
    relevant = [o for o in locals_sorted if o.status in RELEVANT_LOCAL_STATUSES]
    for order in relevant:
        if order.client_order_id in conflicted_locals:
            continue
        ownership = classify_client_order_id(order.client_order_id, namespace=namespace)
        if ownership is not ClientOrderIdOwnership.OURS:
            local_conflict(_LOCAL_ID_REASON[ownership], (order,))
    by_client = {o.client_order_id: o for o in locals_sorted}  # unique unless conflicted
    relevant_by_client = {o.client_order_id: o for o in relevant}
    by_exchange_id = {o.exchange_order_id: o for o in locals_sorted if o.exchange_order_id}

    # 2. Exchange duplicates (the API does not rely on the snapshot's validation).
    for reason, duplicated, key in (
        (
            IdentityConflictReason.DUPLICATE_EXCHANGE_ORDER_ID,
            _duplicates(o.exchange_order_id for o in exchanges_sorted),
            "exchange_order_id",
        ),
        (
            IdentityConflictReason.DUPLICATE_EXCHANGE_CLIENT_ID,
            _duplicates(o.client_order_id for o in exchanges_sorted),
            "client_order_id",
        ),
    ):
        for value in sorted(duplicated):
            draft = _Draft(reason)
            drafts.append(draft)
            for ex in exchanges_sorted:
                if getattr(ex, key) == value:
                    draft.add_exchange(ex)
                    conflicted_exchange.add(ex.exchange_order_id)
                    local = relevant_by_client.get(ex.client_order_id or "")
                    if local is not None:
                        draft.add_local(local)
                        conflicted_locals.setdefault(local.client_order_id, draft)

    # 3. Every other exchange order.
    candidates: list[ManagedOrderMatch] = []
    foreign: list[ExchangeOrder] = []
    lost: list[LostManagedOrder] = []
    for ex in exchanges_sorted:
        if ex.exchange_order_id in conflicted_exchange:
            continue
        ex_cid = ex.client_order_id
        local = None if ex_cid is None else by_client.get(ex_cid)
        owner = by_exchange_id.get(ex.exchange_order_id)
        if local is not None and local.client_order_id in conflicted_locals:
            # The local order of this client id already fails closed: so does this.
            conflicted_locals[local.client_order_id].add_exchange(ex)
            conflicted_exchange.add(ex.exchange_order_id)
            continue
        if owner is not None and owner.client_order_id != ex_cid:
            # The exchange id belongs to another local order (any status).
            draft = exchange_conflict(IdentityConflictReason.REVERSE_EXCHANGE_ID_COLLISION, ex)
            draft.add_local(owner)
            conflicted_locals.setdefault(owner.client_order_id, draft)
            if local is not None:
                draft.add_local(local)
                conflicted_locals.setdefault(local.client_order_id, draft)
            continue
        ownership = classify_client_order_id(ex_cid, namespace=namespace)
        if local is not None and ownership is ClientOrderIdOwnership.OTHER:
            # Not ours, yet a local order (of any status) carries this id: a local
            # identity problem (e.g. a legacy id), never a foreign order.
            draft = exchange_conflict(IdentityConflictReason.LOCAL_UNMANAGED_ID, ex)
            draft.add_local(local)
            conflicted_locals.setdefault(local.client_order_id, draft)
            continue
        if ownership in _EXCHANGE_ID_REASON:
            exchange_conflict(_EXCHANGE_ID_REASON[ownership], ex)
            continue
        if ownership is not ClientOrderIdOwnership.OURS:  # ABSENT / OTHER
            foreign.append(ex)
            continue
        if local is None or local.status not in RELEVANT_LOCAL_STATUSES:
            parsed = parse_client_order_id(ex_cid).parsed
            if parsed is None:  # pragma: no cover - OURS always parses
                raise RuntimeError(f"managed id {ex_cid} did not parse")
            lost.append(LostManagedOrder(exchange_order=ex, identity=parsed, local_order=local))
            continue
        if local.exchange_order_id not in (None, ex.exchange_order_id):
            draft = exchange_conflict(IdentityConflictReason.EXCHANGE_ID_MISMATCH, ex)
            draft.add_local(local)
            conflicted_locals.setdefault(local.client_order_id, draft)
            continue
        terms = order_terms_mismatch(local, ex)
        if terms:
            draft = exchange_conflict(IdentityConflictReason.TERMS_MISMATCH, ex, terms)
            draft.add_local(local)
            conflicted_locals.setdefault(local.client_order_id, draft)
            continue
        candidates.append(
            ManagedOrderMatch(
                local_order=local,
                exchange_order=ex,
                exchange_id_completion_required=local.exchange_order_id is None,
            )
        )

    # 4. A candidate whose local order was drawn into a conflict meanwhile is not managed.
    managed: list[ManagedOrderMatch] = []
    for match in candidates:
        held = conflicted_locals.get(match.local_order.client_order_id)
        if held is None:
            managed.append(match)
        else:
            held.add_exchange(match.exchange_order)
            conflicted_exchange.add(match.exchange_order.exchange_order_id)

    matched = {match.local_order.client_order_id for match in managed}
    missing = tuple(
        order
        for order in relevant
        if order.client_order_id not in matched and order.client_order_id not in conflicted_locals
    )
    conflicts = sorted(
        (draft.freeze() for draft in drafts),
        key=lambda c: (_REASON_RANK[c.reason], c.client_order_ids, c.exchange_order_ids),
    )
    return OpenOrderClassification(
        managed=tuple(managed),
        foreign=tuple(foreign),
        lost_managed=tuple(lost),
        identity_conflicts=tuple(conflicts),
        missing_local_active=missing,
    )
