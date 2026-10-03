"""Authoritative open-order discovery (docs/ARCHITECTURE.md 13.7, 13.10).

One attempt::

    poison check -> reader.list_open_orders() (no account lock held)
    -> account lock: poison check -> classify_open_orders(current local orders,
       snapshot) -> only if the WHOLE classification is clean: complete the
       proven exchange order ids in ONE change -> classify again (final)

The reader contract (``ExchangeStateReader.list_open_orders``) returns the
COMPLETE open-order set of its scope or raises; ``OpenOrdersSnapshot`` has no
partial form, so there is no completeness flag to check here.

The classification is the pure matcher (``app.execution.recovery_matching``),
always against the CURRENT local orders under the account lock, never against
a local view from before the network read. Findings are results, not errors:

* foreign, lost-managed, identity conflicts (malformed / unsupported ids,
  duplicates, exchange-id mismatch, reverse collisions, terms mismatches) and
  missing local active orders make the result BLOCKED. Nothing is imported,
  canceled, created or transitioned; a missing local order is only "absent
  from this snapshot", never terminal;
* with a clean classification the only correction is the identity completion
  of managed matches whose local order does not know its exchange id yet
  (``exchange_id_completion_required``): all of them in ONE durable change,
  all or nothing (``complete_exchange_order_ids``). A blocked classification
  completes nothing, not even an otherwise provable id.

``clean`` means only "no blocking discovery finding against this snapshot". It
changes no safety gate and does not prove that the exchange stayed unchanged
after the snapshot (the recovery coordinator's final verification). No clock,
no id generation, no order action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.domain.errors import DomainValidationError
from app.domain.orders import Order
from app.exchanges.protocols import ExchangeStateReader
from app.exchanges.recovery import OpenOrdersSnapshot
from app.execution.account_state import (
    AccountStateError,
    ExchangeOrderIdCompletion,
    InMemoryAccountState,
)
from app.execution.client_order_id import ClientOrderNamespace
from app.execution.recovery_matching import OpenOrderClassification, classify_open_orders


class OpenOrderDiscoveryError(AccountStateError):
    """The reader broke its contract (no ``OpenOrdersSnapshot``); nothing changed."""


class OpenOrderDiscoveryInvariantError(AccountStateError):
    """Internal invariant broken: after the identity completion the same snapshot
    does not classify clean. A programming error, never a recovery finding."""


class OpenOrderDiscoveryOutcome(StrEnum):
    CLEAN = "clean"
    """Clean classification, nothing to correct."""
    CORRECTED = "corrected"
    """Clean classification; missing exchange order ids were completed."""
    BLOCKED = "blocked"
    """At least one blocking finding; nothing was changed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenOrderDiscoveryResult:
    snapshot: OpenOrdersSnapshot
    classification: OpenOrderClassification
    """The final classification (after the completion, if any)."""
    exchange_ids_completed: tuple[Order, ...]
    """The orders whose exchange id was recorded, sorted by client id."""
    revision_before: int
    """The revision the snapshot was classified against (under the lock)."""
    revision_after: int
    outcome: OpenOrderDiscoveryOutcome

    @property
    def clean(self) -> bool:
        """No blocking discovery finding against this snapshot (not a gate)."""
        return self.outcome is not OpenOrderDiscoveryOutcome.BLOCKED


async def discover_open_orders(
    *,
    account_state: InMemoryAccountState,
    reader: ExchangeStateReader,
    namespace: ClientOrderNamespace,
) -> OpenOrderDiscoveryResult:
    """Read one authoritative open-order snapshot and reconcile the local order
    identities with it (see the module doc).

    Raises:
        AccountStatePoisonedError: before the read, or under the lock after it.
        OpenOrderDiscoveryError: the reader returned no ``OpenOrdersSnapshot``.
        OpenOrderDiscoveryInvariantError: internal invariant broken.
        Exception: whatever the reader or the store raises (reader errors leave
            the account untouched and do not poison it).
    """
    if type(account_state) is not InMemoryAccountState:
        raise DomainValidationError("account_state must be an InMemoryAccountState")
    if not callable(getattr(reader, "list_open_orders", None)):
        raise DomainValidationError("reader must provide list_open_orders()")
    if type(namespace) is not ClientOrderNamespace:
        raise DomainValidationError("namespace must be a ClientOrderNamespace")
    async with account_state.account_lock() as locked:
        locked.ensure_mutations_allowed()  # no pointless read for a poisoned account
    snapshot: object = await reader.list_open_orders()
    if type(snapshot) is not OpenOrdersSnapshot:
        raise OpenOrderDiscoveryError(
            f"reader returned {type(snapshot).__name__}, not an OpenOrdersSnapshot"
        )
    async with account_state.account_lock() as locked:
        locked.ensure_mutations_allowed()
        revision_before = locked.revision
        classification = classify_open_orders(
            local_orders=locked.orders(), exchange_orders=snapshot.orders, namespace=namespace
        )
        if not classification.is_clean:
            outcome = OpenOrderDiscoveryOutcome.BLOCKED
            completed: tuple[Order, ...] = ()
        else:
            completions = tuple(
                sorted(
                    (
                        ExchangeOrderIdCompletion(
                            client_order_id=match.local_order.client_order_id,
                            exchange_order_id=match.exchange_order.exchange_order_id,
                        )
                        for match in classification.managed
                        if match.exchange_id_completion_required
                    ),
                    key=lambda c: (c.client_order_id, c.exchange_order_id),
                )
            )
            completed = ()
            if completions:
                completed = await locked.complete_exchange_order_ids(
                    completions, expected_revision=revision_before
                )
                classification = classify_open_orders(
                    local_orders=locked.orders(),
                    exchange_orders=snapshot.orders,
                    namespace=namespace,
                )
                if not classification.is_clean or any(
                    match.exchange_id_completion_required for match in classification.managed
                ):
                    raise OpenOrderDiscoveryInvariantError(
                        "the snapshot does not classify clean after the exchange id completion"
                    )
            outcome = (
                OpenOrderDiscoveryOutcome.CORRECTED
                if completed
                else OpenOrderDiscoveryOutcome.CLEAN
            )
        return OpenOrderDiscoveryResult(
            snapshot=snapshot,
            classification=classification,
            exchange_ids_completed=completed,
            revision_before=revision_before,
            revision_after=locked.revision,
            outcome=outcome,
        )
