"""Automatic reconciliation of the local positions with a supplied exchange
``PositionSnapshot`` (docs/ARCHITECTURE.md 13.9). No reader, no network: the
snapshot was obtained elsewhere.

Known is not reconciled. A complete snapshot makes every position of the scope
KNOWN (the exchange is authoritative for the current quantity); a position is
EXPLAINED only when the local durable evidence accounts for it:

* durable projection known and equal to the exchange quantity ->
  DURABLE_PROJECTION_MATCH (explained). The projection already contains every
  committed fill, including fills recovered after a restart: fills are never
  added a second time;
* durable projection known and different -> DURABLE_PROJECTION_MISMATCH
  (unexplained; the durable projection is NOT overwritten);
* no known durable projection, exchange flat, no committed fill of the symbol ->
  UNKNOWN_FLAT_WITH_NO_EVIDENCE (explained);
* no known durable projection, exchange flat, but committed fills exist ->
  UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE (their relation to the unknown
  pre-history cannot be proven);
* no known durable projection, exchange nonzero -> UNEXPLAINED_NONZERO.

Scope (complete snapshot): every symbol of the snapshot, of the durable
projections, of the runtime positions and of the committed fills; a symbol
missing from a COMPLETE snapshot has exchange quantity 0. A partial snapshot
(``complete=False``) proves no scope: nothing is inferred, nothing is
published, every local or reported symbol is INCOMPLETE_SNAPSHOT and the result
is not reconciled.

For a complete snapshot every resulting quantity is published as the RUNTIME
position in one synchronous step (``publish_exchange_positions``): no store
commit, no revision change, durable projections untouched; a restart forgets
it (reconcile again). Mismatches and unexplained positions are ordinary
outcomes, never exceptions; they block readiness until resolved (a future
explicit baseline acceptance).

The local evidence is read, classified and published under ONE account lock
without any await, so a concurrent fill is either fully before it or after it.
The result reconciles the SUPPLIED snapshot only: it does not prove that the
exchange stayed unchanged afterwards (the recovery coordinator's final
verification). It changes no safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final

from app.domain.errors import DomainValidationError
from app.exchanges.recovery import PositionSnapshot
from app.execution.account_state import InMemoryAccountState, PositionEvidence

_ZERO: Final = Decimal(0)


class PositionExplanation(StrEnum):
    DURABLE_PROJECTION_MATCH = "durable_projection_match"
    DURABLE_PROJECTION_MISMATCH = "durable_projection_mismatch"
    UNKNOWN_FLAT_WITH_NO_EVIDENCE = "unknown_flat_with_no_evidence"
    UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE = "unexplained_flat_with_local_evidence"
    UNEXPLAINED_NONZERO = "unexplained_nonzero"
    INCOMPLETE_SNAPSHOT = "incomplete_snapshot"


_EXPLAINED: Final = frozenset(
    {
        PositionExplanation.DURABLE_PROJECTION_MATCH,
        PositionExplanation.UNKNOWN_FLAT_WITH_NO_EVIDENCE,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionReconciliation:
    """The reconciliation of one symbol."""

    symbol: str
    exchange_qty: Decimal | None
    """The authoritative quantity (0 for a symbol missing from a complete
    snapshot); None when a partial snapshot does not report the symbol."""
    durable_present: bool
    durable_known: bool
    durable_qty: Decimal | None
    has_fill_evidence: bool
    runtime_qty: Decimal | None
    """The runtime position after this reconciliation (published for a complete
    snapshot, unchanged for a partial one)."""
    known: bool
    explained: bool
    explanation: PositionExplanation


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionReconciliationResult:
    """Per-symbol results (sorted by symbol) and the account-level verdict."""

    snapshot_complete: bool
    positions: tuple[PositionReconciliation, ...]
    all_known: bool
    all_explained: bool
    reconciled: bool
    """``snapshot_complete and all_known and all_explained``."""


def classify_positions(
    snapshot: PositionSnapshot, evidence: PositionEvidence
) -> PositionReconciliationResult:
    """Pure classification of ``snapshot`` against ``evidence`` (no publication)."""
    if type(snapshot) is not PositionSnapshot:
        raise DomainValidationError("snapshot must be a PositionSnapshot")
    if type(evidence) is not PositionEvidence:
        raise DomainValidationError("evidence must be a PositionEvidence")
    reported = {position.symbol: position.qty for position in snapshot.positions}
    universe = sorted(
        set(reported)
        | set(evidence.durable_positions)
        | set(evidence.runtime_positions)
        | evidence.fill_symbols
    )
    results: list[PositionReconciliation] = []
    for symbol in universe:
        row = evidence.durable_positions.get(symbol)
        durable_known = row is not None and row.known
        durable_qty = row.qty if row is not None and row.known else None
        has_fills = symbol in evidence.fill_symbols
        if not snapshot.complete:
            results.append(
                PositionReconciliation(
                    symbol=symbol,
                    durable_present=row is not None,
                    durable_known=durable_known,
                    durable_qty=durable_qty,
                    has_fill_evidence=has_fills,
                    exchange_qty=reported.get(symbol),
                    runtime_qty=evidence.runtime_positions.get(symbol),
                    known=False,
                    explained=False,
                    explanation=PositionExplanation.INCOMPLETE_SNAPSHOT,
                )
            )
            continue
        exchange_qty = reported.get(symbol, _ZERO)
        if durable_qty is not None:
            explanation = (
                PositionExplanation.DURABLE_PROJECTION_MATCH
                if durable_qty == exchange_qty
                else PositionExplanation.DURABLE_PROJECTION_MISMATCH
            )
        elif exchange_qty != 0:
            explanation = PositionExplanation.UNEXPLAINED_NONZERO
        elif has_fills:
            explanation = PositionExplanation.UNEXPLAINED_FLAT_WITH_LOCAL_EVIDENCE
        else:
            explanation = PositionExplanation.UNKNOWN_FLAT_WITH_NO_EVIDENCE
        results.append(
            PositionReconciliation(
                symbol=symbol,
                durable_present=row is not None,
                durable_known=durable_known,
                durable_qty=durable_qty,
                has_fill_evidence=has_fills,
                exchange_qty=exchange_qty,
                runtime_qty=exchange_qty,
                known=True,
                explained=explanation in _EXPLAINED,
                explanation=explanation,
            )
        )
    all_known = snapshot.complete and all(result.known for result in results)
    all_explained = snapshot.complete and all(result.explained for result in results)
    return PositionReconciliationResult(
        snapshot_complete=snapshot.complete,
        positions=tuple(results),
        all_known=all_known,
        all_explained=all_explained,
        reconciled=snapshot.complete and all_known and all_explained,
    )


async def reconcile_positions(
    *, account_state: InMemoryAccountState, snapshot: PositionSnapshot
) -> PositionReconciliationResult:
    """Reconcile ``snapshot`` and, if it is complete, publish the runtime
    positions; one account lock, no await inside it, no store access."""
    if type(account_state) is not InMemoryAccountState:
        raise DomainValidationError("account_state must be an InMemoryAccountState")
    if type(snapshot) is not PositionSnapshot:
        raise DomainValidationError("snapshot must be a PositionSnapshot")
    async with account_state.account_lock() as locked:
        locked.ensure_mutations_allowed()
        evidence = locked.position_evidence()
        result = classify_positions(snapshot, evidence)
        if result.snapshot_complete:
            locked.publish_exchange_positions(
                {
                    position.symbol: position.runtime_qty
                    for position in result.positions
                    if position.runtime_qty is not None
                },
                expected_revision=evidence.revision,
            )
    return result
