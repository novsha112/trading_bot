"""Recover the missing fills of ONE managed order from the exchange's execution
history (docs/ARCHITECTURE.md 13.8). Not an account-wide recovery: the caller
chooses the order (a ``ManagedOrderMatch`` from ``recovery_matching``) and the
fixed query window.

``recover_missing_fills(account_state=..., reader=..., match=..., query=..., clock=...)``:

1. preconditions under the account lock (no network): not poisoned; the local
   order still exists with the same client id and a compatible exchange id;
   ``query`` names this order's exchange id and symbol;
2. collect the COMPLETE history outside the lock: every page of ``query``; each
   page must be an ``ExecutionPage`` echoing ``query`` exactly; a repeated cursor
   (a cycle) is a protocol error; any exchange error propagates. No artificial
   page limit;
3. preflight the whole history before ANY mutation: ``exec_id`` dedup (the same
   id with an identical payload is one execution, a different payload is a
   conflict); only ``ExecutionKind.TRADE`` (any other kind blocks: it is never
   turned into a normal fill); every trade belongs to the order (exchange id,
   client id present and equal, symbol, side); the exact history quantity equals
   the exchange order's ``cum_filled_qty``;
4. under the account lock: re-read the order (stale-match protection) and its
   applied fills; an already applied ``exec_id`` must equal the authoritative
   fill exactly (all persisted ``Fill`` fields), else conflict; the local
   execution must be a subset of the history (local fills + notional equal to
   the present executions); then apply the missing trades in
   (exchange_ts, exec_id) order through ``LockedAccountState.apply_fill`` (each
   one an atomic durable mutation, the existing accounting path), and verify the
   result: local ``filled_qty`` == history qty == exchange cum qty and local
   notional == exact history notional.

Any failure in steps 1-3 applies nothing. In step 4 the checks run before the
first ``apply_fill``. A definite store failure keeps the fills committed before
it (a rerun dedups them and applies the rest); ``StoreUncertainError`` poisons
the account (existing invariant) and stops immediately. Nothing is compensated
or rolled back. The whole step 4 holds the account lock (the durable commits are
the only awaits under it), so a concurrent recovery of the same order sees the
fills already applied and changes nothing.

Notional: the generic invariant is execution history notional == local
notional. ``ExchangeOrder.cum_filled_notional`` is NOT used: its semantics
depend on the reader (a future adapter must prove them as an explicit
capability before it can become a safety invariant).

The clock is read only to apply a fill (``apply_fill`` needs the local change
time; fallback-safe ``change_time``, never before the order's ``updated_at``).
Read-only towards the exchange: only ``list_executions`` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException
from enum import StrEnum
from typing import Final

from app.domain.clock import Clock
from app.domain.errors import DomainValidationError
from app.domain.fill_math import accumulate_execution
from app.domain.fills import Fill
from app.domain.orders import Order
from app.exchanges.protocols import ExchangeStateReader
from app.exchanges.recovery import (
    ExchangeExecution,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
)
from app.execution.account_state import (
    AccountStateError,
    InMemoryAccountState,
    LockedAccountState,
)
from app.execution.recovery_matching import ManagedOrderMatch
from app.execution.timing import change_time

_ZERO: Final = Decimal(0)


class ExecutionRecoveryError(AccountStateError):
    """Fill recovery of an order is blocked; nothing (more) was applied by it."""


class ExecutionHistoryProtocolError(ExecutionRecoveryError):
    """The reader broke the pagination contract (wrong query echo, a cursor
    cycle, not an ``ExecutionPage``): the history is not provably complete."""


class ExecutionIdentityConflictError(ExecutionRecoveryError):
    """An execution contradicts the order or another record of the same
    ``exec_id`` (in the history or already applied locally)."""


class UnsupportedExecutionKindError(ExecutionRecoveryError):
    """The history contains a non-TRADE execution; it is never applied as a
    normal fill and blocks automatic recovery."""


class ExecutionCumulativeMismatchError(ExecutionRecoveryError):
    """Exact cumulative quantity / notional of history, exchange order and local
    state disagree."""


class ExecutionRecoveryOutcome(StrEnum):
    ALREADY_COMPLETE = "already_complete"
    """Every execution was already applied: no commit, no revision change."""
    RECOVERED = "recovered"
    """Missing fills were applied."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionRecoveryResult:
    """Observability of one recovery attempt (not a safety gate by itself)."""

    client_order_id: str
    exchange_order_id: str
    outcome: ExecutionRecoveryOutcome
    pages_read: int
    executions_seen: int
    unique_executions: int
    already_present: int
    fills_applied: int
    final_filled_qty: Decimal
    final_notional: Decimal


def _fill_of(execution: ExchangeExecution) -> Fill:
    """The domain fill of a TRADE execution: every field taken as reported."""
    return Fill(
        exec_id=execution.exec_id,
        exchange_order_id=execution.exchange_order_id,
        client_order_id=execution.client_order_id,
        symbol=execution.symbol,
        side=execution.side,
        price=execution.price,
        qty=execution.qty,
        fee=execution.fee,
        fee_asset=execution.fee_asset,
        is_maker=execution.is_maker,
        exchange_ts=execution.exchange_ts,
    )


def _totals(fills: list[Fill]) -> tuple[Decimal, Decimal]:
    """Exact (qty, notional) of ``fills`` through the shared accounting rule."""
    qty, notional = _ZERO, _ZERO
    try:
        for fill in fills:
            totals = accumulate_execution(
                filled_qty=qty, filled_notional=notional, price=fill.price, qty=fill.qty
            )
            qty, notional = totals.filled_qty, totals.filled_notional
    except DecimalException:
        raise ExecutionCumulativeMismatchError(
            "the execution totals cannot be computed exactly"
        ) from None
    return qty, notional


def _check_order(locked: LockedAccountState, match: ManagedOrderMatch) -> Order:
    """The current local order of ``match`` (stale-match protection)."""
    expected = match.local_order
    current = locked.order(expected.client_order_id)
    if current is None:
        raise ExecutionIdentityConflictError(f"local order {expected.client_order_id} is gone")
    exchange_id = match.exchange_order.exchange_order_id
    if current.exchange_order_id not in (None, exchange_id):
        raise ExecutionIdentityConflictError(
            f"local order {current.client_order_id} has exchange id "
            f"{current.exchange_order_id}, the exchange reports {exchange_id}"
        )
    return current


async def _collect(
    reader: ExchangeStateReader, query: ExecutionQuery
) -> tuple[list[ExecutionPage], list[ExchangeExecution]]:
    pages: list[ExecutionPage] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    while True:
        page = await reader.list_executions(query, cursor=cursor)
        if type(page) is not ExecutionPage:
            raise ExecutionHistoryProtocolError("the reader returned no ExecutionPage")
        if page.query != query:
            raise ExecutionHistoryProtocolError("an execution page does not echo its query")
        pages.append(page)
        if page.next_cursor is None:
            break
        if page.next_cursor in seen_cursors:
            raise ExecutionHistoryProtocolError("the execution cursors repeat (a cycle)")
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor
    return pages, [execution for page in pages for execution in page.executions]


def _preflight_history(
    executions: list[ExchangeExecution], match: ManagedOrderMatch
) -> list[ExchangeExecution]:
    """The unique trades of the order in (exchange_ts, exec_id) order."""
    unique: dict[str, ExchangeExecution] = {}
    for execution in executions:
        known = unique.setdefault(execution.exec_id, execution)
        if known != execution:
            raise ExecutionIdentityConflictError(
                f"exec_id {execution.exec_id} is reported with different data"
            )
    order, exchange_order = match.local_order, match.exchange_order
    for execution in unique.values():
        if execution.kind is not ExecutionKind.TRADE:
            raise UnsupportedExecutionKindError(
                f"execution {execution.exec_id} is {execution.kind.value}, not a trade"
            )
    for execution in unique.values():
        if (
            execution.exchange_order_id != exchange_order.exchange_order_id
            or execution.client_order_id is None
            or execution.client_order_id != order.client_order_id
            or execution.symbol != order.symbol
            or execution.side is not order.side
        ):
            raise ExecutionIdentityConflictError(
                f"execution {execution.exec_id} does not belong to order "
                f"{order.client_order_id} / {exchange_order.exchange_order_id}"
            )
    return sorted(unique.values(), key=lambda e: (e.exchange_ts, e.exec_id))


async def recover_missing_fills(
    *,
    account_state: InMemoryAccountState,
    reader: ExchangeStateReader,
    match: ManagedOrderMatch,
    query: ExecutionQuery,
    clock: Clock,
) -> ExecutionRecoveryResult:
    """Apply the missing trades of ``match`` from the complete history of ``query``."""
    if type(account_state) is not InMemoryAccountState:
        raise DomainValidationError("account_state must be an InMemoryAccountState")
    if type(match) is not ManagedOrderMatch:
        raise DomainValidationError("match must be a ManagedOrderMatch")
    if type(query) is not ExecutionQuery:
        raise DomainValidationError("query must be an ExecutionQuery")
    if not callable(getattr(reader, "list_executions", None)):
        raise DomainValidationError("reader must provide list_executions()")
    if not callable(getattr(clock, "now", None)):
        raise DomainValidationError("clock must provide now()")
    exchange_order = match.exchange_order
    if query.exchange_order_id != exchange_order.exchange_order_id:
        raise DomainValidationError("query must name the exchange id of the matched order")
    if query.symbol != match.local_order.symbol:
        raise DomainValidationError("query symbol differs from the order's symbol")

    async with account_state.account_lock() as locked:
        locked.ensure_mutations_allowed()
        _check_order(locked, match)

    pages, executions = await _collect(reader, query)  # no lock across the network
    history = _preflight_history(executions, match)
    authoritative = [_fill_of(execution) for execution in history]
    history_qty, history_notional = _totals(authoritative)
    if history_qty != exchange_order.cum_filled_qty:
        raise ExecutionCumulativeMismatchError(
            f"history quantity {history_qty} differs from the exchange cumulative "
            f"{exchange_order.cum_filled_qty} of {exchange_order.exchange_order_id}"
        )

    async with account_state.account_lock() as locked:
        locked.ensure_mutations_allowed()
        order = _check_order(locked, match)
        present: list[Fill] = []
        missing: list[Fill] = []
        for fill in authoritative:
            applied = locked.fill(fill.exec_id)
            if applied is None:
                missing.append(fill)
            elif applied != fill:
                raise ExecutionIdentityConflictError(
                    f"exec_id {fill.exec_id} was applied locally with different data"
                )
            else:
                present.append(fill)
        present_qty, present_notional = _totals(present)
        local_notional = locked.filled_notional(order.client_order_id)
        if order.filled_qty > history_qty:
            raise ExecutionCumulativeMismatchError(
                f"local filled {order.filled_qty} exceeds the exchange history {history_qty}"
            )
        if order.filled_qty != present_qty or local_notional != present_notional:
            raise ExecutionCumulativeMismatchError(
                f"local execution of {order.client_order_id} ({order.filled_qty}, "
                f"{local_notional}) is not the applied part of the history "
                f"({present_qty}, {present_notional})"
            )
        for fill in missing:
            current = _check_order(locked, match)
            await locked.apply_fill(fill, at=change_time(clock, floor=current.updated_at))
        final = _check_order(locked, match)
        final_notional = locked.filled_notional(final.client_order_id)
        if final.filled_qty != history_qty or final_notional != history_notional:
            raise ExecutionCumulativeMismatchError(  # pragma: no cover - accounting invariant
                f"after recovery {final.client_order_id} has ({final.filled_qty}, "
                f"{final_notional}), the history ({history_qty}, {history_notional})"
            )

    return ExecutionRecoveryResult(
        client_order_id=final.client_order_id,
        exchange_order_id=exchange_order.exchange_order_id,
        outcome=(
            ExecutionRecoveryOutcome.RECOVERED
            if missing
            else ExecutionRecoveryOutcome.ALREADY_COMPLETE
        ),
        pages_read=len(pages),
        executions_seen=len(executions),
        unique_executions=len(history),
        already_present=len(present),
        fills_applied=len(missing),
        final_filled_qty=final.filled_qty,
        final_notional=history_notional,
    )
