"""Explicit position baseline acceptance (docs/ARCHITECTURE.md 13.9).

A PRIVILEGED durable action of the operator / recovery layer, never automatic:
an unexplained position (``known=True, explained=False`` after automatic
reconciliation) is NOT accepted because it is unexplained; only an explicit
per-symbol ``PositionBaselineAcceptance`` with an audit reason accepts it.

What is accepted is the CURRENT runtime position, i.e. the quantity the last
exchange reconciliation published, never an arbitrary new quantity: the
command names the quantity it saw (``expected_exchange_qty``) and the revision
it saw (``expected_revision``). Runtime publication does not change the
revision, so both are checked: a reconciliation that published another
quantity since makes the command stale.

Under ONE account lock, in this order: poison -> revision -> runtime position
exists -> expected quantity == runtime -> already accepted (no-op) -> one
``clock.now()`` and one new ``baseline_id`` -> ONE durable change (known
durable position + ``PositionBaselineRecord`` + revision R -> R+1) -> publish.
The no-op (durable known projection already equal to the runtime position)
reads no clock, takes no id and commits nothing.

The acceptance makes the durable projection agree with the runtime position
again (later fills move both). It is not a lasting proof of the exchange
position: after a restart the position is unknown again until an exchange
reconciliation explains it against the accepted baseline. It changes no
safety gate (the recovery coordinator decides on ``mark_exchange_reconciled``).
There is no operator identity yet: the reason is the V1 provenance.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from app.domain.clock import Clock
from app.domain.errors import DomainValidationError
from app.domain.validation import require_text
from app.execution.account_state import (
    BaselineQtyMismatchError,
    BaselineRuntimeUnknownError,
    InMemoryAccountState,
    StaleRevisionError,
)
from app.execution.models import (
    PositionBaselineRecord,
    require_audit_reason,
    require_baseline_id,
)
from app.execution.timing import strict_change_time

BaselineIdSource = Callable[[], str]
"""Returns a new, unique ``baseline_id`` on every call."""

BASELINE_ID_PREFIX: Final = "bl_"
DEFAULT_BASELINE_ID_BYTES: Final = 16
"""Entropy of a default baseline id: 128 bits as 32 lowercase hex characters."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def default_baseline_id() -> str:
    """A new unpredictable baseline id (``secrets``, OS randomness)."""
    return BASELINE_ID_PREFIX + secrets.token_hex(DEFAULT_BASELINE_ID_BYTES)


class BaselineOutcome(StrEnum):
    ACCEPTED = "accepted"
    """A new baseline record and known durable position were committed."""
    ALREADY_ACCEPTED = "already_accepted"
    """The durable projection already equals the runtime position: no change."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionBaselineAcceptance:
    """The explicit command: accept the runtime position of ONE symbol."""

    symbol: str
    expected_exchange_qty: Decimal
    """The runtime (exchange-published) quantity the decision was based on."""
    expected_revision: int
    """The account revision the decision was based on."""
    reason: str
    """Audit reason (canonical, see ``require_audit_reason``; never a secret)."""

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        qty = self.expected_exchange_qty
        if type(qty) is not Decimal or not qty.is_finite():
            raise DomainValidationError(
                f"expected_exchange_qty must be an exact, finite Decimal, got {qty!r}"
            )
        revision = self.expected_revision
        if type(revision) is not int or revision < 0:
            raise DomainValidationError(f"expected_revision must be an int >= 0, got {revision!r}")
        require_audit_reason(self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionBaselineAcceptanceResult:
    symbol: str
    qty: Decimal
    """The accepted (or already accepted) quantity: the runtime position."""
    outcome: BaselineOutcome
    baseline_record: PositionBaselineRecord | None
    """The new record for ACCEPTED; None for ALREADY_ACCEPTED."""
    revision: int
    """The account revision after the call."""


async def accept_position_baseline(
    *,
    account_state: InMemoryAccountState,
    command: PositionBaselineAcceptance,
    clock: Clock,
    baseline_ids: BaselineIdSource = default_baseline_id,
) -> PositionBaselineAcceptanceResult:
    """Accept the current runtime position of ``command.symbol`` as its durable
    baseline (see the module doc for the order of the checks).

    Raises:
        AccountStatePoisonedError: the account is poisoned (before the clock).
        StaleRevisionError: the revision moved since the command was formed.
        BaselineRuntimeUnknownError: no runtime position of the symbol.
        BaselineQtyMismatchError: the runtime position is not the expected one.
        BaselineIdConflictError: the new ``baseline_id`` is already recorded.
        DomainValidationError: invalid arguments, clock time or baseline id.
        Exception: whatever ``clock.now()``, the id source or the store raises.
    """
    if type(account_state) is not InMemoryAccountState:
        raise DomainValidationError("account_state must be an InMemoryAccountState")
    if type(command) is not PositionBaselineAcceptance:
        raise DomainValidationError("command must be a PositionBaselineAcceptance")
    if not callable(getattr(clock, "now", None)):
        raise DomainValidationError("clock must provide now()")
    if not callable(baseline_ids):
        raise DomainValidationError("baseline_ids must be callable")
    symbol = command.symbol
    async with account_state.account_lock() as locked:
        locked.ensure_mutations_allowed()
        if command.expected_revision != locked.revision:
            raise StaleRevisionError(
                f"baseline acceptance based on revision {command.expected_revision}, "
                f"current revision is {locked.revision}"
            )
        runtime = locked.position_qty(symbol)
        if runtime is None:
            raise BaselineRuntimeUnknownError(
                f"no runtime position of {symbol}: reconcile it with the exchange first"
            )
        if runtime != command.expected_exchange_qty:
            raise BaselineQtyMismatchError(
                f"runtime position of {symbol} is {runtime}, the acceptance expected "
                f"{command.expected_exchange_qty}"
            )
        durable = locked.durable_position(symbol)
        if durable is not None and durable.known and durable.qty == runtime:
            return PositionBaselineAcceptanceResult(
                symbol=symbol,
                qty=runtime,
                outcome=BaselineOutcome.ALREADY_ACCEPTED,
                baseline_record=None,
                revision=locked.revision,
            )
        history = locked.position_baselines()
        floor = max((record.accepted_at for record in history), default=_EPOCH)
        accepted_at = strict_change_time(clock, floor=floor)
        record = PositionBaselineRecord(
            baseline_id=require_baseline_id(baseline_ids()),
            symbol=symbol,
            qty=runtime,
            reason=command.reason,
            accepted_at=accepted_at,
            account_revision=locked.revision + 1,
        )
        await locked.commit_position_baseline(record)
        return PositionBaselineAcceptanceResult(
            symbol=symbol,
            qty=runtime,
            outcome=BaselineOutcome.ACCEPTED,
            baseline_record=record,
            revision=locked.revision,
        )
