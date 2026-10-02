"""Fail-closed runtime safety: requested vs effective ``TradingState`` and the
startup recovery gates (docs/ARCHITECTURE.md 9.2 / 12).

``TradingState`` stays the trading mode (RUNNING, REDUCE_ONLY, PAUSED, HALTED);
recovery readiness is a separate safety dimension, never a ``TradingState``.
``effective_trading_state`` combines them, strongest first:

1. requested HALTED -> HALTED (absolute; nothing un-halts it automatically);
2. account poisoned -> PAUSED (the durable state may be ahead of RAM);
3. recovery incomplete -> PAUSED (REDUCE_ONLY too: position and exchange state
   may still be unknown);
4. otherwise the requested state (PAUSED stays PAUSED).

``SafetyController`` is bound to ONE account state (one startup / recovery
session: a re-hydrated account gets a new controller). It owns the requested
state (runtime-only, PAUSED by default: a restart never restores RUNNING) and
the readiness gates. Gates only go ``False -> True``: there is no API to lower
one (a later loss of trust is a runtime-health concern, not a gate).
The exchange gates (orders, positions, open orders, fills) are confirmed
together by ONE ``mark_exchange_reconciled`` after a recovery's final
verification, and only after ``mark_hydrated`` (an orchestration bug
otherwise); the fields stay separate for observability.

``submission_allowed`` is the pure send permission of one order under an
effective state: RUNNING -> any order; REDUCE_ONLY -> reduce-only orders only;
PAUSED and HALTED -> none (HALTED blocks reduce-only too). The order submitter
checks it before the write-ahead marker and again right before the send.

Hydrated is not ready: ``InMemoryAccountState.hydrate`` never touches a
controller; the future bootstrap marks the gates explicitly. Effective state is
computed on every read from the account's public ``is_poisoned``, so a poison
after readiness takes effect immediately; the controller never clears poison.

Fully synchronous, single event loop: no lock. No exchange, no persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from app.domain.errors import DomainValidationError
from app.execution.account_state import InMemoryAccountState
from app.risk.models import TradingState


class SafetyStateError(Exception):
    """Base class of safety-controller errors that are not invalid arguments."""


class RecoveryGateOrderError(SafetyStateError):
    """An exchange recovery gate was marked before the account was hydrated."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RecoveryReadiness:
    """Startup recovery gates of one session; all False by default."""

    hydrated: bool = False
    orders_reconciled: bool = False
    positions_reconciled: bool = False
    open_orders_reconciled: bool = False
    fills_complete: bool = False

    def __post_init__(self) -> None:
        for item in fields(self):
            if type(getattr(self, item.name)) is not bool:
                raise DomainValidationError(f"{item.name} must be a bool")

    @property
    def complete(self) -> bool:
        """Every gate confirmed."""
        return all(getattr(self, item.name) for item in fields(self))


def _require_state(value: object, field: str) -> TradingState:
    if type(value) is not TradingState:
        raise DomainValidationError(f"{field} must be a TradingState, got {value!r}")
    return value


def effective_trading_state(
    *, requested: TradingState, readiness: RecoveryReadiness, account_poisoned: bool
) -> TradingState:
    """The state placements must use (pure; see the module doc for the order)."""
    requested = _require_state(requested, "requested")
    if type(readiness) is not RecoveryReadiness:
        raise DomainValidationError("readiness must be a RecoveryReadiness")
    if type(account_poisoned) is not bool:
        raise DomainValidationError("account_poisoned must be a bool")
    if requested is TradingState.HALTED:
        return TradingState.HALTED
    if account_poisoned or not readiness.complete:
        return TradingState.PAUSED
    return requested


def submission_allowed(state: TradingState, *, reduce_only: bool) -> bool:
    """May an order with ``reduce_only`` be sent under the effective ``state``?"""
    state = _require_state(state, "state")
    if type(reduce_only) is not bool:
        raise DomainValidationError("reduce_only must be a bool")
    if state is TradingState.RUNNING:
        return True
    return state is TradingState.REDUCE_ONLY and reduce_only


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetySnapshot:
    """One consistent, immutable read of the controller."""

    requested_state: TradingState
    effective_state: TradingState
    readiness: RecoveryReadiness
    account_poisoned: bool


class SafetyController:
    """Requested state and recovery gates of one account state (runtime-only)."""

    __slots__ = ("_account", "_readiness", "_requested")

    def __init__(self, *, account_state: InMemoryAccountState) -> None:
        if type(account_state) is not InMemoryAccountState:
            raise DomainValidationError("account_state must be an InMemoryAccountState")
        self._account = account_state
        self._requested = TradingState.PAUSED
        self._readiness = RecoveryReadiness()

    @property
    def account_state(self) -> InMemoryAccountState:
        """The account state whose poison and readiness this controller covers."""
        return self._account

    def request_state(self, state: TradingState) -> None:
        """Remember the operator's requested state; the effective state may stay
        stricter (an incomplete recovery is not an error)."""
        self._requested = _require_state(state, "state")

    def snapshot(self) -> SafetySnapshot:
        """Requested, effective (from the current poison flag), readiness."""
        poisoned = self._account.is_poisoned
        return SafetySnapshot(
            requested_state=self._requested,
            effective_state=effective_trading_state(
                requested=self._requested, readiness=self._readiness, account_poisoned=poisoned
            ),
            readiness=self._readiness,
            account_poisoned=poisoned,
        )

    @property
    def effective_state(self) -> TradingState:
        return self.snapshot().effective_state

    def mark_hydrated(self) -> None:
        """The durable local state was loaded and crash-classified."""
        self._readiness = replace(self._readiness, hydrated=True)

    def mark_exchange_reconciled(self) -> None:
        """Confirm ALL exchange gates at once (orders, positions, open orders,
        fills), after the final verification of a recovery. There is no API to
        confirm them one by one: no partial readiness state exists. Idempotent;
        before ``mark_hydrated`` a ``RecoveryGateOrderError``."""
        if not self._readiness.hydrated:
            raise RecoveryGateOrderError(
                "exchange gates cannot be confirmed before the account is hydrated"
            )
        self._readiness = RecoveryReadiness(
            hydrated=True,
            orders_reconciled=True,
            positions_reconciled=True,
            open_orders_reconciled=True,
            fills_complete=True,
        )
