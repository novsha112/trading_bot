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
Exchange gates (orders, positions, open orders, fills) complete in any order,
but only after ``mark_hydrated`` (an orchestration bug otherwise).

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

    def mark_orders_reconciled(self) -> None:
        """Every local active order was confirmed against the exchange."""
        self._mark_exchange_gate("orders_reconciled")

    def mark_positions_reconciled(self) -> None:
        """Every position was confirmed against the exchange."""
        self._mark_exchange_gate("positions_reconciled")

    def mark_open_orders_reconciled(self) -> None:
        """The exchange's open orders were compared with the local ones."""
        self._mark_exchange_gate("open_orders_reconciled")

    def mark_fills_complete(self) -> None:
        """No execution is missing locally."""
        self._mark_exchange_gate("fills_complete")

    def _mark_exchange_gate(self, gate: str) -> None:
        if not self._readiness.hydrated:
            raise RecoveryGateOrderError(
                f"{gate} cannot be confirmed before the account is hydrated"
            )
        self._readiness = replace(self._readiness, **{gate: True})
