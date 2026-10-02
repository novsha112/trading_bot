"""Placement coordinator: the atomic ``snapshot -> evaluate -> reserve`` boundary.

Under ONE account lock (``InMemoryAccountState.account_lock``):

1. replay check by ``intent_id`` (before any evaluation, id or clock read): an
   equal recorded intent returns its ``PlacementRecord``; different data raises
   ``PlacementConflictError``;
2. revision R, the position of the intent's symbol, its active orders and the
   account order count are read from the held handle: one consistent local
   state (fills change orders and positions under the same lock);
   ``snapshot_id = "<account_scope_id>:<R>"``;
3. ``build_risk_snapshot`` -> ``evaluate`` (pure);
4. rejected -> ``register_rejected`` (no Order, revision stays R);
5. approved -> ``client_order_id`` from the generator, then ``clock.now()``, then
   ``register_approved``: the ``Order(NEW)`` is visible before the lock is
   released and the revision becomes R + 1.

Nothing is awaited inside the lock and nothing is sent anywhere: the result is
the ``PlacementRecord``. The position is never taken from the caller (an unknown
position stays ``None`` -> UNKNOWN_POSITION); ``TradingState`` has no owner yet
and is still passed per placement. ``account_scope_id`` must not contain ``:``,
the delimiter of the snapshot id.

Failures: an exactly unrepresentable snapshot (``ExposureCalculationError``)
is a ``PlacementPreparationError`` (cause chained), not a Risk rejection; other
errors (invalid inputs, evaluator contract violations, generator or clock
failures, invalid or duplicate ids, a reservation time before the intent)
propagate unchanged. In every failure case nothing is recorded and the revision
does not change; ids are not retried.
"""

from __future__ import annotations

from typing import Final, Protocol

from app.domain.clock import Clock
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.validation import require_text
from app.execution.account_state import InMemoryAccountState
from app.execution.models import PlacementRecord
from app.risk.exposure import ExposureCalculationError
from app.risk.manager import evaluate
from app.risk.models import RiskPolicy, TradingState
from app.risk.snapshots import build_risk_snapshot

_SNAPSHOT_ID_DELIMITER: Final = ":"


class ClientOrderIdGenerator(Protocol):
    """Source of new client order ids, asked only after a Risk approval."""

    def next_id(self, *, intent: PlaceOrderIntent) -> str: ...


class PlacementPreparationError(RuntimeError):
    """The risk snapshot could not be prepared: nothing was evaluated or reserved."""


class PlacementCoordinator:
    """Serializes placements of one account scope through its account-state lock."""

    __slots__ = ("_account", "_account_scope_id", "_clock", "_ids", "_policy")

    def __init__(
        self,
        *,
        account_scope_id: str,
        account_state: InMemoryAccountState,
        policy: RiskPolicy,
        clock: Clock,
        client_order_id_generator: ClientOrderIdGenerator,
    ) -> None:
        scope = require_text(account_scope_id, "account_scope_id")
        if _SNAPSHOT_ID_DELIMITER in scope:
            raise DomainValidationError(
                f"account_scope_id must not contain {_SNAPSHOT_ID_DELIMITER!r}, got {scope!r}"
            )
        if type(account_state) is not InMemoryAccountState:
            raise DomainValidationError("account_state must be an InMemoryAccountState")
        if type(policy) is not RiskPolicy:
            raise DomainValidationError("policy must be a RiskPolicy")
        if not callable(getattr(clock, "now", None)):
            raise DomainValidationError("clock must provide now()")
        if not callable(getattr(client_order_id_generator, "next_id", None)):
            raise DomainValidationError("client_order_id_generator must provide next_id()")
        self._account_scope_id = scope
        self._account = account_state
        self._policy = policy
        self._clock = clock
        self._ids = client_order_id_generator

    @property
    def account_scope_id(self) -> str:
        return self._account_scope_id

    async def place(
        self,
        *,
        intent: PlaceOrderIntent,
        trading_state: TradingState,
    ) -> PlacementRecord:
        """Evaluate ``intent`` and reserve it if approved, atomically per account."""
        if type(intent) is not PlaceOrderIntent:
            raise DomainValidationError("intent must be a PlaceOrderIntent")
        async with self._account.account_lock() as locked:
            replay = locked.replay_of(intent)
            if replay is not None:
                return replay
            revision = locked.revision
            try:
                snapshot = build_risk_snapshot(
                    snapshot_id=f"{self._account_scope_id}{_SNAPSHOT_ID_DELIMITER}{revision}",
                    symbol=intent.symbol,
                    trading_state=trading_state,
                    position_qty=locked.position_qty(intent.symbol),
                    orders=locked.active_orders(intent.symbol),
                    account_open_order_count=locked.account_active_order_count(),
                )
            except ExposureCalculationError as error:
                raise PlacementPreparationError(
                    f"risk snapshot of {intent.symbol} at revision {revision} "
                    "cannot be computed exactly"
                ) from error
            decision = evaluate(intent=intent, snapshot=snapshot, policy=self._policy)
            if not decision.approved:
                return locked.register_rejected(
                    intent=intent, decision=decision, expected_revision=revision
                )
            client_order_id = self._ids.next_id(intent=intent)
            at = self._clock.now()
            return locked.register_approved(
                intent=intent,
                decision=decision,
                client_order_id=client_order_id,
                expected_revision=revision,
                at=at,
            )
