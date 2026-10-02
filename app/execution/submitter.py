"""Safe submission of an approved ``Order(NEW)`` reservation (one attempt, no retry).

A reservation is not a permission to send forever: the CURRENT effective trading
state (``SafetyController``) is checked twice, ``submission_allowed``: RUNNING
sends any order, REDUCE_ONLY only reduce-only orders, PAUSED and HALTED nothing.

Phase A, under the account lock:
    poison check; find the order; it must be NEW (anything else ->
    ``OrderAlreadySubmittedError``, never a second request); build the
    ``OrderRequest`` (pure); FIRST safety check. Refused -> one clock read,
    durable ``NEW -> FAILED`` with a ``SafetyBlockRecord`` (BEFORE_WRITE_AHEAD),
    ``OrderSubmissionBlockedError``. Allowed -> one clock read, durable
    ``NEW -> SUBMITTING`` (write-ahead marker). Any failure here sends nothing.
Phase A2, under the account lock again (after the durable await):
    poison check (``AccountStatePoisonedError`` wins, no mutation); the order must
    still be SUBMITTING; FINAL safety check. Refused -> durable
    ``SUBMITTING -> FAILED`` (definite NOT_SENT) with a ``SafetyBlockRecord``
    (BEFORE_SEND), ``OrderSubmissionBlockedError``. Allowed -> the lock is
    released and ``place_order`` is called with NO await in between.
Phase B, outside the lock: ``await client.place_order(request)`` exactly once.
Phase C, under the account lock again: the outcome is recorded.
    * ack -> its ``client_order_id`` must be this order's; ``exchange_order_id`` is
      recorded as metadata (status unchanged, never OPEN on an ack). A mismatch
      is ambiguous: SUBMITTING -> UNKNOWN, ``OrderAckMismatchError`` is raised.
    * ``ExchangeNotSentError`` (incl. ``ExchangeRequestValidationError``) -> FAILED;
    * ``ExchangeRejectedError`` (incl. ``ExchangeAuthenticationError``) -> REJECTED;
    * anything else (``ExchangeAmbiguousResultError``, ``ExchangeDuplicateOrderError``,
      any other error or cancellation: the request may exist) -> UNKNOWN, which
      stays active exposure until a resolver decides it.
    Exceptions are classified by type only and re-raised after the outcome is
    recorded. Confirmed exchange progress (fills applied while the request was in
    flight) is stronger than the transport outcome and is never rolled back (see
    ``LockedAccountState.record_submission_outcome``).

Race boundary: the safety gate guarantees no send if the state was already
disallowed at the final synchronous pre-send check. It cannot make an external
network side effect atomic with an in-memory operator state change (a pause
after that check still lets this one request go); a runtime execution gate /
private stream / serialized command loop will close that later. The account
lock is never held across the network.

Persistence: every recorded step (SUBMITTING, safety block, ack, outcome) is
durable before it is visible. A store error before the network (including on a
safety-block path) leaves the published order as it was and sends nothing; the
store error propagates unchanged. A store error after the network propagates
(the transport exception, if any, stays as its ``__context__``); the published
order keeps its previous state, the request is never re-sent and the exchange
is never "rolled back". ``StoreUncertainError`` poisons the account state:
later submissions fail with ``AccountStatePoisonedError`` before the clock, the
marker or the network.

A safety-blocked order is FAILED (definitely not sent), its reservation is
released and its approved ``PlacementRecord`` stays: a replay of the same intent
returns that record; a new trading decision needs a new ``intent_id``.

Time: the clock is read only once a mutation is decided: phase A (block or
marker, strict), phase A2 refusal and phase C outcomes fall back to the order's
own ``updated_at`` when the clock fails or goes back, so a known outcome is
always recorded. TradingState is never changed here (e.g. on an authentication
error): the submitter only reads the effective state.
"""

from __future__ import annotations

from datetime import datetime

from app.domain.clock import Clock
from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError
from app.domain.orders import Order
from app.domain.validation import require_text
from app.exchanges.errors import ExchangeNotSentError, ExchangeRejectedError
from app.exchanges.models import OrderAck
from app.exchanges.protocols import TradingClient
from app.execution.account_state import (
    AccountStateError,
    InMemoryAccountState,
    LockedAccountState,
    OrderAckMismatchError,
)
from app.execution.models import SubmissionBlockStage, SubmissionOutcome
from app.execution.requests import order_request_from_order
from app.execution.safety import SafetyController, SafetyStateError, submission_allowed
from app.execution.timing import change_time
from app.risk.models import TradingState


class OrderAlreadySubmittedError(AccountStateError):
    """The order is not NEW: it was (or may have been) sent already; nothing sent."""


class OrderSubmissionBlockedError(SafetyStateError):
    """The effective trading state refused to send the order: it was NOT sent and
    is now FAILED with a durable ``SafetyBlockRecord``. A local safety decision,
    not an exchange error."""

    def __init__(
        self,
        *,
        client_order_id: str,
        effective_state: TradingState,
        reduce_only: bool,
        stage: SubmissionBlockStage,
    ) -> None:
        super().__init__(
            f"order {client_order_id} (reduce_only={reduce_only}) not sent: effective "
            f"trading state is {effective_state.value} ({stage.value}); it is FAILED"
        )
        self.client_order_id = client_order_id
        self.effective_state = effective_state
        self.reduce_only = reduce_only
        self.stage = stage


class OrderSubmitter:
    """Sends approved reservations of one account through a ``TradingClient``,
    only while the effective trading state allows it."""

    __slots__ = ("_account", "_client", "_clock", "_safety")

    def __init__(
        self,
        *,
        account_state: InMemoryAccountState,
        safety: SafetyController,
        client: TradingClient,
        clock: Clock,
    ) -> None:
        if type(account_state) is not InMemoryAccountState:
            raise DomainValidationError("account_state must be an InMemoryAccountState")
        if type(safety) is not SafetyController:
            raise DomainValidationError("safety must be a SafetyController")
        if safety.account_state is not account_state:
            raise DomainValidationError("safety must cover the same account_state")
        if not callable(getattr(client, "place_order", None)):
            raise DomainValidationError("client must provide place_order()")
        if not callable(getattr(clock, "now", None)):
            raise DomainValidationError("clock must provide now()")
        self._account = account_state
        self._safety = safety
        self._client = client
        self._clock = clock

    async def submit(self, *, client_order_id: str) -> Order:
        """Send the NEW order once; returns it after the ack (still SUBMITTING
        unless fills advanced it). Re-raises the transport error after recording
        its outcome."""
        require_text(client_order_id, "client_order_id")
        async with self._account.account_lock() as locked:
            # A poisoned account fails before the clock, the marker or the network.
            locked.ensure_mutations_allowed()
            order = locked.order(client_order_id)
            if order is None:
                raise AccountStateError(f"no local order {client_order_id}")
            if order.status is not OrderStatus.NEW:
                raise OrderAlreadySubmittedError(
                    f"order {client_order_id} is {order.status.value}: not sent again"
                )
            request = order_request_from_order(order)
            effective = self._safety.snapshot().effective_state
            if not submission_allowed(effective, reduce_only=order.reduce_only):
                # Never sent: NEW -> FAILED with its durable reason. A store error
                # propagates unchanged (the order stays NEW) and nothing is sent.
                await locked.record_safety_block(
                    client_order_id, effective_state=effective, at=self._clock.now()
                )
                raise self._blocked(order, effective, SubmissionBlockStage.BEFORE_WRITE_AHEAD)
            # Write-ahead: SUBMITTING is durable before any request may be sent.
            # Any store error propagates here and nothing is sent.
            await locked.mark_submitting(client_order_id, at=self._clock.now())

        async with self._account.account_lock() as locked:
            # Final pre-send check, after the durable await: poison wins (no
            # mutation of a poisoned account), then the current effective state.
            locked.ensure_mutations_allowed()
            order = locked.order(client_order_id)
            if order is None or order.status is not OrderStatus.SUBMITTING:
                raise AccountStateError(
                    f"order {client_order_id} changed before it was sent; not sent"
                )
            effective = self._safety.snapshot().effective_state
            if not submission_allowed(effective, reduce_only=order.reduce_only):
                # Definitely not sent: SUBMITTING -> FAILED (NOT_SENT) with its reason.
                await locked.record_safety_block(
                    client_order_id,
                    effective_state=effective,
                    at=change_time(self._clock, floor=order.updated_at),
                )
                raise self._blocked(order, effective, SubmissionBlockStage.BEFORE_SEND)
        # No await between the final check and the send (the lock release does
        # not suspend): the request goes out under the state that was checked.
        try:
            ack = await self._client.place_order(request)
        except ExchangeNotSentError:
            await self._record(client_order_id, SubmissionOutcome.NOT_SENT)
            raise
        except ExchangeRejectedError:
            await self._record(client_order_id, SubmissionOutcome.REJECTED)
            raise
        except BaseException:
            # Ambiguous, duplicate, unexpected error or cancellation: the request
            # may have reached the exchange, so the reservation stays active.
            await self._record(client_order_id, SubmissionOutcome.AMBIGUOUS)
            raise
        return await self._record_ack(client_order_id, ack)

    @staticmethod
    def _blocked(
        order: Order, effective: TradingState, stage: SubmissionBlockStage
    ) -> OrderSubmissionBlockedError:
        return OrderSubmissionBlockedError(
            client_order_id=order.client_order_id,
            effective_state=effective,
            reduce_only=order.reduce_only,
            stage=stage,
        )

    async def _record(self, client_order_id: str, outcome: SubmissionOutcome) -> Order:
        async with self._account.account_lock() as locked:
            return await locked.record_submission_outcome(
                client_order_id, outcome, at=self._outcome_time(locked, client_order_id)
            )

    async def _record_ack(self, client_order_id: str, ack: object) -> Order:
        async with self._account.account_lock() as locked:
            at = self._outcome_time(locked, client_order_id)
            try:
                if type(ack) is not OrderAck:
                    raise OrderAckMismatchError(f"place_order returned no OrderAck: {ack!r}")
                if ack.client_order_id != client_order_id:
                    raise OrderAckMismatchError(
                        f"ack for {ack.client_order_id} does not belong to order {client_order_id}"
                    )
                return await locked.record_ack(
                    client_order_id, exchange_order_id=ack.exchange_order_id, at=at
                )
            except OrderAckMismatchError:
                # The request was sent and the answer is unusable: ambiguous.
                await locked.record_submission_outcome(
                    client_order_id, SubmissionOutcome.AMBIGUOUS, at=at
                )
                raise

    def _outcome_time(self, locked: LockedAccountState, client_order_id: str) -> datetime:
        """Local time of an outcome; never fails and never moves the order back."""
        order = locked.order(client_order_id)
        if order is None:
            raise AccountStateError(f"no local order {client_order_id}")
        return change_time(self._clock, floor=order.updated_at)
