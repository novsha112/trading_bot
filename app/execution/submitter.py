"""Safe submission of an approved ``Order(NEW)`` reservation (one attempt, no retry).

Phase A, under the account lock (nothing awaited):
    find the order; it must be NEW (anything else -> ``OrderAlreadySubmittedError``,
    never a second request); build the ``OrderRequest`` (pure); read the clock;
    ``NEW -> SUBMITTING`` (write-ahead marker, +1 revision). Any failure here
    leaves the order NEW and nothing is sent.
Phase B, after the lock is released:
    ``await client.place_order(request)`` exactly once.
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

Persistence: every recorded step (SUBMITTING, ack, outcome) is durable before
it is visible. A store error before the network leaves the order NEW and sends
nothing. A store error after the network propagates (the transport exception,
if any, stays as its ``__context__``); the published order keeps its previous
state, the request is never re-sent and the exchange is never "rolled back".
After ``StoreUncertainError`` the account state must not be mutated further
until reloaded (no runtime poison flag yet).

Time: the submission time is one clock read in phase A. The outcome time is read
in phase C, but a clock failure or an invalid / earlier value there falls back
to the order's own ``updated_at``, so a known outcome is always recorded. TradingState
is not changed here (e.g. on an authentication error): that is a safety
controller's job.
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
from app.execution.models import SubmissionOutcome
from app.execution.requests import order_request_from_order
from app.execution.timing import change_time


class OrderAlreadySubmittedError(AccountStateError):
    """The order is not NEW: it was (or may have been) sent already; nothing sent."""


class OrderSubmitter:
    """Sends approved reservations of one account through a ``TradingClient``."""

    __slots__ = ("_account", "_client", "_clock")

    def __init__(
        self, *, account_state: InMemoryAccountState, client: TradingClient, clock: Clock
    ) -> None:
        if type(account_state) is not InMemoryAccountState:
            raise DomainValidationError("account_state must be an InMemoryAccountState")
        if not callable(getattr(client, "place_order", None)):
            raise DomainValidationError("client must provide place_order()")
        if not callable(getattr(clock, "now", None)):
            raise DomainValidationError("clock must provide now()")
        self._account = account_state
        self._client = client
        self._clock = clock

    async def submit(self, *, client_order_id: str) -> Order:
        """Send the NEW order once; returns it after the ack (still SUBMITTING
        unless fills advanced it). Re-raises the transport error after recording
        its outcome."""
        require_text(client_order_id, "client_order_id")
        async with self._account.account_lock() as locked:
            order = locked.order(client_order_id)
            if order is None:
                raise AccountStateError(f"no local order {client_order_id}")
            if order.status is not OrderStatus.NEW:
                raise OrderAlreadySubmittedError(
                    f"order {client_order_id} is {order.status.value}: not sent again"
                )
            request = order_request_from_order(order)
            # Write-ahead: SUBMITTING is durable before any request may be sent.
            # Any store error propagates here and nothing is sent.
            await locked.mark_submitting(client_order_id, at=self._clock.now())

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
