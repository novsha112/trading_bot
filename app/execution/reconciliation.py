"""Exchange order reports into the local state, and single-shot UNKNOWN resolution.

* ``exchange_state_from_update(update)``: pure normalization of an exchange
  ``OrderUpdate`` into the execution-local ``ExchangeOrderState`` (a status the
  exchange cannot report, e.g. a local lifecycle status, is rejected). The
  account state then applies it (``LockedAccountState.apply_exchange_state``):
  never synthesizing fills, never regressing stronger local progress.
* ``UnknownOrderReconciler.reconcile(client_order_id=...)``: one attempt to resolve
  an order in UNKNOWN. It is reconciliation, never a resend:

      lock:   the order must be UNKNOWN (else ``OrderNotUnknownError``, no read)
      unlock: one ``get_order(OrderRef(symbol, client_order_id, exchange_order_id))``
      lock:   apply the report to the CURRENT local order (durable commit, then
              publication; the store commit is the only await under the lock)

  Exactly one read per call: no retry, backoff, grace period or loop. A read
  error propagates and changes nothing. ``None`` (the exchange confirms no such
  order) is classified against the order as it is when the answer is processed,
  changing nothing: still UNKNOWN with the same exchange id ->
  ``OrderStillUnknownError`` (it keeps its exposure; "not found -> FAILED" needs a
  separate grace policy); anything newer (fills, another report, a new exchange
  id) -> ``ExchangeStateMismatchError``. While
  the read is in flight the order may advance (fills): the report is then
  applied to that newer state by the same rules, so a stale answer is a
  mismatch, never a rollback.
"""

from __future__ import annotations

from typing import NoReturn

from app.domain.clock import Clock
from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError
from app.domain.orders import Order, OrderUpdate
from app.domain.validation import require_text
from app.exchanges.models import OrderRef
from app.exchanges.protocols import TradingClient
from app.execution.account_state import (
    AccountStateError,
    ExchangeStateMismatchError,
    InMemoryAccountState,
)
from app.execution.models import ExchangeOrderState
from app.execution.timing import change_time


class OrderNotUnknownError(AccountStateError):
    """Only an UNKNOWN order is reconciled here; nothing was read."""


class OrderStillUnknownError(AccountStateError):
    """The exchange confirmed that no such order exists; the order stays UNKNOWN
    (and active) until a grace policy decides otherwise."""


def exchange_state_from_update(update: OrderUpdate) -> ExchangeOrderState:
    """Normalize an exchange report (field by field; nothing is corrected)."""
    if type(update) is not OrderUpdate:
        raise DomainValidationError("update must be an OrderUpdate")
    return ExchangeOrderState(
        client_order_id=update.client_order_id,
        exchange_order_id=update.exchange_order_id,
        status=update.status,
        filled_qty=update.cum_filled_qty,
        avg_fill_price=update.avg_fill_price,
        exchange_ts=update.exchange_ts,
    )


class UnknownOrderReconciler:
    """Resolves UNKNOWN orders of one account with one exchange read per call."""

    __slots__ = ("_account", "_client", "_clock")

    def __init__(
        self, *, account_state: InMemoryAccountState, client: TradingClient, clock: Clock
    ) -> None:
        if type(account_state) is not InMemoryAccountState:
            raise DomainValidationError("account_state must be an InMemoryAccountState")
        if not callable(getattr(client, "get_order", None)):
            raise DomainValidationError("client must provide get_order()")
        if not callable(getattr(clock, "now", None)):
            raise DomainValidationError("clock must provide now()")
        self._account = account_state
        self._client = client
        self._clock = clock

    async def _classify_not_found(self, client_order_id: str, ref: OrderRef) -> NoReturn:
        """Raise for a "no such order" answer; never changes anything.

        Still unresolved only if the order is STILL UNKNOWN with the same exchange
        identity when the answer is processed. Any exchange fact observed locally
        meanwhile (fills, another report, a newly known exchange id) contradicts
        "not found": a reconciliation mismatch, not "still unknown".
        """
        async with self._account.account_lock() as locked:
            current = locked.order(client_order_id)
        if current is None:
            raise AccountStateError(
                f"local order {client_order_id} disappeared during reconciliation"
            )
        if (
            current.status is OrderStatus.UNKNOWN
            and current.exchange_order_id == ref.exchange_order_id
        ):
            raise OrderStillUnknownError(
                f"exchange reports no order {client_order_id}; it stays unknown"
            )
        raise ExchangeStateMismatchError(
            f"exchange reports no order {client_order_id}, but it is locally "
            f"{current.status.value} (exchange id {current.exchange_order_id}) after "
            "newer exchange facts; state kept"
        )

    async def reconcile(self, *, client_order_id: str) -> Order:
        """Read the exchange state of the UNKNOWN order once and apply it."""
        require_text(client_order_id, "client_order_id")
        async with self._account.account_lock() as locked:
            # The read is meant to mutate the account: a poisoned account stops here.
            locked.ensure_mutations_allowed()
            order = locked.order(client_order_id)
            if order is None:
                raise AccountStateError(f"no local order {client_order_id}")
            if order.status is not OrderStatus.UNKNOWN:
                raise OrderNotUnknownError(
                    f"order {client_order_id} is {order.status.value}, not unknown"
                )
            ref = OrderRef(
                symbol=order.symbol,
                client_order_id=client_order_id,
                exchange_order_id=order.exchange_order_id,
            )

        update = await self._client.get_order(ref)  # errors propagate; nothing changed

        if update is None:
            await self._classify_not_found(client_order_id, ref)
        try:
            report = exchange_state_from_update(update)
        except DomainValidationError as error:
            raise ExchangeStateMismatchError(
                f"unusable exchange report for order {client_order_id}: {error}"
            ) from error
        if report.client_order_id != client_order_id:
            raise ExchangeStateMismatchError(
                f"exchange report for {report.client_order_id} does not belong to "
                f"order {client_order_id}"
            )
        async with self._account.account_lock() as locked:
            current = locked.order(client_order_id)
            if current is None:  # pragma: no cover - orders are never removed
                raise AccountStateError(f"no local order {client_order_id}")
            return await locked.apply_exchange_state(
                report, at=change_time(self._clock, floor=current.updated_at)
            )
