"""Async exchange contracts, split by capability.

* ``MarketDataClient``: public snapshots (no credentials).
* ``AccountClient``: balances and positions.
* ``TradingClient``: order placement, cancellation and queries; used only by the
  execution layer (and the kill switch).
* ``ExchangeStateReader``: read-only recovery capabilities (complete open-order
  snapshot, position snapshot, execution history pages); no mutation at all, so
  recovery code depending on it cannot send or cancel anything.

Protocols are structural and checked statically by mypy (no runtime_checkable).
Errors are reported with the classes from ``app.exchanges.errors``; a failure to
obtain data is never reported as an empty or ``None`` result.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.balances import Balance
from app.domain.instrument import InstrumentSpec
from app.domain.market import Ticker
from app.domain.orders import OrderUpdate
from app.domain.positions import Position
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.recovery import (
    ExecutionPage,
    ExecutionQuery,
    OpenOrdersSnapshot,
    PositionSnapshot,
)


class MarketDataClient(Protocol):
    async def get_instrument(self, symbol: str) -> InstrumentSpec:
        """Trading constraints of ``symbol``. Unknown symbol: ExchangeRejectedError."""
        ...

    async def get_ticker(self, symbol: str) -> Ticker:
        """Latest price snapshot of ``symbol``."""
        ...


class AccountClient(Protocol):
    async def get_balances(self) -> tuple[Balance, ...]:
        """Balances of the trading account."""
        ...

    async def get_positions(self) -> tuple[Position, ...]:
        """Open positions of the trading account."""
        ...


class TradingClient(Protocol):
    async def place_order(self, order: OrderRequest) -> OrderAck:
        """Submit a placement request.

        ``order.client_order_id`` is the idempotency key; the adapter sends it with
        the request, never generates its own and never re-sends on its own.

        Returns an acknowledgement only; the order state is confirmed later by an
        OrderUpdate.

        Raises:
            ExchangeNotSentError: the request definitely did not reach the exchange.
            ExchangeRejectedError: refused; the order does not exist on the exchange.
            ExchangeAmbiguousResultError: may have been placed; reconcile through
                ``get_order(OrderRef(symbol, client_order_id))``, never re-send blindly.
        """
        ...

    async def cancel_order(self, order: OrderRef) -> None:
        """Request cancellation of the referenced order.

        The adapter chooses the identifier its exchange needs (client order id, or
        exchange order id when known). Returning means the request was accepted, not
        that the order is canceled: the final state (CANCELED, or FILLED in a race)
        comes from an OrderUpdate.

        Raises:
            ExchangeNotSentError: the request definitely did not reach the exchange.
            ExchangeRejectedError: refused (e.g. order already final or unknown);
                check the order state with ``get_order``.
            ExchangeAmbiguousResultError: may have been accepted; reconcile.
        """
        ...

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        """Current state of the referenced order (works with only ``client_order_id``).

        ``None`` only when the exchange confirms that no such order exists. Any
        failure to get an answer raises an ExchangeError instead.
        """
        ...

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        """All open orders of ``symbol``; an empty tuple means none are open."""
        ...


class ExchangeStateReader(Protocol):
    """Read-only exchange state for startup recovery (docs/ARCHITECTURE.md 13.3).

    The scope (account, category, settlement) is a property of the reader's own
    configuration. Read failures raise exchange errors; a partial open-order set,
    an empty page instead of an error or an unknown cursor never pass silently.
    """

    async def list_open_orders(self) -> OpenOrdersSnapshot:
        """The COMPLETE set of open orders of the scope (all pages read)."""
        ...

    async def get_position_snapshot(self) -> PositionSnapshot:
        """Positions of the scope; ``complete`` says whether an absent symbol is
        flat (True) or unknown (False)."""
        ...

    async def list_executions(
        self, query: ExecutionQuery, *, cursor: str | None = None
    ) -> ExecutionPage:
        """One page of ``query`` (``cursor`` from the previous page of the SAME
        query, None for the first). ``next_cursor is None`` = exhausted.

        Raises:
            ExchangeRejectedError: unknown cursor or a cursor of another query.
            ExchangeResponseError: no valid answer for this page (nothing returned).
        """
        ...
