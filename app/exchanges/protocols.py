"""Async exchange contracts, split by capability.

* ``MarketDataClient``: public snapshots (no credentials).
* ``AccountClient``: balances and positions.
* ``TradingClient``: order placement, cancellation and queries; used only by the
  execution layer (and the kill switch).

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
