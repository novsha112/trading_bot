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
from app.domain.orders import Order, OrderUpdate
from app.domain.positions import Position
from app.exchanges.models import OrderAck


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
    async def place_order(self, order: Order) -> OrderAck:
        """Submit ``order`` (already persisted as SUBMITTING by the execution layer).

        The order carries the client order id: the idempotency key the exchange
        stores with the order, so an ambiguous outcome can be resolved by
        ``get_order``. The adapter never generates its own id and never re-sends.

        Returns an acknowledgement only; the order state is confirmed later by an
        OrderUpdate.

        Raises:
            ExchangeRejectedError: refused; the order does not exist on the exchange.
            ExchangeUnavailableError: definitely not sent.
            ExchangeAmbiguousResultError: may have been placed; reconcile, do not retry.
        """
        ...

    async def cancel_order(self, order: Order) -> None:
        """Request cancellation of ``order``.

        The adapter chooses the identifier its exchange needs: the client order id,
        or ``order.exchange_order_id`` when it is known. Returning means the request
        was accepted, not that the order is canceled: the final state (CANCELED or
        FILLED in a race) comes from an OrderUpdate.

        Raises:
            ExchangeRejectedError: refused (e.g. order already final or unknown);
                check the order state with ``get_order``.
            ExchangeUnavailableError: definitely not sent.
            ExchangeAmbiguousResultError: may have been accepted; reconcile.
        """
        ...

    async def get_order(self, *, symbol: str, client_order_id: str) -> OrderUpdate | None:
        """Current state of the order with ``client_order_id``.

        ``None`` only when the exchange confirms that no such order exists. Any
        failure to get an answer raises an ExchangeError instead.
        """
        ...

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        """All open orders of ``symbol``; an empty tuple means none are open."""
        ...
