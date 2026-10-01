"""Deterministic, exchange-neutral simulated exchange (``TradingClient``).

Purpose: exercise the order lifecycle (place, cancel, query) in tests and in the
future paper mode without network access or secrets. The simulator is the
authoritative exchange state of its own instance; callers reconcile against it
exactly as against a real exchange.

Current scope:
* LIMIT GTC / POST_ONLY orders are accepted and rest OPEN until filled or canceled.
* Fills come only from an explicit simulation input,
  ``fill_crossed_limit_orders(symbol, execution_price)`` (not part of
  ``TradingClient``, not market data): every OPEN order of that symbol with
  BUY ``execution_price <= limit`` or SELL ``execution_price >= limit`` is filled
  completely at ``execution_price`` (price-improvement model). There is no order
  book, bid/ask, depth, spread, slippage or latency: the caller fully determines
  the execution price. Fee, fee asset and liquidity role are unknown (``None``),
  never invented; POST_ONLY is not claimed to be maker.
* No partial fills, fees, balances or positions. Requests whose outcome cannot be
  determined without them are refused (``ExchangeRejectedError``): MARKET (no
  execution price model), LIMIT IOC / FOK (never rest on the book, their outcome
  depends on matching), ``reduce_only`` (no position model).

Semantics:
* Idempotency by ``client_order_id`` (unique per instance, across symbols): a repeat
  with value-equal terms returns the original ``OrderAck`` and creates nothing, in
  any state of the order; different terms raise ``ExchangeDuplicateOrderError``.
* Exchange order ids are opaque strings from a per-instance monotonic sequence,
  assigned only to accepted orders.
* ``cancel_order``: OPEN -> CANCELED. Unknown order or an already final order ->
  ``ExchangeRejectedError`` without any state change (deterministic on repeats).
* ``get_order``: lookup by symbol + ``client_order_id``; another symbol is a
  different namespace -> ``None``. A given ``exchange_order_id`` that contradicts the
  order -> ``ExchangeRejectedError`` (never ``None``: the order exists).
* ``get_open_orders``: active orders of one symbol, by (created_at, client_order_id).
* Fill batches: crossed orders are processed in (created_at, client_order_id)
  order (a simulation determinism rule, not exchange price-time priority) with
  consecutive execution ids from a per-instance sequence. The clock is read once
  per batch that fills anything; all fills and order updates of the batch carry
  that time. A batch is all-or-nothing: every fill and new order record is built
  first, then the state and the execution sequence are committed together.
* All timestamps come from the injected ``Clock``. Nothing is ever ambiguous: there
  is no transport.

Methods never await, so each call is atomic within one event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Final

from app.domain.clock import Clock
from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.orders import OrderUpdate
from app.domain.validation import require_text, require_utc
from app.exchanges.errors import (
    ExchangeDuplicateOrderError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest

EXCHANGE_ORDER_ID_PREFIX: Final = "SIM-"
EXEC_ID_PREFIX: Final = "SIM-EXEC-"
_ACTIVE_STATUSES: Final = frozenset({OrderStatus.OPEN})
_RESTING_TIME_IN_FORCE: Final = frozenset({TimeInForce.GTC, TimeInForce.POST_ONLY})
_ZERO: Final = Decimal(0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SimulatedOrder:
    """Exchange-side record: the accepted request plus exchange-owned state only
    (no strategy id, version or local lifecycle states such as SUBMITTING)."""

    request: OrderRequest
    exchange_order_id: str
    status: OrderStatus
    filled_qty: Decimal
    avg_fill_price: Decimal | None
    created_at: datetime
    updated_at: datetime

    def to_ack(self) -> OrderAck:
        return OrderAck(
            client_order_id=self.request.client_order_id,
            exchange_order_id=self.exchange_order_id,
            exchange_ts=self.created_at,
        )

    def to_update(self) -> OrderUpdate:
        return OrderUpdate(
            client_order_id=self.request.client_order_id,
            exchange_order_id=self.exchange_order_id,
            status=self.status,
            cum_filled_qty=self.filled_qty,
            avg_fill_price=self.avg_fill_price,
            reject_reason=None,
            exchange_ts=self.updated_at,
        )


def _crosses(record: _SimulatedOrder, execution_price: Decimal) -> bool:
    limit = record.request.price
    if limit is None:  # only LIMIT orders are ever accepted
        return False
    if record.request.side is Side.BUY:
        return execution_price <= limit
    return execution_price >= limit


def _build_fill(
    record: _SimulatedOrder, *, exec_id: str, execution_price: Decimal, exchange_ts: datetime
) -> Fill:
    """Full fill of ``record``. Fee data and liquidity role are not modeled."""
    return Fill(
        exec_id=exec_id,
        exchange_order_id=record.exchange_order_id,
        client_order_id=record.request.client_order_id,
        symbol=record.request.symbol,
        side=record.request.side,
        price=execution_price,
        qty=record.request.qty,
        fee=None,
        fee_asset=None,
        is_maker=None,
        exchange_ts=exchange_ts,
    )


def _require_symbol(symbol: object) -> str:
    try:
        return require_text(symbol, "symbol")
    except DomainValidationError:
        raise ExchangeRequestValidationError("simulated exchange: invalid symbol") from None


class SimulatedExchange:
    """In-memory ``TradingClient`` with deterministic ids and injected time."""

    __slots__ = ("_clock", "_exec_sequence", "_orders", "_sequence")

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock
        self._orders: dict[str, _SimulatedOrder] = {}
        self._sequence = 0
        self._exec_sequence = 0

    def __repr__(self) -> str:
        return f"SimulatedExchange(orders={len(self._orders)})"

    def _now(self) -> datetime:
        return require_utc(self._clock.now(), "clock.now()")

    def _next_exchange_order_id(self) -> str:
        self._sequence += 1
        return f"{EXCHANGE_ORDER_ID_PREFIX}{self._sequence:010d}"

    def _find(self, ref: object) -> _SimulatedOrder | None:
        """The order matching symbol + client id, or None. A contradicting
        exchange_order_id raises: the order exists, the reference is wrong."""
        if not isinstance(ref, OrderRef):
            raise ExchangeRequestValidationError("simulated exchange: expected an OrderRef")
        record = self._orders.get(ref.client_order_id)
        if record is None or record.request.symbol != ref.symbol:
            return None
        if ref.exchange_order_id is not None and ref.exchange_order_id != record.exchange_order_id:
            raise ExchangeRejectedError(
                f"simulated exchange: exchange_order_id mismatch for {ref.client_order_id}"
            )
        return record

    async def place_order(self, order: OrderRequest) -> OrderAck:
        if not isinstance(order, OrderRequest):
            raise ExchangeRequestValidationError("simulated exchange: expected an OrderRequest")
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            if existing.request == order:
                return existing.to_ack()
            raise ExchangeDuplicateOrderError(
                f"simulated exchange: client_order_id {order.client_order_id} already exists "
                f"with different terms"
            )
        if order.order_type is not OrderType.LIMIT:
            raise ExchangeRejectedError(
                "simulated exchange: market orders are unsupported (no execution price model)"
            )
        if order.time_in_force not in _RESTING_TIME_IN_FORCE:
            raise ExchangeRejectedError(
                f"simulated exchange: time_in_force {order.time_in_force.value} is unsupported "
                f"(no matching engine)"
            )
        if order.reduce_only:
            raise ExchangeRejectedError(
                "simulated exchange: reduce_only is unsupported (no position model)"
            )
        now = self._now()
        record = _SimulatedOrder(
            request=order,
            exchange_order_id=self._next_exchange_order_id(),
            status=OrderStatus.OPEN,
            filled_qty=_ZERO,
            avg_fill_price=None,
            created_at=now,
            updated_at=now,
        )
        self._orders[order.client_order_id] = record
        return record.to_ack()

    async def cancel_order(self, order: OrderRef) -> None:
        record = self._find(order)
        if record is None:
            raise ExchangeRejectedError(
                f"simulated exchange: order {order.client_order_id} not found"
            )
        if record.status not in _ACTIVE_STATUSES:
            raise ExchangeRejectedError(
                f"simulated exchange: order {order.client_order_id} is already "
                f"{record.status.value}"
            )
        self._orders[record.request.client_order_id] = replace(
            record, status=OrderStatus.CANCELED, updated_at=self._now()
        )

    async def get_order(self, order: OrderRef) -> OrderUpdate | None:
        record = self._find(order)
        return None if record is None else record.to_update()

    async def get_open_orders(self, *, symbol: str) -> tuple[OrderUpdate, ...]:
        _require_symbol(symbol)
        active = sorted(
            (
                record
                for record in self._orders.values()
                if record.request.symbol == symbol and record.status in _ACTIVE_STATUSES
            ),
            key=lambda record: (record.created_at, record.request.client_order_id),
        )
        return tuple(record.to_update() for record in active)

    async def fill_crossed_limit_orders(
        self, *, symbol: str, execution_price: Decimal
    ) -> tuple[Fill, ...]:
        """Simulation input: fill every OPEN limit order of ``symbol`` crossed by
        ``execution_price`` completely at that price. Returns the new fills in
        processing order; an empty tuple when nothing crossed."""
        _require_symbol(symbol)
        if (
            type(execution_price) is not Decimal
            or not execution_price.is_finite()
            or execution_price <= 0
        ):
            raise ExchangeRequestValidationError(
                "simulated exchange: execution_price must be a finite Decimal > 0"
            )
        crossed = sorted(
            (
                record
                for record in self._orders.values()
                if record.request.symbol == symbol
                and record.status in _ACTIVE_STATUSES
                and _crosses(record, execution_price)
            ),
            key=lambda record: (record.created_at, record.request.client_order_id),
        )
        if not crossed:
            return ()

        # Prepare the whole batch before touching any state.
        batch_ts = self._now()
        sequence = self._exec_sequence
        fills: list[Fill] = []
        filled_records: list[_SimulatedOrder] = []
        for record in crossed:
            sequence += 1
            fills.append(
                _build_fill(
                    record,
                    exec_id=f"{EXEC_ID_PREFIX}{sequence:010d}",
                    execution_price=execution_price,
                    exchange_ts=batch_ts,
                )
            )
            filled_records.append(
                replace(
                    record,
                    status=OrderStatus.FILLED,
                    filled_qty=record.request.qty,
                    avg_fill_price=execution_price,
                    updated_at=batch_ts,
                )
            )

        # Commit.
        for filled in filled_records:
            self._orders[filled.request.client_order_id] = filled
        self._exec_sequence = sequence
        return tuple(fills)
