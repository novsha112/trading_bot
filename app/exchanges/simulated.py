"""Deterministic, exchange-neutral simulated exchange (``TradingClient``).

Purpose: exercise the order lifecycle (place, cancel, query) in tests and in the
future paper mode without network access or secrets. The simulator is the
authoritative exchange state of its own instance; callers reconcile against it
exactly as against a real exchange.

Current scope:
* Instrument registry: ``SimulatedExchange(clock=..., instruments=(spec, ...))``,
  copied privately and immutable for the simulator's lifetime; duplicate symbols
  are a constructor error. An order for an unregistered symbol is rejected (fail
  closed, also with an empty registry).
* LIMIT GTC / POST_ONLY orders are accepted and rest OPEN until filled or canceled,
  after pre-trade checks against their ``InstrumentSpec``: price a multiple of
  ``tick_size``, qty a multiple of ``qty_step``, ``min_qty <= qty <= max_qty``,
  exact ``price * qty >= min_notional`` (no fees, leverage or margin). Nothing is
  corrected; a violation is ``ExchangeRejectedError`` with no side effect (no id,
  no clock read, nothing stored). These checks run only for a new client order id:
  an identical retry of an accepted order returns its ack first.
* Fills come only from an explicit simulation input,
  ``fill_crossed_limit_orders(symbol, execution_price, available_qty=None)`` (not
  part of ``TradingClient``, not market data): every OPEN / PARTIALLY_FILLED order
  of that symbol with BUY ``execution_price <= limit`` or SELL
  ``execution_price >= limit`` executes at ``execution_price`` (price-improvement
  model). ``available_qty=None`` fills each crossed order's remaining quantity;
  otherwise it is the total quantity for the whole batch, allocated in
  (created_at, client_order_id) order: ``min(remaining, still available)`` rounded
  DOWN to ``qty_step`` (an order whose share rounds to zero gets no fill, and the
  unused budget is not carried over), one fill per order per call. The remaining
  quantity is always a step multiple, so the last fill closes it exactly. The
  execution price is not checked against ``tick_size``: it is an external
  observation, not a new order price. ``Fill.qty`` is the quantity of that execution. There
  is no order book, bid/ask, depth, spread, slippage or latency: the caller fully
  determines price and liquidity. Fee data and liquidity role come only from an
  explicit fee policy (below); POST_ONLY alone is never claimed to be maker.
* Average fill price = exact cumulative notional / cumulative qty, divided with an
  explicit context (``AVERAGE_PRICE_PRECISION`` significant digits,
  ROUND_HALF_EVEN), independent of the global decimal context; the first fill's
  average is its execution price. Quantities and notionals are exact (fail closed
  if they need more than ``_EXACT_PRECISION`` digits).
* Positions: every committed fill is applied to a one-way net position ledger
  (``simulated_positions.SimulatedPositionLedger``), read through the
  simulation-only ``get_position(symbol)`` (None before the first fill). Gross
  realized PnL only; no mark price, unrealized PnL unknown while open.
* Reduce-only (deterministic simulator policy, not a claim about any exchange):
  a new reduce-only order must reduce the current position (SELL a long, BUY a
  short), else ``ExchangeRejectedError``; its size may exceed the position. At
  every fill the position prepared so far in the batch is authoritative: the fill
  is capped at the reducible quantity, never opening, increasing or reversing a
  position. A crossed reduce-only order that can no longer reduce (position flat
  or on the other side, before or after its fill) is CANCELED with its fills kept;
  such a cancel uses no execution id and no budget.
* Fees are opt-in: ``fees=SimulatedFeePolicy(schedule, liquidity_role)`` gives
  every fill ``fee = execution price * fill qty * rate`` of the configured role
  (``simulated_fees``), its fee asset and ``is_maker`` from that assumed role;
  without it fee, fee asset and liquidity role stay None (unknown, not zero).
  Fees are computed while preparing the batch. Position realized PnL stays gross.
* No balances, leverage or margin. Requests whose outcome cannot be
  determined without them are refused (``ExchangeRejectedError``): MARKET (no
  execution price model), LIMIT IOC / FOK (never rest on the book, their outcome
  depends on matching).

Semantics:
* Idempotency by ``client_order_id`` (unique per instance, across symbols): a repeat
  with value-equal terms returns the original ``OrderAck`` and creates nothing, in
  any state of the order; different terms raise ``ExchangeDuplicateOrderError``.
* Exchange order ids are opaque strings from a per-instance monotonic sequence,
  assigned only to accepted orders.
* ``cancel_order``: OPEN / PARTIALLY_FILLED -> CANCELED, keeping the filled
  quantity and average. Unknown order or an already final order ->
  ``ExchangeRejectedError`` without any state change (deterministic on repeats).
* ``get_order``: lookup by symbol + ``client_order_id``; another symbol is a
  different namespace -> ``None``. A given ``exchange_order_id`` that contradicts the
  order -> ``ExchangeRejectedError`` (never ``None``: the order exists).
* ``get_open_orders``: active (OPEN, PARTIALLY_FILLED) orders of one symbol, by
  (created_at, client_order_id).
* Fill batches: crossed orders are processed in (created_at, client_order_id)
  order (a simulation determinism rule, not exchange price-time priority) with
  consecutive execution ids from a per-instance sequence. The clock is read once
  per batch that changes anything (a fill or a reduce-only auto-cancel); all fills
  and order updates of the batch carry that time. A batch is all-or-nothing:
  orders are prepared one after another on working copies (each sees the position
  after the previous fills), then positions, orders and the execution sequence
  are committed together. A record violating the status / fill invariants raises
  before anything is committed and is never repaired.
* All timestamps come from the injected ``Clock``. Nothing is ever ambiguous: there
  is no transport.

Methods never await, so each call is atomic within one event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
)
from typing import Final

from app.domain.clock import Clock
from app.domain.enums import OrderStatus, OrderType, RoundingDirection, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.fills import Fill
from app.domain.instrument import InstrumentSpec
from app.domain.orders import OrderUpdate
from app.domain.positions import Position
from app.domain.rounding import (
    is_price_aligned,
    is_qty_aligned,
    meets_min_notional,
    meets_min_qty,
    round_qty,
)
from app.domain.validation import require_text, require_utc
from app.exchanges.errors import (
    ExchangeDuplicateOrderError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.simulated_fees import SimulatedFeePolicy
from app.exchanges.simulated_positions import SimulatedPositionLedger

EXCHANGE_ORDER_ID_PREFIX: Final = "SIM-"
EXEC_ID_PREFIX: Final = "SIM-EXEC-"
_ACTIVE_STATUSES: Final = frozenset({OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED})
_RESTING_TIME_IN_FORCE: Final = frozenset({TimeInForce.GTC, TimeInForce.POST_ONLY})
_ZERO: Final = Decimal(0)
# Quantities and notionals are computed exactly; a value needing more digits than
# this fails closed instead of being rounded.
_EXACT_PRECISION: Final = 80
# Average fill price: significant digits, ROUND_HALF_EVEN (only a repeating
# quotient is ever rounded; the notional it is computed from stays exact).
AVERAGE_PRICE_PRECISION: Final = 40


def _exact_context() -> Context:
    return Context(
        prec=_EXACT_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
    )


def _average_context() -> Context:
    return Context(
        prec=AVERAGE_PRICE_PRECISION,
        rounding=ROUND_HALF_EVEN,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _SimulatedOrder:
    """Exchange-side record: the accepted request plus exchange-owned state only
    (no strategy id, version or local lifecycle states such as SUBMITTING)."""

    request: OrderRequest
    exchange_order_id: str
    status: OrderStatus
    filled_qty: Decimal
    """Cumulative executed quantity."""
    avg_fill_price: Decimal | None
    """Weighted average execution price (``filled_notional / filled_qty``)."""
    filled_notional: Decimal
    """Exact sum of price * qty over all fills; source of the average, so rounding
    of one average never carries into the next."""
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        self.check_invariants()

    def check_invariants(self) -> None:
        """Exchange-side consistency of status and fill state (not a second state
        machine: transitions are decided by the simulator)."""
        qty = self.request.qty
        filled = self.filled_qty
        avg = self.avg_fill_price
        has_fills = filled > 0
        consistent = (
            0 <= filled <= qty
            and (avg is None) == (not has_fills)
            and (avg is None or avg > 0)
            and (self.filled_notional > 0) == has_fills
            and self.filled_notional >= 0
        )
        if self.status is OrderStatus.OPEN:
            consistent = consistent and not has_fills
        elif self.status is OrderStatus.PARTIALLY_FILLED:
            consistent = consistent and 0 < filled < qty
        elif self.status is OrderStatus.FILLED:
            consistent = consistent and filled == qty
        elif self.status is OrderStatus.CANCELED:
            consistent = consistent and filled < qty
        else:
            consistent = False
        if not consistent:
            raise RuntimeError(
                f"simulated exchange: order state invariant violated for "
                f"{self.request.client_order_id} ({self.status.value}, filled {filled} of {qty})"
            )

    def remaining_qty(self) -> Decimal:
        return _exact_context().subtract(self.request.qty, self.filled_qty)

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
    record: _SimulatedOrder,
    *,
    exec_id: str,
    execution_price: Decimal,
    qty: Decimal,
    exchange_ts: datetime,
    fees: SimulatedFeePolicy | None,
) -> Fill:
    """One execution of ``qty`` of ``record``. Without a fee policy the fee, fee
    asset and liquidity role are unknown (None), never zero."""
    fee: Decimal | None = None
    fee_asset: str | None = None
    is_maker: bool | None = None
    if fees is not None:
        fee, fee_asset, is_maker = fees.fill_fee(price=execution_price, qty=qty)
    return Fill(
        exec_id=exec_id,
        exchange_order_id=record.exchange_order_id,
        client_order_id=record.request.client_order_id,
        symbol=record.request.symbol,
        side=record.request.side,
        price=execution_price,
        qty=qty,
        fee=fee,
        fee_asset=fee_asset,
        is_maker=is_maker,
        exchange_ts=exchange_ts,
    )


def _apply_fill(
    record: _SimulatedOrder, *, execution_price: Decimal, qty: Decimal, at: datetime
) -> _SimulatedOrder:
    """The record after one execution of ``qty`` (0 < qty <= remaining)."""
    exact = _exact_context()
    filled = exact.add(record.filled_qty, qty)
    notional = exact.add(record.filled_notional, exact.multiply(execution_price, qty))
    if record.filled_qty == 0:
        average = execution_price
    else:
        average = _average_context().divide(notional, filled)
    return replace(
        record,
        status=OrderStatus.FILLED if filled == record.request.qty else OrderStatus.PARTIALLY_FILLED,
        filled_qty=filled,
        avg_fill_price=average,
        filled_notional=notional,
        updated_at=at,
    )


def _require_positive_decimal(value: object, field: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise ExchangeRequestValidationError(
            f"simulated exchange: {field} must be a finite Decimal > 0"
        )
    return value


def _check_instrument_rules(order: OrderRequest, spec: InstrumentSpec) -> None:
    """Pre-trade checks of a new LIMIT order against its instrument. Values are
    never corrected; any violation rejects the order (``ExchangeRejectedError``).
    The domain helpers compute exactly in a private context (global decimal
    context neither read nor modified)."""
    price = order.price
    if price is None:  # only LIMIT orders reach this point
        raise ExchangeRejectedError("simulated exchange: limit price required")
    try:
        problems = [
            (is_price_aligned(price, spec), f"price {price} is not a multiple of tick_size"),
            (is_qty_aligned(order.qty, spec), f"qty {order.qty} is not a multiple of qty_step"),
            (meets_min_qty(order.qty, spec), f"qty {order.qty} is below min_qty {spec.min_qty}"),
            (order.qty <= spec.max_qty, f"qty {order.qty} is above max_qty {spec.max_qty}"),
            (
                meets_min_notional(price, order.qty, spec),
                f"notional is below min_notional {spec.min_notional}",
            ),
        ]
    except DomainValidationError:
        raise ExchangeRejectedError(
            f"simulated exchange: order {order.client_order_id} cannot be validated exactly "
            f"against {spec.symbol}"
        ) from None
    for passed, message in problems:
        if not passed:
            raise ExchangeRejectedError(
                f"simulated exchange: order {order.client_order_id} rejected: {message}"
            )


def _reducible_qty(side: Side, position_qty: Decimal) -> Decimal:
    """How much a reduce-only order of ``side`` may execute against the signed
    position: a SELL reduces a long, a BUY reduces a short; otherwise nothing."""
    if side is Side.SELL and position_qty > 0:
        return position_qty
    if side is Side.BUY and position_qty < 0:
        return -position_qty
    return _ZERO


def _fill_qty(
    record: _SimulatedOrder, spec: InstrumentSpec, left: Decimal | None, cap: Decimal | None
) -> Decimal:
    """Quantity of this order's fill: the remaining quantity, limited by ``cap``
    (reduce-only: what the position allows) and by the budget ``left``, rounded
    DOWN to ``qty_step`` (0 if less than one step)."""
    remaining = record.remaining_qty()
    try:
        aligned = is_qty_aligned(remaining, spec)
    except DomainValidationError:
        aligned = False
    if not aligned:
        raise RuntimeError(
            f"simulated exchange: order state invariant violated for "
            f"{record.request.client_order_id}: remaining {remaining} is not a multiple "
            f"of qty_step {spec.qty_step}"
        )
    limit = remaining if cap is None else min(remaining, cap)
    if left is not None:
        limit = min(limit, left)
    if limit == remaining:
        return remaining
    if limit < spec.qty_step:
        return _ZERO
    try:
        return round_qty(limit, spec, RoundingDirection.DOWN)
    except DomainValidationError:
        raise ExchangeRequestValidationError(
            "simulated exchange: available_qty cannot be allocated exactly"
        ) from None


def _require_symbol(symbol: object) -> str:
    try:
        return require_text(symbol, "symbol")
    except DomainValidationError:
        raise ExchangeRequestValidationError("simulated exchange: invalid symbol") from None


class SimulatedExchange:
    """In-memory ``TradingClient`` with deterministic ids and injected time."""

    __slots__ = (
        "_clock",
        "_exec_sequence",
        "_fees",
        "_instruments",
        "_orders",
        "_positions",
        "_sequence",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        instruments: tuple[InstrumentSpec, ...] = (),
        fees: SimulatedFeePolicy | None = None,
    ) -> None:
        if fees is not None and not isinstance(fees, SimulatedFeePolicy):
            raise TypeError("fees must be a SimulatedFeePolicy or None")
        registry: dict[str, InstrumentSpec] = {}
        for spec in tuple(instruments):
            if not isinstance(spec, InstrumentSpec):
                raise TypeError("instruments must contain only InstrumentSpec")
            if spec.symbol in registry:
                raise ValueError(f"duplicate instrument {spec.symbol}")
            registry[spec.symbol] = spec
        self._clock = clock
        # Private copy, immutable for the lifetime of the simulator.
        self._instruments = registry
        self._orders: dict[str, _SimulatedOrder] = {}
        self._sequence = 0
        self._exec_sequence = 0
        self._positions = SimulatedPositionLedger()
        # Immutable for the lifetime of the simulator; None = fees not modeled.
        self._fees = fees

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
        spec = self._instruments.get(order.symbol)
        if spec is None:
            raise ExchangeRejectedError(f"simulated exchange: unknown instrument {order.symbol}")
        _check_instrument_rules(order, spec)
        if order.reduce_only:
            self._check_reduce_only_direction(order)
        now = self._now()
        record = _SimulatedOrder(
            request=order,
            exchange_order_id=self._next_exchange_order_id(),
            status=OrderStatus.OPEN,
            filled_qty=_ZERO,
            avg_fill_price=None,
            filled_notional=_ZERO,
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
        self,
        *,
        symbol: str,
        execution_price: Decimal,
        available_qty: Decimal | None = None,
    ) -> tuple[Fill, ...]:
        """Simulation input: execute OPEN / PARTIALLY_FILLED limit orders of
        ``symbol`` crossed by ``execution_price``, at that price.

        ``available_qty=None``: unlimited, every crossed order fills its remaining
        quantity. Otherwise the total quantity for the whole batch, allocated in
        (created_at, client_order_id) order. At most one fill per order per call.
        Returns the new fills in processing order; empty when nothing executed.
        """
        _require_symbol(symbol)
        _require_positive_decimal(execution_price, "execution_price")
        if available_qty is not None:
            _require_positive_decimal(available_qty, "available_qty")
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

        # Prepare the whole batch sequentially without touching any state: each
        # order sees the position prepared after the previous orders of the batch.
        if not crossed:
            return ()
        spec = self._instruments.get(symbol)
        if spec is None:
            raise RuntimeError(
                f"simulated exchange: order state invariant violated: orders of "
                f"{symbol} have no registered instrument"
            )
        positions = self._positions.begin_batch()
        batch_ts: datetime | None = None  # read once, at the first state change
        sequence = self._exec_sequence
        fills: list[Fill] = []
        new_records: list[_SimulatedOrder] = []
        left = available_qty
        try:
            for record in crossed:
                record.check_invariants()  # never repair a corrupted record
                side = record.request.side
                cap: Decimal | None = None
                if record.request.reduce_only:
                    cap = _reducible_qty(side, positions.signed_qty(symbol))
                    if cap == 0:
                        # Cannot reduce anything any more: cancel, no fill, no budget used.
                        if batch_ts is None:
                            batch_ts = self._now()
                        new_records.append(
                            replace(record, status=OrderStatus.CANCELED, updated_at=batch_ts)
                        )
                        continue
                qty = _fill_qty(record, spec, left, cap)
                if qty == 0:
                    continue  # budget below one step
                if batch_ts is None:
                    batch_ts = self._now()
                sequence += 1
                fill = _build_fill(
                    record,
                    exec_id=f"{EXEC_ID_PREFIX}{sequence:010d}",
                    execution_price=execution_price,
                    qty=qty,
                    exchange_ts=batch_ts,
                    fees=self._fees,
                )
                filled = _apply_fill(record, execution_price=execution_price, qty=qty, at=batch_ts)
                positions.apply(fill)
                if (
                    record.request.reduce_only
                    and filled.status is OrderStatus.PARTIALLY_FILLED
                    and _reducible_qty(side, positions.signed_qty(symbol)) == 0
                ):
                    # The position is closed: the rest could only open or reverse it.
                    filled = replace(filled, status=OrderStatus.CANCELED)
                fills.append(fill)
                new_records.append(filled)
                if left is not None:
                    left = _exact_context().subtract(left, qty)
        except DecimalException:
            raise ExchangeRequestValidationError(
                "simulated exchange: fill quantities cannot be computed exactly"
            ) from None
        if not new_records:
            return ()

        # Commit: positions, orders and the execution sequence together.
        self._positions.commit(positions.prepared())
        for new_record in new_records:
            self._orders[new_record.request.client_order_id] = new_record
        self._exec_sequence = sequence
        return tuple(fills)

    def _check_reduce_only_direction(self, order: OrderRequest) -> None:
        """A new reduce-only order must reduce the current position: SELL a long,
        BUY a short. Its size is not limited here; fills are capped at execution."""
        position = self._positions.get_position(order.symbol)
        position_qty = _ZERO if position is None else position.qty
        if _reducible_qty(order.side, position_qty) == 0:
            state = "no" if position_qty == 0 else ("a long" if position_qty > 0 else "a short")
            raise ExchangeRejectedError(
                f"simulated exchange: reduce_only {order.side.value} order "
                f"{order.client_order_id} would not reduce {state} {order.symbol} position"
            )

    async def get_position(self, *, symbol: str) -> Position | None:
        """Simulation-only read (not part of ``TradingClient``): the net position of
        ``symbol``, or None if it never had a fill. Never reads the clock."""
        return self._positions.get_position(_require_symbol(symbol))
