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
  explicit context (``fill_math.AVERAGE_PRICE_PRECISION`` significant digits,
  ROUND_HALF_EVEN), independent of the global decimal context; the first fill's
  average is its execution price (shared rule: ``app.domain.fill_math``).
  Quantities and notionals are exact (fail closed
  if they need more than ``_EXACT_PRECISION`` digits).
* Positions: every committed fill is applied to a one-way net position ledger
  (``simulated_positions.SimulatedPositionLedger``), read through the
  simulation-only ``get_position(symbol)`` (None before the first fill). Gross
  realized PnL only.
* Mark-to-market: ``set_mark_price(symbol, mark_price)`` is an explicit simulation
  input for a registered instrument (never inferred from executions, limits or
  entries); the mark is stored per symbol, may precede the first fill and
  survives flat. Positions are valued at the stored mark from the exact basis;
  without a mark an open position's unrealized PnL is unknown (None). Marks and
  fills have separate time streams: a stale mark is rejected, a newer mark never
  makes a fill stale; ``Position.updated_at`` is the later of both. Marks never
  move cash.
* Equity: ``get_equity_state()`` is a read model (never stored) of exact cash plus
  the exact unrealized PnL of all positions at the stored marks; None without
  cash accounting or while any open position has no mark. Domain ``Balance`` is
  not used (no available / margin semantics).
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
* Cash accounting is opt-in: ``cash=SimulatedCashConfig(asset, starting_cash)``
  keeps ``starting_cash + gross realized PnL - fees`` of one asset
  (``simulated_accounting``), fed per fill with the exact realized delta from the
  position ledger and the fill's fee. It requires a fee policy in the same asset
  and instruments quoted in it (checked at construction). Opening notional does
  not move cash; no unrealized PnL, equity, funding or margin.
* No ``Balance`` (it would need equity / available), leverage or margin.
  Requests whose outcome cannot be determined without a matching model are
  refused (``ExchangeRejectedError``): MARKET (no execution price model), LIMIT
  IOC / FOK (never rest on the book, their outcome depends on matching).

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
  after the previous fills), then positions, cash, orders and the execution
  sequence are committed together. A record violating the status / fill invariants raises
  before anything is committed and is never repaired.
* All timestamps come from the injected ``Clock``. Nothing is ever ambiguous: there
  is no transport.

Recovery reads (``ExchangeStateReader``; the simulator's whole state is its scope):
* ``list_open_orders()``: every OPEN / PARTIALLY_FILLED order (bot-placed and
  external) with its full terms and exact cumulative notional, by
  (created_ts, exchange_order_id); ``server_ts`` is one clock read.
* ``get_position_snapshot()``: the signed quantity of every symbol that ever had
  a fill (flat included), ``complete=True``; after the test input
  ``restrict_position_snapshot(symbols)`` only those symbols and
  ``complete=False``.
* ``list_executions(query, cursor=None)``: the execution history (every fill as
  a TRADE, plus test-recorded executions) matching the query, inclusive window,
  ordered by (exchange_ts, exec_id) for reproducibility only (no exchange
  ordering is claimed), in pages of ``execution_page_size``. Cursors are opaque,
  deterministic and bound to their query: an unknown cursor or one of another
  query -> ``ExchangeRejectedError``. ``set_execution_page_failures({n, ...})``
  makes page n fail with ``ExchangeResponseError`` on every request (nothing
  returned). A page the DTO rejects (e.g. one exec id twice) is a response error.
* Test inputs (not part of any protocol): ``add_external_order`` (an order placed
  outside the bot, optionally without client id: listed, never matched or
  filled) and ``record_external_execution`` (a raw history record; no effect on
  orders, positions or cash, duplicates allowed).

Methods never await, so each call is atomic within one event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import (
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
from app.domain.fill_math import accumulate_execution
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
    ExchangeResponseError,
)
from app.exchanges.models import OrderAck, OrderRef, OrderRequest
from app.exchanges.recovery import (
    ExchangeExecution,
    ExchangeOrder,
    ExchangePosition,
    ExecutionKind,
    ExecutionPage,
    ExecutionQuery,
    OpenOrdersSnapshot,
    PositionSnapshot,
)
from app.exchanges.simulated_accounting import (
    CashState,
    EquityState,
    SimulatedCashConfig,
    SimulatedCashLedger,
    equity_state,
)
from app.exchanges.simulated_fees import SimulatedFeePolicy
from app.exchanges.simulated_positions import MarkQuote, SimulatedPositionLedger

EXCHANGE_ORDER_ID_PREFIX: Final = "SIM-"
EXEC_ID_PREFIX: Final = "SIM-EXEC-"
CURSOR_PREFIX: Final = "SIM-CURSOR-"
DEFAULT_EXECUTION_PAGE_SIZE: Final = 50
"""Simulator default only; not any exchange's limit."""
_ACTIVE_STATUSES: Final = frozenset({OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED})
_RESTING_TIME_IN_FORCE: Final = frozenset({TimeInForce.GTC, TimeInForce.POST_ONLY})
_ZERO: Final = Decimal(0)
# Quantities and notionals are computed exactly; a value needing more digits than
# this fails closed instead of being rounded.
_EXACT_PRECISION: Final = 80


def _exact_context() -> Context:
    return Context(
        prec=_EXACT_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
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

    def to_exchange_order(self) -> ExchangeOrder:
        request = self.request
        return ExchangeOrder(
            exchange_order_id=self.exchange_order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            price=request.price,
            qty=request.qty,
            time_in_force=request.time_in_force,
            reduce_only=request.reduce_only,
            status=self.status,
            cum_filled_qty=self.filled_qty,
            cum_filled_notional=self.filled_notional,
            avg_fill_price=self.avg_fill_price,
            created_ts=self.created_at,
            updated_ts=self.updated_at,
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
    totals = accumulate_execution(
        filled_qty=record.filled_qty,
        filled_notional=record.filled_notional,
        price=execution_price,
        qty=qty,
    )
    return replace(
        record,
        status=(
            OrderStatus.FILLED
            if totals.filled_qty == record.request.qty
            else OrderStatus.PARTIALLY_FILLED
        ),
        filled_qty=totals.filled_qty,
        avg_fill_price=totals.avg_fill_price,
        filled_notional=totals.filled_notional,
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
        return position_qty.copy_negate()  # exact; -x would use the global context
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


def _trade_execution(fill: Fill) -> ExchangeExecution:
    return ExchangeExecution(
        exec_id=fill.exec_id,
        exchange_order_id=fill.exchange_order_id,
        client_order_id=fill.client_order_id,
        symbol=fill.symbol,
        side=fill.side,
        price=fill.price,
        qty=fill.qty,
        fee=fill.fee,
        fee_asset=fill.fee_asset,
        is_maker=fill.is_maker,
        kind=ExecutionKind.TRADE,
        exchange_ts=fill.exchange_ts,
    )


def _require_symbol(symbol: object) -> str:
    try:
        return require_text(symbol, "symbol")
    except DomainValidationError:
        raise ExchangeRequestValidationError("simulated exchange: invalid symbol") from None


class SimulatedExchange:
    """In-memory ``TradingClient`` with deterministic ids and injected time."""

    __slots__ = (
        "_cash",
        "_clock",
        "_cursor_sequence",
        "_cursors",
        "_exec_sequence",
        "_executions",
        "_external_orders",
        "_failing_pages",
        "_fees",
        "_instruments",
        "_marks",
        "_orders",
        "_page_size",
        "_position_scope",
        "_positions",
        "_sequence",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        instruments: tuple[InstrumentSpec, ...] = (),
        fees: SimulatedFeePolicy | None = None,
        cash: SimulatedCashConfig | None = None,
        execution_page_size: int = DEFAULT_EXECUTION_PAGE_SIZE,
    ) -> None:
        if type(execution_page_size) is not int or execution_page_size <= 0:
            raise ValueError("execution_page_size must be an int > 0")
        if fees is not None and not isinstance(fees, SimulatedFeePolicy):
            raise TypeError("fees must be a SimulatedFeePolicy or None")
        if cash is not None and not isinstance(cash, SimulatedCashConfig):
            raise TypeError("cash must be a SimulatedCashConfig or None")
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
        # External mark prices (simulation input), independent of positions.
        self._marks: dict[str, MarkQuote] = {}
        # Immutable for the lifetime of the simulator; None = fees not modeled.
        self._fees = fees
        if cash is not None:
            # Cash must stay fully known: every fill needs a known fee in the cash
            # asset, and realized PnL of every instrument is in its quote asset.
            if fees is None:
                raise ValueError("cash accounting requires a fee policy")
            if fees.schedule.fee_asset != cash.asset:
                raise ValueError(
                    f"fee asset {fees.schedule.fee_asset} differs from cash asset {cash.asset}"
                )
            for spec in registry.values():
                if spec.quote_asset != cash.asset:
                    raise ValueError(
                        f"{spec.symbol} quote asset {spec.quote_asset} differs from "
                        f"cash asset {cash.asset} (no conversion)"
                    )
        self._cash = None if cash is None else SimulatedCashLedger(cash)
        # Recovery reads: execution history, external orders, cursors, test inputs.
        self._executions: list[ExchangeExecution] = []
        self._external_orders: dict[str, ExchangeOrder] = {}
        self._page_size = execution_page_size
        self._cursors: dict[str, tuple[ExecutionQuery, int]] = {}
        self._cursor_sequence = 0
        self._failing_pages: frozenset[int] = frozenset()
        self._position_scope: tuple[str, ...] | None = None

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
        if any(o.client_order_id == order.client_order_id for o in self._external_orders.values()):
            raise ExchangeDuplicateOrderError(
                f"simulated exchange: client_order_id {order.client_order_id} belongs to "
                "an external order"
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
        cash = None if self._cash is None else self._cash.begin_batch()
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
                realized_delta = positions.apply(fill)
                if cash is not None:
                    if realized_delta is None:  # fresh exec ids are never replays
                        raise RuntimeError(
                            f"simulated exchange: fill {fill.exec_id} was not applied"
                        )
                    cash.apply(
                        exec_id=fill.exec_id,
                        realized_delta=realized_delta,
                        fee=fill.fee,
                        fee_asset=fill.fee_asset,
                    )
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

        # The published position (valued at the stored mark) must be computable
        # before anything is committed.
        positions.position(symbol, mark=self._marks.get(symbol))

        # Commit: positions, cash, orders and the execution sequence together.
        prepared_positions = positions.prepared()
        prepared_cash = None if cash is None else cash.prepared()
        self._positions.commit(prepared_positions)
        if self._cash is not None and prepared_cash is not None:
            self._cash.commit(prepared_cash)
        for new_record in new_records:
            self._orders[new_record.request.client_order_id] = new_record
        self._exec_sequence = sequence
        self._executions.extend(_trade_execution(fill) for fill in fills)
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

    # --- recovery reads (ExchangeStateReader) -------------------------------------

    async def list_open_orders(self) -> OpenOrdersSnapshot:
        """Every open order of the simulator (bot-placed and external)."""
        orders = [
            record.to_exchange_order()
            for record in self._orders.values()
            if record.status in _ACTIVE_STATUSES
        ]
        orders.extend(self._external_orders.values())
        orders.sort(key=lambda order: (order.created_ts, order.exchange_order_id))
        return OpenOrdersSnapshot(orders=tuple(orders), server_ts=self._now())

    async def get_position_snapshot(self) -> PositionSnapshot:
        """Signed positions of every symbol that ever had a fill; partial (and
        ``complete=False``) only after ``restrict_position_snapshot``."""
        quantities = self._positions.signed_quantities()
        scope = self._position_scope
        positions = tuple(
            ExchangePosition(symbol=symbol, qty=qty)
            for symbol, qty in quantities.items()
            if scope is None or symbol in scope
        )
        return PositionSnapshot(positions=positions, complete=scope is None, server_ts=self._now())

    async def list_executions(
        self, query: ExecutionQuery, *, cursor: str | None = None
    ) -> ExecutionPage:
        """One page of the execution history matching ``query``."""
        if type(query) is not ExecutionQuery:
            raise ExchangeRequestValidationError("simulated exchange: expected an ExecutionQuery")
        offset = 0
        if cursor is not None:
            known = self._cursors.get(cursor) if type(cursor) is str else None
            if known is None:
                raise ExchangeRejectedError("simulated exchange: unknown execution cursor")
            if known[0] != query:
                raise ExchangeRejectedError(
                    "simulated exchange: execution cursor belongs to another query"
                )
            offset = known[1]
        page_number = offset // self._page_size + 1
        if page_number in self._failing_pages:
            raise ExchangeResponseError(
                f"simulated exchange: execution page {page_number} failed (injected)"
            )
        matching = sorted(
            (execution for execution in self._executions if query.matches(execution)),
            key=lambda execution: (execution.exchange_ts, execution.exec_id),
        )
        end = offset + self._page_size
        next_cursor: str | None = None
        if end < len(matching):
            self._cursor_sequence += 1
            next_cursor = f"{CURSOR_PREFIX}{self._cursor_sequence:010d}"
            self._cursors[next_cursor] = (query, end)
        try:
            return ExecutionPage(
                query=query, executions=tuple(matching[offset:end]), next_cursor=next_cursor
            )
        except DomainValidationError as error:
            raise ExchangeResponseError(
                f"simulated exchange: invalid execution page {page_number}: {error}"
            ) from None

    # --- test inputs for recovery scenarios (not part of any protocol) ------------

    def add_external_order(
        self,
        *,
        client_order_id: str | None,
        symbol: str,
        side: Side,
        price: Decimal,
        qty: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> ExchangeOrder:
        """Simulation input: an OPEN limit order placed outside the bot (any
        symbol, optionally without client id). It is listed by
        ``list_open_orders`` but never matched, filled or canceled here. A client
        id already used by any order of this simulator is refused."""
        if client_order_id is not None and (
            client_order_id in self._orders
            or any(o.client_order_id == client_order_id for o in self._external_orders.values())
        ):
            raise ExchangeDuplicateOrderError(
                f"simulated exchange: client_order_id {client_order_id} already exists"
            )
        now = self._now()
        try:
            order = ExchangeOrder(
                exchange_order_id=self._next_exchange_order_id(),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                qty=qty,
                time_in_force=time_in_force,
                reduce_only=reduce_only,
                status=OrderStatus.OPEN,
                cum_filled_qty=_ZERO,
                cum_filled_notional=None,
                avg_fill_price=None,
                created_ts=now,
                updated_ts=now,
            )
        except DomainValidationError as error:
            raise ExchangeRequestValidationError(
                f"simulated exchange: invalid external order: {error}"
            ) from None
        self._external_orders[order.exchange_order_id] = order
        return order

    def record_external_execution(self, execution: ExchangeExecution) -> None:
        """Simulation input: append a raw execution record to the history (no
        effect on orders, positions or cash; the same exec id may be recorded
        again, identical or not)."""
        if type(execution) is not ExchangeExecution:
            raise ExchangeRequestValidationError(
                "simulated exchange: expected an ExchangeExecution"
            )
        self._executions.append(execution)

    def set_execution_page_failures(self, pages: frozenset[int]) -> None:
        """Simulation input: every request for one of these (1-based) execution
        pages raises ``ExchangeResponseError`` until replaced (empty = none)."""
        if type(pages) is not frozenset or not all(
            type(page) is int and page > 0 for page in pages
        ):
            raise ValueError("pages must be a frozenset of ints > 0")
        self._failing_pages = pages

    def restrict_position_snapshot(self, symbols: tuple[str, ...] | None) -> None:
        """Simulation input: later position snapshots contain only ``symbols``
        and are partial (``complete=False``); None restores the complete one."""
        if symbols is not None:
            if type(symbols) is not tuple:
                raise ValueError("symbols must be a tuple or None")
            for symbol in symbols:
                _require_symbol(symbol)
        self._position_scope = symbols

    async def get_cash_state(self) -> CashState | None:
        """Simulation-only read (not part of ``TradingClient``): cash components of
        the accounting asset, or None when cash accounting is not configured. Cash
        is not equity (no unrealized PnL). Never reads the clock."""
        return None if self._cash is None else self._cash.state()

    async def get_equity_state(self) -> EquityState | None:
        """Simulation-only derived read model (not part of ``TradingClient``):
        ``equity = exact cash + exact unrealized PnL of all positions`` at the
        stored marks, rounded only when published.

        None when cash accounting is off, or when any open position has no mark
        (equity is unknown, never computed with a zero in its place). Flat
        positions need no mark. Pure: no clock read, no state change."""
        if self._cash is None:
            return None
        unrealized = self._positions.exact_unrealized_total(self._marks)
        if unrealized is None:
            return None
        return equity_state(
            asset=self._cash.config.asset,
            exact_cash=self._cash.exact_cash(),
            exact_unrealized=unrealized,
        )

    async def get_position(self, *, symbol: str) -> Position | None:
        """Simulation-only read (not part of ``TradingClient``): the net position of
        ``symbol``, or None if it never had a fill. Never reads the clock."""
        symbol = _require_symbol(symbol)
        return self._positions.get_position(symbol, mark=self._marks.get(symbol))

    async def set_mark_price(self, *, symbol: str, mark_price: Decimal) -> Position | None:
        """Simulation-only input (not part of ``TradingClient``, not inferred from
        executions): the mark price of a registered instrument, observed now.

        Revalues the position (exact basis) without changing quantities, basis,
        realized PnL, cash, orders or ids. Returns the revalued position, or None if
        the symbol has never had a fill (the mark is kept for later). A mark older
        than the stored one is an invariant error; equal times are allowed."""
        symbol = _require_symbol(symbol)
        _require_positive_decimal(mark_price, "mark_price")
        if symbol not in self._instruments:
            raise ExchangeRejectedError(f"simulated exchange: unknown instrument {symbol}")
        quote = MarkQuote(price=mark_price, at=self._now())
        previous = self._marks.get(symbol)
        if previous is not None and quote.at < previous.at:
            raise RuntimeError(
                f"simulated exchange: invariant violated: stale mark for {symbol} "
                f"({quote.at.isoformat()} < {previous.at.isoformat()})"
            )
        revalued = self._positions.get_position(symbol, mark=quote)  # before commit
        self._marks[symbol] = quote
        return revalued
