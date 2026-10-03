"""Exchange-neutral read DTOs for startup recovery (docs/ARCHITECTURE.md 13.4).

What a read-only ``ExchangeStateReader`` (``app.exchanges.protocols``) reports
about the exchange: the complete open-order set of its scope, a position
snapshot with an explicit completeness flag, and pages of execution history.
They describe exchange facts only: no local lifecycle, no matching, no
recovery decision (those belong to the execution layer).

Validation follows the project conventions, strictly: exact finite ``Decimal``
(``type(x) is Decimal``: no float, int or subclass), exact ``bool``, members of
the domain enums, aware UTC ``datetime``, tuples (not lists). Values are kept
as given: no rounding, quantization or normalization of representations
(``Decimal("4.000")`` stays ``4.000``). A violation is a
``DomainValidationError``; an adapter that cannot build a valid DTO from an
exchange answer reports a response error instead.

* ``ExchangeOrder``: one exchange order with its full terms and cumulative
  execution. Only exchange-reportable statuses (``EXCHANGE_REPORTED_STATUSES``);
  the cumulative quantity must fit the status (the domain's status / fill
  rules); ``avg_fill_price`` is informational only.
* ``OpenOrdersSnapshot``: the COMPLETE set of open orders of the reader's scope
  (a partial set is never returned); unique exchange ids and unique non-None
  client ids; only OPEN / PARTIALLY_FILLED orders; ``server_ts`` not before any
  order's ``updated_ts``.
* ``ExchangePosition`` / ``PositionSnapshot``: signed quantities; ``complete``
  says whether a symbol absent from the snapshot may later be read as flat
  (True) or is unknown (False). Nothing here infers a missing symbol as zero.
* ``ExecutionKind`` / ``ExchangeExecution``: one execution; only ``TRADE`` is an
  ordinary trade. Not converted to a domain ``Fill`` here.
* ``ExecutionQuery`` / ``ExecutionPage``: a fixed query window ``[start, end]``
  (both inclusive) and one page of its answer. ``next_cursor is None`` is the
  only signal that the query is exhausted; the cursor is opaque. A page echoes
  its query exactly, contains only executions matching it and no exec id
  twice; uniqueness across pages is checked by the recovery layer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final, TypeVar

from app.domain.enums import OrderStatus, OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.order_state import EXCHANGE_REPORTED_STATUSES
from app.domain.validation import require_enum, require_text, require_utc

_T = TypeVar("_T")

OPEN_ORDER_STATUSES: Final = frozenset({OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED})
"""Statuses of an order in an open-order snapshot."""


def _exact(value: object, field: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise DomainValidationError(f"{field} must be an exact, finite Decimal, got {value!r}")
    return value


def _positive(value: object, field: str) -> Decimal:
    decimal = _exact(value, field)
    if decimal <= 0:
        raise DomainValidationError(f"{field} must be > 0, got {decimal}")
    return decimal


def _non_negative(value: object, field: str) -> Decimal:
    decimal = _exact(value, field)
    if decimal < 0:
        raise DomainValidationError(f"{field} must be >= 0, got {decimal}")
    return decimal


def _optional_text(value: object, field: str) -> str | None:
    return None if value is None else require_text(value, field)


def _exact_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise DomainValidationError(f"{field} must be a bool, got {value!r}")
    return value


def _tuple_of(value: object, kind: type[_T], field: str) -> tuple[_T, ...]:
    if type(value) is not tuple:
        raise DomainValidationError(f"{field} must be a tuple, got {type(value).__name__}")
    for item in value:
        if type(item) is not kind:
            raise DomainValidationError(f"{field} must contain only {kind.__name__}")
    return value


def _require_status_fill(status: OrderStatus, filled: Decimal, qty: Decimal) -> None:
    """The domain's status / fill rules for an exchange-reported status."""
    if status in (OrderStatus.OPEN, OrderStatus.REJECTED):
        valid = filled == 0
    elif status is OrderStatus.PARTIALLY_FILLED:
        valid = 0 < filled < qty
    elif status is OrderStatus.FILLED:
        valid = filled == qty
    else:  # CANCELED, EXPIRED: any execution short of the full quantity
        valid = filled < qty
    if not valid:
        raise DomainValidationError(
            f"cum_filled_qty {filled} of {qty} is not possible for status {status.value}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExchangeOrder:
    """One exchange order: identity, full terms and cumulative execution."""

    exchange_order_id: str
    client_order_id: str | None
    """None: the exchange reports no client id (e.g. an order placed elsewhere)."""
    symbol: str
    side: Side
    order_type: OrderType
    price: Decimal | None
    """Limit price; None exactly for a MARKET order."""
    qty: Decimal
    """Original order quantity."""
    time_in_force: TimeInForce
    reduce_only: bool
    status: OrderStatus
    cum_filled_qty: Decimal
    cum_filled_notional: Decimal | None
    """Cumulative executed value, only when the reader has proven its semantics."""
    avg_fill_price: Decimal | None
    """Informational only; never a reconciliation invariant."""
    created_ts: datetime
    updated_ts: datetime

    def __post_init__(self) -> None:
        require_text(self.exchange_order_id, "exchange_order_id")
        _optional_text(self.client_order_id, "client_order_id")
        require_text(self.symbol, "symbol")
        require_enum(self.side, Side, "side")
        order_type = require_enum(self.order_type, OrderType, "order_type")
        require_enum(self.time_in_force, TimeInForce, "time_in_force")
        if order_type is OrderType.LIMIT:
            _positive(self.price, "price of a limit order")
        elif self.price is not None:
            raise DomainValidationError(f"a market order has no price, got {self.price!r}")
        if self.time_in_force is TimeInForce.POST_ONLY and order_type is not OrderType.LIMIT:
            raise DomainValidationError("post_only time_in_force requires a limit order")
        qty = _positive(self.qty, "qty")
        _exact_bool(self.reduce_only, "reduce_only")
        status = require_enum(self.status, OrderStatus, "status")
        if status not in EXCHANGE_REPORTED_STATUSES:
            raise DomainValidationError(f"status {status.value} is not an exchange-reported status")
        filled = _non_negative(self.cum_filled_qty, "cum_filled_qty")
        if filled > qty:
            raise DomainValidationError(f"cum_filled_qty {filled} exceeds qty {qty}")
        _require_status_fill(status, filled, qty)
        notional = self.cum_filled_notional
        if notional is not None:
            _non_negative(notional, "cum_filled_notional")
            if (notional == 0) != (filled == 0):
                raise DomainValidationError(
                    f"cum_filled_notional {notional} is inconsistent with cum_filled_qty {filled}"
                )
        if self.avg_fill_price is not None:
            _positive(self.avg_fill_price, "avg_fill_price")
            if filled == 0:
                raise DomainValidationError("avg_fill_price requires an execution")
        created = require_utc(self.created_ts, "created_ts")
        if require_utc(self.updated_ts, "updated_ts") < created:
            raise DomainValidationError("updated_ts is before created_ts")


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenOrdersSnapshot:
    """The complete set of open orders of the reader's scope at ``server_ts``."""

    orders: tuple[ExchangeOrder, ...]
    server_ts: datetime

    def __post_init__(self) -> None:
        orders = _tuple_of(self.orders, ExchangeOrder, "orders")
        server_ts = require_utc(self.server_ts, "server_ts")
        exchange_ids: set[str] = set()
        client_ids: set[str] = set()
        for order in orders:
            if order.status not in OPEN_ORDER_STATUSES:
                raise DomainValidationError(
                    f"order {order.exchange_order_id} is {order.status.value}, not open"
                )
            if order.exchange_order_id in exchange_ids:
                raise DomainValidationError(
                    f"exchange_order_id {order.exchange_order_id} appears more than once"
                )
            exchange_ids.add(order.exchange_order_id)
            if order.client_order_id is not None:
                if order.client_order_id in client_ids:
                    raise DomainValidationError(
                        f"client_order_id {order.client_order_id} appears more than once"
                    )
                client_ids.add(order.client_order_id)
            if order.updated_ts > server_ts:
                raise DomainValidationError(
                    f"order {order.exchange_order_id} was updated after the snapshot time"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExchangePosition:
    """Signed net position of one symbol (0 = an explicit flat entry)."""

    symbol: str
    qty: Decimal

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        _exact(self.qty, "qty")


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionSnapshot:
    """Positions of the reader's scope.

    ``complete=True``: the reader guarantees the snapshot covers its whole scope,
    so a symbol absent from it may be read as flat by the recovery layer.
    ``complete=False``: partial; an absent symbol is UNKNOWN, never flat.
    """

    positions: tuple[ExchangePosition, ...]
    complete: bool
    server_ts: datetime

    def __post_init__(self) -> None:
        positions = _tuple_of(self.positions, ExchangePosition, "positions")
        _exact_bool(self.complete, "complete")
        require_utc(self.server_ts, "server_ts")
        symbols: set[str] = set()
        for position in positions:
            if position.symbol in symbols:
                raise DomainValidationError(f"symbol {position.symbol} appears more than once")
            symbols.add(position.symbol)


class ExecutionKind(StrEnum):
    """Exchange-neutral classification of an execution by the reader."""

    TRADE = "trade"
    """An ordinary trade of an order: the only kind applied as a normal fill."""
    LIQUIDATION = "liquidation"
    ADL = "adl"
    """Auto-deleveraging."""
    BUST = "bust"
    """A bankruptcy / takeover execution."""
    OTHER = "other"
    """Anything else that the reader cannot classify more precisely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ExchangeExecution:
    """One execution as reported by the exchange."""

    exec_id: str
    exchange_order_id: str
    client_order_id: str | None
    symbol: str
    side: Side
    price: Decimal
    qty: Decimal
    fee: Decimal | None
    """Any sign (a negative fee is a rebate); None = unknown, never zero."""
    fee_asset: str | None
    """None exactly when ``fee`` is None (the domain ``Fill`` rule)."""
    is_maker: bool | None
    kind: ExecutionKind
    exchange_ts: datetime

    def __post_init__(self) -> None:
        require_text(self.exec_id, "exec_id")
        require_text(self.exchange_order_id, "exchange_order_id")
        _optional_text(self.client_order_id, "client_order_id")
        require_text(self.symbol, "symbol")
        require_enum(self.side, Side, "side")
        _positive(self.price, "price")
        _positive(self.qty, "qty")
        if self.fee is not None:
            _exact(self.fee, "fee")
        _optional_text(self.fee_asset, "fee_asset")
        if (self.fee is None) != (self.fee_asset is None):
            raise DomainValidationError("fee and fee_asset must be both known or both None")
        if self.is_maker is not None:
            _exact_bool(self.is_maker, "is_maker")
        require_enum(self.kind, ExecutionKind, "kind")
        require_utc(self.exchange_ts, "exchange_ts")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionQuery:
    """Executions of ``symbol`` (optionally of one exchange order) with
    ``start <= exchange_ts <= end``."""

    symbol: str
    exchange_order_id: str | None
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        require_text(self.symbol, "symbol")
        _optional_text(self.exchange_order_id, "exchange_order_id")
        start = require_utc(self.start, "start")
        if require_utc(self.end, "end") < start:
            raise DomainValidationError("query end is before its start")

    def matches(self, execution: ExchangeExecution) -> bool:
        """Does ``execution`` belong to this query (inclusive window)?"""
        return (
            execution.symbol == self.symbol
            and self.exchange_order_id in (None, execution.exchange_order_id)
            and self.start <= execution.exchange_ts <= self.end
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionPage:
    """One page of an execution query; ``next_cursor is None`` = exhausted."""

    query: ExecutionQuery
    executions: tuple[ExchangeExecution, ...]
    next_cursor: str | None
    """Opaque continuation token of the same query; never parsed by the caller."""

    def __post_init__(self) -> None:
        if type(self.query) is not ExecutionQuery:
            raise DomainValidationError("query must be an ExecutionQuery")
        executions = _tuple_of(self.executions, ExchangeExecution, "executions")
        if self.next_cursor is not None and (
            type(self.next_cursor) is not str or not self.next_cursor
        ):
            raise DomainValidationError("next_cursor must be None or a non-empty str")
        seen: set[str] = set()
        for execution in executions:
            if execution.exec_id in seen:
                raise DomainValidationError(
                    f"exec_id {execution.exec_id} appears more than once in one page"
                )
            seen.add(execution.exec_id)
            if not self.query.matches(execution):
                raise DomainValidationError(
                    f"execution {execution.exec_id} does not match the page's query"
                )
