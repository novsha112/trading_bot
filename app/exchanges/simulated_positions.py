"""Deterministic one-way (net) position accounting from confirmed fills.

A building block of the simulated exchange; it knows only domain ``Fill`` and
``Position`` (no orders, instruments, exchange contracts or time source).

Model:
* One signed position per symbol: BUY adds ``+qty``, SELL adds ``-qty``.
* Same-side fills increase the position; the entry price is the weighted average
  of execution prices.
* Opposite-side fills close up to the open size first, realizing
  ``(exit - basis) * closed`` for a long and ``(basis - exit) * closed`` for a
  short; any excess opens the other side at the fill's execution price.
* ``realized_pnl`` is gross (execution prices only): fill fees and funding are
  never included. It is kept through flat and later positions.
* Mark-to-market is a read-time valuation with an externally supplied
  ``MarkQuote``: unrealized PnL comes from the exact cost basis
  (long ``mark * qty - cost``, short ``cost - mark * |qty|``), published like the
  other values. Without a mark an open position has ``mark_price=None`` and
  ``unrealized_pnl=None`` (unknown); a flat one always has ``unrealized_pnl=0``
  and keeps a known mark price.
* The fill stream has its own watermark (time of the last applied fill); a fill
  older than it is an error, equal times are allowed. A published position's
  ``updated_at`` is the later of that time and the mark time, so a newer mark
  never makes a fill stale.
* Idempotency by ``exec_id``: re-applying an identical fill changes nothing; the
  same id with a different payload is an identity conflict.

Exactness: the entry cost and realized PnL are kept as exact rationals
(``fractions.Fraction``), so closing part of a position with a non-terminating
average basis (e.g. 310/3) leaves no rounding residue. Only the published
``entry_price``, ``realized_pnl`` and ``unrealized_pnl`` are rounded, to
``POSITION_PRICE_PRECISION``
significant digits with ROUND_HALF_EVEN in an explicit context; the global
decimal context is neither read nor modified. Quantities stay exact ``Decimal``.

Batches: ``begin_batch()`` returns a working copy whose reads see its own
prepared fills (needed when a later decision depends on the position after an
earlier fill, e.g. reduce-only); ``prepare(fills)`` is the same for a fixed list.
``commit(prepared)`` installs the result only on the ledger version it was built
on, so a caller can make several fills and its own state changes all-or-nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
from fractions import Fraction
from typing import Final

from app.domain.fill_math import next_position_qty, signed_quantity
from app.domain.fills import Fill
from app.domain.positions import Position

# Same published precision as the simulator's average fill price.
POSITION_PRICE_PRECISION: Final = 40
_ZERO_DECIMAL: Final = Decimal(0)
_ZERO: Final = Fraction(0)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkQuote:
    """An externally supplied mark price and the time it was observed. Valuation
    input only: it never changes quantities, cost basis or realized PnL."""

    price: Decimal
    at: datetime

    def __post_init__(self) -> None:
        price = self.price
        if type(price) is not Decimal or not price.is_finite() or price <= 0:
            raise ValueError("mark price must be a finite Decimal > 0")
        if not isinstance(self.at, datetime) or self.at.utcoffset() != timedelta(0):
            raise ValueError("mark time 'at' must be a UTC datetime")


class PositionAccountingError(RuntimeError):
    """Internal accounting invariant violated (stale or conflicting fill, corrupted
    state). Not an exchange answer; nothing was changed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _PositionState:
    symbol: str
    qty: Decimal
    """Signed open quantity."""
    entry_cost: Fraction
    """Exact cost basis of the open quantity (>= 0; 0 when flat)."""
    realized: Fraction
    """Exact gross realized PnL."""
    last_fill_at: datetime
    """Exchange time of the last applied fill: the watermark of the fill stream only
    (mark updates never move it)."""


@dataclass(frozen=True, slots=True)
class PreparedPositions:
    """Result of ``prepare``; valid only for the ledger version it was built on."""

    base_version: int
    states: Mapping[str, _PositionState]
    applied: Mapping[str, Fill]


def _publish(value: Fraction) -> Decimal:
    context = Context(
        prec=POSITION_PRICE_PRECISION,
        rounding=ROUND_HALF_EVEN,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )
    return context.divide(Decimal(value.numerator), Decimal(value.denominator))


def _apply_fill(state: _PositionState | None, fill: Fill) -> _PositionState:
    """State after one fill (pure)."""
    if state is not None and fill.exchange_ts < state.last_fill_at:
        raise PositionAccountingError(
            f"fill {fill.exec_id} is older than the {fill.symbol} position "
            f"({fill.exchange_ts.isoformat()} < {state.last_fill_at.isoformat()})"
        )
    qty = state.qty if state is not None else _ZERO_DECIMAL
    cost = state.entry_cost if state is not None else _ZERO
    realized = state.realized if state is not None else _ZERO
    price = Fraction(fill.price)
    fill_qty = Fraction(fill.qty)
    # Signed quantity and the new net position: the shared exact rule
    # (app.domain.fill_math); copy_abs below is exact and context-free too.
    signed = signed_quantity(fill.side, fill.qty)

    if qty == 0 or (qty > 0) == (signed > 0):
        cost += price * fill_qty  # open or increase
    else:
        open_abs = Fraction(qty.copy_abs())
        closed = min(fill_qty, open_abs)
        basis = cost / open_abs
        pnl_per_unit = price - basis if qty > 0 else basis - price
        realized += pnl_per_unit * closed
        cost -= basis * closed
        opened = fill_qty - closed
        if opened > 0:
            cost = price * opened  # reversal: fresh basis on the new side
    try:
        new_qty = next_position_qty(qty, side=fill.side, qty=fill.qty)
    except DecimalException:
        raise PositionAccountingError(
            f"{fill.symbol} position quantity cannot be computed exactly"
        ) from None
    if new_qty == 0:
        cost = _ZERO
    return _PositionState(
        symbol=fill.symbol,
        qty=new_qty,
        entry_cost=cost,
        realized=realized,
        last_fill_at=fill.exchange_ts,
    )


def _exact_unrealized(state: _PositionState, mark: MarkQuote | None) -> Fraction | None:
    """The only valuation formula: exact unrealized PnL from the cost basis.
    Long ``mark * qty - cost``, short ``cost - mark * |qty|``, flat 0; None when an
    open position has no mark."""
    if state.qty == 0:
        return _ZERO
    if mark is None:
        return None
    value = Fraction(mark.price) * Fraction(state.qty.copy_abs())
    return value - state.entry_cost if state.qty > 0 else state.entry_cost - value


def _to_position(state: _PositionState, mark: MarkQuote | None) -> Position:
    """Published position, valued at ``mark`` (if known) from the exact basis."""
    is_flat = state.qty == 0
    open_qty = Fraction(state.qty.copy_abs())
    exact_unrealized = _exact_unrealized(state, mark)
    unrealized = None if exact_unrealized is None else _publish(exact_unrealized)
    updated_at = state.last_fill_at
    if mark is not None and mark.at > updated_at:
        updated_at = mark.at
    return Position(
        symbol=state.symbol,
        qty=state.qty,
        entry_price=None if is_flat else _publish(state.entry_cost / open_qty),
        mark_price=None if mark is None else mark.price,
        unrealized_pnl=unrealized,
        realized_pnl=_publish(state.realized),
        updated_at=updated_at,
    )


class PositionBatch:
    """Working state on top of a ledger: fills applied here are visible to later
    reads of the same batch, never to the ledger itself (until ``commit``)."""

    __slots__ = ("_applied", "_base_version", "_ledger", "_states")

    def __init__(self, ledger: SimulatedPositionLedger) -> None:
        self._ledger = ledger
        self._base_version = ledger._version
        self._states: dict[str, _PositionState] = {}
        self._applied: dict[str, Fill] = {}

    def _state(self, symbol: str) -> _PositionState | None:
        return self._states.get(symbol) or self._ledger._states.get(symbol)

    def signed_qty(self, symbol: str) -> Decimal:
        """Signed position quantity as prepared so far (0 if none)."""
        state = self._state(symbol)
        return _ZERO_DECIMAL if state is None else state.qty

    def apply(self, fill: Fill) -> Fraction | None:
        """Apply one fill to the batch; on error the batch is unchanged.

        Returns the exact gross realized PnL delta of this fill (before any public
        rounding), or None for an identical replay that changed nothing."""
        if not isinstance(fill, Fill):
            raise PositionAccountingError("expected a Fill")
        known = self._applied.get(fill.exec_id) or self._ledger._applied.get(fill.exec_id)
        if known is not None:
            if known != fill:
                raise PositionAccountingError(
                    f"exec_id {fill.exec_id} was already applied with a different payload"
                )
            return None  # identical fill: already accounted for
        current = self._state(fill.symbol)
        new_state = _apply_fill(current, fill)
        self._states[fill.symbol] = new_state
        self._applied[fill.exec_id] = fill
        return new_state.realized - (current.realized if current is not None else _ZERO)

    def position(self, symbol: str, *, mark: MarkQuote | None) -> Position | None:
        """The position as prepared so far, valued at ``mark`` (None if never
        filled). Used to prove the published state is computable before commit."""
        state = self._state(symbol)
        return None if state is None else _to_position(state, mark)

    def prepared(self) -> PreparedPositions:
        return PreparedPositions(
            base_version=self._base_version,
            states=dict(self._states),
            applied=dict(self._applied),
        )


class SimulatedPositionLedger:
    """Net position per symbol, built only from confirmed fills."""

    __slots__ = ("_applied", "_states", "_version")

    def __init__(self) -> None:
        self._states: dict[str, _PositionState] = {}
        self._applied: dict[str, Fill] = {}
        self._version = 0

    def __repr__(self) -> str:
        return f"SimulatedPositionLedger(symbols={len(self._states)})"

    def exact_unrealized_total(self, marks: Mapping[str, MarkQuote]) -> Fraction | None:
        """Exact sum of unrealized PnL over all positions valued at ``marks``
        (before any rounding). Flat positions count 0 and need no mark; None if any
        open position has no mark (an unknown value is never read as zero)."""
        total = _ZERO
        for symbol, state in self._states.items():
            value = _exact_unrealized(state, marks.get(symbol))
            if value is None:
                return None
            total += value
        return total

    def get_position(self, symbol: str, *, mark: MarkQuote | None = None) -> Position | None:
        """Current position, or None if no fill of ``symbol`` was ever applied."""
        state = self._states.get(symbol)
        return None if state is None else _to_position(state, mark)

    def begin_batch(self) -> PositionBatch:
        """A working copy for preparing several fills; reads see the batch's own
        prepared fills. The ledger is not changed until ``commit``."""
        return PositionBatch(self)

    def prepare(self, fills: Sequence[Fill]) -> PreparedPositions:
        """Apply ``fills`` in the given order to a copy of the state. The ledger is
        not changed; any error leaves it exactly as it was."""
        batch = self.begin_batch()
        for fill in fills:
            batch.apply(fill)
        return batch.prepared()

    def commit(self, prepared: PreparedPositions) -> None:
        """Install a prepared batch built on the current ledger version."""
        if prepared.base_version != self._version:
            raise PositionAccountingError("prepared positions are stale")
        self._states.update(prepared.states)
        self._applied.update(prepared.applied)
        self._version += 1

    def apply_fill(self, fill: Fill) -> Position:
        """Apply one fill atomically; returns the resulting position (unchanged for
        an identical repeat)."""
        self.commit(self.prepare((fill,)))
        position = self.get_position(fill.symbol)
        if position is None:  # unreachable: a Fill always leaves a state
            raise PositionAccountingError(f"no {fill.symbol} position after a fill")
        return position
