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
* No mark-to-market: an open position has ``mark_price=None`` and
  ``unrealized_pnl=None`` (unknown); a flat one has ``unrealized_pnl=0``.
* ``updated_at`` is the exchange time of the last applied fill; a fill older than
  the symbol's current position is an error, equal times are allowed.
* Idempotency by ``exec_id``: re-applying an identical fill changes nothing; the
  same id with a different payload is an identity conflict.

Exactness: the entry cost and realized PnL are kept as exact rationals
(``fractions.Fraction``), so closing part of a position with a non-terminating
average basis (e.g. 310/3) leaves no rounding residue. Only the published
``entry_price`` and ``realized_pnl`` are rounded, to ``POSITION_PRICE_PRECISION``
significant digits with ROUND_HALF_EVEN in an explicit context; the global
decimal context is neither read nor modified. Quantities stay exact ``Decimal``.

Batches: ``prepare(fills)`` computes the resulting state without touching the
ledger and ``commit(prepared)`` installs it, so a caller can make several fills
and its own state changes all-or-nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from fractions import Fraction
from typing import Final

from app.domain.enums import Side
from app.domain.fills import Fill
from app.domain.positions import Position

# Same published precision as the simulator's average fill price.
POSITION_PRICE_PRECISION: Final = 40
_EXACT_PRECISION: Final = 80
_ZERO_DECIMAL: Final = Decimal(0)
_ZERO: Final = Fraction(0)


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
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PreparedPositions:
    """Result of ``prepare``; valid only for the ledger version it was built on."""

    base_version: int
    states: Mapping[str, _PositionState]
    applied: Mapping[str, Fill]


def _exact() -> Context:
    return Context(
        prec=_EXACT_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
    )


def _publish(value: Fraction) -> Decimal:
    context = Context(
        prec=POSITION_PRICE_PRECISION,
        rounding=ROUND_HALF_EVEN,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )
    return context.divide(Decimal(value.numerator), Decimal(value.denominator))


def _apply_fill(state: _PositionState | None, fill: Fill) -> _PositionState:
    """State after one fill (pure)."""
    if state is not None and fill.exchange_ts < state.updated_at:
        raise PositionAccountingError(
            f"fill {fill.exec_id} is older than the {fill.symbol} position "
            f"({fill.exchange_ts.isoformat()} < {state.updated_at.isoformat()})"
        )
    qty = state.qty if state is not None else _ZERO_DECIMAL
    cost = state.entry_cost if state is not None else _ZERO
    realized = state.realized if state is not None else _ZERO
    price = Fraction(fill.price)
    fill_qty = Fraction(fill.qty)
    signed = fill.qty if fill.side is Side.BUY else -fill.qty

    if qty == 0 or (qty > 0) == (signed > 0):
        cost += price * fill_qty  # open or increase
    else:
        open_abs = Fraction(abs(qty))
        closed = min(fill_qty, open_abs)
        basis = cost / open_abs
        pnl_per_unit = price - basis if qty > 0 else basis - price
        realized += pnl_per_unit * closed
        cost -= basis * closed
        opened = fill_qty - closed
        if opened > 0:
            cost = price * opened  # reversal: fresh basis on the new side
    try:
        new_qty = _exact().add(qty, signed)
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
        updated_at=fill.exchange_ts,
    )


def _to_position(state: _PositionState) -> Position:
    is_flat = state.qty == 0
    return Position(
        symbol=state.symbol,
        qty=state.qty,
        entry_price=None if is_flat else _publish(state.entry_cost / Fraction(abs(state.qty))),
        mark_price=None,
        unrealized_pnl=_ZERO_DECIMAL if is_flat else None,
        realized_pnl=_publish(state.realized),
        updated_at=state.updated_at,
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

    def get_position(self, symbol: str) -> Position | None:
        """Current position, or None if no fill of ``symbol`` was ever applied."""
        state = self._states.get(symbol)
        return None if state is None else _to_position(state)

    def prepare(self, fills: Sequence[Fill]) -> PreparedPositions:
        """Apply ``fills`` in the given order to a copy of the state. The ledger is
        not changed; any error leaves it exactly as it was."""
        states: dict[str, _PositionState] = {}
        applied: dict[str, Fill] = {}
        for fill in fills:
            if not isinstance(fill, Fill):
                raise PositionAccountingError("expected a Fill")
            known = applied.get(fill.exec_id) or self._applied.get(fill.exec_id)
            if known is not None:
                if known != fill:
                    raise PositionAccountingError(
                        f"exec_id {fill.exec_id} was already applied with a different payload"
                    )
                continue  # identical fill: already accounted for
            current = states.get(fill.symbol) or self._states.get(fill.symbol)
            states[fill.symbol] = _apply_fill(current, fill)
            applied[fill.exec_id] = fill
        return PreparedPositions(base_version=self._version, states=states, applied=applied)

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
