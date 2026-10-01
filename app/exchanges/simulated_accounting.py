"""Deterministic single-asset cash accounting for the simulator (opt-in).

Derivatives-style model: cash moves only by realized trading PnL and trading fees.
Opening or increasing a position does not move notional cash; there is no
mark-to-market, unrealized PnL, equity, funding, margin, deposits or withdrawals.

    cash = starting_cash + gross_realized_pnl - trading_fees

The three components are kept separately (cash is derived, so the equation holds
by construction). A negative fee (rebate) lowers ``trading_fees`` and raises cash.

Inputs per execution come from elsewhere and are never recomputed here: the exact
realized PnL delta from the position ledger (a ``Fraction``, before any public
rounding) and the fee from the fill. A fee must be known and in the ledger's asset
(no FX); otherwise the execution is refused. Realized PnL is kept as an exact
rational; published values are rounded to ``CASH_PRECISION`` significant digits,
ROUND_HALF_EVEN, in an explicit context. Settlement rounding is not modeled.

``begin_batch()`` / ``commit()`` make several executions all-or-nothing together
with the caller's other state; each exec_id is applied once (an identical replay
is ignored, a different payload for a known exec_id is an error).
"""

from __future__ import annotations

from dataclasses import dataclass
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

from app.domain.validation import require_text

# Published precision, same as the simulator's prices and positions.
CASH_PRECISION: Final = 40
_EXACT_PRECISION: Final = 120


class CashAccountingError(RuntimeError):
    """Internal accounting invariant violated (unknown or foreign-asset fee,
    conflicting execution, stale batch). Not an exchange answer; nothing changed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulatedCashConfig:
    """Accounting asset and starting cash; both explicit, no defaults."""

    asset: str
    starting_cash: Decimal

    def __post_init__(self) -> None:
        require_text(self.asset, "asset")
        value = self.starting_cash
        if type(value) is not Decimal or not value.is_finite() or value < 0:
            raise ValueError("starting_cash must be a finite Decimal >= 0")


@dataclass(frozen=True, slots=True, kw_only=True)
class CashState:
    """Published cash components. ``cash`` is not equity: no unrealized PnL."""

    asset: str
    starting_cash: Decimal
    gross_realized_pnl: Decimal
    trading_fees: Decimal
    """Cumulative fees; negative fees (rebates) reduce it."""
    cash: Decimal


@dataclass(frozen=True, slots=True)
class _Entry:
    realized_delta: Fraction
    fee: Decimal


@dataclass(frozen=True, slots=True)
class _Totals:
    realized: Fraction
    fees: Decimal


@dataclass(frozen=True, slots=True)
class PreparedCash:
    """Result of a batch; valid only for the ledger version it was built on."""

    base_version: int
    totals: _Totals
    entries: dict[str, _Entry]


def _exact() -> Context:
    return Context(
        prec=_EXACT_PRECISION, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
    )


def _publish(value: Fraction) -> Decimal:
    context = Context(
        prec=CASH_PRECISION,
        rounding=ROUND_HALF_EVEN,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )
    return context.divide(Decimal(value.numerator), Decimal(value.denominator))


class CashBatch:
    """Working totals on top of a ledger; invisible to the ledger until commit."""

    __slots__ = ("_base_version", "_entries", "_ledger", "_totals")

    def __init__(self, ledger: SimulatedCashLedger) -> None:
        self._ledger = ledger
        self._base_version = ledger._version
        self._totals = ledger._totals
        self._entries: dict[str, _Entry] = {}

    def apply(
        self,
        *,
        exec_id: str,
        realized_delta: Fraction,
        fee: Decimal | None,
        fee_asset: str | None,
    ) -> None:
        """Record one execution; on error the batch is unchanged."""
        if not isinstance(realized_delta, Fraction):
            raise CashAccountingError("realized_delta must be an exact Fraction")
        if fee is None:
            raise CashAccountingError(f"execution {exec_id}: unknown fee cannot enter cash")
        if type(fee) is not Decimal or not fee.is_finite():
            raise CashAccountingError(f"execution {exec_id}: fee must be a finite Decimal")
        asset = self._ledger.config.asset
        if fee_asset != asset:
            raise CashAccountingError(
                f"execution {exec_id}: fee asset {fee_asset} is not the cash asset {asset}"
            )
        entry = _Entry(realized_delta=realized_delta, fee=fee)
        known = self._entries.get(exec_id) or self._ledger._entries.get(exec_id)
        if known is not None:
            if known != entry:
                raise CashAccountingError(
                    f"exec_id {exec_id} was already accounted with a different payload"
                )
            return  # identical replay: already accounted for
        try:
            fees = _exact().add(self._totals.fees, fee)
        except DecimalException:
            raise CashAccountingError("trading fees cannot be summed exactly") from None
        self._totals = _Totals(realized=self._totals.realized + realized_delta, fees=fees)
        self._entries[exec_id] = entry

    def prepared(self) -> PreparedCash:
        return PreparedCash(
            base_version=self._base_version, totals=self._totals, entries=dict(self._entries)
        )


class SimulatedCashLedger:
    """Cash of one accounting asset, built from executions' realized PnL and fees."""

    __slots__ = ("_entries", "_totals", "_version", "config")

    def __init__(self, config: SimulatedCashConfig) -> None:
        if not isinstance(config, SimulatedCashConfig):
            raise TypeError("config must be a SimulatedCashConfig")
        self.config = config
        self._totals = _Totals(realized=Fraction(0), fees=Decimal(0))
        self._entries: dict[str, _Entry] = {}
        self._version = 0

    def __repr__(self) -> str:
        return f"SimulatedCashLedger(asset={self.config.asset!r}, executions={len(self._entries)})"

    def state(self) -> CashState:
        starting = self.config.starting_cash
        realized = self._totals.realized
        fees = self._totals.fees
        return CashState(
            asset=self.config.asset,
            starting_cash=starting,
            gross_realized_pnl=_publish(realized),
            trading_fees=fees,
            cash=_publish(Fraction(starting) + realized - Fraction(fees)),
        )

    def begin_batch(self) -> CashBatch:
        return CashBatch(self)

    def commit(self, prepared: PreparedCash) -> None:
        if prepared.base_version != self._version:
            raise CashAccountingError("prepared cash is stale")
        self._totals = prepared.totals
        self._entries.update(prepared.entries)
        self._version += 1

    def apply_execution(
        self,
        *,
        exec_id: str,
        realized_delta: Fraction,
        fee: Decimal | None,
        fee_asset: str | None,
    ) -> CashState:
        """Record one execution atomically; returns the resulting state."""
        batch = self.begin_batch()
        batch.apply(exec_id=exec_id, realized_delta=realized_delta, fee=fee, fee_asset=fee_asset)
        self.commit(batch.prepared())
        return self.state()
