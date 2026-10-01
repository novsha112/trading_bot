"""Deterministic single-asset cash accounting: starting cash, gross realized PnL
and trading fees kept separately (derivatives-style, no mark-to-market)."""

from __future__ import annotations

import ast
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from app.domain.errors import DomainValidationError
from app.exchanges import simulated_accounting
from app.exchanges.simulated_accounting import (
    CashAccountingError,
    CashState,
    SimulatedCashConfig,
    SimulatedCashLedger,
)

D = Decimal
F = Fraction
ZERO_FEE = Decimal("0")


def ledger(starting: str = "10000") -> SimulatedCashLedger:
    return SimulatedCashLedger(SimulatedCashConfig(asset="USDT", starting_cash=D(starting)))


def apply(
    cash: SimulatedCashLedger,
    exec_id: str,
    realized: Fraction | int = 0,
    fee: Any = ZERO_FEE,
    fee_asset: Any = "USDT",
) -> CashState:
    return cash.apply_execution(
        exec_id=exec_id, realized_delta=F(realized), fee=fee, fee_asset=fee_asset
    )


def parts(state: CashState) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    return state.starting_cash, state.gross_realized_pnl, state.trading_fees, state.cash


# --- configuration -------------------------------------------------------------


@pytest.mark.parametrize("starting", ["10000", "0", "0.000001"])
def test_starting_cash(starting: str) -> None:
    state = ledger(starting).state()

    assert parts(state) == (D(starting), D("0"), D("0"), D(starting))
    assert state.asset == "USDT"


@pytest.mark.parametrize(
    "value", [D("-1"), D("NaN"), D("sNaN"), D("Infinity"), 1.5, 100, True, "100", None]
)
def test_invalid_starting_cash(value: object) -> None:
    with pytest.raises(ValueError, match="starting_cash"):
        SimulatedCashConfig(asset="USDT", starting_cash=value)  # type: ignore[arg-type]


def test_starting_cash_decimal_subclass_rejected() -> None:
    class Sub(Decimal):
        pass

    with pytest.raises(ValueError, match="starting_cash"):
        SimulatedCashConfig(asset="USDT", starting_cash=Sub("1"))


@pytest.mark.parametrize("asset", ["", " USDT", None, 1])
def test_invalid_asset(asset: object) -> None:
    with pytest.raises(DomainValidationError, match="asset"):
        SimulatedCashConfig(asset=asset, starting_cash=D("1"))  # type: ignore[arg-type]


def test_ledger_requires_a_config() -> None:
    with pytest.raises(TypeError, match="SimulatedCashConfig"):
        SimulatedCashLedger("USDT")  # type: ignore[arg-type]


# --- cash flows -----------------------------------------------------------------


def test_opening_fill_moves_cash_by_fee_only() -> None:
    cash = ledger()

    state = apply(cash, "open", realized=0, fee=D("1"))

    assert parts(state) == (D("10000"), D("0"), D("1"), D("9999"))


def test_open_then_close_with_fees() -> None:
    cash = ledger()
    apply(cash, "open", realized=0, fee=D("1"))

    state = apply(cash, "close", realized=100, fee=D("1.1"))

    assert parts(state) == (D("10000"), D("100"), D("2.1"), D("10097.9"))


def test_loss_with_fees() -> None:
    cash = ledger()
    apply(cash, "open", fee=D("1"))

    state = apply(cash, "close", realized=-100, fee=D("1"))

    assert parts(state) == (D("10000"), D("-100"), D("2"), D("9898"))


def test_partial_closes_accumulate() -> None:
    cash = ledger()
    apply(cash, "open", fee=D("1"))
    apply(cash, "c1", realized=40, fee=D("0.44"))
    state = apply(cash, "c2", realized=-30, fee=D("0.27"))

    assert parts(state) == (D("10000"), D("10"), D("1.71"), D("10008.29"))


def test_zero_fee_and_rebate() -> None:
    cash = ledger()
    apply(cash, "zero", fee=D("0"))
    state = apply(cash, "rebate", fee=D("-0.5"))

    assert parts(state) == (D("10000"), D("0"), D("-0.5"), D("10000.5"))


def test_exact_realized_delta_is_kept_exactly() -> None:
    cash = ledger("0")
    for i in range(3):
        apply(cash, f"t{i}", realized=F(50, 3))

    state = cash.state()
    assert state.gross_realized_pnl == D("50")  # 3 * 50/3, no rounding residue
    assert state.cash == D("50")


def test_non_terminating_realized_is_published_at_40_digits() -> None:
    cash = ledger("0")

    state = apply(cash, "t", realized=F(50, 3))

    assert state.gross_realized_pnl == D("16.66666666666666666666666666666666666667")
    assert state.cash == state.gross_realized_pnl


def test_cash_equation_holds() -> None:
    cash = ledger("2500.25")
    flows = [(F(0), D("0.3")), (F(125, 2), D("0.7")), (F(-20), D("-0.05")), (F(7), D("0"))]
    for i, (realized, fee) in enumerate(flows):
        state = apply(cash, f"e{i}", realized=realized, fee=fee)
        assert state.cash == state.starting_cash + state.gross_realized_pnl - state.trading_fees


def test_results_ignore_the_global_decimal_context() -> None:
    def run() -> CashState:
        cash = ledger("1234.5678")
        apply(cash, "a", realized=F(123456789, 1000), fee=D("0.123456789"))
        return apply(cash, "b", realized=F(-1, 7), fee=D("-0.0000001"))

    baseline = run()
    with localcontext() as context:
        context.prec = 2
        context.rounding = "ROUND_UP"
        low = run()

    assert low == baseline
    assert str(low.cash) == str(baseline.cash)


# --- rejected inputs ----------------------------------------------------------------


def test_unknown_fee_is_rejected() -> None:
    cash = ledger()
    before = cash.state()

    with pytest.raises(CashAccountingError, match="unknown fee"):
        apply(cash, "x", fee=None, fee_asset=None)

    assert cash.state() == before


def test_other_fee_asset_is_rejected() -> None:
    cash = ledger()

    with pytest.raises(CashAccountingError, match="BTC"):
        apply(cash, "x", fee=D("0.001"), fee_asset="BTC")

    assert cash.state() == ledger().state()


@pytest.mark.parametrize("fee", [1, 0.5, D("NaN"), "1"])
def test_invalid_fee_is_rejected(fee: object) -> None:
    with pytest.raises(CashAccountingError, match="fee"):
        apply(ledger(), "x", fee=fee)


def test_realized_delta_must_be_a_fraction() -> None:
    with pytest.raises(CashAccountingError, match="realized_delta"):
        ledger().apply_execution(
            exec_id="x",
            realized_delta=D("1"),  # type: ignore[arg-type]
            fee=D("0"),
            fee_asset="USDT",
        )


# --- duplicate executions -------------------------------------------------------------


def test_identical_execution_replay_is_idempotent() -> None:
    cash = ledger()
    first = apply(cash, "x", realized=100, fee=D("1"))

    again = apply(cash, "x", realized=100, fee=D("1"))

    assert again == first


def test_same_exec_id_with_other_payload_is_a_conflict() -> None:
    cash = ledger()
    first = apply(cash, "x", realized=100, fee=D("1"))

    with pytest.raises(CashAccountingError, match="exec_id x"):
        apply(cash, "x", realized=100, fee=D("2"))

    assert cash.state() == first


def test_error_is_internal_not_an_exchange_error() -> None:
    from app.exchanges.errors import ExchangeError

    assert issubclass(CashAccountingError, RuntimeError)
    assert not issubclass(CashAccountingError, ExchangeError)


# --- batches ------------------------------------------------------------------------


def test_batch_is_invisible_until_commit_and_atomic() -> None:
    cash = ledger()
    batch = cash.begin_batch()
    batch.apply(exec_id="a", realized_delta=F(10), fee=D("1"), fee_asset="USDT")

    assert cash.state() == ledger().state()
    with pytest.raises(CashAccountingError):
        batch.apply(exec_id="b", realized_delta=F(0), fee=None, fee_asset=None)
    cash.commit(batch.prepared())  # the failed entry left no trace in the batch

    assert parts(cash.state()) == (D("10000"), D("10"), D("1"), D("10009"))


def test_stale_batch_commit_is_rejected() -> None:
    cash = ledger()
    batch = cash.begin_batch()
    batch.apply(exec_id="a", realized_delta=F(10), fee=D("1"), fee_asset="USDT")
    apply(cash, "b")

    with pytest.raises(CashAccountingError, match="stale"):
        cash.commit(batch.prepared())


def test_cash_state_has_no_equity() -> None:
    assert not hasattr(ledger().state(), "equity")


def test_module_has_no_forbidden_dependencies() -> None:
    source = Path(simulated_accounting.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert not name.startswith("app.") or name.startswith("app.domain"), name
    for banned in (
        "float(",
        "getcontext",
        "setcontext",
        "localcontext",
        "Side.",
        "entry_price",
        "qty",
    ):
        assert banned not in source, banned
