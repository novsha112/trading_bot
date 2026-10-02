"""Deterministic one-way position accounting from confirmed fills (gross PnL)."""

from __future__ import annotations

import ast
import itertools
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from app.domain import fill_math
from app.domain.enums import PositionSide, Side
from app.domain.fills import Fill
from app.domain.positions import Position
from app.exchanges import simulated_positions
from app.exchanges.simulated_positions import (
    MarkQuote,
    PositionAccountingError,
    SimulatedPositionLedger,
)

D = Decimal
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
_ids = itertools.count(1)


def fill(
    side: Side,
    qty: str,
    price: str,
    *,
    exec_id: str | None = None,
    symbol: str = "BTCUSDT",
    ts: datetime = T0,
    fee: Decimal | None = None,
    fee_asset: str | None = None,
) -> Fill:
    return Fill(
        exec_id=exec_id or f"E-{next(_ids)}",
        exchange_order_id="SIM-1",
        client_order_id="c-1",
        symbol=symbol,
        side=side,
        price=D(price),
        qty=D(qty),
        fee=fee,
        fee_asset=fee_asset,
        is_maker=None,
        exchange_ts=ts,
    )


def buy(qty: str, price: str, **kwargs: Any) -> Fill:
    return fill(Side.BUY, qty, price, **kwargs)


def sell(qty: str, price: str, **kwargs: Any) -> Fill:
    return fill(Side.SELL, qty, price, **kwargs)


def run(*fills: Fill) -> Position:
    ledger = SimulatedPositionLedger()
    position = None
    for f in fills:
        position = ledger.apply_fill(f)
    assert position is not None
    return position


def summary(p: Position) -> tuple[Decimal, Decimal | None, Decimal]:
    return p.qty, p.entry_price, p.realized_pnl


# --- initial state / opening -------------------------------------------------


def test_no_position_before_any_fill() -> None:
    ledger = SimulatedPositionLedger()

    assert ledger.get_position("BTCUSDT") is None


def test_first_buy_opens_long() -> None:
    p = run(buy("5", "100", ts=T0 + timedelta(seconds=3)))

    assert p == Position(
        symbol="BTCUSDT",
        qty=D("5"),
        entry_price=D("100"),
        mark_price=None,
        unrealized_pnl=None,
        realized_pnl=D("0"),
        updated_at=T0 + timedelta(seconds=3),
    )
    assert p.side is PositionSide.LONG


def test_first_sell_opens_short() -> None:
    p = run(sell("5", "100"))

    assert summary(p) == (D("-5"), D("100"), D("0"))
    assert (p.mark_price, p.unrealized_pnl) == (None, None)
    assert p.side is PositionSide.SHORT


def test_positions_are_per_symbol() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100"))
    ledger.apply_fill(sell("2", "50", symbol="ETHUSDT"))

    btc = ledger.get_position("BTCUSDT")
    eth = ledger.get_position("ETHUSDT")
    assert btc is not None
    assert eth is not None
    assert (btc.qty, eth.qty) == (D("1"), D("-2"))
    assert ledger.get_position("SOLUSDT") is None


# --- same-side increase -------------------------------------------------------


def test_long_increase_weighted_entry() -> None:
    p = run(buy("2", "100"), buy("1", "110"))

    assert p.qty == D("3")
    assert p.entry_price == D("103.3333333333333333333333333333333333333")  # 310/3


def test_short_increase_weighted_entry() -> None:
    p = run(sell("1", "100"), sell("3", "120"))

    assert summary(p) == (D("-4"), D("115"), D("0"))


def test_repeating_entry_rounds_half_even_to_40_digits() -> None:
    p = run(buy("3", "100"), buy("3", "100"), buy("3", "101"))  # 903/9 = 100.333...

    assert p.entry_price == D("100.3333333333333333333333333333333333333")
    assert p.entry_price is not None
    assert len(p.entry_price.as_tuple().digits) == 40


def test_exact_basis_avoids_drift_from_rounded_entry() -> None:
    # 310/3 is rounded publicly, but closing all three units realizes exactly
    # 3 * 120 - 310 = 50, not 3 * (120 - rounded entry).
    p = run(buy("2", "100"), buy("1", "110"), sell("1", "120"), sell("2", "120"))

    assert summary(p) == (D("0"), None, D("50"))


def test_partial_close_after_repeating_entry_keeps_exact_remaining_basis() -> None:
    p = run(buy("2", "100"), buy("1", "110"), sell("1", "120"))

    assert p.qty == D("2")
    assert p.entry_price == D("103.3333333333333333333333333333333333333")
    assert p.realized_pnl == D("16.66666666666666666666666666666666666667")


# --- partial close ------------------------------------------------------------


@pytest.mark.parametrize(
    ("fills", "expected"),
    [
        ((buy("10", "100"), sell("4", "110")), (D("6"), D("100"), D("40"))),
        ((buy("10", "100"), sell("4", "90")), (D("6"), D("100"), D("-40"))),
        ((sell("10", "100"), buy("4", "90")), (D("-6"), D("100"), D("40"))),
        ((sell("10", "100"), buy("4", "110")), (D("-6"), D("100"), D("-40"))),
    ],
    ids=["long-profit", "long-loss", "short-profit", "short-loss"],
)
def test_partial_close(fills: tuple[Fill, ...], expected: tuple[Decimal, ...]) -> None:
    p = run(*fills)

    assert summary(p) == expected
    assert p.unrealized_pnl is None


# --- exact close ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fills",
    [(buy("10", "100"), sell("10", "110")), (sell("10", "100"), buy("10", "90"))],
    ids=["long", "short"],
)
def test_exact_close_is_flat_with_realized(fills: tuple[Fill, ...]) -> None:
    p = run(*fills)

    assert p == Position(
        symbol="BTCUSDT",
        qty=D("0"),
        entry_price=None,
        mark_price=None,
        unrealized_pnl=D("0"),
        realized_pnl=D("100"),
        updated_at=T0,
    )
    assert p.side is PositionSide.FLAT


# --- reversal ---------------------------------------------------------------------


def test_long_to_short_reversal() -> None:
    p = run(buy("10", "100"), sell("15", "110"))

    assert summary(p) == (D("-5"), D("110"), D("100"))
    assert p.unrealized_pnl is None


def test_short_to_long_reversal() -> None:
    p = run(sell("10", "100"), buy("15", "90"))

    assert summary(p) == (D("5"), D("90"), D("100"))


def test_reversed_side_has_fresh_basis() -> None:
    # New short opened at 110; closing it at 100 realizes (110-100)*5 = 50 more.
    p = run(buy("10", "100"), sell("15", "110"), buy("5", "100"))

    assert summary(p) == (D("0"), None, D("150"))


# --- realized accumulation ------------------------------------------------------


def test_realized_accumulates_through_flat_and_reopen() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("10", "100"))
    ledger.apply_fill(sell("4", "110"))  # +40
    flat = ledger.apply_fill(sell("6", "95"))  # -30 -> flat, 10
    reopened = ledger.apply_fill(sell("2", "200"))  # new short, realized kept
    closed = ledger.apply_fill(buy("2", "150"))  # +100

    assert summary(flat) == (D("0"), None, D("10"))
    assert flat.unrealized_pnl == D("0")
    assert summary(reopened) == (D("-2"), D("200"), D("10"))
    assert reopened.unrealized_pnl is None
    assert summary(closed) == (D("0"), None, D("110"))


# --- fees / mark ----------------------------------------------------------------


def test_known_fee_is_ignored_gross_only() -> None:
    p = run(
        buy("10", "100", fee=D("1.5"), fee_asset="USDT"),
        sell("10", "110", fee=D("-0.2"), fee_asset="USDT"),
    )

    assert p.realized_pnl == D("100")


def test_mark_and_unrealized_semantics() -> None:
    open_p = run(buy("1", "100"))
    flat_p = run(buy("1", "100"), sell("1", "120"))

    assert (open_p.mark_price, open_p.unrealized_pnl) == (None, None)
    assert (flat_p.mark_price, flat_p.unrealized_pnl) == (None, D("0"))


# --- decimal context ------------------------------------------------------------


def test_results_ignore_the_global_decimal_context() -> None:
    def scenario() -> list[Position]:
        ledger = SimulatedPositionLedger()
        return [
            ledger.apply_fill(f)
            for f in (
                buy("2", "100.123", exec_id="a"),
                buy("1", "110.7", exec_id="b"),
                sell("1.5", "120.05", exec_id="c"),
                sell("3", "99.99", exec_id="d"),
            )
        ]

    baseline = scenario()
    with localcontext() as context:
        context.prec = 2
        context.rounding = "ROUND_UP"
        low = scenario()

    assert low == baseline
    assert [str(p.realized_pnl) for p in low] == [str(p.realized_pnl) for p in baseline]


def test_precision_matches_the_simulator_average_policy() -> None:
    assert simulated_positions.POSITION_PRICE_PRECISION == fill_math.AVERAGE_PRICE_PRECISION


# --- exec-id idempotency ----------------------------------------------------------


def test_exact_duplicate_fill_is_idempotent() -> None:
    ledger = SimulatedPositionLedger()
    first = buy("10", "100", exec_id="X")
    ledger.apply_fill(first)
    after = ledger.apply_fill(sell("4", "110"))

    again = ledger.apply_fill(first)

    assert again == after
    assert ledger.get_position("BTCUSDT") == after


def test_same_exec_id_with_other_payload_is_a_conflict() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("10", "100", exec_id="X"))
    before = ledger.get_position("BTCUSDT")

    with pytest.raises(PositionAccountingError, match="exec_id X"):
        ledger.apply_fill(buy("10", "101", exec_id="X"))

    assert ledger.get_position("BTCUSDT") == before


def test_duplicate_exec_id_inside_one_batch() -> None:
    ledger = SimulatedPositionLedger()
    f = buy("1", "100", exec_id="X")

    prepared = ledger.prepare((f, f))
    ledger.commit(prepared)
    assert ledger.get_position("BTCUSDT") == run(buy("1", "100"))

    with pytest.raises(PositionAccountingError, match="exec_id Y"):
        ledger.prepare((buy("1", "100", exec_id="Y"), buy("2", "100", exec_id="Y")))


def test_error_is_not_an_exchange_error() -> None:
    from app.exchanges.errors import ExchangeError

    assert issubclass(PositionAccountingError, RuntimeError)
    assert not issubclass(PositionAccountingError, ExchangeError)


# --- timestamps -------------------------------------------------------------------


def test_stale_fill_is_rejected_without_mutation() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=10)))
    before = ledger.get_position("BTCUSDT")

    with pytest.raises(PositionAccountingError, match="older"):
        ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=9)))

    assert ledger.get_position("BTCUSDT") == before


def test_equal_timestamps_are_accepted() -> None:
    p = run(buy("1", "100"), buy("1", "102"), sell("1", "101"))

    assert summary(p) == (D("1"), D("101"), D("0"))
    assert p.updated_at == T0


def test_stale_check_is_per_symbol() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=10)))

    p = ledger.apply_fill(buy("1", "100", symbol="ETHUSDT", ts=T0))

    assert p.updated_at == T0


# --- preparation / commit ---------------------------------------------------------


def test_prepare_does_not_change_the_ledger() -> None:
    ledger = SimulatedPositionLedger()

    prepared = ledger.prepare((buy("1", "100"), buy("1", "102")))

    assert ledger.get_position("BTCUSDT") is None
    ledger.commit(prepared)
    p = ledger.get_position("BTCUSDT")
    assert p is not None
    assert summary(p) == (D("2"), D("101"), D("0"))


def test_failed_preparation_leaves_ledger_unchanged() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=5)))
    before = ledger.get_position("BTCUSDT")

    with pytest.raises(PositionAccountingError):
        ledger.prepare(
            (
                buy("1", "100", exec_id="ok", ts=T0 + timedelta(seconds=6)),
                buy("1", "100", exec_id="stale", ts=T0 + timedelta(seconds=1)),
            )
        )

    assert ledger.get_position("BTCUSDT") == before
    # Neither exec id was recorded: both can still be applied.
    p = ledger.apply_fill(buy("1", "100", exec_id="ok", ts=T0 + timedelta(seconds=6)))
    assert p.qty == D("2")


def test_stale_prepared_batch_cannot_be_committed() -> None:
    ledger = SimulatedPositionLedger()
    prepared = ledger.prepare((buy("1", "100"),))
    ledger.apply_fill(buy("1", "100"))

    with pytest.raises(PositionAccountingError, match="stale"):
        ledger.commit(prepared)


def test_non_fill_is_rejected() -> None:
    ledger = SimulatedPositionLedger()

    with pytest.raises(PositionAccountingError, match="Fill"):
        ledger.apply_fill("not a fill")  # type: ignore[arg-type]


# --- module boundaries ------------------------------------------------------------


def test_ledger_has_no_forbidden_dependencies() -> None:
    source = Path(simulated_positions.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    for name in imports:
        assert name.split(".")[0] in {
            "__future__",
            "dataclasses",
            "decimal",
            "fractions",
            "collections",
            "datetime",
            "typing",
            "app",
        }, name
        assert not name.startswith("app.") or name.startswith("app.domain"), name
    for banned in ("float(", "getcontext", "setcontext", "localcontext", ".now(", ".fee"):
        assert banned not in source, banned


# --- working batch ------------------------------------------------------------------


def test_batch_reads_its_own_prepared_state_without_touching_the_ledger() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("10", "100"))
    batch = ledger.begin_batch()

    assert batch.signed_qty("BTCUSDT") == D("10")
    assert batch.signed_qty("ETHUSDT") == D("0")
    batch.apply(sell("7", "110"))
    assert batch.signed_qty("BTCUSDT") == D("3")  # sees the previous prepared fill
    batch.apply(sell("3", "110"))
    assert batch.signed_qty("BTCUSDT") == D("0")

    original = ledger.get_position("BTCUSDT")
    assert original is not None
    assert original.qty == D("10")  # untouched until commit
    ledger.commit(batch.prepared())
    p = ledger.get_position("BTCUSDT")
    assert p is not None
    assert summary(p) == (D("0"), None, D("100"))


def test_failed_batch_apply_keeps_ledger_and_batch_consistent() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=5)))
    batch = ledger.begin_batch()
    batch.apply(buy("1", "100", exec_id="ok", ts=T0 + timedelta(seconds=6)))

    with pytest.raises(PositionAccountingError):
        batch.apply(buy("1", "100", exec_id="stale", ts=T0))

    assert batch.signed_qty("BTCUSDT") == D("2")  # the failed fill left no trace
    original = ledger.get_position("BTCUSDT")
    assert original is not None
    assert original.qty == D("1")


def test_stale_batch_commit_is_rejected() -> None:
    ledger = SimulatedPositionLedger()
    batch = ledger.begin_batch()
    batch.apply(buy("1", "100"))
    ledger.apply_fill(buy("1", "100"))

    with pytest.raises(PositionAccountingError, match="stale"):
        ledger.commit(batch.prepared())


def test_batch_keeps_exec_id_idempotency() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", exec_id="X"))
    batch = ledger.begin_batch()

    batch.apply(buy("1", "100", exec_id="X"))  # identical: ignored
    assert batch.signed_qty("BTCUSDT") == D("1")
    with pytest.raises(PositionAccountingError, match="exec_id X"):
        batch.apply(buy("2", "100", exec_id="X"))


# --- exact realized delta per fill --------------------------------------------------


def test_batch_apply_returns_exact_realized_delta() -> None:
    from fractions import Fraction

    ledger = SimulatedPositionLedger()
    batch = ledger.begin_batch()

    assert batch.apply(buy("2", "100")) == Fraction(0)  # open
    assert batch.apply(buy("1", "110")) == Fraction(0)  # increase
    assert batch.apply(sell("1", "120")) == Fraction(50, 3)  # 120 - 310/3, exact
    assert batch.apply(sell("4", "100")) == Fraction(-20, 3)  # closes 2 @ 310/3, opens 2
    replay = sell("1", "100", exec_id="R")
    assert batch.apply(replay) == Fraction(0)
    assert batch.apply(replay) is None  # identical replay: nothing applied


def test_delta_sum_equals_exact_realized() -> None:
    ledger = SimulatedPositionLedger()
    batch = ledger.begin_batch()
    deltas = [
        batch.apply(f)
        for f in (buy("3", "100"), sell("1", "101"), sell("1", "102"), sell("1", "103.5"))
    ]

    ledger.commit(batch.prepared())
    p = ledger.get_position("BTCUSDT")
    assert p is not None
    assert sum(d for d in deltas if d is not None) == p.realized_pnl == D("6.5")


# --- valuation (mark-to-market) -------------------------------------------------------


def mark(price: str, seconds: int = 0) -> MarkQuote:
    return MarkQuote(price=D(price), at=T0 + timedelta(seconds=seconds))


def valued(*fills: Fill, quote: MarkQuote) -> Position:
    ledger = SimulatedPositionLedger()
    for f in fills:
        ledger.apply_fill(f)
    p = ledger.get_position("BTCUSDT", mark=quote)
    assert p is not None
    return p


@pytest.mark.parametrize(
    ("side", "price", "expected"),
    [
        (Side.BUY, "90", "-100"),
        (Side.BUY, "100", "0"),
        (Side.BUY, "110", "100"),
        (Side.SELL, "90", "100"),
        (Side.SELL, "100", "0"),
        (Side.SELL, "110", "-100"),
    ],
)
def test_unrealized_long_and_short(side: Side, price: str, expected: str) -> None:
    p = valued(fill(side, "10", "100"), quote=mark(price))

    assert (p.mark_price, p.unrealized_pnl) == (D(price), D(expected))
    assert p.unrealized_pnl is not None  # a known zero stays known


def test_unrealized_uses_exact_basis_not_rounded_entry() -> None:
    p = valued(buy("2", "100"), buy("1", "110"), quote=mark("120"))

    # 3 * 120 - 310 = 50 exactly; the rounded entry (103.33...) would not give it.
    assert p.unrealized_pnl == D("50")
    assert p.entry_price == D("103.3333333333333333333333333333333333333")


def test_unrealized_non_terminating_is_published_at_40_digits() -> None:
    p = valued(buy("3", "100"), sell("1", "100"), quote=mark("100.5"))

    # 2 * 100.5 - 2 * 100 = 1 exactly
    assert p.unrealized_pnl == D("1")
    q = valued(buy("2", "100"), buy("1", "101"), sell("2", "100"), quote=mark("100"))
    # remaining 1 @ 301/3: 100 - 100.333... = -1/3
    assert q.unrealized_pnl == D("-0.3333333333333333333333333333333333333333")


def test_flat_keeps_mark_and_zero_unrealized() -> None:
    p = valued(buy("1", "100"), sell("1", "110"), quote=mark("105"))

    assert (p.qty, p.entry_price, p.mark_price, p.unrealized_pnl) == (
        D("0"),
        None,
        D("105"),
        D("0"),
    )


def test_updated_at_is_latest_of_fill_and_mark() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=5)))

    later_mark = ledger.get_position("BTCUSDT", mark=mark("101", 9))
    earlier_mark = ledger.get_position("BTCUSDT", mark=mark("101", 2))

    assert later_mark is not None
    assert earlier_mark is not None
    assert later_mark.updated_at == T0 + timedelta(seconds=9)
    assert earlier_mark.updated_at == T0 + timedelta(seconds=5)


def test_mark_never_moves_the_fill_watermark() -> None:
    # A newer mark must not make an older-than-mark fill look stale.
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=1)))
    ledger.get_position("BTCUSDT", mark=mark("101", 10))

    p = ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=2)))

    assert p.qty == D("2")
    with pytest.raises(PositionAccountingError, match="older"):
        ledger.apply_fill(buy("1", "100", ts=T0 + timedelta(seconds=1, microseconds=-1)))


def test_valuation_ignores_the_global_decimal_context() -> None:
    def run() -> Position:
        return valued(buy("2", "100.123"), buy("1", "110.7"), quote=mark("105.05"))

    baseline = run()
    with localcontext() as context:
        context.prec = 2
        context.rounding = "ROUND_UP"
        low = run()

    assert low == baseline
    assert str(low.unrealized_pnl) == str(baseline.unrealized_pnl)


@pytest.mark.parametrize("price", [D("0"), D("-1"), D("NaN"), D("Infinity"), 1, 1.5, "1", None])
def test_mark_quote_validation(price: object) -> None:
    with pytest.raises(ValueError, match="mark"):
        MarkQuote(price=price, at=T0)  # type: ignore[arg-type]


def test_mark_quote_requires_utc_time() -> None:
    with pytest.raises(ValueError, match="at"):
        MarkQuote(price=D("1"), at=datetime(2026, 1, 15))  # noqa: DTZ001


def test_batch_position_uses_prepared_state_and_mark() -> None:
    ledger = SimulatedPositionLedger()
    batch = ledger.begin_batch()
    batch.apply(buy("10", "100"))

    p = batch.position("BTCUSDT", mark=mark("110"))

    assert p is not None
    assert (p.qty, p.unrealized_pnl) == (D("10"), D("100"))
    assert ledger.get_position("BTCUSDT") is None
    assert batch.position("ETHUSDT", mark=None) is None


# --- exact unrealized aggregate --------------------------------------------------------


def test_exact_unrealized_total_none_without_positions_is_zero() -> None:
    from fractions import Fraction

    assert SimulatedPositionLedger().exact_unrealized_total({}) == Fraction(0)


def test_exact_unrealized_total_sums_exact_values() -> None:
    from fractions import Fraction

    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("2", "100"))
    ledger.apply_fill(buy("1", "101"))  # basis 301/3
    ledger.apply_fill(sell("1", "50", symbol="ETHUSDT"))

    total = ledger.exact_unrealized_total({"BTCUSDT": mark("100"), "ETHUSDT": mark("60")})

    # BTC: 300 - 301 = -1; ETH short: 50 - 60 = -10
    assert total == Fraction(-11)


def test_exact_unrealized_total_unknown_if_any_open_position_lacks_a_mark() -> None:
    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100"))
    ledger.apply_fill(sell("1", "50", symbol="ETHUSDT"))

    assert ledger.exact_unrealized_total({"BTCUSDT": mark("110")}) is None


def test_exact_unrealized_total_flat_needs_no_mark() -> None:
    from fractions import Fraction

    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("1", "100"))
    ledger.apply_fill(sell("1", "120"))

    assert ledger.exact_unrealized_total({}) == Fraction(0)


def test_exact_unrealized_total_matches_published_values() -> None:
    from fractions import Fraction

    ledger = SimulatedPositionLedger()
    ledger.apply_fill(buy("2", "100"))
    ledger.apply_fill(buy("1", "110"))  # basis 310/3
    quote = mark("105")

    total = ledger.exact_unrealized_total({"BTCUSDT": quote})
    p = ledger.get_position("BTCUSDT", mark=quote)

    assert total == Fraction(5)
    assert p is not None
    assert p.unrealized_pnl == D("5")
