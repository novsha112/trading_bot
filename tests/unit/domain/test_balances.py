"""Balance snapshot of one asset."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.domain.balances import Balance
from app.domain.errors import DomainValidationError

D = Decimal
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

BALANCE: dict[str, Any] = {
    "asset": "USDT",
    "wallet_balance": D("1000"),
    "equity": D("1010.5"),
    "available": D("800"),
    "updated_at": TS,
}


def balance(**overrides: Any) -> Balance:
    return Balance(**{**BALANCE, **overrides})


def test_structure_frozen_slotted_keyword_only() -> None:
    b = balance()
    for name, value in BALANCE.items():
        assert getattr(b, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.equity = D("1")  # type: ignore[misc]
    assert not hasattr(b, "__dict__")
    with pytest.raises(TypeError):
        Balance(*BALANCE.values())  # type: ignore[call-arg]


def test_no_sign_constraints_yet() -> None:
    # Sign semantics depend on the account model; not constrained until confirmed.
    b = balance(wallet_balance=D("-1"), equity=D("-2"), available=D("-3"))
    assert (b.wallet_balance, b.equity, b.available) == (D("-1"), D("-2"), D("-3"))


@pytest.mark.parametrize("field", ["wallet_balance", "equity", "available"])
@pytest.mark.parametrize("value", [1.5, 1, True, None, D("NaN"), D("sNaN"), D("-Infinity")])
def test_numeric_fields_must_be_finite_decimal(field: str, value: Any) -> None:
    with pytest.raises(DomainValidationError, match=rf"^{field} must be"):
        balance(**{field: value})


@pytest.mark.parametrize("value", ["", " USDT", None])
def test_asset_validated(value: Any) -> None:
    with pytest.raises(DomainValidationError, match=r"^asset "):
        balance(asset=value)


def test_updated_at_must_be_utc() -> None:
    with pytest.raises(DomainValidationError, match=r"^updated_at "):
        balance(updated_at=datetime(2026, 1, 15))  # noqa: DTZ001 - deliberately naive
