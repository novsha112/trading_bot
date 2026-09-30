"""Clock is the only source of current time for trading logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.domain.clock import Clock, ManualClock, SystemClock
from app.domain.errors import DomainValidationError

START = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def test_system_clock_returns_current_utc_time() -> None:
    clock: Clock = SystemClock()
    before = datetime.now(UTC)  # noqa: TID251 - reference value for the real clock
    now = clock.now()
    after = datetime.now(UTC)  # noqa: TID251 - reference value for the real clock

    assert now.utcoffset() == timedelta(0)
    assert before <= now <= after


def test_manual_clock_starts_at_given_time() -> None:
    clock: Clock = ManualClock(START)
    assert clock.now() == START


@pytest.mark.parametrize(
    "start",
    [
        datetime(2026, 1, 15, 12, 0),  # noqa: DTZ001 - deliberately naive
        datetime(2026, 1, 15, 12, 0, tzinfo=timezone(timedelta(hours=3))),
    ],
)
def test_manual_clock_rejects_non_utc_start(start: datetime) -> None:
    with pytest.raises(DomainValidationError):
        ManualClock(start)


def test_advance_moves_forward() -> None:
    clock = ManualClock(START)
    clock.advance(timedelta(seconds=90))
    clock.advance(timedelta(microseconds=1))
    assert clock.now() == START + timedelta(seconds=90, microseconds=1)


def test_advance_by_zero_is_allowed() -> None:
    clock = ManualClock(START)
    clock.advance(timedelta(0))
    assert clock.now() == START


def test_advance_backwards_rejected_and_time_unchanged() -> None:
    clock = ManualClock(START)
    with pytest.raises(DomainValidationError, match="backwards"):
        clock.advance(timedelta(microseconds=-1))
    assert clock.now() == START


@pytest.mark.parametrize("delta", [1, 1.5, "1s", None])
def test_advance_requires_timedelta(delta: object) -> None:
    clock = ManualClock(START)
    with pytest.raises(DomainValidationError, match="timedelta"):
        clock.advance(delta)  # type: ignore[arg-type]
    assert clock.now() == START


def test_set_moves_forward_and_accepts_same_time() -> None:
    clock = ManualClock(START)
    clock.set(START)
    later = START + timedelta(hours=1)
    clock.set(later)
    assert clock.now() == later


def test_set_backwards_rejected_and_time_unchanged() -> None:
    clock = ManualClock(START)
    with pytest.raises(DomainValidationError, match="backwards"):
        clock.set(START - timedelta(microseconds=1))
    assert clock.now() == START


def test_set_rejects_non_utc() -> None:
    clock = ManualClock(START)
    with pytest.raises(DomainValidationError):
        clock.set(datetime(2026, 1, 15, 13, 0))  # noqa: DTZ001 - deliberately naive
    assert clock.now() == START
