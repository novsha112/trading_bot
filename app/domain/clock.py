"""Time source for all trading logic.

Code must never read the system time directly: it asks a ``Clock``. Live, paper
and testnet use ``SystemClock``; backtests and tests use ``ManualClock``, so the
same logic runs on simulated time. Direct system time calls outside this module
are rejected by Ruff (TID251, see pyproject.toml).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.domain.errors import DomainValidationError
from app.domain.validation import require_utc


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """Wall-clock time of the host (keep it NTP-synchronized in production)."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """Clock controlled by the caller. Time never moves backwards."""

    def __init__(self, start: datetime) -> None:
        self._now = require_utc(start, "start")

    def now(self) -> datetime:
        return self._now

    def set(self, moment: datetime) -> None:
        """Move to ``moment``; the same time is allowed, an earlier one is not."""
        moment = require_utc(moment, "moment")
        if moment < self._now:
            raise DomainValidationError(
                f"ManualClock cannot move backwards: {moment.isoformat()} < {self._now.isoformat()}"
            )
        self._now = moment

    def advance(self, delta: timedelta) -> None:
        """Move forward by ``delta``; zero is allowed, a negative delta is not."""
        if not isinstance(delta, timedelta):
            raise DomainValidationError(f"delta must be a timedelta, got {type(delta).__name__}")
        if delta < timedelta(0):
            raise DomainValidationError(f"ManualClock cannot move backwards: delta={delta}")
        self._now += delta
