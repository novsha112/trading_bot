"""Local time of an order state change.

* ``change_time``: for a change that must be recorded no matter what. After a
  network call the outcome is known and must be stored; a failing or
  misbehaving clock must not prevent that. It reads the clock once and falls
  back to ``floor`` (the order's own ``updated_at``) when the clock raises,
  returns something that is not a UTC datetime, or a time before ``floor``.
* ``strict_change_time``: for a change that is only made if a valid time
  exists (e.g. a safety block before any send: nothing is lost by not making
  it). It reads the clock once; an exception propagates unchanged and anything
  but an aware UTC ``datetime`` raises ``DomainValidationError`` (never a
  fallback). Only a valid time before ``floor`` is clamped to ``floor``.

In both cases the recorded time never moves an order backwards.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.domain.clock import Clock
from app.domain.validation import require_utc


def change_time(clock: Clock, *, floor: datetime) -> datetime:
    """``clock.now()`` if it is a valid UTC time not before ``floor``, else ``floor``."""
    try:
        now = clock.now()
    except Exception:
        return floor
    if not isinstance(now, datetime) or now.utcoffset() != timedelta(0) or now < floor:
        return floor
    return now


def strict_change_time(clock: Clock, *, floor: datetime) -> datetime:
    """``clock.now()`` validated as an aware UTC ``datetime``, clamped to ``floor``.

    Raises:
        Exception: whatever ``clock.now()`` raises, unchanged.
        DomainValidationError: not a datetime, naive, or not UTC.
    """
    now = require_utc(clock.now(), "clock.now()")
    return now if now >= floor else floor
