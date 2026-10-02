"""Local time of a state change that must be recorded no matter what.

After a network call the outcome is known and must be stored; a failing or
misbehaving clock must not prevent that. ``change_time`` reads the clock once and
falls back to ``floor`` (the order's own ``updated_at``) when the clock raises,
returns something that is not a UTC datetime, or a time before ``floor``: the
recorded time never moves an order backwards.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.domain.clock import Clock


def change_time(clock: Clock, *, floor: datetime) -> datetime:
    """``clock.now()`` if it is a valid UTC time not before ``floor``, else ``floor``."""
    try:
        now = clock.now()
    except Exception:
        return floor
    if not isinstance(now, datetime) or now.utcoffset() != timedelta(0) or now < floor:
        return floor
    return now
