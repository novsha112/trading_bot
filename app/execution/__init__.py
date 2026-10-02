"""Execution layer: local order state between Risk approval and the exchange.

The in-memory account state (docs/ARCHITECTURE.md 7.0, the account-level
serialization boundary), the single-attempt order submission lifecycle (7.2)
and single-shot UNKNOWN reconciliation (7.3) through the ``TradingClient``
abstraction, and the account-state persistence port (``persistence.py``) that
storage adapters implement; every account mutation is committed through it
before it is published, and startup hydration (``recovery.py``) rebuilds it
from durable state with the locally provable crash classification only; the
runtime ``SafetyController`` (``safety.py``) gates the effective trading state
on recovery readiness and poison. This package never imports
``app.persistence``. No retries, exchange recovery or full reconciliation yet.
"""
