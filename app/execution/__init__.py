"""Execution layer: local order state between Risk approval and the exchange.

The in-memory account state (docs/ARCHITECTURE.md 7.0, the account-level
serialization boundary), the single-attempt order submission lifecycle (7.2)
and single-shot UNKNOWN reconciliation (7.3) through the ``TradingClient``
abstraction, and the account-state persistence port (``persistence.py``) that
storage adapters implement; every account mutation is committed through it
before it is published. This package never imports ``app.persistence``. No
retries, startup recovery or full reconciliation yet.
"""
