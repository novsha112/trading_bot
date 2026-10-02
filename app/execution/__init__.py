"""Execution layer: local order state between Risk approval and the exchange.

The in-memory account state (docs/ARCHITECTURE.md 7.0, the account-level
serialization boundary) and the single-attempt order submission lifecycle
(7.2) through the ``TradingClient`` abstraction. No resolver, reconciliation or
persistence yet.
"""
