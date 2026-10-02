"""Execution layer: local order state between Risk approval and the exchange.

So far only the in-memory account state (docs/ARCHITECTURE.md 7.0): the
account-level serialization boundary for ``snapshot -> evaluate -> reserve``.
No submission, exchange client, resolver or persistence yet.
"""
