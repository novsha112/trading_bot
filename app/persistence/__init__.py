"""Persistence (docs/ARCHITECTURE.md 11, 11.0).

Storage adapters of the execution-owned persistence port
(``app.execution.persistence``) and physical-storage codecs: pure codecs (exact
Decimal, UTC datetime) and the deterministic in-memory reference store. No
database driver or migrations yet; the execution layer does not use a store yet.
"""
