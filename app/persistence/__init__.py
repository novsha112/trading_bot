"""Persistence (docs/ARCHITECTURE.md 11, 11.0).

Pure storage codecs (exact Decimal, UTC datetime) and the account state store
contract with its deterministic in-memory reference implementation. No database
driver or migrations yet; the execution layer does not use the store yet.
"""
