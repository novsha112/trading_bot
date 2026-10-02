"""Orchestration services that combine layers (docs/ARCHITECTURE.md 7.0).

So far only the placement coordinator: the atomic snapshot -> evaluate ->
reserve boundary. No submission, exchange access or persistence.
"""
