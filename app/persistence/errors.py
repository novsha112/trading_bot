"""Errors of the account state store contract (docs/ARCHITECTURE.md 11.0)."""

from __future__ import annotations


class PersistenceStoreError(Exception):
    """Base class of account state store errors."""


class StoreValidationError(PersistenceStoreError):
    """The change set is malformed (types, revisions, references); nothing was
    written. A caller bug, never a storage condition."""


class StoreConflictError(PersistenceStoreError):
    """The change contradicts the durable state (stale ``expected_revision``, or
    an identity already stored with different data); nothing was written."""


class StoreCommitError(PersistenceStoreError):
    """The commit definitely did not happen: the durable state is unchanged."""


class StoreUncertainError(PersistenceStoreError):
    """The outcome of the commit is unknown (it may have been applied in full).
    The caller must treat its in-memory state as unusable until it reloads."""
