"""The durable persistence boundary of the account aggregate (ARCHITECTURE 11.0).

One transactional store for the whole risk-relevant account state, not a set of
per-table repositories. It is NOT a multi-writer coordination mechanism: the
account aggregate serializes its mutations (one writer per account scope in
V1); ``expected_revision`` is a compare-and-set guard against stale state.
"""

from __future__ import annotations

from typing import Protocol

from app.persistence.models import AccountStateChange, PersistedAccountState


class AccountStateStore(Protocol):
    async def load(self, *, account_scope_id: str) -> PersistedAccountState | None:
        """The committed state of the account, or None if nothing was ever committed."""
        ...

    async def commit(self, change: AccountStateChange) -> None:
        """Apply the whole change atomically, or nothing.

        Raises:
            StoreValidationError: malformed change (checked before anything else).
            StoreConflictError: stale ``expected_revision`` or an identity already
                stored with different data.
            StoreCommitError: the commit definitely did not happen.
            StoreUncertainError: the outcome is unknown; it may be fully applied.
        """
        ...
