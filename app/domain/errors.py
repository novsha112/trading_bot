"""Domain error hierarchy."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class DomainValidationError(DomainError, ValueError):
    """A value violates a domain invariant (wrong type, sign, non-finite, non-UTC...)."""


class InvalidOrderTransition(DomainError):  # noqa: N818 - name fixed by the domain contract
    """A status change or fill update that the order state machine does not allow."""
