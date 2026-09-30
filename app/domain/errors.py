"""Domain error hierarchy."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class DomainValidationError(DomainError, ValueError):
    """A value violates a domain invariant (wrong type, sign, non-finite, non-UTC...)."""
