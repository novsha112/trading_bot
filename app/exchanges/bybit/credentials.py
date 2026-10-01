"""Bybit API credentials, kept out of every text representation."""

from __future__ import annotations

from dataclasses import dataclass


def _require_clean(value: object, name: str) -> None:
    # The value is never part of the message: it is a credential.
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")


@dataclass(frozen=True, slots=True, repr=False)
class BybitCredentials:
    """API key and secret for HMAC signing.

    A plain wrapper (no pydantic dependency in the adapter): ``repr``/``str`` are
    masked. The composition layer builds it from ``EnvSettings`` secrets; only the
    private transport reads the fields. The secret is used for signing and is
    never sent.
    """

    api_key: str
    api_secret: str

    def __post_init__(self) -> None:
        _require_clean(self.api_key, "api_key")
        _require_clean(self.api_secret, "api_secret")

    def __repr__(self) -> str:
        return "BybitCredentials(api_key='**********', api_secret='**********')"

    __str__ = __repr__
