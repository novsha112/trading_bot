"""Exchange errors, classified by what is known about the request's effect.

* ``ExchangeRejectedError``: the exchange definitively refused the request; it had
  no effect. Not retried.
* ``ExchangeUnavailableError``: the request was definitely not executed (not sent,
  connection refused, rate-limited with an explicit answer). May be retried later
  under the retry policy.
* ``ExchangeAmbiguousResultError``: a mutating request may or may not have been
  executed (timeout after sending, connection lost before the answer). Never
  retried blindly: the outcome must be established through reconciliation
  (e.g. ``get_order`` by client order id). If an adapter cannot prove that a
  mutating request did not reach the exchange, it must raise this error.

Messages must not contain credentials, signatures, raw headers or raw response
bodies.
"""

from __future__ import annotations


class ExchangeError(Exception):
    """Base class of all exchange errors."""


class ExchangeRejectedError(ExchangeError):
    """The exchange definitively rejected the request; nothing was executed."""


class ExchangeAuthenticationError(ExchangeRejectedError):
    """Credentials or permissions were refused; nothing was executed."""


class ExchangeUnavailableError(ExchangeError):
    """The request was definitely not executed (e.g. not sent, explicit rate limit)."""


class ExchangeAmbiguousResultError(ExchangeError):
    """A mutating request may have been executed; the outcome is unknown."""
