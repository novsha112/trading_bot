"""Exchange errors, classified by what is known about the request's effect.

Three sibling outcome categories; none is a subclass of another, so handling one
can never swallow another:

* ``ExchangeNotSentError``: the request definitely did not reach the exchange
  (local validation before sending, connection not established before any byte
  was sent, local rate limiter refused). The caller's policy may retry it.
* ``ExchangeRejectedError``: the exchange answered and definitively refused the
  request (including an explicit rate-limit refusal); nothing was accepted.
  Not retried blindly.
* ``ExchangeAmbiguousResultError``: the request may have been accepted, but no
  confirmed result exists (timeout or connection loss after sending, or the
  adapter cannot prove the request was not sent). Never retried blindly: the
  outcome is established first by reconciliation through ``client_order_id``.

A timeout is never classified as "not sent" unless the adapter can prove it.

Messages must not contain credentials, signatures, raw headers or raw response
bodies.
"""

from __future__ import annotations


class ExchangeError(Exception):
    """Base class of all exchange errors."""


class ExchangeNotSentError(ExchangeError):
    """The request definitely did not reach the exchange."""


class ExchangeRejectedError(ExchangeError):
    """The exchange definitively refused the request; nothing was accepted."""


class ExchangeAuthenticationError(ExchangeRejectedError):
    """The exchange refused authentication or authorization.

    Only for an exchange answer; missing local credentials are a configuration error.
    """


class ExchangeAmbiguousResultError(ExchangeError):
    """A request may have been accepted by the exchange; the outcome is unknown."""
