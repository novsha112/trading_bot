"""Exchange error taxonomy: what is known about the effect of a request."""

from __future__ import annotations

import itertools

import pytest

from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeAuthenticationError,
    ExchangeError,
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeRequestValidationError,
    ExchangeResponseError,
)

OUTCOMES = [
    ExchangeNotSentError,
    ExchangeRejectedError,
    ExchangeAmbiguousResultError,
    ExchangeResponseError,
]


@pytest.mark.parametrize("error", [*OUTCOMES, ExchangeAuthenticationError])
def test_all_are_exchange_errors(error: type[ExchangeError]) -> None:
    assert issubclass(error, ExchangeError)


@pytest.mark.parametrize(("a", "b"), list(itertools.permutations(OUTCOMES, 2)))
def test_outcome_categories_are_siblings(a: type[ExchangeError], b: type[ExchangeError]) -> None:
    # No category can be caught as another: an ambiguous outcome is never mistaken
    # for a safe "not sent", and vice versa.
    assert not issubclass(a, b)


def test_authentication_is_a_rejection() -> None:
    assert issubclass(ExchangeAuthenticationError, ExchangeRejectedError)
    assert not issubclass(ExchangeAuthenticationError, ExchangeNotSentError)
    assert not issubclass(ExchangeAuthenticationError, ExchangeAmbiguousResultError)


def test_old_unavailable_name_is_gone() -> None:
    import app.exchanges.errors as errors

    assert not hasattr(errors, "ExchangeUnavailableError")


def blind_retry_allowed(error: ExchangeError) -> bool:
    """The policy the contracts encode (no retry engine exists yet)."""
    try:
        raise error
    except ExchangeAmbiguousResultError:
        return False  # reconcile by client_order_id first
    except ExchangeRejectedError:
        return False  # definitive answer; repeating changes nothing
    except ExchangeNotSentError:
        return True  # nothing reached the exchange; the caller's policy may retry


@pytest.mark.parametrize(
    ("error", "allowed"),
    [
        (ExchangeNotSentError("local rate limiter: no token"), True),
        (ExchangeRejectedError("rate limit exceeded (exchange response)"), False),
        (ExchangeAuthenticationError("invalid signature"), False),
        (ExchangeAmbiguousResultError("timeout after request was sent"), False),
    ],
)
def test_blind_retry_policy(error: ExchangeError, allowed: bool) -> None:
    assert blind_retry_allowed(error) is allowed


def test_errors_carry_only_a_message() -> None:
    error = ExchangeRejectedError("order rejected: insufficient balance")
    assert error.args == ("order rejected: insufficient balance",)


def test_response_error_is_read_only_and_not_ambiguous() -> None:
    # A failed read has no side effect: it must not be confused with an ambiguous
    # mutating request nor with a definitive rejection.
    assert not issubclass(ExchangeResponseError, ExchangeAmbiguousResultError)
    assert not issubclass(ExchangeResponseError, ExchangeRejectedError)
    assert "Never raised for mutating requests" in (ExchangeResponseError.__doc__ or "")


def test_request_validation_error_is_a_local_not_sent_error() -> None:
    # Refused before any transport call: never ambiguous, never an exchange answer.
    assert issubclass(ExchangeRequestValidationError, ExchangeNotSentError)
    for other in (ExchangeRejectedError, ExchangeAmbiguousResultError, ExchangeResponseError):
        assert not issubclass(ExchangeRequestValidationError, other)
