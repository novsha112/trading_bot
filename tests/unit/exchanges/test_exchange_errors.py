"""Exchange error taxonomy: what is known about the effect of a request."""

from __future__ import annotations

import pytest

from app.exchanges.errors import (
    ExchangeAmbiguousResultError,
    ExchangeAuthenticationError,
    ExchangeError,
    ExchangeRejectedError,
    ExchangeUnavailableError,
)

ALL = [
    ExchangeRejectedError,
    ExchangeAuthenticationError,
    ExchangeUnavailableError,
    ExchangeAmbiguousResultError,
]


@pytest.mark.parametrize("error", ALL)
def test_all_are_exchange_errors(error: type[ExchangeError]) -> None:
    assert issubclass(error, ExchangeError)
    assert issubclass(ExchangeError, Exception)


def test_authentication_is_a_definitive_rejection() -> None:
    assert issubclass(ExchangeAuthenticationError, ExchangeRejectedError)


def test_ambiguous_result_is_not_a_safe_failure() -> None:
    # Catching "rejected" or "unavailable" must never swallow an ambiguous outcome:
    # the request may have been executed.
    assert not issubclass(ExchangeAmbiguousResultError, ExchangeRejectedError)
    assert not issubclass(ExchangeAmbiguousResultError, ExchangeUnavailableError)
    assert not issubclass(ExchangeRejectedError, ExchangeUnavailableError)
    assert not issubclass(ExchangeUnavailableError, ExchangeRejectedError)


def test_errors_carry_only_a_message() -> None:
    error = ExchangeRejectedError("order rejected: insufficient balance")
    assert str(error) == "order rejected: insufficient balance"
    assert error.args == ("order rejected: insufficient balance",)


def test_handler_order_example() -> None:
    def classify(error: ExchangeError) -> str:
        try:
            raise error
        except ExchangeAmbiguousResultError:
            return "reconcile"
        except ExchangeRejectedError:
            return "rejected"
        except ExchangeUnavailableError:
            return "not executed"

    assert classify(ExchangeAmbiguousResultError("timeout after send")) == "reconcile"
    assert classify(ExchangeAuthenticationError("invalid key")) == "rejected"
    assert classify(ExchangeUnavailableError("connection refused")) == "not executed"
