"""Secrets must never reach log output, whatever path they take."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from app.monitoring.logging import (
    REDACTED,
    SecretRedactor,
    configure_logging,
    is_sensitive_name,
)

API_KEY = "AK-test-3f9a1c7e5b2d"
API_SECRET = "SK-test-8e4b6d2a0c9f1e7b"
BOT_TOKEN = "123456:TG-test-token-value"
ALL_SECRETS = (API_KEY, API_SECRET, BOT_TOKEN)


class FakeSecretStr:
    """Mimics pydantic.SecretStr without adding pydantic as a dependency."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_trading_bot_handler", False):
            root.removeHandler(handler)
    root.setLevel(logging.WARNING)


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


def _configure(stream: io.StringIO, log_format: str = "json") -> None:
    configure_logging(
        level="DEBUG",
        log_format=log_format,  # type: ignore[arg-type]
        secrets=ALL_SECRETS,
        stream=stream,
    )


def _json_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def _assert_no_secrets(output: str) -> None:
    for secret in ALL_SECRETS:
        assert secret not in output


@pytest.mark.parametrize(
    "name",
    [
        "api_key",
        "apiKey",
        "API_SECRET",
        "secret",
        "bot_token",
        "X-BAPI-API-KEY",
        "X-BAPI-SIGN",
        "signature",
        "Authorization",
        "password",
        "passphrase",
        "Cookie",
    ],
)
def test_sensitive_names_detected(name: str) -> None:
    assert is_sensitive_name(name)


@pytest.mark.parametrize(
    "name",
    ["event", "signal", "symbol", "client_order_id", "order_link_id", "price", "qty", "monkey"],
)
def test_regular_names_not_detected(name: str) -> None:
    assert not is_sensitive_name(name)


def test_sensitive_field_masked_by_name(stream: io.StringIO) -> None:
    # Not a registered secret: only the field name protects it.
    configure_logging(log_format="json", stream=stream)
    structlog.get_logger().info("auth", api_key="unregistered-key-value", symbol="BTCUSDT")

    [record] = _json_lines(stream)
    assert record["api_key"] == REDACTED
    assert record["symbol"] == "BTCUSDT"
    assert "unregistered-key-value" not in stream.getvalue()


def test_nested_headers_masked(stream: io.StringIO) -> None:
    configure_logging(level="DEBUG", log_format="json", stream=stream)
    structlog.get_logger().debug(
        "http_request",
        headers={"X-BAPI-API-KEY": "k-value", "X-BAPI-SIGN": "s-value", "Content-Type": "json"},
        attempts=[{"token": "t-value", "status": 200}],
    )

    [record] = _json_lines(stream)
    assert record["headers"] == {
        "X-BAPI-API-KEY": REDACTED,
        "X-BAPI-SIGN": REDACTED,
        "Content-Type": "json",
    }
    assert record["attempts"] == [{"token": REDACTED, "status": 200}]


def test_registered_secret_masked_in_event_and_free_text(stream: io.StringIO) -> None:
    _configure(stream)
    structlog.get_logger().warning(
        f"request failed for key {API_KEY}",
        detail=f"url=https://example.invalid/?api_key={API_KEY}&sign={API_SECRET}",
    )

    output = stream.getvalue()
    _assert_no_secrets(output)
    [record] = _json_lines(stream)
    assert record["event"] == f"request failed for key {REDACTED}"


def test_secret_in_exception_message_and_traceback(stream: io.StringIO) -> None:
    _configure(stream)
    log = structlog.get_logger()
    try:
        try:
            raise ConnectionError(f"auth failed with secret {API_SECRET}")
        except ConnectionError as exc:
            raise RuntimeError(f"exchange error, key={API_KEY}") from exc
    except RuntimeError:
        log.exception("order_submit_failed", client_order_id="grid-1-buy-0001")

    output = stream.getvalue()
    _assert_no_secrets(output)
    [record] = _json_lines(stream)
    assert "RuntimeError" in record["exception"]
    assert "ConnectionError" in record["exception"]
    assert REDACTED in record["exception"]
    assert record["client_order_id"] == "grid-1-buy-0001"


def test_exception_object_as_field_value(stream: io.StringIO) -> None:
    _configure(stream)
    structlog.get_logger().error("api_error", error=ValueError(f"bad key {API_KEY}"))

    _assert_no_secrets(stream.getvalue())
    [record] = _json_lines(stream)
    assert record["error"] == f"bad key {REDACTED}"


def test_secret_wrapper_object_masked(stream: io.StringIO) -> None:
    configure_logging(log_format="json", stream=stream)
    structlog.get_logger().info("settings_loaded", exchange_credential_ref=FakeSecretStr("x-y-z"))

    [record] = _json_lines(stream)
    assert record["exchange_credential_ref"] == REDACTED


def test_stdlib_logger_secrets_masked(stream: io.StringIO) -> None:
    """Third-party libraries log through stdlib logging; they must be scrubbed too."""
    _configure(stream)
    foreign = logging.getLogger("some.third_party.http")
    foreign.debug("sending headers %s", {"X-BAPI-API-KEY": API_KEY})
    try:
        raise OSError(f"socket closed, token={BOT_TOKEN}")
    except OSError:
        foreign.exception("request failed")

    output = stream.getvalue()
    _assert_no_secrets(output)
    records = _json_lines(stream)
    assert [r["logger"] for r in records] == ["some.third_party.http"] * 2
    assert "OSError" in records[1]["exception"]


def test_contextvars_are_redacted(stream: io.StringIO) -> None:
    _configure(stream)
    structlog.contextvars.bind_contextvars(mode="paper", session=f"s-{API_KEY}")
    try:
        structlog.get_logger().info("tick")
    finally:
        structlog.contextvars.clear_contextvars()

    _assert_no_secrets(stream.getvalue())
    [record] = _json_lines(stream)
    assert record["mode"] == "paper"


def test_console_format_is_redacted(stream: io.StringIO) -> None:
    _configure(stream, log_format="console")
    log = structlog.get_logger()
    log.info("startup", api_secret=API_SECRET, note=f"token {BOT_TOKEN}")
    try:
        raise RuntimeError(API_KEY)
    except RuntimeError:
        log.exception("boom")

    output = stream.getvalue()
    _assert_no_secrets(output)
    assert REDACTED in output


def test_empty_sensitive_values_are_kept_for_diagnostics() -> None:
    redactor = SecretRedactor(secrets=["", API_KEY])
    result = redactor(None, "info", {"event": "config", "api_key": None, "api_secret": ""})
    assert result == {"event": "config", "api_key": None, "api_secret": ""}


def test_overlapping_secrets_fully_masked() -> None:
    short, long = "abcdef", "abcdef-123456"
    redactor = SecretRedactor(secrets=[short, long])
    result = redactor(None, "info", {"event": f"value {long}"})
    assert result["event"] == f"value {REDACTED}"


def test_log_level_filtering(stream: io.StringIO) -> None:
    configure_logging(level=logging.WARNING, log_format="json", stream=stream)
    log = structlog.get_logger()
    log.info("hidden")
    log.warning("shown")

    assert [r["event"] for r in _json_lines(stream)] == ["shown"]


def test_reconfiguration_does_not_duplicate_handlers(stream: io.StringIO) -> None:
    configure_logging(log_format="json", stream=io.StringIO())
    configure_logging(log_format="json", stream=stream)
    structlog.get_logger().info("once")

    assert len(_json_lines(stream)) == 1


def test_invalid_level_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging(level="LOUD")


def test_invalid_format_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown log format"):
        configure_logging(log_format="xml")  # type: ignore[arg-type]
