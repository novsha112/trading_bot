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


def _keep_only_own_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach pytest's capture handlers: they format stdlib records on their own
    and would fail on the deliberately broken records, masking our handler's result."""
    root = logging.getLogger()
    own = [h for h in root.handlers if getattr(h, "_trading_bot_handler", False)]
    monkeypatch.setattr(root, "handlers", own)


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
        # Plural and short forms.
        "apiKeys",
        "APIKeys",
        "api_keys",
        "secrets",
        "tokens",
        "access_tokens",
        "pwd",
        "db_pwd",
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
    # Neither the field name nor a registered secret may be what masks the value:
    # only the get_secret_value() branch.
    assert not is_sensitive_name("loaded_value")
    configure_logging(log_format="json", stream=stream)
    structlog.get_logger().info(
        "settings_loaded", loaded_value=FakeSecretStr("unregistered-wrapped-value")
    )

    [record] = _json_lines(stream)
    assert record["loaded_value"] == REDACTED
    assert "unregistered-wrapped-value" not in stream.getvalue()


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


# --- Regression tests for issues found in the Phase 0 review ---------------------------


def test_stdlib_positional_args_masked_by_name_before_formatting(stream: io.StringIO) -> None:
    """Structured %-args of stdlib logging are redacted before the message is rendered."""
    configure_logging(level="DEBUG", log_format="json", stream=stream)
    foreign = logging.getLogger("some.third_party.http")
    foreign.debug("headers %s", {"X-BAPI-API-KEY": "unregistered-header-key"})
    foreign.debug("auth %s %s", "BTCUSDT", {"api_secret": "unregistered-api-secret"})

    output = stream.getvalue()
    assert "unregistered-header-key" not in output
    assert "unregistered-api-secret" not in output
    first, second = _json_lines(stream)
    assert first["event"] == f"headers {{'X-BAPI-API-KEY': '{REDACTED}'}}"
    assert second["event"] == f"auth BTCUSDT {{'api_secret': '{REDACTED}'}}"


def test_stdlib_named_args_still_formatted(stream: io.StringIO) -> None:
    configure_logging(log_format="json", stream=stream)
    logging.getLogger("lib").info("order %(id)s token %(token)s", {"id": 7, "token": "tk-value"})

    [record] = _json_lines(stream)
    assert record["event"] == f"order 7 token {REDACTED}"


def test_stdlib_bad_format_args_do_not_raise(
    stream: io.StringIO, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(log_format="json", stream=stream)
    _keep_only_own_handlers(monkeypatch)
    logging.getLogger("lib").info("value %d", "not-a-number")

    [record] = _json_lines(stream)
    assert "value %d" in record["event"]
    assert "Logging error" not in capsys.readouterr().err


def test_reconfiguration_updates_existing_logger_instances(stream: io.StringIO) -> None:
    secret = "late-registered-secret-42"
    configure_logging(log_format="json", stream=io.StringIO())
    log = structlog.get_logger()
    log.info("before secrets are known")

    configure_logging(log_format="json", secrets=[secret], stream=stream)
    log.info(f"after reconfigure {secret}")

    assert secret not in stream.getvalue()
    [record] = _json_lines(stream)
    assert record["event"] == f"after reconfigure {REDACTED}"


def test_secret_as_mapping_key_masked(stream: io.StringIO) -> None:
    _configure(stream)
    structlog.get_logger().info(
        "balances", by_account={API_KEY: 100, API_SECRET: 200, "sub-account": 300}
    )

    _assert_no_secrets(stream.getvalue())
    [record] = _json_lines(stream)
    # Structure and non-secret keys stay visible; colliding redacted keys are kept apart.
    assert record["by_account"] == {REDACTED: 100, f"{REDACTED}#2": 200, "sub-account": 300}


class ExplodingObject:
    def __str__(self) -> str:
        raise RuntimeError("broken __str__")

    def __repr__(self) -> str:
        raise RuntimeError("broken __repr__")


@pytest.mark.parametrize("log_format", ["json", "console"])
def test_unprintable_object_does_not_break_logging(
    stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    log_format: str,
) -> None:
    _configure(stream, log_format=log_format)
    _keep_only_own_handlers(monkeypatch)
    log = structlog.get_logger()

    log.info("order_state", order=ExplodingObject(), nested={"items": [ExplodingObject()]})
    logging.getLogger("lib").info("obj %s", ExplodingObject())

    assert stream.getvalue().count("[UNPRINTABLE]") == 3
    # logging.Handler.handleError reports formatting failures to stderr.
    assert "Logging error" not in capsys.readouterr().err


class ReprLeaksSecret:
    """str() is safe, repr() exposes a registered secret (e.g. a naive client repr)."""

    def __str__(self) -> str:
        return "BybitClient"

    def __repr__(self) -> str:
        return f"BybitClient(api_key={API_KEY!r})"


def test_console_does_not_expose_secret_via_repr(stream: io.StringIO) -> None:
    _configure(stream, log_format="console")
    structlog.get_logger().info("client_ready", client=ReprLeaksSecret())

    _assert_no_secrets(stream.getvalue())
