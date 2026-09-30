"""Structured logging setup with secret redaction.

All log records, both from structlog and from the stdlib ``logging`` module
(third-party libraries), go through the same processor chain. The chain renders
exceptions to text first and only then runs ``SecretRedactor``, so a secret that
ends up inside an exception message or traceback is masked as well.

Redaction works on two levels:

1. By field name: values of fields whose name contains a sensitive token
   (``api_key``, ``X-BAPI-SIGN``, ``bot_token``...) are masked entirely,
   including inside nested dicts and lists (e.g. HTTP headers).
2. By value: every registered secret value is replaced wherever it appears in
   any string (event text, other fields, rendered tracebacks).

Name-based masking alone cannot catch a secret embedded in free text, so the
actual secret values must be registered via ``configure_logging(secrets=...)``.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable, Mapping
from typing import IO, Any, Final, Literal

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

REDACTED: Final = "[REDACTED]"

# Field names are split into lowercase tokens ("X-BAPI-API-KEY" -> x, bapi, api, key;
# "apiKey" -> api, key). A field is sensitive if any token is in this set. Token
# matching (not substring matching) keeps fields like "signal" readable.
SENSITIVE_NAME_TOKENS: Final = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "key",
        "passphrase",
        "passwd",
        "password",
        "secret",
        "sign",
        "signature",
        "token",
    }
)

# structlog / ProcessorFormatter bookkeeping keys that must not be touched.
_META_KEYS: Final = frozenset({"_record", "_from_structlog"})

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")

LogFormat = Literal["json", "console"]

# Marks handlers installed by configure_logging so that reconfiguration replaces
# only them and leaves foreign handlers (e.g. pytest's caplog) alone.
_HANDLER_MARKER: Final = "_trading_bot_handler"


def is_sensitive_name(name: str) -> bool:
    """Return True if a field name looks like it holds a credential."""
    tokens = _NON_ALNUM.split(_CAMEL_BOUNDARY.sub("_", name).lower())
    return any(token in SENSITIVE_NAME_TOKENS for token in tokens)


class SecretRedactor:
    """structlog processor that masks secrets by field name and by value."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        # Longest first, so a secret that contains another secret is fully masked.
        self._secrets: tuple[str, ...] = tuple(
            sorted({s for s in secrets if s}, key=len, reverse=True)
        )

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        return {
            key: value if key in _META_KEYS else self._redact_field(key, value)
            for key, value in event_dict.items()
        }

    def _redact_field(self, name: str, value: Any) -> Any:
        if is_sensitive_name(name) and value is not None and value != "":
            return REDACTED
        return self._redact_value(value)

    def _redact_value(self, value: Any) -> Any:
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self._scrub(value)
        if hasattr(value, "get_secret_value"):
            # pydantic.SecretStr / SecretBytes and similar wrappers.
            return REDACTED
        if isinstance(value, Mapping):
            return {str(k): self._redact_field(str(k), v) for k, v in value.items()}
        if isinstance(value, list | tuple | set | frozenset):
            return [self._redact_value(item) for item in value]
        # Arbitrary objects (exceptions, bytes, models) are rendered with str()
        # later by the renderer; check that text now.
        text = str(value)
        scrubbed = self._scrub(text)
        return scrubbed if scrubbed != text else value

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
        return text


def _build_shared_processors(redactor: SecretRedactor) -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        # Render exceptions to text BEFORE redaction so tracebacks are scrubbed.
        structlog.processors.format_exc_info,
        # Must stay last: nothing may add unredacted data after it.
        redactor,
    ]


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_format: LogFormat = "json",
    secrets: Iterable[str] = (),
    stream: IO[str] | None = None,
) -> None:
    """Configure structlog and the stdlib root logger.

    Args:
        level: minimum log level, e.g. ``logging.INFO`` or ``"DEBUG"``.
        log_format: ``"json"`` for files / production, ``"console"`` for development.
        secrets: actual secret values (API key, API secret, bot token...) to mask
            wherever they appear. Empty values are ignored.
        stream: output stream, ``sys.stderr`` by default.
    """
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    renderer: Processor
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer(default=str)
    elif log_format == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    else:
        raise ValueError(f"Unknown log format: {log_format!r}")

    shared_processors = _build_shared_processors(SecretRedactor(secrets))

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_MARKER, True)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(numeric_level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
