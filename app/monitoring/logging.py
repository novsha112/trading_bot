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
   any string (event text, other fields, mapping keys, rendered tracebacks).

%-style arguments (stdlib ``logger.debug("headers %s", headers)``) are redacted
as structured data first and only then interpolated into the message.

Objects other than primitives are converted to ``str`` during redaction, so the
renderer never calls ``repr()`` / ``str()`` on raw objects. An object that cannot
be converted is logged as ``[UNPRINTABLE]``: a logging call must never raise.

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
UNPRINTABLE: Final = "[UNPRINTABLE]"

# Field names are split into lowercase tokens ("X-BAPI-API-KEY" -> x, bapi, api, key;
# "apiKey" -> api, key). A field is sensitive if any token is in this set. Token
# matching (not substring matching) keeps fields like "signal" readable.
SENSITIVE_NAME_TOKENS: Final = frozenset(
    {
        "apikey",
        "apikeys",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "key",
        "keys",
        "passphrase",
        "passwd",
        "password",
        "passwords",
        "pwd",
        "secret",
        "secrets",
        "sign",
        "signature",
        "token",
        "tokens",
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
        meta = {key: event_dict[key] for key in _META_KEYS if key in event_dict}
        fields = {key: value for key, value in event_dict.items() if key not in _META_KEYS}
        return {**self._redact_mapping(fields), **meta}

    def format_positional_args(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """Redact %-style arguments, then interpolate them into the event text.

        Runs before the main redaction step, so the resulting text is scrubbed
        for registered secrets once more.
        """
        args = event_dict.pop("positional_args", None)
        if not args:
            return event_dict
        try:
            safe_args: Any = (
                self._redact_mapping(args)
                if isinstance(args, Mapping)
                else tuple(self._redact_value(arg) for arg in args)
            )
        except Exception:
            safe_args = (UNPRINTABLE,)
        event = event_dict.get("event")
        try:
            event_dict["event"] = str(event) % safe_args
        except Exception:
            # Mismatched format string: keep both parts instead of failing.
            event_dict["event"] = f"{event} {safe_args!r}"
        return event_dict

    def _redact_mapping(self, mapping: Mapping[Any, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for raw_key, value in mapping.items():
            name = raw_key if isinstance(raw_key, str) else self._to_text(raw_key)
            try:
                redacted = self._redact_field(name, value)
            except Exception:
                redacted = UNPRINTABLE
            key = self._scrub(name)
            if key in result:
                # Two keys collapsed into the same text (e.g. two masked secrets).
                suffix = 2
                while f"{key}#{suffix}" in result:
                    suffix += 1
                key = f"{key}#{suffix}"
            result[key] = redacted
        return result

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
            return self._redact_mapping(value)
        if isinstance(value, list | tuple | set | frozenset):
            return [self._redact_value(item) for item in value]
        # Any other object (exceptions, bytes, Decimal, models): the renderer
        # gets its scrubbed str() and never touches the object itself.
        return self._scrub(self._to_text(value))

    @staticmethod
    def _to_text(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return UNPRINTABLE

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
        redactor.format_positional_args,
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
        # Keep stdlib %-args separate from the message so they are redacted
        # before interpolation (see SecretRedactor.format_positional_args).
        use_get_message=False,
        pass_foreign_args=True,
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
        # No caching: a cached logger would keep the processor chain (and the
        # registered secrets) from the configuration active at its first use.
        cache_logger_on_first_use=False,
    )
