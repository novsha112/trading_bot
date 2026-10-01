"""Environment-level settings: run mode, profile name, logging and credentials.

Secrets are ``SecretStr`` and never appear in ``repr``/``str``/dumps or in
validation errors. ``EnvSettings.secret_values()`` is the only place in config
that deliberately exposes them, for registering with log redaction.

Environment variable names are case-sensitive. Unknown OS variables are ignored;
unknown keys in an explicitly given ``.env`` file and unknown constructor fields
are rejected. A ``.env`` file is read only when passed to ``load_env_settings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.enums import TradingMode

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "console"]

_MODES_REQUIRING_CREDENTIALS = frozenset({TradingMode.TESTNET, TradingMode.LIVE})
_ENV_NAMES = {"bybit_api_key": "BYBIT_API_KEY", "bybit_api_secret": "BYBIT_API_SECRET"}


class ConfigError(Exception):
    """Invalid configuration. The message never contains raw input values."""


def _require_clean_text(value: str, name: str) -> str:
    # The value itself is never part of the message: it may be a secret.
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and without surrounding whitespace")
    return value


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
        env_file=None,  # .env only when explicitly requested (load_env_settings)
    )

    trading_mode: TradingMode = Field(validation_alias="TRADING_MODE")
    config_profile: str = Field(validation_alias="CONFIG_PROFILE")
    """Name of the YAML profile; resolved and loaded by the config loader."""
    log_level: LogLevel = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: LogFormat = Field(default="json", validation_alias="LOG_FORMAT")
    live_trading_enabled: bool = Field(default=False, validation_alias="LIVE_TRADING_ENABLED")
    bybit_api_key: SecretStr | None = Field(default=None, validation_alias="BYBIT_API_KEY")
    bybit_api_secret: SecretStr | None = Field(default=None, validation_alias="BYBIT_API_SECRET")

    @field_validator("config_profile")
    @classmethod
    def _check_profile(cls, value: str) -> str:
        return _require_clean_text(value, "CONFIG_PROFILE")

    @field_validator("live_trading_enabled", mode="before")
    @classmethod
    def _parse_live_flag(cls, value: object) -> bool:
        # Fail closed: only the exact strings "true" / "false" (or empty = false).
        # Pydantic's default bool parsing would also accept "1", "yes", "on", "TRUE".
        if isinstance(value, bool):
            return value
        if value == "true":
            return True
        if value in ("false", ""):
            return False
        raise ValueError("LIVE_TRADING_ENABLED must be exactly 'true' or 'false'")

    @field_validator("bybit_api_key", "bybit_api_secret")
    @classmethod
    def _check_credential(cls, value: SecretStr | None, info: ValidationInfo) -> SecretStr | None:
        if value is not None:
            _require_clean_text(value.get_secret_value(), _ENV_NAMES[str(info.field_name)])
        return value

    @model_validator(mode="after")
    def _check_mode_safety(self) -> Self:
        if (self.bybit_api_key is None) != (self.bybit_api_secret is None):
            raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set together")
        if self.live_trading_enabled and self.trading_mode is not TradingMode.LIVE:
            raise ValueError("LIVE_TRADING_ENABLED=true is only allowed with TRADING_MODE=live")
        if self.trading_mode is TradingMode.LIVE and not self.live_trading_enabled:
            raise ValueError("TRADING_MODE=live requires LIVE_TRADING_ENABLED=true")
        if self.trading_mode in _MODES_REQUIRING_CREDENTIALS and self.bybit_api_key is None:
            raise ValueError(
                f"TRADING_MODE={self.trading_mode.value} "
                "requires BYBIT_API_KEY and BYBIT_API_SECRET"
            )
        return self

    def secret_values(self) -> tuple[str, ...]:
        """Configured secret values, for ``configure_logging(secrets=...)`` only.

        Deterministic order (API key, API secret), unset values skipped, duplicates
        removed. The values must never be logged or stored.
        """
        values: list[str] = []
        for secret in (self.bybit_api_key, self.bybit_api_secret):
            if secret is not None and secret.get_secret_value() not in values:
                values.append(secret.get_secret_value())
        return tuple(values)


def load_env_settings(*, env_file: Path | None = None) -> EnvSettings:
    """Load settings from the OS environment and, if given, a ``.env`` file.

    OS environment variables take precedence over the file. A missing file is
    ignored. Raises ``ConfigError`` whose message lists field names and reasons
    but never input values.
    """
    try:
        return EnvSettings(_env_file=env_file)
    except ValidationError as exc:
        details = "\n".join(
            f"- {'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False, include_context=False)
        )
        raise ConfigError(f"Invalid environment configuration:\n{details}") from None
