"""Environment settings: mode, profile name, logging, credentials, live safety flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.config.settings import ConfigError, EnvSettings, load_env_settings
from app.domain.enums import TradingMode

API_KEY = "AKtest3f9a1c7e5b2d"
API_SECRET = "SKtest8e4b6d2a0c9f1e7b"
ENV_NAMES = (
    "TRADING_MODE",
    "CONFIG_PROFILE",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LIVE_TRADING_ENABLED",
    "BYBIT_API_KEY",
    "BYBIT_API_SECRET",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's real environment must not influence the tests."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(Path(__file__).parent)  # a stray ./.env is never picked up


def set_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def base_env(monkeypatch: pytest.MonkeyPatch, mode: str = "paper", **extra: str) -> None:
    set_env(monkeypatch, TRADING_MODE=mode, CONFIG_PROFILE="paper", **extra)


def with_credentials(monkeypatch: pytest.MonkeyPatch, mode: str, **extra: str) -> None:
    base_env(monkeypatch, mode, BYBIT_API_KEY=API_KEY, BYBIT_API_SECRET=API_SECRET, **extra)


def config_error(message: str) -> pytest.RaisesExc[ConfigError]:
    return pytest.raises(ConfigError, match=message)


def assert_no_secrets(text: str) -> None:
    assert API_KEY not in text
    assert API_SECRET not in text


# --- Defaults and basic fields -----------------------------------------------------------


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    base_env(monkeypatch)
    settings = load_env_settings()
    assert settings.trading_mode is TradingMode.PAPER
    assert settings.config_profile == "paper"
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.live_trading_enabled is False
    assert settings.bybit_api_key is None
    assert settings.bybit_api_secret is None


@pytest.mark.parametrize("mode", ["backtest", "paper"])
def test_modes_without_credentials(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    base_env(monkeypatch, mode)
    assert load_env_settings().trading_mode is TradingMode(mode)


@pytest.mark.parametrize("mode", ["backtest", "paper", "testnet"])
def test_modes_with_credentials(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    with_credentials(monkeypatch, mode)
    assert load_env_settings().trading_mode is TradingMode(mode)


def test_live_with_flag_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    with_credentials(monkeypatch, "live", LIVE_TRADING_ENABLED="true")
    settings = load_env_settings()
    assert settings.trading_mode is TradingMode.LIVE
    assert settings.live_trading_enabled is True


def test_trading_mode_required(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, CONFIG_PROFILE="paper")
    with config_error("TRADING_MODE"):
        load_env_settings()


@pytest.mark.parametrize("mode", ["LIVE", "Paper", "production", "", " paper", "paper "])
def test_trading_mode_exact_lowercase_only(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    base_env(monkeypatch, mode)
    with config_error("TRADING_MODE"):
        load_env_settings()


@pytest.mark.parametrize("profile", ["", "  ", " paper", "paper ", "paper\n"])
def test_config_profile_must_be_clean_text(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    set_env(monkeypatch, TRADING_MODE="paper", CONFIG_PROFILE=profile)
    with config_error("CONFIG_PROFILE"):
        load_env_settings()


def test_config_profile_required_and_not_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, TRADING_MODE="paper")
    with config_error("CONFIG_PROFILE"):
        load_env_settings()
    # Only a name: no file is opened or required in this layer.
    set_env(monkeypatch, CONFIG_PROFILE="does-not-exist")
    assert load_env_settings().config_profile == "does-not-exist"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_level_allowed(monkeypatch: pytest.MonkeyPatch, level: str) -> None:
    base_env(monkeypatch, LOG_LEVEL=level)
    assert load_env_settings().log_level == level


@pytest.mark.parametrize("level", ["info", "WARN", "TRACE", ""])
def test_log_level_invalid(monkeypatch: pytest.MonkeyPatch, level: str) -> None:
    base_env(monkeypatch, LOG_LEVEL=level)
    with config_error("LOG_LEVEL"):
        load_env_settings()


@pytest.mark.parametrize("fmt", ["json", "console"])
def test_log_format_allowed(monkeypatch: pytest.MonkeyPatch, fmt: str) -> None:
    base_env(monkeypatch, LOG_FORMAT=fmt)
    assert load_env_settings().log_format == fmt


@pytest.mark.parametrize("fmt", ["JSON", "text", "xml", ""])
def test_log_format_invalid(monkeypatch: pytest.MonkeyPatch, fmt: str) -> None:
    base_env(monkeypatch, LOG_FORMAT=fmt)
    with config_error("LOG_FORMAT"):
        load_env_settings()


# --- Strict boolean ----------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("", False)])
def test_live_flag_false_values(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    # An empty value (as in .env.example) is explicitly False, never truthy.
    base_env(monkeypatch, LIVE_TRADING_ENABLED=raw)
    assert load_env_settings().live_trading_enabled is expected


@pytest.mark.parametrize(
    "raw", ["1", "0", "yes", "no", "on", "off", "TRUE", "True", "FALSE", " true", "true "]
)
def test_live_flag_rejects_lenient_values(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    base_env(monkeypatch, LIVE_TRADING_ENABLED=raw)
    with config_error("LIVE_TRADING_ENABLED"):
        load_env_settings()


# --- Mode / credential safety matrix -----------------------------------------------------


@pytest.mark.parametrize("mode", ["backtest", "paper", "testnet"])
def test_live_flag_true_outside_live_rejected(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    with_credentials(monkeypatch, mode, LIVE_TRADING_ENABLED="true")
    with config_error("LIVE_TRADING_ENABLED=true is only allowed with TRADING_MODE=live"):
        load_env_settings()


@pytest.mark.parametrize("flag", [None, "false", ""])
def test_live_requires_flag(monkeypatch: pytest.MonkeyPatch, flag: str | None) -> None:
    with_credentials(monkeypatch, "live")
    if flag is not None:
        set_env(monkeypatch, LIVE_TRADING_ENABLED=flag)
    with config_error("TRADING_MODE=live requires LIVE_TRADING_ENABLED=true"):
        load_env_settings()


@pytest.mark.parametrize(("mode", "flag"), [("live", "true"), ("testnet", "false")])
def test_live_and_testnet_require_credentials(
    monkeypatch: pytest.MonkeyPatch, mode: str, flag: str
) -> None:
    base_env(monkeypatch, mode, LIVE_TRADING_ENABLED=flag)
    with config_error(f"TRADING_MODE={mode} requires BYBIT_API_KEY and BYBIT_API_SECRET"):
        load_env_settings()


@pytest.mark.parametrize("mode", ["backtest", "paper", "testnet", "live"])
@pytest.mark.parametrize("present", ["BYBIT_API_KEY", "BYBIT_API_SECRET"])
def test_partial_credential_pair_rejected_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str, present: str
) -> None:
    base_env(monkeypatch, mode, LIVE_TRADING_ENABLED="true" if mode == "live" else "false")
    set_env(monkeypatch, **{present: API_KEY})
    with config_error("must be set together") as info:
        load_env_settings()
    assert_no_secrets(str(info.value))


@pytest.mark.parametrize("name", ["BYBIT_API_KEY", "BYBIT_API_SECRET"])
@pytest.mark.parametrize("value", ["", "   ", f" {API_KEY}", f"{API_KEY} ", f"{API_KEY}\n"])
def test_credential_must_be_clean_when_set(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    with_credentials(monkeypatch, "testnet")
    set_env(monkeypatch, **{name: value})
    with config_error(name) as info:
        load_env_settings()
    assert_no_secrets(str(info.value))


# --- Secret safety -----------------------------------------------------------------------


def test_secrets_hidden_in_repr_str_and_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    with_credentials(monkeypatch, "testnet")
    settings = load_env_settings()

    for text in (repr(settings), str(settings), settings.model_dump_json()):
        assert_no_secrets(text)
    dumped = settings.model_dump()
    assert isinstance(dumped["bybit_api_key"], SecretStr)
    assert_no_secrets(str(dumped))
    as_json = settings.model_dump(mode="json")
    assert as_json["bybit_api_key"] == "**********"
    assert_no_secrets(json.dumps(as_json))


@pytest.mark.parametrize(
    "env",
    [
        {"BYBIT_API_KEY": f" {API_KEY}", "BYBIT_API_SECRET": API_SECRET},  # bad key
        {"BYBIT_API_KEY": API_KEY, "BYBIT_API_SECRET": f"{API_SECRET} "},  # bad secret
        {"BYBIT_API_KEY": API_KEY},  # only one of the pair
        {"BYBIT_API_KEY": API_KEY, "BYBIT_API_SECRET": API_SECRET, "LIVE_TRADING_ENABLED": "yes"},
    ],
)
def test_validation_errors_never_contain_secrets(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    base_env(monkeypatch, "testnet", **env)

    with pytest.raises(ConfigError) as info:
        load_env_settings()
    assert_no_secrets(str(info.value))
    assert_no_secrets(repr(info.value))
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__

    # The raw Pydantic error must be safe to print as well.
    with pytest.raises(ValidationError) as raw:
        EnvSettings()
    assert_no_secrets(str(raw.value))


def test_live_without_credentials_error_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    base_env(monkeypatch, "live", LIVE_TRADING_ENABLED="true", BYBIT_API_KEY=API_KEY)
    with pytest.raises(ConfigError) as info:
        load_env_settings()
    assert_no_secrets(str(info.value))


def test_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    with_credentials(monkeypatch, "testnet")
    assert load_env_settings().secret_values() == (API_KEY, API_SECRET)


def test_secret_values_empty_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    base_env(monkeypatch)
    assert load_env_settings().secret_values() == ()


def test_secret_values_deduplicated_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    base_env(monkeypatch, "testnet", BYBIT_API_KEY=API_KEY, BYBIT_API_SECRET=API_KEY)
    assert load_env_settings().secret_values() == (API_KEY,)


# --- .env, constructor, case sensitivity -------------------------------------------------


def test_dotenv_not_read_unless_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TRADING_MODE=paper\nCONFIG_PROFILE=paper\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with config_error("TRADING_MODE"):
        load_env_settings()
    assert load_env_settings(env_file=tmp_path / ".env").trading_mode is TradingMode.PAPER


def test_os_environment_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TRADING_MODE=paper\nCONFIG_PROFILE=paper\n", encoding="utf-8")
    set_env(monkeypatch, TRADING_MODE="backtest")
    assert load_env_settings(env_file=env_file).trading_mode is TradingMode.BACKTEST


def test_unknown_dotenv_key_rejected_without_leaking(tmp_path: Path) -> None:
    # A typo in a secret name must fail loudly, without echoing the value.
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"TRADING_MODE=paper\nCONFIG_PROFILE=paper\nBYBIT_API_SECRETT={API_SECRET}\n",
        encoding="utf-8",
    )
    with config_error("BYBIT_API_SECRETT") as info:
        load_env_settings(env_file=env_file)
    assert_no_secrets(str(info.value))


def test_missing_dotenv_file_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_env(monkeypatch)
    assert load_env_settings(env_file=tmp_path / "absent.env").trading_mode is TradingMode.PAPER


def test_unknown_os_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    base_env(monkeypatch, SOME_OTHER_TOOL_SETTING="x", BYBIT_API_SECRETT="ignored")
    assert load_env_settings().trading_mode is TradingMode.PAPER


def test_env_names_are_case_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, trading_mode="paper", CONFIG_PROFILE="paper")
    with config_error("TRADING_MODE"):
        load_env_settings()


def test_constructor_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError) as info:
        EnvSettings(TRADING_MODE="paper", CONFIG_PROFILE="paper", UNEXPECTED="x")  # type: ignore[call-arg]
    assert [e["type"] for e in info.value.errors()] == ["extra_forbidden"]


def test_settings_are_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    base_env(monkeypatch)
    settings = load_env_settings()
    with pytest.raises(ValidationError):
        settings.trading_mode = TradingMode.LIVE  # type: ignore[misc]


def test_env_example_lists_only_known_variables() -> None:
    example = Path(__file__).resolve().parents[3] / ".env.example"
    keys = {
        line.split("=", 1)[0]
        for line in example.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    aliases = {field.validation_alias for field in EnvSettings.model_fields.values()}
    assert keys <= aliases
    assert {"TRADING_MODE", "CONFIG_PROFILE"} <= keys
