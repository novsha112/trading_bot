"""Full configuration pipeline: environment -> EnvSettings -> profile path -> AppConfig."""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import sys
from pathlib import Path

import pytest
import structlog

from app.config import bootstrap
from app.config.bootstrap import LoadedConfig, load_config, profile_path
from app.config.settings import ConfigError
from app.domain.enums import TradingMode

REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs"
API_KEY = "AK-bootstrap-7c1e9a2f4b"
API_SECRET = "SK-bootstrap-5d8b3e6a1c9f"
YAML_SECRET = "yaml-secret-31f9c2a7e4"
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
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray ./.env, relative paths are predictable


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    """A private copy of the repository profiles."""
    target = tmp_path / "configs"
    shutil.copytree(REPO_CONFIGS, target)
    return target


def set_env(monkeypatch: pytest.MonkeyPatch, mode: str, profile: str, **extra: str) -> None:
    monkeypatch.setenv("TRADING_MODE", mode)
    monkeypatch.setenv("CONFIG_PROFILE", profile)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def with_credentials(monkeypatch: pytest.MonkeyPatch, mode: str, profile: str) -> None:
    set_env(monkeypatch, mode, profile, BYBIT_API_KEY=API_KEY, BYBIT_API_SECRET=API_SECRET)


def assert_safe(error: ConfigError) -> None:
    for text in (str(error), repr(error)):
        for secret in (API_KEY, API_SECRET, YAML_SECRET):
            assert secret not in text
    assert error.__cause__ is None


def forbid_file_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("profile file must not be read")

    monkeypatch.setattr(Path, "read_text", fail)


# --- Success -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "mode"), [("development", "backtest"), ("development", "paper"), ("paper", "paper")]
)
def test_loads_profiles(monkeypatch: pytest.MonkeyPatch, profile: str, mode: str) -> None:
    set_env(monkeypatch, mode, profile)
    loaded = load_config(config_dir=REPO_CONFIGS)
    assert loaded.env.trading_mode is TradingMode(mode)
    assert loaded.app.profile.name == profile


def test_loads_testnet_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    with_credentials(monkeypatch, "testnet", "testnet")
    loaded = load_config(config_dir=REPO_CONFIGS)
    assert loaded.env.trading_mode is TradingMode.TESTNET
    assert loaded.env.secret_values() == (API_KEY, API_SECRET)
    assert API_KEY not in repr(loaded)


def test_loaded_config_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, "paper", "paper")
    loaded = load_config(config_dir=REPO_CONFIGS)
    assert [f.name for f in dataclasses.fields(LoadedConfig)] == ["env", "app"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.env = loaded.env  # type: ignore[misc]


def test_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "bot.env"
    env_file.write_text("TRADING_MODE=paper\nCONFIG_PROFILE=paper\n", encoding="utf-8")
    assert load_config(config_dir=REPO_CONFIGS, env_file=env_file).app.profile.name == "paper"


def test_os_environment_overrides_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / "bot.env"
    env_file.write_text("TRADING_MODE=paper\nCONFIG_PROFILE=development\n", encoding="utf-8")
    monkeypatch.setenv("TRADING_MODE", "backtest")
    loaded = load_config(config_dir=REPO_CONFIGS, env_file=env_file)
    assert loaded.env.trading_mode is TradingMode.BACKTEST
    assert loaded.app.profile.name == "development"


def test_relative_config_dir(monkeypatch: pytest.MonkeyPatch, configs: Path) -> None:
    set_env(monkeypatch, "paper", "paper")
    assert load_config(config_dir=Path("configs")).app.profile.name == "paper"


def test_absolute_config_dir_with_dot_dot(monkeypatch: pytest.MonkeyPatch, configs: Path) -> None:
    set_env(monkeypatch, "paper", "paper")
    winding = configs / ".." / "configs"
    assert winding.is_absolute()
    assert load_config(config_dir=winding).app.profile.name == "paper"


def test_symlink_inside_config_dir_allowed(monkeypatch: pytest.MonkeyPatch, configs: Path) -> None:
    target = configs / "variants" / "paper-v2.yaml"
    target.parent.mkdir()
    (configs / "paper.yaml").rename(target)
    try:
        (configs / "paper.yaml").symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not supported here")
    set_env(monkeypatch, "paper", "paper")
    assert load_config(config_dir=configs).app.profile.name == "paper"


# --- Profile boundary (no file access) ---------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [
        "production",
        "staging",
        "../paper",
        "development/../../x",
        "/etc/passwd",
        "C:\\configs\\paper",
        "paper.yaml",
        "paper.yaml/../paper",
        "file:///etc/passwd",
        "%2e%2e%2fpaper",
        "Paper",
    ],
)
def test_unknown_profile_rejected_before_any_file_access(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    set_env(monkeypatch, "paper", profile)
    forbid_file_reads(monkeypatch)
    with pytest.raises(ConfigError, match="unknown configuration profile") as info:
        load_config(config_dir=REPO_CONFIGS)
    # The raw (possibly arbitrary) CONFIG_PROFILE value is not echoed.
    assert profile not in str(info.value)


def test_containment_rejects_escaping_name(monkeypatch: pytest.MonkeyPatch, configs: Path) -> None:
    # Second line of defence, independent of the profile name check.
    monkeypatch.setattr(bootstrap, "PROFILE_MODES", {"../escape": frozenset()})
    (configs.parent / "escape.yaml").write_text("x: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="outside the config directory"):
        profile_path(configs, "../escape")


def test_symlink_escape_rejected(
    monkeypatch: pytest.MonkeyPatch, configs: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.yaml"
    outside.write_text(f"profile: {{name: paper}}\nleak: {YAML_SECRET}\n", encoding="utf-8")
    (configs / "paper.yaml").unlink()
    try:
        (configs / "paper.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not supported here")
    set_env(monkeypatch, "paper", "paper")
    forbid_file_reads(monkeypatch)
    with pytest.raises(ConfigError, match="outside the config directory") as info:
        load_config(config_dir=configs)
    assert_safe(info.value)


# --- Other failures at the right boundary ------------------------------------------------


def test_missing_profile_file(monkeypatch: pytest.MonkeyPatch, configs: Path) -> None:
    (configs / "paper.yaml").unlink()
    set_env(monkeypatch, "paper", "paper")
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(config_dir=configs)


def test_invalid_env_fails_before_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, "LIVE", "paper")
    forbid_file_reads(monkeypatch)
    with pytest.raises(ConfigError, match="Invalid environment configuration"):
        load_config(config_dir=REPO_CONFIGS)


@pytest.mark.parametrize(("profile", "mode"), [("paper", "backtest"), ("testnet", "paper")])
def test_invalid_mode_profile_pair(
    monkeypatch: pytest.MonkeyPatch, profile: str, mode: str
) -> None:
    set_env(monkeypatch, mode, profile)
    with pytest.raises(ConfigError, match=f"TRADING_MODE={mode} is not allowed"):
        load_config(config_dir=REPO_CONFIGS)


@pytest.mark.parametrize("profile", ["development", "paper", "testnet", "production"])
def test_live_mode_cannot_be_configured(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    with_credentials(monkeypatch, "live", profile)
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(ConfigError) as info:
        load_config(config_dir=REPO_CONFIGS)
    assert_safe(info.value)


def test_testnet_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, "testnet", "testnet")
    with pytest.raises(ConfigError, match="requires BYBIT_API_KEY and BYBIT_API_SECRET"):
        load_config(config_dir=REPO_CONFIGS)


# --- Secret leakage ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (f"profile:\n  name: testnet\n  note: [{YAML_SECRET}\n", "failed to parse config YAML"),
        (f"profile: {{name: '{YAML_SECRET}'}}\n", "invalid profile config|does not match"),
        (f"profile: {{name: testnet, api_key: {YAML_SECRET}}}\n", "invalid profile config"),
        (
            "profile: {name: testnet}\nexchange: {name: bybit, symbol: BTCUSDT}\n"
            f"strategy: {{type: grid, grid: {{lower_price: 2, upper_price: 1, levels: 2, "
            f"spacing: arithmetic, mode: long, order_qty: 1, token: {YAML_SECRET}}}}}\n",
            "invalid profile config",
        ),
    ],
)
def test_no_secrets_in_errors_after_env_parsing(
    monkeypatch: pytest.MonkeyPatch, configs: Path, content: str, message: str
) -> None:
    (configs / "testnet.yaml").write_text(content, encoding="utf-8")
    with_credentials(monkeypatch, "testnet", "testnet")
    with pytest.raises(ConfigError, match=message) as info:
        load_config(config_dir=configs)
    assert_safe(info.value)


def test_profile_name_mismatch_is_safe(monkeypatch: pytest.MonkeyPatch, configs: Path) -> None:
    text = (configs / "testnet.yaml").read_text(encoding="utf-8")
    (configs / "testnet.yaml").write_text(
        text.replace("name: testnet", "name: paper"), encoding="utf-8"
    )
    with_credentials(monkeypatch, "testnet", "testnet")
    with pytest.raises(ConfigError, match="does not match CONFIG_PROFILE") as info:
        load_config(config_dir=configs)
    assert_safe(info.value)


# --- No side effects ---------------------------------------------------------------------


def test_load_config_has_no_side_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with_credentials(monkeypatch, "testnet", "testnet")
    env_before = dict(os.environ)
    cwd_before = Path.cwd()
    handlers_before = list(logging.getLogger().handlers)
    level_before = logging.getLogger().level
    structlog_before = structlog.get_config()
    modules_before = set(sys.modules)

    load_config(config_dir=REPO_CONFIGS)

    assert dict(os.environ) == env_before
    assert Path.cwd() == cwd_before
    assert logging.getLogger().handlers == handlers_before
    assert logging.getLogger().level == level_before
    assert structlog.get_config() == structlog_before
    forbidden = ("pybit", "ccxt", "httpx", "aiohttp", "requests", "websockets", "sqlalchemy")
    new_modules = set(sys.modules) - modules_before
    assert not [m for m in new_modules if m.split(".")[0] in forbidden]
    assert list(tmp_path.iterdir()) == []  # nothing written to the working directory
