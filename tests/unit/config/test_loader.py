"""Safe YAML loading of profiles and mode/profile cross-validation."""

from __future__ import annotations

import itertools
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.config.loader import PROFILE_MODES, load_profile
from app.config.schema import AppConfig
from app.config.settings import ConfigError, EnvSettings
from app.domain.enums import GridMode

D = Decimal
CONFIGS = Path(__file__).resolve().parents[3] / "configs"
SECRET = "sk-test-0123456789abcdef"

GRID = """\
profile:
  name: {profile}
exchange:
  name: bybit
  symbol: BTCUSDT
strategy:
  type: grid
  grid:
    lower_price: {lower}
    upper_price: 70000
    levels: {levels}
    spacing: arithmetic
    mode: neutral
    order_qty: {qty}
"""


def env(mode: str = "paper", profile: str = "paper") -> EnvSettings:
    values: dict[str, Any] = {"TRADING_MODE": mode, "CONFIG_PROFILE": profile}
    if mode in ("testnet", "live"):
        values |= {"BYBIT_API_KEY": "k-test", "BYBIT_API_SECRET": "s-test"}
    if mode == "live":
        values["LIVE_TRADING_ENABLED"] = "true"
    return EnvSettings(**values)


def write(
    tmp_path: Path,
    text: str | None = None,
    *,
    profile: str = "paper",
    lower: str = "60000",
    levels: str = "10",
    qty: str = "0.001",
) -> Path:
    path = tmp_path / f"{profile}.yaml"
    content = (
        text
        if text is not None
        else GRID.format(profile=profile, lower=lower, levels=levels, qty=qty)
    )
    path.write_text(content, encoding="utf-8")
    return path


def load_grid(tmp_path: Path, **fields: str) -> AppConfig:
    return load_profile(write(tmp_path, **fields), env())


# --- Repository profiles -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        ("development", "backtest"),
        ("development", "paper"),
        ("paper", "paper"),
        ("testnet", "testnet"),
    ],
)
def test_repository_profiles_load(profile: str, mode: str) -> None:
    config = load_profile(CONFIGS / f"{profile}.yaml", env(mode, profile))
    assert config.profile.name == profile
    assert isinstance(config.strategy.grid.order_qty, Decimal)


def test_no_production_profile_in_repository() -> None:
    assert not (CONFIGS / "production.yaml").exists()
    assert sorted(p.name for p in CONFIGS.glob("*.yaml")) == [
        "development.yaml",
        "paper.yaml",
        "testnet.yaml",
    ]


# --- Mode / profile matrix ---------------------------------------------------------------

ALLOWED = {
    ("development", "backtest"),
    ("development", "paper"),
    ("paper", "paper"),
    ("testnet", "testnet"),
}


def test_matrix_constant() -> None:
    pairs = {(p, m.value) for p, modes in PROFILE_MODES.items() for m in modes}
    assert pairs == ALLOWED


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        pair
        for pair in itertools.product(
            ["development", "paper", "testnet"], ["backtest", "paper", "testnet", "live"]
        )
        if pair not in ALLOWED
    ],
)
def test_forbidden_combinations(tmp_path: Path, profile: str, mode: str) -> None:
    path = write(tmp_path, profile=profile)
    with pytest.raises(ConfigError, match=f"TRADING_MODE={mode} is not allowed"):
        load_profile(path, env(mode, profile))


@pytest.mark.parametrize("mode", ["backtest", "paper", "testnet", "live"])
def test_unknown_profile_rejected_before_reading(tmp_path: Path, mode: str) -> None:
    # No production profile exists yet: live cannot be configured at all.
    with pytest.raises(ConfigError, match="CONFIG_PROFILE=production is not a known profile"):
        load_profile(tmp_path / "production.yaml", env(mode, "production"))


def test_profile_name_must_match_config_profile(tmp_path: Path) -> None:
    path = write(tmp_path, profile="development")
    with pytest.raises(ConfigError, match=r"profile\.name .* does not match CONFIG_PROFILE"):
        load_profile(path, env("paper", "paper"))


# --- YAML -> Decimal ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.1", "0.1"),
        ("0.0001", "0.0001"),
        ("100.25", "100.25"),
        ("0.0010", "0.0010"),  # trailing zeros preserved
        ("0.00000001", "0.00000001"),
        ("123456789012345678.123456789", "123456789012345678.123456789"),
        ("1.5e-5", "0.000015"),
        ("1.5E-5", "0.000015"),
        ("2.0e+3", "2000"),
        ("1_000.5", "1000.5"),
        ("5", "5"),
    ],
)
def test_decimal_from_scalar_text(tmp_path: Path, text: str, expected: str) -> None:
    qty = load_grid(tmp_path, qty=text).strategy.grid.order_qty
    assert isinstance(qty, Decimal)
    assert qty == D(expected)
    if "e" not in text.lower() and "_" not in text:
        # Same digits and exponent as the source text (trailing zeros preserved).
        assert qty.as_tuple() == D(text).as_tuple()


def test_no_binary_float_artifact(tmp_path: Path) -> None:
    qty = load_grid(tmp_path, qty="0.1").strategy.grid.order_qty
    assert qty == D("0.1")
    assert qty != D(0.1)  # what a float detour would produce
    assert qty * 3 == D("0.3")


@pytest.mark.parametrize("text", ["1e-5", "1.5e5", "'0.1'", '"0.1"'])
def test_numbers_yaml_keeps_as_strings_are_rejected(tmp_path: Path, text: str) -> None:
    # YAML 1.1 needs a dot and a signed exponent for floats; anything else is a
    # string, and strings are never silently converted to Decimal.
    with pytest.raises(ConfigError, match=r"strategy\.grid\.order_qty"):
        load_grid(tmp_path, qty=text)


@pytest.mark.parametrize("text", [".nan", ".NaN", ".inf", "-.inf", "+.Inf"])
def test_nan_and_infinity_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match=r"strategy\.grid\.order_qty"):
        load_grid(tmp_path, qty=text)


@pytest.mark.parametrize("text", ["1:30.5", "1:30"])
def test_sexagesimal_numbers_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="failed to parse config YAML"):
        load_grid(tmp_path, lower=text)


@pytest.mark.parametrize("text", ["010", "0x1F", "0b101", "+012"])
def test_non_decimal_integers_rejected(tmp_path: Path, text: str) -> None:
    # PyYAML would read 010 as 8 (octal) and 0x1F as 31.
    with pytest.raises(ConfigError, match="failed to parse config YAML"):
        load_grid(tmp_path, levels=text)


def test_yaml_1_1_non_integer_forms_rejected_by_schema(tmp_path: Path) -> None:
    # "0o17" is a string in YAML 1.1 and strings are not accepted as integers.
    with pytest.raises(ConfigError, match=r"strategy\.grid\.levels"):
        load_grid(tmp_path, levels="0o17")


def test_plain_integers(tmp_path: Path) -> None:
    assert load_grid(tmp_path, levels="12").strategy.grid.levels == 12
    assert load_grid(tmp_path, levels="1_2").strategy.grid.levels == 12


def test_loader_does_not_change_global_safe_loader() -> None:
    assert isinstance(yaml.safe_load("v: 0.1")["v"], float)
    assert yaml.safe_load("v: 010")["v"] == 8


# --- YAML structure and safety -----------------------------------------------------------


def test_duplicate_keys_rejected(tmp_path: Path) -> None:
    text = GRID.format(profile="paper", lower="60000", levels="10", qty="0.001")
    text = text.replace("    levels: 10\n", "    levels: 10\n    levels: 20\n")
    with pytest.raises(ConfigError, match="failed to parse config YAML"):
        load_profile(write(tmp_path, text), env())


@pytest.mark.parametrize(
    "tag",
    [
        "!!python/object/apply:os.system ['echo pwned']",
        "!!python/object:builtins.dict {}",
        "!!python/name:os.system",
    ],
)
def test_unsafe_tags_rejected(tmp_path: Path, tag: str) -> None:
    text = GRID.format(profile="paper", lower="60000", levels="10", qty=tag)
    with pytest.raises(ConfigError, match="failed to parse config YAML") as info:
        load_profile(write(tmp_path, text), env())
    assert "os.system" not in str(info.value)


@pytest.mark.parametrize(
    ("text", "message"), [("", "is empty"), ("# only a comment\n", "is empty")]
)
def test_empty_yaml(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_profile(write(tmp_path, text), env())


@pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n", "42\n"])
def test_non_mapping_root(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_profile(write(tmp_path, text), env())


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_profile(tmp_path / "paper.yaml", env())


def test_directory_cannot_be_read(tmp_path: Path) -> None:
    (tmp_path / "paper.yaml").mkdir()
    with pytest.raises(ConfigError, match="config file cannot be read"):
        load_profile(tmp_path / "paper.yaml", env())


def test_permission_error_cannot_be_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path)

    def deny(*args: object, **kwargs: object) -> str:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(Path, "read_text", deny)
    with pytest.raises(ConfigError, match=r"config file cannot be read: .* \(PermissionError\)"):
        load_profile(path, env())


def test_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "paper.yaml"
    path.write_bytes(b"profile:\n  name: \xff\xfe\n")
    with pytest.raises(ConfigError, match="cannot be read"):
        load_profile(path, env())


# --- Error safety ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"profile:\n  name: paper\n  token: [{SECRET}\n",  # malformed YAML
        f"profile: {{name: paper}}\nexchange: {{name: bybit, symbol: '{SECRET} '}}\n",  # schema
        f"profile: {{name: paper, api_key: {SECRET}}}\n",  # extra field with a secret value
        f"profile: {{name: paper}}\nkey: !!python/object/apply:x ['{SECRET}']\n",  # unsafe tag
        f"profile: {{name: '{SECRET}'}}\n",  # mismatching profile name
    ],
)
def test_errors_never_contain_yaml_values(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError) as info:
        load_profile(write(tmp_path, text), env())
    assert SECRET not in str(info.value)
    assert SECRET not in repr(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_parse_error_reports_location_only(tmp_path: Path) -> None:
    path = write(tmp_path, "profile:\n  name: [paper\n")
    with pytest.raises(ConfigError, match=r"line \d+, column \d+") as info:
        load_profile(path, env())
    assert "paper\n" not in str(info.value)


def test_schema_error_names_fields(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"strategy\.grid\.levels") as info:
        load_grid(tmp_path, levels="1")
    assert "invalid profile config" in str(info.value)


def test_grid_mode_is_domain_enum(tmp_path: Path) -> None:
    assert load_grid(tmp_path).strategy.grid.mode is GridMode.NEUTRAL


def test_yaml_float_constructor_is_never_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_float(*args: object) -> float:
        raise AssertionError("YAML float constructor must not be used")

    monkeypatch.setattr(yaml.constructor.SafeConstructor, "construct_yaml_float", no_float)
    for profile, mode in [("development", "backtest"), ("paper", "paper"), ("testnet", "testnet")]:
        config = load_profile(CONFIGS / f"{profile}.yaml", env(mode, profile))
        assert config.strategy.grid.order_qty == D("0.001")
