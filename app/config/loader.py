"""Safe loading of YAML profiles and their cross-validation with EnvSettings.

YAML is parsed with a SafeLoader subclass (no Python object tags) that is
stricter than YAML 1.1 defaults:

* floats become ``Decimal`` built from the scalar text, never via ``float``;
* integers must be plain decimal (``010`` would otherwise be octal 8, ``0x1F`` 31);
* sexagesimal numbers (``1:30``) and duplicate mapping keys are errors.

Every failure is a ``ConfigError`` without file content or input values: YAML
parser messages can quote the document, so only line/column are reported.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, ScalarNode

from app.config.schema import AppConfig
from app.config.settings import ConfigError, EnvSettings, format_validation_errors
from app.domain.enums import TradingMode

# Which trading modes each profile may run. There is no production profile yet,
# so live trading cannot be configured.
PROFILE_MODES: Final[Mapping[str, frozenset[TradingMode]]] = MappingProxyType(
    {
        "development": frozenset({TradingMode.BACKTEST, TradingMode.PAPER}),
        "paper": frozenset({TradingMode.PAPER}),
        "testnet": frozenset({TradingMode.TESTNET}),
    }
)

_PLAIN_INT = re.compile(r"[-+]?(?:0|[1-9][0-9_]*)\Z")


class _ProfileLoader(yaml.SafeLoader):
    """SafeLoader with exact numbers and duplicate-key detection."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        seen: set[str] = set()
        for key_node, _ in node.value:
            if isinstance(key_node, ScalarNode):
                if key_node.value in seen:
                    raise ConstructorError(None, None, "duplicate mapping key", key_node.start_mark)
                seen.add(key_node.value)
        return super().construct_mapping(node, deep=deep)


def _construct_decimal(loader: yaml.SafeLoader, node: yaml.Node) -> Decimal:
    text = str(loader.construct_scalar(node))  # type: ignore[arg-type]
    special = text.lower().lstrip("+-")
    if special == ".inf":
        # Kept as a value so the schema rejects it with the field name.
        return Decimal("-Infinity" if text.startswith("-") else "Infinity")
    if special == ".nan":
        return Decimal("NaN")
    if ":" in text:
        raise ConstructorError(None, None, "sexagesimal numbers are not supported", node.start_mark)
    try:
        return Decimal(text)
    except InvalidOperation:
        raise ConstructorError(None, None, "invalid number", node.start_mark) from None


def _construct_int(loader: yaml.SafeLoader, node: yaml.Node) -> int:
    text = str(loader.construct_scalar(node))  # type: ignore[arg-type]
    if not _PLAIN_INT.match(text):
        raise ConstructorError(
            None, None, "only plain decimal integers are supported", node.start_mark
        )
    return int(text)


_ProfileLoader.add_constructor("tag:yaml.org,2002:float", _construct_decimal)
_ProfileLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)


def load_profile(path: Path, env: EnvSettings) -> AppConfig:
    """Load and validate the profile at ``path`` for the given environment.

    Order: CONFIG_PROFILE/TRADING_MODE matrix (no file access for a forbidden or
    unknown profile), read, parse, schema validation, profile name check.

    Raises:
        ConfigError: with a message that never contains file content or values.
    """
    allowed_modes = PROFILE_MODES.get(env.config_profile)
    if allowed_modes is None:
        known = ", ".join(PROFILE_MODES)
        raise ConfigError(
            f"CONFIG_PROFILE={env.config_profile} is not a known profile (known: {known})"
        )
    if env.trading_mode not in allowed_modes:
        allowed = ", ".join(sorted(mode.value for mode in allowed_modes))
        raise ConfigError(
            f"TRADING_MODE={env.trading_mode.value} is not allowed with "
            f"CONFIG_PROFILE={env.config_profile} (allowed: {allowed})"
        )

    data = _parse(_read(path), path)

    try:
        config = AppConfig.model_validate(data)
    except ValidationError as exc:
        details = format_validation_errors(exc)
        raise ConfigError(f"invalid profile config {path}:\n{details}") from None

    if config.profile.name != env.config_profile:
        raise ConfigError(
            f"profile.name in {path} does not match CONFIG_PROFILE={env.config_profile}"
        )
    return config


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"config file cannot be read: {path} ({type(exc).__name__})") from None


def _parse(text: str, path: Path) -> dict[str, object]:
    try:
        data = yaml.load(text, Loader=_ProfileLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        where = f", line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigError(f"failed to parse config YAML: {path}{where}") from None
    except yaml.YAMLError:
        raise ConfigError(f"failed to parse config YAML: {path}") from None
    if data is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data
