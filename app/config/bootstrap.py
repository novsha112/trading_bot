"""Single entry point for loading the complete configuration.

    environment (+ optional .env) -> EnvSettings
    CONFIG_PROFILE -> profile path inside config_dir
    YAML profile -> AppConfig (cross-validated with EnvSettings)
    -> LoadedConfig

Loading has no runtime side effects: it does not configure logging, create
clients, open databases, change the environment or the working directory.
The caller (runtime bootstrap) decides what to do with the result, e.g.
``configure_logging(secrets=loaded.env.secret_values())``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config.loader import PROFILE_MODES, load_profile
from app.config.schema import AppConfig
from app.config.settings import ConfigError, EnvSettings, load_env_settings


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedConfig:
    """Validated configuration. Secrets stay inside ``env`` as ``SecretStr``."""

    env: EnvSettings
    app: AppConfig


def profile_path(config_dir: Path, profile: str) -> Path:
    """Path of the YAML file for ``profile``, guaranteed to be inside ``config_dir``.

    1. Only names known in PROFILE_MODES are accepted; nothing is built from an
       arbitrary string (no traversal, absolute paths, separators or suffixes).
    2. The resolved file (symlinks followed) must stay inside the resolved
       config directory, checked with path semantics, not string prefixes.

    No file is opened here.
    """
    if profile not in PROFILE_MODES:
        raise ConfigError(f"unknown configuration profile (known: {', '.join(PROFILE_MODES)})")
    candidate = config_dir / f"{profile}.yaml"
    resolved_dir = config_dir.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_dir):
        raise ConfigError(f"profile file resolves outside the config directory: {candidate}")
    return resolved


def load_config(
    *, config_dir: Path = Path("configs"), env_file: Path | None = None
) -> LoadedConfig:
    """Load and validate the full configuration.

    Args:
        config_dir: directory with the YAML profiles (relative to the working directory
            if not absolute).
        env_file: optional ``.env`` file; OS environment variables take precedence.

    Raises:
        ConfigError: on any problem; the message never contains secrets, environment
            values or file content.
    """
    env = load_env_settings(env_file=env_file)
    path = profile_path(config_dir, env.config_profile)
    app = load_profile(path, env)
    return LoadedConfig(env=env, app=app)
