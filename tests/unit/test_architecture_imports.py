"""Module dependency rules from docs/ARCHITECTURE.md, section 1.

Each top-level package under ``app/`` may import only the ``app`` packages listed
for it. Packages marked ``None`` have no import restriction defined yet. A new
top-level package must be added here explicitly, otherwise the test fails.

Packages listed in ``ALLOWED_THIRD_PARTY`` may import, besides the Python standard
library (``sys.stdlib_module_names``) and allowed ``app`` modules, only the listed
third-party top-level packages. Unlisted packages have no third-party restriction yet.

Imports are read with ``ast`` (modules are never executed). Relative imports are
resolved against the importing module; one that would climb above the top-level
package cannot be resolved and is reported (fail closed).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2] / "app"

ALLOWED_APP_IMPORTS: dict[str, frozenset[str] | None] = {
    "domain": frozenset(),
    "config": frozenset({"domain"}),
    "exchanges": frozenset({"domain"}),
    "market_data": frozenset({"domain", "exchanges"}),
    "strategies": frozenset({"domain"}),
    "risk": frozenset({"domain", "portfolio"}),
    "execution": frozenset({"domain", "exchanges"}),
    "portfolio": frozenset({"domain"}),
    "persistence": frozenset({"domain"}),
    # Orchestration / infrastructure packages: restrictions will be refined as
    # their modules appear.
    "backtesting": None,
    "paper_trading": None,
    "monitoring": None,
    "notifications": None,
    "services": None,
}

ALLOWED_THIRD_PARTY: dict[str, frozenset[str]] = {
    # The shared language of the system: no frameworks, SDKs or I/O libraries.
    "domain": frozenset(),
    # Configuration loading only: no logging, network, exchange or database code.
    "config": frozenset({"pydantic", "pydantic_settings", "yaml"}),
    # Pure, deterministic algorithms on domain types: same code in backtest and live.
    "strategies": frozenset(),
    # Exchange-neutral contracts (protocols, DTOs, errors): no third-party code.
    "exchanges": frozenset(),
    # Concrete adapter: the core contracts plus one HTTP client.
    "exchanges.bybit": frozenset({"httpx"}),
    # Pure request mapping and shared JSON types: no HTTP client, no third-party code.
    "exchanges.bybit.order_mapping": frozenset(),
    "exchanges.bybit.types": frozenset(),
}

# Modules that may import only these app modules (prefix match), stricter than their
# package rule: pure request mapping never reaches the transport, credentials or config.
MODULE_APP_ALLOWLIST: dict[str, frozenset[str]] = {
    "exchanges.bybit.order_mapping": frozenset(
        {
            "app.domain",
            "app.exchanges.models",
            "app.exchanges.errors",
            "app.exchanges.bybit.types",
        }
    ),
    "exchanges.bybit.types": frozenset(),
    # Exchange-neutral simulator: domain plus the core exchange contracts only.
    "exchanges.simulated": frozenset(
        {
            "app.domain",
            "app.exchanges.models",
            "app.exchanges.errors",
            "app.exchanges.simulated_positions",
            "app.exchanges.simulated_fees",
            "app.exchanges.simulated_accounting",
        }
    ),
    # Position accounting: domain only (no exchange contracts, no instrument rules).
    "exchanges.simulated_positions": frozenset({"app.domain"}),
    # Fee model: domain only.
    "exchanges.simulated_fees": frozenset({"app.domain"}),
    # Cash accounting: domain only (realized deltas are passed in, not computed).
    "exchanges.simulated_accounting": frozenset({"app.domain"}),
}

# Implementation subpackages that the rest of their own top-level package must not
# import (core contracts never depend on a concrete adapter), mapped to the only
# other implementations allowed to use them.
IMPLEMENTATION_SUBPACKAGES: dict[str, frozenset[str]] = {
    "exchanges.bybit": frozenset(),
    "exchanges.simulated": frozenset(),
    # Position accounting is a building block of the simulator only.
    "exchanges.simulated_positions": frozenset({"exchanges.simulated"}),
    # Fee model likewise: used by the simulator only.
    "exchanges.simulated_fees": frozenset({"exchanges.simulated"}),
    "exchanges.simulated_accounting": frozenset({"exchanges.simulated"}),
}


def _third_party_rule(module: str) -> frozenset[str] | None:
    """Most specific ALLOWED_THIRD_PARTY entry for a module (e.g. exchanges.bybit)."""
    parts = module.split(".")[1:]
    for length in range(len(parts), 0, -1):
        key = ".".join(parts[:length])
        if key in ALLOWED_THIRD_PARTY:
            return ALLOWED_THIRD_PARTY[key]
    return None


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(source: str, module: str, is_package: bool) -> tuple[set[str], list[str]]:
    """Absolute names of all modules imported by the source, relative imports resolved.

    Returns the resolved names and the relative imports that cannot be resolved.
    """
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    imported: set[str] = set()
    unresolved: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Level 1 is the current package, each further dot one parent up.
                # Python rejects climbing above the top-level package; slicing with a
                # negative bound would silently wrap around instead, so fail closed.
                if node.level > len(package_parts):
                    unresolved.append("." * node.level + (node.module or ""))
                    continue
                base_parts = package_parts[: len(package_parts) - node.level + 1]
                base = ".".join([*base_parts, node.module] if node.module else base_parts)
            else:
                base = node.module or ""
            if "." in base:
                imported.add(base)
            else:
                # "from app import strategies" imports a subpackage, not a name.
                imported.update(f"{base}.{alias.name}" for alias in node.names)
    return imported, unresolved


def find_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        module = _module_name(path, root)
        parts = module.split(".")
        if len(parts) < 2:
            continue  # app/__init__.py
        package = parts[1]
        if package not in ALLOWED_APP_IMPORTS:
            violations.append(f"{module}: package 'app.{package}' has no dependency rule")
            continue
        allowed = ALLOWED_APP_IMPORTS[package]
        if allowed is None:
            continue
        imports, unresolved = _imported_modules(
            path.read_text(encoding="utf-8"), module, is_package=path.name == "__init__.py"
        )
        violations.extend(
            f"{module}: unresolvable relative import '{target}' (beyond top-level package)"
            for target in unresolved
        )
        for name in sorted(imports):
            name_parts = name.split(".")
            if name_parts[0] != root.name:
                third_party = _third_party_rule(module)
                top = name_parts[0]
                if (
                    third_party is not None
                    and top not in sys.stdlib_module_names
                    and top not in third_party
                ):
                    violations.append(
                        f"{module}: imports '{name}' (third-party; not allowed in app.{package})"
                    )
                continue
            if len(name_parts) == 1:
                # "import app" gives access to every package.
                violations.append(f"{module}: imports '{name}' (top-level app package)")
                continue
            module_allowlist = MODULE_APP_ALLOWLIST.get(module.partition(".")[2])
            if module_allowlist is not None and not any(
                name == prefix or name.startswith(f"{prefix}.") for prefix in module_allowlist
            ):
                violations.append(f"{module}: imports '{name}' (not in the module allowlist)")
            target = name_parts[1]
            if target != package and target not in allowed:
                violations.append(f"{module}: imports '{name}' (app.{package} -> app.{target})")
            for implementation, importers in IMPLEMENTATION_SUBPACKAGES.items():
                if implementation.split(".")[0] != package:
                    continue  # other packages are governed by ALLOWED_APP_IMPORTS
                prefix = f"{root.name}.{implementation}"
                imports_impl = name == prefix or name.startswith(f"{prefix}.")
                inside_impl = any(
                    module == owner or module.startswith(f"{owner}.")
                    for owner in (prefix, *(f"{root.name}.{i}" for i in importers))
                )
                if imports_impl and not inside_impl:
                    violations.append(
                        f"{module}: imports '{name}' (core module -> implementation {prefix})"
                    )
    return violations


def test_app_respects_dependency_rules() -> None:
    assert find_violations(APP_ROOT) == []


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


@pytest.mark.parametrize(
    ("relative", "source", "expected_target"),
    [
        ("strategies/grid.py", "import app.exchanges.bybit\n", "app.exchanges"),
        ("strategies/grid.py", "from app.execution.engine import X\n", "app.execution"),
        ("strategies/grid.py", "from app import persistence\n", "app.persistence"),
        ("strategies/grid/levels.py", "from ...config import settings\n", "app.config"),
        (
            "domain/models.py",
            "from app.monitoring.logging import configure_logging\n",
            "app.monitoring",
        ),
        ("domain/models.py", "from .. import risk\n", "app.risk"),
        ("risk/limits.py", "if True:\n    import app.execution\n", "app.execution"),
    ],
)
def test_forbidden_imports_detected(
    tmp_path: Path, relative: str, source: str, expected_target: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert len(violations) == 1
    assert f"-> {expected_target})" in violations[0]


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("strategies/grid.py", "from app.domain.models import Order\nimport decimal\n"),
        ("strategies/grid/levels.py", "from . import math\nfrom ..base import Strategy\n"),
        ("risk/limits.py", "from app.portfolio import Portfolio\nfrom app.domain import Side\n"),
        ("execution/engine.py", "from app.exchanges.base import ExchangeAdapter\n"),
        ("services/bootstrap.py", "from app.exchanges import bybit\nfrom app import risk\n"),
    ],
)
def test_allowed_imports_pass(tmp_path: Path, relative: str, source: str) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    assert find_violations(root) == []


def test_unknown_package_requires_rule(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(root, "analytics/report.py", "x = 1\n")

    violations = find_violations(root)

    assert violations == ["app.analytics.report: package 'app.analytics' has no dependency rule"]


def test_top_level_app_import_is_forbidden_for_restricted_packages(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(root, "strategies/grid.py", "import app\n")

    assert len(find_violations(root)) == 1


# --- Domain: standard library only -------------------------------------------------------


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        ("domain/models.py", "import pydantic\n", "'pydantic' (third-party"),
        (
            "domain/models.py",
            "from pydantic import BaseModel\n",
            "'pydantic.BaseModel' (third-party",
        ),
        ("domain/models.py", "import pydantic_settings\n", "'pydantic_settings' (third-party"),
        ("domain/models.py", "import structlog\n", "'structlog' (third-party"),
        ("domain/models.py", "import yaml.constructor\n", "'yaml.constructor' (third-party"),
        ("domain/models.py", "from pybit.unified_trading import HTTP\n", "'pybit"),
        ("domain/models.py", "import typing_extensions\n", "'typing_extensions' (third-party"),
        ("domain/models.py", "def f() -> None:\n    import httpx\n", "'httpx' (third-party"),
        (
            "domain/models.py",
            "from app.monitoring import logging\n",
            "app.domain -> app.monitoring",
        ),
        ("domain/models.py", "import app.config\n", "app.domain -> app.config"),
        ("domain/models.py", "from ..monitoring import configure_logging\n", "-> app.monitoring"),
        ("domain/sub/models.py", "from ...config import x\n", "-> app.config"),
        # Too many dots: beyond the top-level package, cannot be resolved.
        ("domain/models.py", "from ...app.monitoring import x\n", "unresolvable relative import"),
        ("domain/sub/models.py", "from .....pydantic import x\n", "unresolvable relative import"),
        ("domain/__init__.py", "from ... import x\n", "unresolvable relative import"),
    ],
)
def test_domain_violations_detected(
    tmp_path: Path, relative: str, source: str, expected: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert expected in violations[0]


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("domain/models.py", "import datetime\n"),
        ("domain/models.py", "from decimal import Decimal\n"),
        ("domain/models.py", "from __future__ import annotations\n"),
        ("domain/models.py", "import xml.etree.ElementTree\nfrom collections.abc import Mapping\n"),
        ("domain/models.py", "from .enums import Side\n"),
        ("domain/models.py", "from . import enums\n"),
        ("domain/models.py", "from ..domain.enums import Side\n"),
        ("domain/models.py", "from app.domain.enums import Side\n"),
        ("domain/models.py", "import app.domain.enums\n"),
        ("domain/sub/models.py", "from ..validation import require_text\n"),
        ("domain/__init__.py", "from .enums import Side\n"),
    ],
)
def test_domain_allowed_imports_pass(tmp_path: Path, relative: str, source: str) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    assert find_violations(root) == []


def test_third_party_rule_applies_only_to_restricted_packages(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(root, "monitoring/logging.py", "import structlog\n")
    _write(root, "config/settings.py", "from pydantic import BaseModel\n")
    _write(root, "services/bootstrap.py", "import httpx\n")

    assert find_violations(root) == []


# --- Config: domain + configuration libraries only ---------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import structlog\n", "'structlog' (third-party"),
        ("import httpx\n", "'httpx' (third-party"),
        ("from pybit.unified_trading import HTTP\n", "'pybit"),
        ("from sqlalchemy import create_engine\n", "'sqlalchemy.create_engine' (third-party"),
        ("from app.monitoring.logging import configure_logging\n", "app.config -> app.monitoring"),
        ("from app.exchanges import bybit\n", "app.config -> app.exchanges"),
    ],
)
def test_config_violations_detected(tmp_path: Path, source: str, expected: str) -> None:
    root = tmp_path / "app"
    _write(root, "config/bootstrap.py", source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert expected in violations[0]


def test_config_allowed_imports(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "config/loader.py",
        "import yaml\nfrom yaml.constructor import ConstructorError\n"
        "from pydantic import BaseModel\nfrom pydantic_settings import BaseSettings\n"
        "from pathlib import Path\nfrom app.domain.enums import TradingMode\n"
        "from .settings import ConfigError\n",
    )

    assert find_violations(root) == []


# --- Strategies: domain + standard library only ------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from app.config.schema import GridConfig\n", "app.strategies -> app.config"),
        ("from app.exchanges import bybit\n", "app.strategies -> app.exchanges"),
        ("import app.market_data.feed\n", "app.strategies -> app.market_data"),
        ("from app.risk.manager import RiskManager\n", "app.strategies -> app.risk"),
        ("from app.execution import engine\n", "app.strategies -> app.execution"),
        ("from app.persistence import db\n", "app.strategies -> app.persistence"),
        ("from app.monitoring.logging import configure_logging\n", "-> app.monitoring"),
        ("import pydantic\n", "'pydantic' (third-party"),
        ("import yaml\n", "'yaml' (third-party"),
        ("import structlog\n", "'structlog' (third-party"),
        ("import httpx\n", "'httpx' (third-party"),
        ("from pybit.unified_trading import HTTP\n", "'pybit"),
    ],
)
def test_strategy_violations_detected(tmp_path: Path, source: str, expected: str) -> None:
    root = tmp_path / "app"
    _write(root, "strategies/grid/levels.py", source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert expected in violations[0]


def test_strategy_allowed_imports(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "strategies/grid/levels.py",
        "import itertools\nfrom decimal import Decimal, localcontext\n"
        "from app.domain.enums import GridSpacing\nfrom ...domain.validation import require_enum\n"
        "from . import helpers\n",
    )

    assert find_violations(root) == []


# --- Exchanges: domain + standard library only (for now) ---------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from app.config.settings import EnvSettings\n", "app.exchanges -> app.config"),
        ("from app.strategies.grid import levels\n", "app.exchanges -> app.strategies"),
        ("from app.monitoring.logging import configure_logging\n", "-> app.monitoring"),
        ("from app.execution import engine\n", "app.exchanges -> app.execution"),
        ("from app.risk import manager\n", "app.exchanges -> app.risk"),
        ("from app.persistence import db\n", "app.exchanges -> app.persistence"),
        ("import pydantic\n", "'pydantic' (third-party"),
        ("import yaml\n", "'yaml' (third-party"),
        ("import structlog\n", "'structlog' (third-party"),
        ("import httpx\n", "'httpx' (third-party"),
        ("import websockets\n", "'websockets' (third-party"),
        ("import ccxt\n", "'ccxt' (third-party"),
        ("from pybit.unified_trading import HTTP\n", "'pybit"),
    ],
)
def test_exchange_violations_detected(tmp_path: Path, source: str, expected: str) -> None:
    root = tmp_path / "app"
    _write(root, "exchanges/protocols.py", source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert expected in violations[0]


def test_exchange_allowed_imports(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/protocols.py",
        "from typing import Protocol\nfrom app.domain.orders import Order\n"
        "from .models import OrderAck\n",
    )

    assert find_violations(root) == []


# --- Exchange implementation subpackage --------------------------------------------------


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        # Core contracts may not use the adapter's dependency...
        (
            "exchanges/models.py",
            "import httpx\n",
            "'httpx' (third-party; not allowed in app.exchanges)",
        ),
        ("exchanges/protocols.py", "import httpx\n", "'httpx' (third-party"),
        # ...nor import the adapter itself.
        ("exchanges/protocols.py", "from app.exchanges.bybit import market_data\n", "core module"),
        ("exchanges/errors.py", "from .bybit.endpoints import X\n", "core module"),
        # The adapter gets httpx, nothing else.
        ("exchanges/bybit/market_data.py", "import aiohttp\n", "'aiohttp' (third-party"),
        ("exchanges/bybit/market_data.py", "import pybit\n", "'pybit' (third-party"),
        ("exchanges/bybit/market_data.py", "import structlog\n", "'structlog' (third-party"),
        (
            "exchanges/bybit/private_rest.py",
            "from pydantic import SecretStr\n",
            "'pydantic.SecretStr'",
        ),
        ("exchanges/bybit/private_rest.py", "import requests\n", "'requests' (third-party"),
        ("exchanges/bybit/market_data.py", "from app.config.settings import X\n", "-> app.config"),
    ],
)
def test_exchange_subpackage_violations(
    tmp_path: Path, relative: str, source: str, expected: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert expected in violations[0]


def test_bybit_adapter_allowed_imports(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/bybit/market_data.py",
        "import json\nimport httpx\nfrom app.domain.clock import Clock\n"
        "from app.exchanges.errors import ExchangeRejectedError\nfrom ..models import OrderAck\n"
        "from .endpoints import BYBIT_TESTNET_REST_URL\n",
    )

    assert find_violations(root) == []


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        ("exchanges/bybit/order_mapping.py", "import httpx\n", "'httpx' (third-party"),
        ("exchanges/bybit/order_mapping.py", "import pydantic\n", "'pydantic' (third-party"),
        ("exchanges/bybit/order_mapping.py", "import structlog\n", "'structlog' (third-party"),
        (
            "exchanges/bybit/order_mapping.py",
            "from app.exchanges.bybit.private_rest import JsonValue\n",
            "module allowlist",
        ),
        (
            "exchanges/bybit/order_mapping.py",
            "from .private_rest import BybitPrivateRestTransport\n",
            "module allowlist",
        ),
        (
            "exchanges/bybit/order_mapping.py",
            "from .credentials import BybitCredentials\n",
            "module allowlist",
        ),
        (
            "exchanges/bybit/order_mapping.py",
            "from app.exchanges.protocols import TradingClient\n",
            "module allowlist",
        ),
        ("exchanges/bybit/types.py", "import httpx\n", "'httpx' (third-party"),
        ("exchanges/bybit/types.py", "from app.domain.enums import Side\n", "module allowlist"),
    ],
)
def test_pure_mapping_module_violations(
    tmp_path: Path, relative: str, source: str, expected: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert expected in violations[0]


def test_pure_mapping_module_config_import_reported(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(root, "exchanges/bybit/order_mapping.py", "from app.config.settings import X\n")

    violations = find_violations(root)

    assert any("module allowlist" in v for v in violations), violations
    assert any("-> app.config" in v for v in violations), violations


def test_pure_mapping_module_allowed_imports(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/bybit/order_mapping.py",
        "import re\nfrom decimal import Decimal\nfrom app.domain.enums import Side\n"
        "from app.exchanges.models import OrderRequest\n"
        "from app.exchanges.errors import ExchangeRequestValidationError\n"
        "from .types import JsonValue\n",
    )
    _write(root, "exchanges/bybit/types.py", "from __future__ import annotations\n")

    assert find_violations(root) == []


def test_real_order_mapping_module_is_covered_by_the_allowlist() -> None:
    assert (APP_ROOT / "exchanges" / "bybit" / "order_mapping.py").is_file()
    assert (APP_ROOT / "exchanges" / "bybit" / "types.py").is_file()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import httpx\n", "'httpx' (third-party"),
        ("import pydantic\n", "'pydantic' (third-party"),
        ("from app.exchanges.bybit.order_mapping import X\n", "implementation"),
        ("from .bybit import market_data\n", "implementation"),
        ("from app.exchanges.protocols import TradingClient\n", "module allowlist"),
        ("from app.config.settings import X\n", "-> app.config"),
        ("from app.persistence import X\n", "-> app.persistence"),
        ("from app.strategies.grid.levels import X\n", "-> app.strategies"),
    ],
)
def test_simulated_exchange_violations(tmp_path: Path, source: str, expected: str) -> None:
    root = tmp_path / "app"
    _write(root, "exchanges/simulated.py", source)

    violations = find_violations(root)

    assert any(expected in v for v in violations), violations
    assert all(v.startswith("app.exchanges.simulated:") for v in violations), violations


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("exchanges/protocols.py", "from app.exchanges.simulated import SimulatedExchange\n"),
        ("exchanges/errors.py", "from .simulated import SimulatedExchange\n"),
        ("exchanges/bybit/market_data.py", "from ..simulated import SimulatedExchange\n"),
    ],
)
def test_core_and_bybit_must_not_import_simulator(
    tmp_path: Path, relative: str, source: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert len(violations) == 1, violations
    assert "implementation app.exchanges.simulated" in violations[0]


def test_simulated_exchange_allowed_imports(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/simulated.py",
        "from dataclasses import dataclass\nfrom app.domain.clock import Clock\n"
        "from app.domain.orders import OrderUpdate\n"
        "from app.exchanges.errors import ExchangeRejectedError\n"
        "from .models import OrderAck\n",
    )

    assert find_violations(root) == []


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        ("exchanges/protocols.py", "from .simulated_positions import X\n", "implementation"),
        (
            "exchanges/bybit/market_data.py",
            "from app.exchanges.simulated_positions import X\n",
            "implementation",
        ),
        ("exchanges/simulated_positions.py", "import httpx\n", "'httpx' (third-party"),
        (
            "exchanges/simulated_positions.py",
            "from app.exchanges.models import OrderRequest\n",
            "module allowlist",
        ),
        (
            "exchanges/simulated_positions.py",
            "from app.exchanges.simulated import SimulatedExchange\n",
            "module allowlist",
        ),
        ("exchanges/simulated_positions.py", "from app.config import X\n", "-> app.config"),
    ],
)
def test_position_ledger_violations(
    tmp_path: Path, relative: str, source: str, expected: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert any(expected in v for v in violations), violations


def test_simulator_may_use_position_ledger(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/simulated.py",
        "from app.exchanges.simulated_positions import SimulatedPositionLedger\n",
    )
    _write(
        root,
        "exchanges/simulated_positions.py",
        "from fractions import Fraction\nfrom app.domain.fills import Fill\n",
    )

    assert find_violations(root) == []


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        ("exchanges/protocols.py", "from .simulated_fees import X\n", "implementation"),
        (
            "exchanges/simulated_positions.py",
            "from app.exchanges.simulated_fees import X\n",
            "implementation",
        ),
        (
            "exchanges/bybit/market_data.py",
            "from app.exchanges.simulated_fees import X\n",
            "implementation",
        ),
        ("exchanges/simulated_fees.py", "import httpx\n", "'httpx' (third-party"),
        (
            "exchanges/simulated_fees.py",
            "from app.exchanges.simulated import SimulatedExchange\n",
            "module allowlist",
        ),
        (
            "exchanges/simulated_fees.py",
            "from app.exchanges.simulated_positions import X\n",
            "module allowlist",
        ),
        ("exchanges/simulated_fees.py", "from app.config import X\n", "-> app.config"),
        ("exchanges/simulated_fees.py", "from app.strategies import X\n", "-> app.strategies"),
    ],
)
def test_fee_model_violations(tmp_path: Path, relative: str, source: str, expected: str) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert any(expected in v for v in violations), violations


def test_simulator_may_use_fee_model(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/simulated.py",
        "from app.exchanges.simulated_fees import SimulatedFeePolicy\n",
    )
    _write(
        root,
        "exchanges/simulated_fees.py",
        "from decimal import Decimal\nfrom app.domain.validation import require_text\n",
    )

    assert find_violations(root) == []


@pytest.mark.parametrize(
    ("relative", "source", "expected"),
    [
        ("exchanges/protocols.py", "from .simulated_accounting import X\n", "implementation"),
        (
            "exchanges/simulated_positions.py",
            "from app.exchanges.simulated_accounting import X\n",
            "implementation",
        ),
        (
            "exchanges/simulated_fees.py",
            "from app.exchanges.simulated_accounting import X\n",
            "implementation",
        ),
        (
            "exchanges/bybit/market_data.py",
            "from app.exchanges.simulated_accounting import X\n",
            "implementation",
        ),
        (
            "exchanges/simulated_accounting.py",
            "from app.exchanges.simulated import SimulatedExchange\n",
            "module allowlist",
        ),
        (
            "exchanges/simulated_accounting.py",
            "from app.exchanges.simulated_positions import X\n",
            "module allowlist",
        ),
        (
            "exchanges/simulated_accounting.py",
            "from app.exchanges.simulated_fees import X\n",
            "module allowlist",
        ),
        ("exchanges/simulated_accounting.py", "import httpx\n", "'httpx' (third-party"),
        ("exchanges/simulated_accounting.py", "from app.config import X\n", "-> app.config"),
    ],
)
def test_cash_accounting_violations(
    tmp_path: Path, relative: str, source: str, expected: str
) -> None:
    root = tmp_path / "app"
    _write(root, relative, source)

    violations = find_violations(root)

    assert any(expected in v for v in violations), violations


def test_simulator_may_use_cash_accounting(tmp_path: Path) -> None:
    root = tmp_path / "app"
    _write(
        root,
        "exchanges/simulated.py",
        "from app.exchanges.simulated_accounting import SimulatedCashLedger\n",
    )
    _write(
        root,
        "exchanges/simulated_accounting.py",
        "from fractions import Fraction\nfrom app.domain.validation import require_text\n",
    )

    assert find_violations(root) == []
