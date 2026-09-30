"""Module dependency rules from docs/ARCHITECTURE.md, section 1.

Each top-level package under ``app/`` may import only the ``app`` packages listed
for it. Packages marked ``None`` have no import restriction defined yet. A new
top-level package must be added here explicitly, otherwise the test fails.
"""

from __future__ import annotations

import ast
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
    "backtesting": None,
    "paper_trading": None,
    "monitoring": None,
    "notifications": None,
    "services": None,
}


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(source: str, module: str, is_package: bool) -> set[str]:
    """Absolute names of all modules imported by the source, relative imports resolved."""
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package_parts[: len(package_parts) - node.level + 1]
                base = ".".join([*base_parts, node.module] if node.module else base_parts)
            else:
                base = node.module or ""
            if "." in base:
                imported.add(base)
            else:
                # "from app import strategies" imports a subpackage, not a name.
                imported.update(f"{base}.{alias.name}" for alias in node.names)
    return imported


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
        imports = _imported_modules(
            path.read_text(encoding="utf-8"), module, is_package=path.name == "__init__.py"
        )
        for name in sorted(imports):
            name_parts = name.split(".")
            if name_parts[0] != root.name:
                continue
            if len(name_parts) == 1:
                # "import app" gives access to every package.
                violations.append(f"{module}: imports '{name}' (top-level app package)")
                continue
            target = name_parts[1]
            if target != package and target not in allowed:
                violations.append(f"{module}: imports '{name}' (app.{package} -> app.{target})")
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
        ("strategies/grid.py", "from app.domain.models import Order\nimport structlog\n"),
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
