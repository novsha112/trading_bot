"""Smoke test against hidden circular imports between execution, its persistence
port, the persistence adapter and the orchestration modules.

Every order runs in a fresh interpreter (no module is cached from the test run),
so a cycle that only breaks for one particular first import is caught.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "app.execution.account_state",
    "app.execution.persistence",
    "app.execution.state_invariants",
    "app.execution.recovery",
    "app.persistence.memory",
    "app.execution.safety",
    "app.execution.client_order_id",
    "app.services.placement",
    "app.execution.submitter",
    "app.execution.reconciliation",
)
# Each module first, then the rest; plus the full forward and reverse order.
ORDERS = sorted(
    {(first, *[m for m in MODULES if m != first]) for first in MODULES}
    | {MODULES, tuple(reversed(MODULES))}
)


@pytest.mark.parametrize("order", ORDERS, ids=lambda order: order[0].rsplit(".", 1)[-1])
def test_modules_import_in_any_order(order: tuple[str, ...]) -> None:
    code = "import importlib\n" + "".join(f"importlib.import_module({m!r})\n" for m in order)

    result = subprocess.run(  # noqa: S603 - fixed interpreter and generated code only
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
