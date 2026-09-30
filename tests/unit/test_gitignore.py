"""Secrets and local runtime data stay out of Git; test fixtures do not."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    GIT is None or not (REPO_ROOT / ".git").exists(), reason="requires a git checkout"
)


def _is_ignored(path: str) -> bool:
    assert GIT is not None
    # --no-index: evaluate .gitignore rules even for paths that do not exist or are tracked.
    result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [GIT, "check-ignore", "--quiet", "--no-index", path],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed for {path!r}: exit code {result.returncode}")
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "data/candles.csv", "logs/bot.log", "bot.db"],
)
def test_ignored(path: str) -> None:
    assert _is_ignored(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "tests/data/candles.csv",
        "app/backtesting/data/loader.py",
        "tests/logs/sample.log",
    ],
)
def test_not_ignored(path: str) -> None:
    assert not _is_ignored(path)
