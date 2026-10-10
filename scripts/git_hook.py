"""Commit checks, run with the project virtualenv's Python.

The editor's Git process does not have a working ``python`` on PATH. The
Windows Store alias exits 9009. ``.githooks/pre-commit`` starts this file
with ``.venv``'s interpreter instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run ruff, pyright, pytest, and nbstripout. Print nothing on success."""

    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    steps: list[list[str]] = [
        [py, "-m", "ruff", "check", "."],
        [py, "-m", "ruff", "format", "--check", "."],
        [py, "-m", "pyright"],
        [py, "-m", "pytest", "-m", "not gpu", "-q"],
    ]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if notebooks:
        steps.append([py, "-m", "nbstripout", *[str(path) for path in notebooks]])
    for command in steps:
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            print(f"commit check failed: {Path(command[2]).name}", file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
