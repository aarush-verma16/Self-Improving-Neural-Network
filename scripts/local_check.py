"""Run the local checks: ruff, pyright, and CPU tests."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    commands = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "pyright"],
        [sys.executable, "-m", "pytest", "-m", "not gpu"],
    ]
    for command in commands:
        print(">", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
