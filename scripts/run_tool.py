"""Run a tool from the project virtualenv.

Editor git hooks do not have ``uv`` on PATH. Hooks call this script with the
system ``python``, and the script executes ``.venv`` directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tool_path(name: str) -> Path:
    """Path to an installed console script inside ``.venv``."""

    folder = "Scripts" if sys.platform == "win32" else "bin"
    suffix = ".exe" if sys.platform == "win32" else ""
    return ROOT / ".venv" / folder / f"{name}{suffix}"


def main(argv: list[str] | None = None) -> int:
    """Execute ``argv[0]`` from the virtualenv with the remaining arguments."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: run_tool.py <tool> [args...]", file=sys.stderr)
        return 2
    name, tool_args = args[0], args[1:]
    exe = tool_path(name)
    if not exe.is_file():
        print(f"missing {exe}. From the repo root, run: uv sync --group dev", file=sys.stderr)
        return 1
    completed = subprocess.run([str(exe), *tool_args], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
