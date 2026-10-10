"""Repository paths, git identity, and installed library versions."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path


def repo_root() -> Path:
    """Directory that contains ``pyproject.toml``."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml not found above valmorph.core.runtime")


def git_commit(root: Path | None = None) -> str:
    """HEAD sha, with a ``-dirty`` suffix when the work tree is not clean.

    Returns ``no-commit`` when the repository has no commits yet.
    """

    root = repo_root() if root is None else root
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    ).strip()
    if dirty:
        return f"{sha}-dirty"
    return sha


def package_version(name: str) -> str:
    """Installed distribution version, or ``not-installed``."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def library_versions() -> dict[str, str]:
    """Versions recorded in every run header."""

    names = ("valmorph", "torch", "numpy", "pyarrow", "pandas", "omegaconf", "plotly")
    return {name: package_version(name) for name in names}


def platform_string() -> str:
    """OS, release, and machine, for the run header."""

    return f"{platform.system()} {platform.release()} {platform.machine()}"
