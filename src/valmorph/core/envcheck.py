"""Environment check: versions, memory cap, and a real matmul."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from valmorph import __version__
from valmorph.core.compute import apply_compute_cap
from valmorph.core.config import load_run_config
from valmorph.core.device import describe_device, resolve_device, verify_matmul
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.core.seed import seed_everything


def collect_environment(
    preference: str,
    *,
    memory_fraction: float,
    allow_high_memory: bool,
    seed: int,
) -> dict[str, object]:
    """Resolve the device, apply the cap, and run a 2x2 matmul.

    The returned mapping always includes ``matmul_ok``. Callers that persist
    the record should keep failures as well as successes.
    """

    seed_info = seed_everything(seed)
    device = resolve_device(preference)
    info = describe_device(device)
    applied = apply_compute_cap(
        device,
        memory_fraction,
        allow_high_memory=allow_high_memory,
    )
    record: dict[str, object] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valmorph_version": __version__,
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform_string(),
        "processor": platform.processor(),
        "preference": preference,
        "device_type": info.type,
        "device_name": info.name,
        "device_index": info.index,
        "capability": info.capability,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "memory_fraction": applied,
        "seed": seed_info,
        "libraries": library_versions(),
        "matmul_ok": False,
        "matmul_result": None,
        "error": None,
    }
    try:
        record["matmul_result"] = verify_matmul(device)
        record["matmul_ok"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def write_environment(record: dict[str, object], path: Path) -> None:
    """Write ``record`` as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Exit 0 only when the matmul succeeds."""

    parser = argparse.ArgumentParser(description="Check device, versions, and a real matmul.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = repo_root()
    config_path = args.config if args.config is not None else root / "configs" / "setup.yaml"
    cfg = load_run_config(config_path)
    preference = args.device if args.device is not None else cfg.device
    record = collect_environment(
        preference,
        memory_fraction=cfg.compute.memory_fraction,
        allow_high_memory=cfg.compute.allow_high_memory,
        seed=cfg.seed,
    )
    stamp = cast(str, record["created_utc"]).replace(":", "").replace("-", "")
    out = args.out if args.out is not None else root / "results" / "env" / f"{stamp}.json"
    write_environment(record, out)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0 if record["matmul_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
