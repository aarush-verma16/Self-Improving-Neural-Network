# State

Read this file at the start of a session. Update it at the end. It is the handoff between the Windows PC and the MacBook.

## Current phase

Notebook 11 was run by the user (`widen-long-s0-2337dce6c530-20261022T194419Z`). Seven unit-0 widens, five kept, from step 30 through step 70. Final probe accuracy was 0.752 with no edits, 0.806 on the extra-steps curve, and 0.750 with the live widens. The live curve stays with no edits. Notebook 12 is ready: the same starting weights, one copy widened to 37 before training, one grown on the schedule, one left at 32. A local check finished at 0.752, 0.754, and 0.750. The user runs it. Still CPU.

## Done

- Package name `valmorph`, MIT license, open-source metadata (`CITATION.cff`, `CONTRIBUTING.md`).
- Docs `00` through `09`, this file, and the ADR log.
- Cursor rules `.cursor/rules/00`–`10` and `AGENTS.md`.
- uv project pinned to Python 3.12. Torch 2.14.1+cu132 on this PC. macOS resolves the PyPI wheel (ADR-003). `uv.lock` is present.
- Phase 0 library: device, seed, config hash, memory cap, Parquet ledger, synthetic demo, Plotly/matplotlib timeline, provenance manifests.
- Notebook `notebooks/00_setup.py` (jupytext percent format) and the generated `notebooks/00_setup.ipynb`.
- Pre-commit hook installed locally (ruff, pyright, pytest excluding the gpu marker, nbstripout). It calls `.venv` through `scripts/run_tool.py`, because the editor's git process does not have `uv` on PATH. No GitHub Actions.
- Verification on this PC, 2026-10-10: ruff clean, pyright clean, 25 pytest passed, CUDA matmul succeeded, setup notebook wrote a synthetic figure.
- Mac verification, 2026-10-19: uv 0.13.1, Python 3.12.6, torch 2.14.1 (PyPI), `env_check` device `mps` with `matmul_ok` true, pytest 32 passed and 1 skipped (the CUDA env check), setup notebook wrote a synthetic figure, gate notebook wrote a non-synthetic ledger.

## Next

1. Open `notebooks/12_fixed_width.ipynb` as a new tab, restart the kernel, and run it. The three curves start from the same weights. Width 37 is five copies made before training. Grow-to-35 widens at steps 10, 20, and 30. There is no gate.
2. Phase 0b citation gaps before any published baseline number is used.
3. `apply_compute_cap` sets both `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` to the configured fraction. That fix is in this merge. The Mac `env_check` on 2026-10-19 passed after it (`results/env/20261019T071452Z.json`).
4. Notebooks `00` through `03` stay short CPU runs. The next time a notebook is heavy training, stop and switch to the CUDA PC. Do not treat an MPS run as the canonical result.

## Open questions

- Electricity's license string on the OpenML dataset page itself, as opposed to the datahub mirror.
- Which single learned controller is L1 (bandit, small RL, or evolution). Decide in a Phase 3 ADR, not now.
- Published Permuted MNIST and Split CIFAR-100 numbers. Blank until the protocol papers are opened.

## Environment — Windows PC

Last verified: 2026-10-10, `results/env/20261010T150959Z.json` (gitignored).

- OS: Windows 11 AMD64
- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0, device index 0
- Project Python: 3.12.15 via uv (not the Miniforge 3.14 base)
- Torch: 2.14.1+cu132, `torch.version.cuda` 13.2
- `matmul_ok`: true, result `[[1.0, 2.0], [3.0, 4.0]]`
- Memory fraction applied: 0.2
- Git commit: `cc4b8ec` on `main`. Not pushed.
- Setup note: uv 0.13's TLS client rejected GitHub and PyPI with `UnknownIssuer` on this PC. curl could reach GitHub. The downloads used `uv --allow-insecure-host` for those hosts. That flag is not part of the project config. If a later `uv sync` fails the same way, repeat the flag rather than changing the lockfile.

## Environment — MacBook Pro

Last verified: 2026-10-19, `results/env/20261019T071452Z.json` (gitignored). An earlier check the same minute, `results/env/20261019T071321Z.json`, found MPS and then failed the matmul because only the high watermark was set.

- Machine: MacBook Pro Mac16,8, Apple M4 Pro, 24 GB unified memory, 994 GB SSD
- OS: Darwin 27.0.0 arm64
- Project Python: 3.12.6 at `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`, via uv 0.13.1. `.python-version` is `3.12`.
- Torch: 2.14.1 from PyPI, not the cu132 index. `torch.version.cuda` is null.
- `device_type`: `mps`. `matmul_ok`: true, result `[[1.0, 2.0], [3.0, 4.0]]`
- Memory fraction applied: 0.2, on both `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO`
- Git commit: `954818b` on `main`, dirty with the watermark fix. Not pushed.
- Hooks: local `core.hooksPath` is `.githooks`.
- Milestone 2 ledger: `results/ledgers/gate-s0-2337dce6c530-20261019T071507Z`. Provenance `results/figures/02_gate_timeline.provenance.json`, `synthetic` false. Probe gain for `prune-junk` was −143.9385 (accepted). Probe gain for `prune-useful` was +0.3496 (rejected).

## Mac checklist

On the Mac, from a fresh clone of this repo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# open a new shell so uv is on PATH
cd path/to/SINN
uv python pin 3.12
uv sync --group dev
git config core.hooksPath .githooks
uv run python scripts/env_check.py
uv run pytest
uv run python notebooks/00_setup.py
uv run jupytext --sync notebooks/00_setup.py
```

`env_check` must print `device_type` `mps` and `matmul_ok` true. If it prints `cpu`, MPS is not available to this torch build: stop and record the torch version. Do not copy `results/` from the Windows machine into the Mac tree. Do not commit `results/`.

## Session protocol

1. Read this file, then `docs/06_ROADMAP.md` for the current phase's exit check.
2. Do the phase. Keep smoke runs inside the memory cap.
3. Update this file and add a dated entry to `docs/09_JOURNAL.md`.
4. A paper sentence written in the session points at a provenance manifest or it does not get written.
