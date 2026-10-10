# State

Read this file at the start of a session. Update it at the end. It is the handoff between the Windows PC and the MacBook.

## Current phase

Milestone 1 of Phase 1 is done on CPU: a small MLP can widen by splitting a unit and can insert an identity layer, and the logits stay the same. The acceptance gate is the next milestone. Citation gaps in `docs/02_RELATED_WORK.md` stay open and do not block the gate.

## Done

- Package name `valmorph`, MIT license, open-source metadata (`CITATION.cff`, `CONTRIBUTING.md`).
- Docs `00` through `09`, this file, and the ADR log.
- Cursor rules `.cursor/rules/00`–`10` and `AGENTS.md`.
- uv project pinned to Python 3.12. Torch 2.14.1+cu132 on this PC. macOS resolves the PyPI wheel (ADR-003). `uv.lock` is present.
- Phase 0 library: device, seed, config hash, memory cap, Parquet ledger, synthetic demo, Plotly/matplotlib timeline, provenance manifests.
- Notebook `notebooks/00_setup.py` (jupytext percent format) and the generated `notebooks/00_setup.ipynb`.
- Pre-commit hook installed locally (ruff, pyright, pytest excluding the gpu marker, nbstripout). It calls `.venv` through `scripts/run_tool.py`, because the editor's git process does not have `uv` on PATH. No GitHub Actions.
- Verification on this PC, 2026-10-10: ruff clean, pyright clean, 25 pytest passed, CUDA matmul succeeded, setup notebook wrote a synthetic figure.

## Next

1. Milestone 2: the acceptance gate. A toy problem where a helpful edit is accepted, a harmful edit is rejected, both are logged, and the live network changes only after accept.
2. Phase 0b citation gaps before any published baseline number is used.
3. The Mac checklist is still unverified.

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

## Environment — MacBook Pro M4

Last verified: never.

- Expected: Apple M4, 24 GB unified memory, 1 TB storage, device `mps`
- Torch: PyPI 2.14.1 wheel, not the cu132 index
- `matmul_ok`:
- env JSON path:

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
