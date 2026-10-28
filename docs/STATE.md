# State

Read this file at the start of a session. Update it at the end. It is the handoff between the Windows PC and the MacBook.

## Current phase

The GEM-sized smoke is `gem-protocol-s0-2337dce6c530-20261028T022302Z`. Twenty permutations, 1000 training images, one pass, width 100, minibatch 10, learning rate 0.03. Mean of the last test-accuracy row is 0.489. Backward transfer is −0.197. Task 0 is 0.333 after its own pass and 0.138 at the end. GEM's appendix prints 0.602 and −0.198 for their single model, and 0.784 on the first task after its own pass. This run is not that table.

The gated comparison on the same sizes is CPU, five seeds, not a Level 1 claim. Probe is training rows 1000–1499. The test set is scored after each task and is not an input to the gate. Seed 0 detail is `gem-gate-s0-2337dce6c530-20261028T030244Z`: no edits 0.489 and −0.197, L1 0.513 and −0.161 (14 kept), random 0.457 and −0.089 (11 kept). L1's keeps are 12 widens and 2 hidden-layer halvings. Across seeds 0–4 the L1 last-row mean minus no edits is +0.024, −0.017, −0.001, −0.075, +0.020. Random's mean gap is −0.032, −0.087, −0.028, −0.088, −0.050. Provenance `results/figures/20_gem_gate_seeds.provenance.json`.

The same sizes with stored rows in later batches are `gem-rehearsal0` through `gem-rehearsal4`, timestamps `20261028T032423Z` through `20261028T032444Z`. Half of each later batch is kept training rows. Each new example is still seen once. Seed 0 goes from mean 0.489 and task 0 at 0.138 to mean 0.546 and task 0 at 0.845. Task 19 goes from 0.753 to 0.744. Mean minus the plain pass, seeds 0–4: +0.057, +0.049, +0.050, +0.049, +0.035. Task 0: +0.707, +0.707, +0.598, +0.723, +0.649. Task 19: −0.009, +0.016, −0.012, +0.013, −0.005. Provenance `results/figures/20_gem_rehearsal_seeds.provenance.json`. CPU. Not a Level 1 claim. No GPU run is waiting.

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

1. Short CPU notebooks run in the session. A handoff is for a long GPU run.
2. The 20-task naive run, the five-seed gated comparison, and the five-seed rehearsal mix all finished on CPU. The gate does not raise the last-row mean on every seed. The rehearsal mix does, and task 19 stays inside 2 pp. EWC's Fisher formula was not transcribed. Rolnick et al. does not supply a supervised replay formula. The store size is not searched against this test set. No GPU handoff is waiting.
3. Phase 0b's named papers are opened. Kirkpatrick's permutation section was not the source of the task count. Electricity's OpenML license line is still unread.
4. `apply_compute_cap` sets both `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` to the configured fraction. That fix is in this merge. The Mac `env_check` on 2026-10-19 passed after it (`results/env/20261019T071452Z.json`).
5. Notebooks `00` through `03` stay short CPU runs. The next time a notebook is heavy training, stop and switch to the CUDA PC. Do not treat an MPS run as the canonical result.

## Open questions

- Electricity's license string on the OpenML dataset page itself, as opposed to the datahub mirror.
- L1 is the two-arm bandit in ADR-011. Small RL and evolution were the alternatives and were not chosen.
- Which Split CIFAR-100 schedule to freeze. GEM and iCaRL were both opened and they differ. Permuted MNIST is ADR-013. Published accuracies are not sanity-check targets.

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
