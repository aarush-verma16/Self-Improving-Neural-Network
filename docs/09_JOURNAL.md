# Journal

Narrative log. Empirical claims also go in the ledger `findings` table once a real run exists. This file is safe to write before any run.

## 2026-10-10 — Phase 0 foundation

The repo was empty: `main`, no commits, remote `https://github.com/aarush-verma16/Self-Improving-Neural-Network.git`, upstream `origin/main` missing. Nothing was pushed.

Machine check the day before, confirmed in this session's environment script when it passes: Windows 11 build 26300, Miniforge Python 3.14.7 on the base interpreter, uv not yet installed, torch not yet installed, RTX 5080 (16 GB, compute capability 12.0, driver 610.88, CUDA UMD 13.3). The GPU was idle for compute at that check. The project does not use the Miniforge base. It uses uv and CPython 3.12.

Decisions locked in `docs/07_DECISIONS.md`: package name `valmorph`, MIT license, torch 2.14.1 with the cu132 wheel on Windows, PyPI torch on the M4, memory fraction 0.20, citations only from opened sources.

Related-work pass opened Schmidhuber 1993, Schlag/Irie/Schmidhuber 2021, Nested Learning / HOPE (NeurIPS 2025), Net2Net, splitting steepest descent, Firefly, GROWN, SMGrNN, the Darwin Gödel Machine, EWC, and Dohare et al. on loss of plasticity. GEM, experience replay, OGD, CKA, Network Morphism, and the image-protocol papers were not opened. They are a Phase 0b list, not a bibliography.

Dataset shortlist: Electricity (ELEC2) recommended, license ODC-PDDL on the datahub mirror, still to be confirmed on the OpenML page before download. Weather OpenML 42690 is the CC0 alternate on a mirror. Insects has no license on the River page that was opened.

The M4 (24 GB, 1 TB) is a first-class development machine. Smoke configs are required to run there. Canonical CUDA numbers stay on the 5080 and are not pooled with MPS.

Phase 0 code is the frame only. No substrate, no training loop.

Verification on this PC, after the install:

- `uv run ruff check` clean, `uv run pyright` clean, `uv run pytest` 25 passed.
- `scripts/env_check.py` on `auto`: device `cuda`, NVIDIA GeForce RTX 5080, capability 12.0, torch 2.14.1+cu132, CUDA 13.2, memory fraction 0.2, `matmul_ok` true. Record: `results/env/20261010T150959Z.json`.
- `notebooks/00_setup.py` wrote `results/figures/00_setup_gate_timeline.html`, the png twin, and a provenance file. The manifest is `synthetic: true` and is not citable.
- The cu132 wheel does run a real matmul on this 5080. That answers the open question from the install audit.

Nothing has been committed or pushed.

The first commit attempt from the editor failed in the ruff hook because that process does not have `uv` on PATH. The hook said `Executable uv not found` before ruff ran. Hooks now call the project `.venv` through `scripts/run_tool.py`.
