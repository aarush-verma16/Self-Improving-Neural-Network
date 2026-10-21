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

Public README and CONTRIBUTING no longer name the lab machines or the phase log. Those stay in this journal and in `docs/STATE.md`. The always-on integrity rule now requires an analytical ledger finding and a visual provenance record for anything that would enter the paper.

The first commit is `cc4b8ec` on `main`. It has not been pushed. The editor commit failed because that process does not have `uv` on PATH, so the hook stopped at `Executable uv not found` before ruff ran. Hooks now call the project `.venv` through `scripts/run_tool.py`.

## 2026-10-10 — Milestone 1, the network

Started the neural network, not another tooling pass. `ValmorphMLP` is a small ReLU MLP. `widen` copies one hidden unit and halves the outgoing weights of the parent and the child. `identity_deepen` inserts a linear identity with no activation. Both leave the logits unchanged.

CPU tests, including a hand-checked case whose logit stays 6.5 and a Hypothesis sweep over small random widths: the full suite passed. A split writes a lineage row. Notebook `notebooks/01_substrate.py` measured a max absolute logit change of about `3e-8` on one CPU draw, which is float32 roundoff, and recorded that as a method finding. This is not a learning result and not Level 1.

Commit from the editor was failing with exit 9009 because every hook started with `python`, and that name is the Windows Store alias. The hook is now `.githooks/pre-commit`, and this repo's local `core.hooksPath` points there. It runs `.venv`'s Python by path.

## 2026-10-11 — Visual working process

The runs are meant to be watched. Each milestone notebook has to show the data and the process while it happens: edits, gate decisions, and learning curves as figures. A final scalar is not enough. Paper figures stay the static copy of those same views, with a provenance manifest.

## 2026-10-19 — Where training runs

Short work stays on CPU: tests, the substrate checks, and `notebooks/02_gate.ipynb`. Those run on the Mac. Heavy training is a switch to the CUDA PC, under the 0.20 memory cap. Say that switch before the run. A short MPS smoke on the Mac stays labeled MPS and is not pooled with CUDA.

## 2026-10-19 — Milestone 2, the gate

The gate trains a copy of the network after a proposed edit and keeps the edit only when probe loss falls. A unit test removes a harmful hidden unit and accepts that prune, and removes a unit the fit needs and rejects that prune. The live network changes only on accept. `notebooks/02_gate.ipynb` is the run to watch: a sine fit, a damaged extra unit, one kept prune, one refused prune.

## 2026-10-19 — Mac setup and the gate run

This machine had no `uv` and the system Python was 3.9.6. Installed uv 0.13.1, pinned Python 3.12, and `uv sync --group dev` used the existing CPython 3.12.6 framework build. Torch resolved to the PyPI 2.14.1 wheel. Local `core.hooksPath` is `.githooks`.

The first `env_check` selected MPS and then failed the matmul: `RuntimeError: invalid low watermark ratio 1.4`. PyTorch's default low watermark is 1.4, and it must be at most the high watermark. Setting only `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2` makes the first MPS allocation illegal. `apply_compute_cap` now sets both watermarks to the configured fraction. The retry wrote `results/env/20261019T071452Z.json`: device `mps`, `matmul_ok` true, result `[[1.0, 2.0], [3.0, 4.0]]`. pytest: 32 passed, 1 skipped (CUDA env check).

`notebooks/00_setup.py` wrote a synthetic timeline. `notebooks/02_gate.py` then ran on CPU. Training loss fell from 0.625 to 0.011 over 60 steps. The damaged unit left the fit near a constant around 11–13 (mean squared error about 144 against the sine). The gate accepted `prune-junk` (probe gain −143.9385) and the live fit came back to the sine (mean squared error about 0.011). It rejected `prune-useful` (probe gain +0.3496), so the live network stayed on the repaired fit. Ledger `gate-s0-2337dce6c530-20261019T071507Z`, finding `gate-toy`, provenance `results/figures/02_gate_timeline.provenance.json` with `synthetic` false. This is the toy gate on one seed. It is not a Level 1 result, and it is not pooled with the CUDA machine.

The user's run `gate-s0-2337dce6c530-20261021T021223Z` matches that: repair gain -143.9385 accepted, useful-unit prune gain +0.3496 refused. Lineage contains only the kept prune. Training loss fell from about 0.62 to near 0. The refused point sits near the zero line on the chart because the kept point is about -144; the ledger value is the one that counts.

## 2026-10-20 — Milestone 3, diagnostics

While the sine net trains, the notebook records three of our own measurements: dead-unit fraction, singular-entropy rank of the hidden activations, and gradient norm. CKA is still not implemented. The user's run `diagnostics-s0-2337dce6c530-20261021T021711Z` ended at dead fraction 0.5, rank 2.327, gradient norm 0.029.

## 2026-10-20 — Milestone 4, one permuted-MNIST task

Going forward, a paper-worthy figure gets a ledger finding and a provenance manifest in the same session. Only `claim` figures are copied into `paper/figures/`. Downloads and run outputs stay gitignored (`data/`, `results/`). The user's notebook 04 run reached probe accuracy 0.634 from 0.224. Loss fell from 2.32 to 1.53. The permuted digits are scrambled. Method check only. Notebook 05 then put a planted damaged unit through the gate: accuracy fell to 0.084 and returned to 0.630 when that unit was removed. The useful-unit prune was refused.

## 2026-10-20 — Milestone 6, proposals from silent units

The controller now looks at the current batch. If a hidden unit is silent, it proposes pruning that unit. The existing gate still keeps the edit only when probe loss falls. The user's run `l0-mnist-s0-2337dce6c530-20261021T033055Z` proposed 2 and accepted 0. Gains were about +1.1e-5 and -7.6e-6, both inside the refusal band. Final probe accuracy was 0.634 with and without those proposals, so the two curves sit on top of each other. Lineage is empty. CPU, synthetic false, seed 0. Removing a silent unit does not lower probe loss, so this gate refuses the cleanup. The static gain chart no longer draws a line through unrelated decisions, and each point is labeled with its gain.

## 2026-10-20 — Milestone 7, widen with matched shadow training

A function-preserving widen has immediate gain 0, so the gate that only looks at the edit itself will refuse it. `train_with_widen_trials` proposes `widen` of hidden unit 0 every 10 steps. The gate then trains the widened copy and an unedited copy for 8 full-batch cross-entropy steps and keeps the widen only when the widened copy has lower probe loss. Shadow training for classification uses cross-entropy. The sine gate still uses mean squared error.

The user's run `widen-mnist-s0-2337dce6c530-20261021T033931Z` proposed 3 and accepted 1. `widen-10` gain +0.00436 and `widen-20` gain +0.00103 were refused. `widen-30` gain -0.000589 was kept, lineage `u0` split to `u42`. Final probe accuracy was 0.634 with no edits and 0.668 with the proposals. CPU, synthetic false, seed 0. The accuracy curves lie on top of each other until the kept step. That keep also adopted the 8 extra steps, so the 0.034 accuracy gap is not a matched-step comparison.

The follow-up stays in notebook 07. A third copy now receives the same extra steps when a widen is kept, and does not widen. The kept step is a dotted line. The extra-steps curve is dashed so it stays visible when it matches the widen curve. A local check of that revision ended at 0.634, 0.668, and 0.668. The widen curve and the extra-steps curve differed by at most 0.004 (steps 31, 36, 37, 38) and finished equal. That revision is commit `dacc942`.

The later user run `widen-mnist-s0-2337dce6c530-20261021T035219Z` has the same three decisions and the same 0.634 / 0.668 accuracies. It has no `step_metrics` table, and the finding sentence is the two-curve sentence. The saved accuracy figure has two lines. Jupyter executed the notebook that was already open, not the file on disk. Still CPU. Reload and restart, then run again.

## 2026-10-20 — Milestone 5, gate on permuted MNIST

The gate now scores digit classification by probe cross-entropy. Notebook 05 trains the same small permuted-MNIST net, breaks it with one added unit, then offers a prune that removes that unit and a prune that removes a useful unit. A local smoke restored accuracy from 0.084 to 0.630 and refused the second prune. The user runs the notebook. Still CPU.
