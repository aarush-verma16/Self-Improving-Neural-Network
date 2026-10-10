# Contributing

valmorph is open source under the MIT license. Contributions should keep the study reproducible and the claims honest.

## Setup

Follow the README. `uv run python scripts/env_check.py` must succeed on your machine before you add experiment code. On a Mac, the device line should say `mps`. On the RTX 5080 machine, it should say `cuda`.

## What to change

- Behavior and types live in `src/valmorph`. Notebooks stay thin.
- Edit `notebooks/*.py`, then `uv run jupytext --sync` the file you changed.
- A new config field is added to `RunConfig` and to the YAML, with the config hash tests still passing.
- A new metric name is defined in `docs/05_METRICS.md` before it is logged.
- A new citation is added to `paper/refs.bib` only after you have opened the source, and the opening is noted in `docs/09_JOURNAL.md`.

## What not to change

- Do not raise the default memory fraction to make a run fit.
- Do not pool CUDA and MPS results.
- Do not retune a gate or a controller on the test set.
- Do not delete a failed ledger to make a table look cleaner. Write a `negative` finding.
- Do not commit `results/`, datasets, or secrets.

## Checks

`uv run python scripts/local_check.py` before you consider a change done. Pre-commit runs the same tools.

## Findings

If you learn something that a later session or the paper might repeat, add a `findings` row when a run ledger exists, and add a short entry to `docs/09_JOURNAL.md`. Session handoff goes in `docs/STATE.md`.
