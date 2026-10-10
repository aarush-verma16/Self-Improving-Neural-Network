# Contributing

valmorph is open source under the MIT license. Changes should stay reproducible, and any claim should trace to the ledger.

## Setup

Follow the README. `uv run python scripts/env_check.py` must succeed on the machine you are using before you add experiment code. The printed device should be the one that configuration requested.

## Code

- Behavior and types live in `src/valmorph`. Notebooks stay thin.
- Edit `notebooks/*.py`, then run `uv run jupytext --sync` on that file.
- A new config field is added to `RunConfig` and to the YAML.
- A new metric name is defined in `docs/05_METRICS.md` before it is logged.
- A citation is added to `paper/refs.bib` only after the source has been opened. Note that opening in `docs/09_JOURNAL.md`.

## Findings

Record what the run showed, in both forms:

- Analytical: metrics, intervals, effect sizes, acceptance rates, validated gains, failed seeds, and negative results. These are ledger rows. A dropped seed is a finding, not a deletion.
- Visual: every paper figure is exported from a notebook cell with a provenance manifest. The finding that the figure supports points at that manifest.

`docs/09_JOURNAL.md` is the narrative. `docs/STATE.md` is the session handoff. Neither one replaces a ledger row.

## Boundaries

- The default memory fraction stays 0.20 unless a config sets the override on purpose.
- CUDA runs and MPS runs are reported in separate tables.
- The gate and the controller do not read the test set.
- `results/` and dataset files stay untracked.

## Checks

Run `uv run python scripts/local_check.py` before considering a change done. Pre-commit runs the same tools.
