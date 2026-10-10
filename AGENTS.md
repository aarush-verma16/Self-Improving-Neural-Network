# Agents

valmorph is a small-scale, validation-gated self-modifying network, studied in this repository only. The CUDA machine is an RTX 5080 (16 GB). The other development machine is a MacBook Pro M4 (24 GB, 1 TB). Light compute is a design constraint. Canonical CUDA results come from the 5080 and are never pooled with MPS.

## Session protocol

1. Read `docs/STATE.md`.
2. Do only the current phase in that file. The map is `docs/06_ROADMAP.md`.
3. Update `docs/STATE.md` and add an entry to `docs/09_JOURNAL.md` before stopping.
4. Claims require a ledger-backed provenance manifest. Synthetic output does not count.

## Rules

| Rule | When it applies |
| --- | --- |
| `00-charter` | Always |
| `01-integrity` | Always |
| `02-python` | `src/`, `tests/`, `scripts/` |
| `03-notebooks` | `notebooks/` |
| `04-experiments` | By description, when designing or running experiments |
| `05-compute` | Always |
| `06-cross-machine` | Always |
| `07-testing` | `tests/` |
| `08-viz` | By description, when making figures |
| `09-paper` | `paper/` |
| `10-git` | Always |

## Docs

| Doc | Contents |
| --- | --- |
| `docs/00_VISION.md` | Frozen definition of Level 1–3 and the compute constraint |
| `docs/01_RESEARCH_QUESTIONS.md` | Pre-registered hypotheses |
| `docs/02_RELATED_WORK.md` | Opened sources only, plus the gap list |
| `docs/03_ARCHITECTURE.md` | Ledger, devices, packages |
| `docs/04_EXPERIMENT_PROTOCOL.md` | Splits, smoke, datasets, baselines |
| `docs/05_METRICS.md` | Formulas and required figures |
| `docs/06_ROADMAP.md` | Phases 0 through 7 and their exit checks |
| `docs/07_DECISIONS.md` | ADR log |
| `docs/08_PAPER_OUTLINE.md` | Paper map, no results yet |
| `docs/09_JOURNAL.md` | Session narrative |
| `docs/STATE.md` | Cross-machine handoff |

## Commands

```bash
uv sync --group dev
uv run python scripts/env_check.py
uv run pytest
uv run python scripts/local_check.py
uv run python notebooks/00_setup.py
uv run jupytext --sync notebooks/00_setup.py
```

Phase 0 does not train a model.
