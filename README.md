# valmorph

Validation-gated self-modifying networks for continual learning.

A valmorph network can change its structure or a learning-rule setting while it trains. Every proposed edit is applied to a shadow copy, scored on a held-out probe split, and accepted or rejected. Accepted and rejected proposals are both written to a Parquet ledger, with the lineage of the units those edits create. Held-out comparisons use matched training budgets, a random-edit control, and the usual continual-learning baselines.

The scientific record is the ledger. Figures for exploration are interactive. Figures for the paper are static, colorblind-safe, and written with a provenance manifest: git commit, config hash, seeds, and ledger hashes. Analytical findings and visual findings are stored as ledger rows, including negative results.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv python pin 3.12
uv sync --group dev
git config core.hooksPath .githooks
uv run python scripts/env_check.py
uv run pytest
```

Windows and Linux install PyTorch from the CUDA 13.2 wheel index. macOS installs the PyPI wheel and uses MPS. `env_check` resolves `cuda`, `mps`, or `cpu`, applies the default memory fraction of 0.20, and runs a real matmul. It prints `matmul_ok: true` when the selected device executed the op. An explicit device request is never silently replaced with another device.

## Notebooks

Edit the percent-format source, then sync the notebook:

```bash
uv run python notebooks/00_setup.py
uv run jupytext --sync notebooks/00_setup.py
```

Logic lives in `src/valmorph`. Notebooks load the ledger, run checks, and export figures. Pre-commit strips notebook outputs.

## Layout

| Path | Role |
| --- | --- |
| `src/valmorph` | Device handling, ledger, edits, diagnostics, controllers, gate, metrics, figures |
| `configs/` | Typed run configs |
| `notebooks/` | Analysis and figure export |
| `paper/` | Manuscript, bibliography, and exported figures |
| `docs/` | Definitions, protocol, metrics, and decisions |
| `results/` | Local ledgers and environment records (not committed) |

## Reproducibility

Each run stores its seed, config hash, git commit, device, and library versions. Results from different devices are reported separately. Claims in the paper cite a provenance manifest. Synthetic demos are labeled and are not evidence.

Checks before a change is finished:

```bash
uv run python scripts/local_check.py
```

## License

MIT. See `LICENSE`. Cite this repository with `CITATION.cff`. Dataset files keep their own licenses and are not committed.

## Contributing

See `CONTRIBUTING.md`.
