# Decisions

An ADR is append-only. Supersede an old one by writing a new one that names it. Do not rewrite history after a confirmatory run.

## ADR-001 — Package name `valmorph`

Date: 2026-10-10.

The repo folder stays `SINN`. The import name is `valmorph`: validation-gated morphism. The name records the mechanism under test. `sinn` was the alternative that matches the folder and states the hoped-for outcome. `editline` was the alternative that names the ledger. The mechanism name is the one we have to live with if Level 1 fails.

## ADR-002 — MIT license

Date: 2026-10-10.

MIT, so the code and the docs can be reused with a short notice. Apache-2.0 was the alternative if we later want an explicit patent grant. Data files keep their own licenses and are not relicensed by this ADR.

## ADR-003 — Python 3.12 and torch 2.14.1

Date: 2026-10-10.

The project interpreter is a uv-managed CPython 3.12, not the machine's Miniforge 3.14 base. Torch is 2.14.1. Windows and Linux resolve it from the cu132 index. macOS resolves it from PyPI, which is the MPS build. Checked on 2026-10-09: the live PyTorch selector's stable Windows CUDA wheels are cu126, cu130, and cu132; cu128 has no 2.14.1 wheel; the static install page that still says torch 2.7 and cu118 is stale.

If the 5080 matmul fails on cu132, stop and write the failure here. The next candidate is cu130. The code does not switch wheels by itself.

## ADR-004 — Light compute is a constraint, and the GPU is a 5080

Date: 2026-10-10.

The CUDA machine is an RTX 5080, 16303 MiB, compute capability 12.0, driver 610.88, CUDA UMD 13.3 at the time of the check. The second machine is a MacBook Pro M4 with 24 GB unified memory and 1 TB storage. Default memory fraction is 0.20. Fractions above 0.25 raise unless `allow_high_memory` is set in the config on purpose.

A result that needs a larger model or a long uncapped run is outside the study. The M4 must be able to execute the smoke path of every phase.

## ADR-005 — Probe split and matched compute

Date: 2026-10-10.

The gate sees the probe split only. Probe, validation, and test are disjoint. Primary comparisons match the number of training steps. A pre-registered robustness check matches training steps plus the gate's probe budget, as specified in H1. Extra probe compute is not silently called a smarter editor.

## ADR-006 — Citations are opened sources only

Date: 2026-10-10.

`paper/refs.bib` and factual sentences in `docs/02_RELATED_WORK.md` come from pages or PDFs opened on 2026-10-10. Unopened papers stay on the gap list. A secondary citation of GEM's arXiv id was not enough to add a bib entry.

## ADR-007 — Findings are ledger rows

Date: 2026-10-10.

Empirical notes that should survive a session go in the `findings` table: observation, negative, claim, or method. The journal is the narrative. A paper claim is a finding of kind `claim` plus a citable provenance manifest. Synthetic demos may write findings only to say they are not evidence.

## ADR-009 — Substrate morphisms before the citation pass

Date: 2026-10-10.

Identity widen and identity deepen do not depend on the unopened protocol papers. The construction is the function-preserving split and the inserted identity map. Phase 0b stays open for citations that would support a baseline number. It does not block this milestone.

## ADR-008 — Pyright standard mode

Date: 2026-10-10.

Pyright's `strict` mode reports the PyArrow, Plotly, and matplotlib APIs as unknown because those packages do not ship stubs. The project uses `standard` mode plus Ruff's type-related lint, and our own functions stay annotated. Turning `strict` on would require casts around the ledger and the figures without making those calls safer.

## ADR-010 — CPU for short work, GPU for heavy training

Date: 2026-10-19.

Tests, morphism checks, and toy notebooks stay on CPU. Training that would occupy the CPU for a long time uses CUDA here, under the 0.20 memory fraction, and MPS on the Mac. The GPU stays shared. A heavy run is not moved onto the CPU to avoid the GPU, and a short notebook is not moved onto the GPU by default. `notebooks/02_gate.ipynb` remains a short CPU run.
