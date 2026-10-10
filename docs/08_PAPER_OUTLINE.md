# Paper outline

Working title: Validation-gated self-modification in small continual learners.

No results section is filled until Phase 6 produces citable manifests. This outline is the map, not a draft of claims.

## Abstract

One paragraph. State the question, the gate, the two tasks, and which of Level 1, Level 2, and Level 3 held. If a level failed, the abstract says it failed.

## 1. Introduction

Catastrophic forgetting and loss of plasticity are different failures. Growing a network and editing a learning rule are old ideas. The question is whether a logged, probe-gated editor improves held-out performance against matched baselines, and whether the editor itself improves.

## 2. Related work

Compress `docs/02_RELATED_WORK.md`. Keep the novelty paragraph's caution. Do not cite the gap list.

## 3. Method

Substrate and the function-preserving property. Diagnostics actually implemented. Controllers, with the random controller specified as a control. Gate, probe split, and the matched-compute rule. What the ledger stores.

## 4. Experimental setup

Protocols, with citations opened before the numbers were computed. Seeds. Bootstrap and the primary family. Hardware: RTX 5080 for canonical CUDA, M4 called out when a figure is MPS. Memory cap. The sanity-check gap from Phase 5, whether or not we matched the published number.

## 5. Results

Level 1 curves. Level 2 curves. Forgetting matrices. Edit timelines. Lineage and a topology snapshot. Tables report intervals, not stars. Every figure caption ends with the manifest's ledger hash and git commit.

## 6. Negative results and limitations

Rejected edits. Failed seeds. Levels that did not hold. The probe/validation correlation if it was weak. Device non-equivalence. The scale we refused to run.

## 7. Conclusion

Which levels hold, and what was not shown. No future-work paragraph that walks back a negative result.

## Figures

The required list is in `docs/05_METRICS.md`. A figure is added to `paper/figures/` only with its provenance JSON.
