# Vision

Status: frozen for Phase 0 on 2026-10-10. Change this definition only by an ADR in `docs/07_DECISIONS.md` written before the run that would depend on the change.

## What this project is

valmorph is a small-scale study of a validation-gated self-modifying network. The network may change its structure or a learning-rule setting during training. Every proposed change is tried on a shadow copy, scored on a held-out probe split, and accepted or rejected. Accepted and rejected proposals are both written to a typed Parquet ledger, with the lineage of units those edits create.

The scientific product is the ledger and the figures drawn from it. A working system that cannot show, with seeds and baselines, whether it got better is a failed experiment, not a partial success. A clean negative result is a success.

This repository is standalone. It does not import, copy, or take design from any other project on the machine.

## Levels

Level 1. The network modifies its own structure or learning-rule settings during training, and held-out performance improves relative to matched-compute baselines, including a random-edit controller.

Level 2. The quality of the self-edits improves over time. Acceptance rate, validated probe gain per edit, and edit efficiency trend upward from the early third of proposals to the late third.

Level 3, stretch. A self-improvement policy learned on one task family transfers to an unseen task family without retuning.

The paper reports which levels hold. Level 3 is not required for the project to succeed.

## Compute is part of the design

The question is whether a small network's edits help, not whether a large model can memorize the benchmark. Light compute is a validity condition.

- Development machine: Windows PC with an RTX 5080, 16 GB, compute capability 12.0. This is the canonical CUDA device. Another job may be using the GPU, so the process memory fraction defaults to 0.20 (about 3.3 GB) and the code refuses a fraction above 0.25 unless a config explicitly overrides it.
- Development machine: MacBook Pro, Apple M4, 24 GB unified memory, 1 TB storage. The same configs run here on MPS. The MPS allocator watermark uses the same 0.20 fraction.
- Every confirmatory comparison has a short CPU smoke first. The heavy run itself uses the GPU under the 0.20 memory cap. A result that needs an uncapped GPU is out of scope. A long schedule is not left on the CPU.
- Runs record device, capability, torch build, and a content hash of the ledger. Results from the 5080 and the M4 are never pooled.
- Determinism is best-effort. We do not claim bit-exact equality across CUDA, MPS, and CPU.

Quality comes from matched baselines, at least five seeds, bootstrap intervals, a random-edit control, and a complete edit log. It does not come from model scale.

## System hypothesis

Four parts, kept only while they earn their place:

1. Substrate. Function-preserving edits: widen by neuron split, identity deepen, prune, prune-and-regrow, per-layer learning-rate or regularization change, consolidate or freeze. A zero-parameter morphism leaves the output unchanged. That property is unit-tested with Hypothesis, not checked by eye.
2. Introspection. Diagnostics during training: gradient noise, dead-neuron fraction, effective rank, activation covariance, a Hessian-trace proxy, representation drift, and plasticity-loss indicators. Each diagnostic names the paper that defines it once that paper has been opened. Until then the name is a planned measurement, not a citation.
3. Controller. L0 heuristic rules. L1 a learned controller (bandit, small RL, or evolutionary search — pick one in Phase 3 and record the choice). L2 a controller that can also change its own hyperparameters. A random-edit controller is a mandatory control, not an optional ablation.
4. Acceptance gate. Shadow copy, held-out probe, accept or reject. The probe split is disjoint from validation and from test. The correlation between probe gain and full validation gain is itself logged, so a proxy that stops tracking validation is visible.

Matched compute means the same number of training steps on the data. Probe evaluations spent by the gate are counted. If Level 1 appears only because the gated run received extra probe passes, a robustness check gives the baselines the same extra budget. A win that disappears under that check is not counted as self-improvement. The primary comparison remains matched training steps; the robustness check is pre-registered in `docs/01_RESEARCH_QUESTIONS.md`.

## What we are not building

- A language model, a nested-memory architecture at that scale, or a reimplementation of HOPE.
- A coding agent that rewrites its own repository.
- A method that is allowed to look at the test set while choosing edits.

## Two outputs

1. A system that runs, at smoke scale and then at the pre-registered scale, on continual image classification and on one real concept-drift stream.
2. A reproducible record: ledger, figures with provenance manifests, and a paper in which every number points at that record.
