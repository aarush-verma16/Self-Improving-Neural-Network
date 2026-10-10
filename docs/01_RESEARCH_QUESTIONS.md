# Research questions

Status: pre-registered on 2026-10-10, before any training run. No confirmatory data exists. Do not edit a hypothesis after the run that tests it. A change before that run is an ADR, and the ADR has to say that no confirmatory ledger for the hypothesis exists yet.

Smoke runs in later phases debug code. They are not evidence for or against these hypotheses. A smoke ledger is marked in its notes and is not entered in the primary analysis.

## Endpoints

The unit of analysis is the seed. Intervals are percentile bootstrap, 10,000 resamples of seeds, 95% interval. Report the interval and the point estimate. Seeds are chosen in advance and listed in the config. Failed runs stay in the ledger and in the paper's negative-results section. They are not dropped because the curve looks bad.

Average accuracy, forgetting, plasticity, and edit statistics are defined in `docs/05_METRICS.md`.

Smallest effect we will call practically relevant, chosen before data: 2 percentage points of average accuracy on an image protocol, and 1 percentage point on the stream. An interval that excludes zero but sits below that bar is reported as detectable and small. The bar is not a license to fish for a larger effect.

## Primary family

Four comparisons, Bonferroni-adjusted and raw intervals both reported:

1. Permuted MNIST: gated controller minus random-edit controller, average accuracy.
2. Permuted MNIST: gated controller minus naive sequential, average accuracy.
3. The chosen stream: gated controller minus random-edit controller, average accuracy.
4. The chosen stream: gated controller minus naive sequential, average accuracy.

The image protocol's task count, architecture width, and step budget are frozen in an ADR after the source paper for that protocol is opened (Phase 4). They are not frozen by memory here. Split CIFAR-100 is a second image protocol used for Level 3 and as a secondary Level 1 check. It is outside the family of four.

## H1 — Level 1

On Permuted MNIST and on the stream, the gated controller's average accuracy exceeds the random-edit controller and the naive sequential baseline at a matched training-step budget.

Success: the Bonferroni-adjusted 95% interval for that seed-level difference lies entirely above 0, on at least five seeds, all from the same device. The RTX 5080 is the canonical device if the run uses CUDA. An MPS run is a separate result and is labeled as such.

Robustness, not primary: repeat the comparison after giving each baseline extra training steps equal to the gated run's probe-evaluation budget. If the primary interval is above 0 and the robustness interval is not, H1 is not claimed.

## H2 — Level 2

Split proposals into the earliest third and the latest third by proposal index, within a run. Compare mean validated probe gain, and compare acceptance rate.

Success: for the gated controller, the 95% interval of the late-minus-early difference, across seeds, lies entirely above 0 for validated gain. Acceptance rate is reported the same way and is secondary. If a third contains fewer than 10 proposals, that seed is inconclusive for H2, not a failure and not a success.

The random-edit controller is plotted on the same axes. Level 2 is about the editor getting better, so a rising curve that the random controller matches is not Level 2.

## H3 — Level 3, stretch

Freeze the controller selected on Permuted MNIST. Apply it, without retuning, to Split CIFAR-100 and to the stream. Compare with a random-edit controller on the target task.

Success uses the same interval rule as H1 against random-edit only. Failure of H3 is an expected possible outcome and is reported as a stretch result. It does not reopen H1 or H2.

## H0

If H1 or H2 fails, that is the result. The gate threshold, the controller, and the architecture are not revised against the test set to recover a win. A revision is a new pre-registered experiment.

## Controls that must exist before H1 is tested

Naive sequential training. A fixed-size network at the starting width. A fixed-size network at the largest width the editor is allowed to reach. EWC. Experience replay. A fixed growth schedule. Magnitude pruning. The random-edit controller.

Published baseline numbers are reproduced, or the miss is documented, before any of those controls is used in a comparison. The papers those numbers come from are opened in Phase 4 and cited only then.

## What is not a hypothesis

The setup notebook's synthetic ledger. Environment-check matmuls. Any single-seed smoke run.
