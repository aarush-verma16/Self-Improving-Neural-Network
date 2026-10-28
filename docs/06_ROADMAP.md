# Roadmap

One phase at a time. A phase ends when its exit check is true and `docs/STATE.md` says so. The next phase does not start by widening the model. Smoke stays inside the 0.20 memory cap and finishes in minutes on the M4.

## Phase 0 — Foundation. Exit met on the Windows PC on 2026-10-10: pytest 25 passed, `env_check` matmul succeeded on the RTX 5080 with torch 2.14.1+cu132, and `notebooks/00_setup.py` wrote a synthetic figure plus a provenance file. No training.

## Phase 0b — Citation gaps

Open the primary sources listed at the bottom of `docs/02_RELATED_WORK.md`. Add a bib entry only after the PDF or the publisher page is opened. Rewrite any sentence that would have depended on an unopened paper.

Exit met on 2026-10-27: Network Morphism, GEM, Rolnick et al. on experience replay, OGD, CKA, Goodfellow et al. on a two-task permutation, and iCaRL on iCIFAR-100 are in `paper/refs.bib`, with the opening note in `docs/09_JOURNAL.md`. Rolnick et al. is recorded as a reinforcement-learning method and is not used as a supervised replay formula. Split CIFAR-100's schedule is not frozen, because GEM and iCaRL disagree; that is written in ADR-013.

Compute: none.

## Phase 1 — Substrate

Milestone 1, met on CPU on 2026-10-10: widen by neuron split and identity deepen leave the MLP logits unchanged. Hypothesis covers random small widths. Lineage for a split is written to the ledger. No training claim.

Still inside Phase 1 before the gate: edits that are allowed to change the function (prune, prune-and-regrow) need their own tests, and those tests must not call them function-preserving.

Exit met on 2026-10-28: widen and identity deepen keep the logits, and prune and prune-and-regrow each have a test that the logits change. Prune-and-regrow keeps the width. Multiply-adds for one example are `forward_macs`. Regularization change and consolidate remain ledger names without an edit. Compute: CPU only.

## Phase 2 — Introspection

Each planned diagnostic has a function, a docstring that states the tensor shapes, a unit test on a hand-built tensor with a known value, and a ledger write on a smoke training run of a few steps.

Exit: the smoke ledger contains one row per diagnostic name. Diagnostics whose defining paper is still unopened are not given a "standard" formula from memory; they wait.

Not met. The sine smoke `diagnostics-s0-2337dce6c530-20261028T042552Z` logs dead-unit fraction, singular-entropy rank, gradient norm, gradient dispersion, representation drift, and mean off-diagonal covariance. A Hessian-trace proxy and a plasticity-loss indicator still wait.

Compute: smoke, width 32, one tiny dataset or synthetic tensors. Device `auto`.

## Phase 3 — Controllers and the gate

L0 rules, the random controller, and the shadow gate. L1 is one learned controller, chosen and written into an ADR before it is trained. L2 is in scope only after L0, random, and the gate have tests for both accept and reject.

Exit: a toy objective produces a ledger in which a helpful edit is accepted, a harmful edit is rejected, both rows exist, and the live network's output matches the shadow only after accept.

Compute: toy tensors or a few hundred MNIST rows. Minutes.

## Phase 4 — Tasks and baselines

Loaders for Permuted MNIST, Split CIFAR-100, and the recommended stream after the license is copied into an ADR. All mandatory baselines, smoke configs only.

Exit: one seed, tiny config, each baseline finishes and writes metrics. Published numbers are not claimed yet.

Compute: smoke. CIFAR is a small subset, not the full training schedule.

## Phase 5 — Sanity check

Pick one published baseline number from a paper opened in Phase 0b or Phase 4. Run our implementation at the closest smoke-or-small setting the paper allows. Write the target number, our number, the gap, and the protocol differences.

Exit: a ledger-backed table. A miss stays in the table. No valmorph-versus-baseline claim is made before this exit.

Compute: the smallest setting that the paper's protocol still recognizes. If that setting does not fit in the 0.20 cap, the sanity check is "not run, too large," which is a result, and the paper says so.

## Phase 6 — Confirmatory runs

Only the pre-registered hypotheses. At least five seeds. Bootstrap intervals. Level 1 and Level 2 figures with manifests. The robustness check from H1 if the primary interval is positive.

Exit: `docs/09_JOURNAL.md` states which levels hold. `paper/` can start.

Compute: still the capped, small models. Runs are resumable. A run that needs more than the cap is out of scope, not a reason to raise the cap after seeing a curve.

## Phase 7 — Paper

Write only from manifests. Include the negative-results section even if every hypothesis holds, because failed seeds and rejected edits belong there.

Exit: every number in the PDF resolves to a provenance file whose `synthetic` field is false.
