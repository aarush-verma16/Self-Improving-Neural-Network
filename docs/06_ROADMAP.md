# Roadmap

One phase at a time. A phase ends when its exit check is true and `docs/STATE.md` says so. The next phase does not start by widening the model. Smoke stays inside the 0.20 memory cap and finishes in minutes on the M4.

## Phase 0 — Foundation. Exit met on the Windows PC on 2026-10-10: pytest 25 passed, `env_check` matmul succeeded on the RTX 5080 with torch 2.14.1+cu132, and `notebooks/00_setup.py` wrote a synthetic figure plus a provenance file. No training.

## Phase 0b — Citation gaps

Open the primary sources listed at the bottom of `docs/02_RELATED_WORK.md`. Add a bib entry only after the PDF or the publisher page is opened. Rewrite any sentence that would have depended on an unopened paper.

Exit: Network Morphism, GEM, experience replay, one OGD paper, CKA, and one protocol paper each for Permuted MNIST and Split CIFAR-100 are either in `refs.bib` with a journal note, or explicitly dropped with a reason.

Compute: none.

## Phase 1 — Substrate

A small MLP with the v1 edit vocabulary. Identity morphisms (widen with zero extra function change, identity deepen) satisfy output-unchanged under Hypothesis, on CPU, for random inputs and random widths inside a tiny bound.

Exit: property tests pass on CPU. No claim about learning. Lineage rows are produced by the edits that create units.

Compute: CPU only.

## Phase 2 — Introspection

Each planned diagnostic has a function, a docstring that states the tensor shapes, a unit test on a hand-built tensor with a known value, and a ledger write on a smoke training run of a few steps.

Exit: the smoke ledger contains one row per diagnostic name. Diagnostics whose defining paper is still unopened are not given a "standard" formula from memory; they wait.

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
