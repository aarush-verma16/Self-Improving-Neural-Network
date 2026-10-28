# Metrics

Names in the ledger are the strings below. A new name is allowed; a redefined name is not. If a formula changes, the new name gets a suffix and an ADR.

Let `a_{i,j}` be test accuracy on task `j` measured after training on task `i`. `T` is the number of tasks. The stream uses the same letters with a pre-registered window in place of a task. The window length is part of the config, set before the run.

## Performance

Average accuracy after the last task:

```
A_T = (1 / T) * sum_{j=1..T} a_{T,j}
```

Ledger name: `average_accuracy`, split `test`.

Forgetting of task `j` (j < T):

```
f_j = max_{i in j..T-1} a_{i,j} - a_{T,j}
```

Average forgetting:

```
F = (1 / (T - 1)) * sum_{j=1..T-1} f_j
```

Ledger name: `forgetting`. Higher means more forgotten.

Backward transfer:

```
BWT = (1 / (T - 1)) * sum_{j=1..T-1} (a_{T,j} - a_{j,j})
```

Ledger name: `bwt_end_minus_diag`. Negative when the model is worse on old tasks than it was at the end of those tasks. This is the same expression as Lopez-Paz and Ranzato, equation 3. The code lives in `valmorph.metrics`.

Learning accuracy: `a_{j,j}` averaged over tasks. Ledger name: `learning_accuracy`.

Few-shot plasticity: accuracy on the new task after `K` training steps, `K` fixed in the config. Ledger name: `plasticity_at_k`.

## Cost

`param_count`: trainable parameters after each accepted edit.

`forward_flops`: multiply-adds for one example at the current architecture. `ValmorphMLP.forward_macs` counts `out * in` for each linear map and leaves out the bias adds.

## Editor

Acceptance rate: accepted proposals divided by all proposals. Rejections stay in the denominator. Ledger name: `acceptance_rate`.

Validated gain: `probe_after - probe_before` on the shadow copy. The ledger writer rejects a row where this identity fails. Report the mean over accepted edits and, separately, over rejected edits.

Edit efficiency: validated gain divided by parameters added. Zero-parameter edits (identity deepen, freeze, a learning-rate change) are reported as gain per edit, not divided by zero.

Level 1 curve: `average_accuracy` against environment step, one line per method, with `param_count` available at the same steps.

Level 2 curve: validated gain and acceptance rate against proposal index. The early third and the late third are pre-registered in `docs/01_RESEARCH_QUESTIONS.md`. A rolling mean is a display choice. The test uses the thirds, not a rolling window chosen after looking at the plot.

## Intervals

Percentile bootstrap, 10,000 resamples, seed is the unit, 95% interval. The primary family of four comparisons is Bonferroni-adjusted. Raw intervals are shown beside adjusted ones. Effect size is the paired difference of seed-level means, not a p-value alone.

## Figures that must exist before a claim

Each of these is exported by a notebook cell with a provenance manifest:

- Level 1 self-improvement curve against the mandatory controls
- Level 2 editor curve, gated versus random
- Edit timeline (accept and reject)
- Forgetting matrix `a_{i,j}`
- Diagnostic heatmap over layers and time
- Neuron lineage for one seed, as a tree
- Topology over time, as an animation, for one seed

A claim in `docs/` or in `paper/` names the manifest path and the ledger hash. A synthetic manifest does not qualify. `assert_citable` enforces that in code.
