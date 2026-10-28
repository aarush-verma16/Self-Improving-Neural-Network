# Architecture

Phase 0 implements the frame: device, seed, config, memory cap, ledger, provenance. Models, introspection, controllers, and the gate now have code. `metrics` holds the task-matrix summaries. `baselines` is still empty.

## Layout

```
src/valmorph/
  core/           device, seed, config, compute cap, ledger, env check
  models/         Phase 1 substrate
  introspect/     Phase 2 diagnostics
  controllers/    Phase 3, including random
  gate/           Phase 3 shadow trial
  benchmarks/     Phase 4 tasks
  baselines/      Phase 4 controls
  metrics/        Phase 4 formulas in code
  viz/            Plotly exploration, matplotlib paper figures, manifests
```

Notebooks in `notebooks/` only call this package. Configs in `configs/` are YAML against `RunConfig`. Evidence lives in `results/`, which is gitignored. A paper figure's provenance file records the ledger content hash so the claim can be checked against the machine that produced it.

## Devices

`resolve_device` accepts `auto`, `cuda`, `mps`, and `cpu`. `auto` prefers CUDA, then MPS, then CPU. An explicit request never falls back. A Mac run that asks for `cuda` fails. A Windows run that asks for `mps` fails. That failure is preferable to a silent device change.

`apply_compute_cap` runs before the first tensor on that device. CUDA uses `torch.cuda.set_per_process_memory_fraction`. MPS sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` to the same fraction. The low ratio has to be at most the high ratio, and PyTorch's default low ratio is 1.4, so the high ratio cannot be lowered alone. CPU still rejects an illegal fraction, and otherwise does nothing, because there is no CUDA-style allocator cap.

## Ledger

Version 1. One directory per run:

| File | What a row is |
| --- | --- |
| `run.json` | Machine, seed, config hash, git commit, library versions, synthetic flag |
| `step_metrics.parquet` | One finite scalar, split in {train, probe, val, test} |
| `diagnostics.parquet` | One finite scalar, with a layer name or an empty layer |
| `edit_proposals.parquet` | One proposal. Controller in {L0, L1, L2, random} |
| `gate_decisions.parquet` | One accept or reject. Gain must equal probe_after − probe_before |
| `lineage.parquet` | One parent unit to one child unit |
| `findings.parquet` | One observation, negative result, claim, or method note |

`assert_same_machine` refuses to pool runs that differ in device type, device name, torch version, or platform. Synthetic runs use device type `synthetic` and are refused unless the caller passes `allow_synthetic=True`. They still cannot be pooled with a real run.

Edit types in v1: `widen`, `identity_deepen`, `prune`, `prune_regrow`, `lr_change`, `regularization_change`, `consolidate`, `freeze`. Adding a type is an ADR plus a vocabulary change, not a silent string.

## Gate, when it exists

A proposal is applied to a copy. The copy is scored on the probe split only. The live network changes only after an accept. The proposal row is written before the decision row, including when the decision is reject. Validation may select a checkpoint under a rule written in the config before the run. Test is touched once, after that rule has been applied.

## Figures

Exploration figures are Plotly. Paper figures are matplotlib or seaborn, Okabe–Ito, and each export writes a provenance JSON: git commit, config hash, seeds, ledger version, ledger content hashes, device. `assert_citable` raises on a synthetic manifest. The setup notebook's figure is synthetic and fails that check on purpose.

## Resuming

`Ledger.reopen` appends. Training loops in later phases write the step they reached into `step_metrics` so a crash continues from the last step rather than from zero. A resumed run keeps the original `run_id`, config hash, and device. A resume on a different machine is a new run.
