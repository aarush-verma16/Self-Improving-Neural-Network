# Experiment protocol

No confirmatory run starts until `docs/01_RESEARCH_QUESTIONS.md` still matches the config that will be used. If it does not, write an ADR first.

## Machines

| Machine | Role | Device | Memory cap |
| --- | --- | --- | --- |
| Windows PC, RTX 5080, 16 GB, cc 12.0 | Canonical CUDA results | `cuda` | 0.20 of VRAM |
| MacBook Pro, M4, 24 GB, 1 TB | Full development, MPS results labeled as MPS | `mps` | 0.20 watermark |

`configs/setup.yaml` is the Phase 0 check: it resolves `auto`, so the 5080 run records CUDA and the M4 run records MPS. Those two JSON files are not a comparison.

Record in every run header: device type and name, compute capability when CUDA, torch version, CUDA version when present, Python, OS, git commit, config hash, seed, library versions. `scripts/env_check.py` is the first command on a new machine. If the matmul fails, stop. Do not fall back to another CUDA wheel inside the script.

Torch pinning, decided in ADR-003: 2.14.1+cu132 on Windows and Linux, and the PyPI wheel on macOS. Checked against the PyTorch install selector on 2026-10-09. There is no 2.14.1 cu128 wheel. On 2026-10-10, `scripts/env_check.py` on this 5080 reported capability 12.0, torch 2.14.1+cu132, and `matmul_ok: true`.

## Splits

For each task, freeze three disjoint subsets before training: train, probe, and test. Validation, if used, is a fourth disjoint subset. The probe is the gate's only acceptance signal. Test is not an input to the controller, the gate, or checkpoint selection. The correlation of probe gain with validation gain is a logged diagnostic (`probe_val_gain_correlation`). If it collapses, say so. Do not quietly retune the probe.

## Smoke, then confirmatory

Every later phase ships a CPU smoke that:

- uses a tiny width (32 unless a phase ADR says otherwise),
- uses two tasks or a few hundred stream rows,
- finishes in minutes,
- writes a real ledger.

The heavy run uses the same code with `device: cuda` on the CUDA machine, or `device: mps` on the Mac. It stays inside the 0.20 cap, uses the pre-registered width and step budget, and at least five seeds (ten when a smoke shows the run is cheap). It is resumable. It is not started in the same session as an untested code change. It is not left on the CPU.

## Image protocols

Permuted MNIST for H1 is ADR-013: Lopez-Paz and Ranzato, NIPS 2017, section 4.1. Twenty tasks, 1000 examples each, one pass, two hidden layers of 100 ReLU units, minibatch 10, test-partition evaluation. The three-task width-32 notebooks are smokes, not that protocol. The 20-task run has not been started. Published accuracies from that paper are not the sanity-check target.

Split CIFAR-100 is not frozen. GEM's incremental CIFAR-100 and iCaRL's iCIFAR-100, both opened on 2026-10-27, use different batch sizes, architectures, and epoch counts. ADR-013 records both and does not choose.

## Stream

Shortlist opened on 2026-10-10:

1. Electricity (ELEC2). About 45,312 rows, Australian NSW wholesale prices, label UP/DOWN against a 24-hour moving average. The datahub.io mirror states license ODC-PDDL. This is the recommended dataset: real drift, small enough for the M4 and for the GPU cap, long history as a stream benchmark.
2. OpenML weather, dataset id 42690. A GitLab mirror of that OpenML entry carries a CC0 license file. Second choice if Electricity's license on the OpenML page itself is not a free license.
3. Insects, via RiverML, Souza et al. 2020. The River documentation was opened and does not state a license. Do not download it until a license is found.

Covertype is a larger backup. OpenML's documentation uses it as an example uploaded under CC0. That example is not the dataset's own license field. Read the field before any download.

Phase 4 download rule: open the dataset page, copy the license string into the run notes and into an ADR, and stop if the license is not one we can ship with an open-source repo (MIT code, data under its own free license, data not committed if it is large).

## Baselines

Implemented in Phase 4, smoke-tested before any comparison: naive sequential, fixed-size at the start width, fixed-size at the maximum allowed width, EWC, experience replay, fixed growth schedule, magnitude pruning, random-edit controller.

The random controller proposes from the same edit vocabulary as the gated controller, at the same proposal times, and goes through the same gate. Only the choice of edit is random.

## Statistics

See `docs/01_RESEARCH_QUESTIONS.md` and `docs/05_METRICS.md`. Seed is the resample unit. Do not bootstrap minibatches and call them independent samples.

## Failures

A crashed run is reopened or marked failed in `findings` with kind `negative`. Deleting a bad ledger to tidy the directory is not allowed. `results/` is local; the finding and the journal entry are what the next session sees.
