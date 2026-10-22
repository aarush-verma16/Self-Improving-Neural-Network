# Journal

Narrative log. Empirical claims also go in the ledger `findings` table once a real run exists. This file is safe to write before any run.

## 2026-10-10 — Phase 0 foundation

The repo was empty: `main`, no commits, remote `https://github.com/aarush-verma16/Self-Improving-Neural-Network.git`, upstream `origin/main` missing. Nothing was pushed.

Machine check the day before, confirmed in this session's environment script when it passes: Windows 11 build 26300, Miniforge Python 3.14.7 on the base interpreter, uv not yet installed, torch not yet installed, RTX 5080 (16 GB, compute capability 12.0, driver 610.88, CUDA UMD 13.3). The GPU was idle for compute at that check. The project does not use the Miniforge base. It uses uv and CPython 3.12.

Decisions locked in `docs/07_DECISIONS.md`: package name `valmorph`, MIT license, torch 2.14.1 with the cu132 wheel on Windows, PyPI torch on the M4, memory fraction 0.20, citations only from opened sources.

Related-work pass opened Schmidhuber 1993, Schlag/Irie/Schmidhuber 2021, Nested Learning / HOPE (NeurIPS 2025), Net2Net, splitting steepest descent, Firefly, GROWN, SMGrNN, the Darwin Gödel Machine, EWC, and Dohare et al. on loss of plasticity. GEM, experience replay, OGD, CKA, Network Morphism, and the image-protocol papers were not opened. They are a Phase 0b list, not a bibliography.

Dataset shortlist: Electricity (ELEC2) recommended, license ODC-PDDL on the datahub mirror, still to be confirmed on the OpenML page before download. Weather OpenML 42690 is the CC0 alternate on a mirror. Insects has no license on the River page that was opened.

The M4 (24 GB, 1 TB) is a first-class development machine. Smoke configs are required to run there. Canonical CUDA numbers stay on the 5080 and are not pooled with MPS.

Phase 0 code is the frame only. No substrate, no training loop.

Verification on this PC, after the install:

- `uv run ruff check` clean, `uv run pyright` clean, `uv run pytest` 25 passed.
- `scripts/env_check.py` on `auto`: device `cuda`, NVIDIA GeForce RTX 5080, capability 12.0, torch 2.14.1+cu132, CUDA 13.2, memory fraction 0.2, `matmul_ok` true. Record: `results/env/20261010T150959Z.json`.
- `notebooks/00_setup.py` wrote `results/figures/00_setup_gate_timeline.html`, the png twin, and a provenance file. The manifest is `synthetic: true` and is not citable.
- The cu132 wheel does run a real matmul on this 5080. That answers the open question from the install audit.

Public README and CONTRIBUTING no longer name the lab machines or the phase log. Those stay in this journal and in `docs/STATE.md`. The always-on integrity rule now requires an analytical ledger finding and a visual provenance record for anything that would enter the paper.

The first commit is `cc4b8ec` on `main`. It has not been pushed. The editor commit failed because that process does not have `uv` on PATH, so the hook stopped at `Executable uv not found` before ruff ran. Hooks now call the project `.venv` through `scripts/run_tool.py`.

## 2026-10-10 — Milestone 1, the network

Started the neural network, not another tooling pass. `ValmorphMLP` is a small ReLU MLP. `widen` copies one hidden unit and halves the outgoing weights of the parent and the child. `identity_deepen` inserts a linear identity with no activation. Both leave the logits unchanged.

CPU tests, including a hand-checked case whose logit stays 6.5 and a Hypothesis sweep over small random widths: the full suite passed. A split writes a lineage row. Notebook `notebooks/01_substrate.py` measured a max absolute logit change of about `3e-8` on one CPU draw, which is float32 roundoff, and recorded that as a method finding. This is not a learning result and not Level 1.

Commit from the editor was failing with exit 9009 because every hook started with `python`, and that name is the Windows Store alias. The hook is now `.githooks/pre-commit`, and this repo's local `core.hooksPath` points there. It runs `.venv`'s Python by path.

## 2026-10-11 — Visual working process

The runs are meant to be watched. Each milestone notebook has to show the data and the process while it happens: edits, gate decisions, and learning curves as figures. A final scalar is not enough. Paper figures stay the static copy of those same views, with a provenance manifest.

## 2026-10-19 — Where training runs

Short work stays on CPU: tests, the substrate checks, and `notebooks/02_gate.ipynb`. Those run on the Mac. Heavy training is a switch to the CUDA PC, under the 0.20 memory cap. Say that switch before the run. A short MPS smoke on the Mac stays labeled MPS and is not pooled with CUDA.

## 2026-10-19 — Milestone 2, the gate

The gate trains a copy of the network after a proposed edit and keeps the edit only when probe loss falls. A unit test removes a harmful hidden unit and accepts that prune, and removes a unit the fit needs and rejects that prune. The live network changes only on accept. `notebooks/02_gate.ipynb` is the run to watch: a sine fit, a damaged extra unit, one kept prune, one refused prune.

## 2026-10-19 — Mac setup and the gate run

This machine had no `uv` and the system Python was 3.9.6. Installed uv 0.13.1, pinned Python 3.12, and `uv sync --group dev` used the existing CPython 3.12.6 framework build. Torch resolved to the PyPI 2.14.1 wheel. Local `core.hooksPath` is `.githooks`.

The first `env_check` selected MPS and then failed the matmul: `RuntimeError: invalid low watermark ratio 1.4`. PyTorch's default low watermark is 1.4, and it must be at most the high watermark. Setting only `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2` makes the first MPS allocation illegal. `apply_compute_cap` now sets both watermarks to the configured fraction. The retry wrote `results/env/20261019T071452Z.json`: device `mps`, `matmul_ok` true, result `[[1.0, 2.0], [3.0, 4.0]]`. pytest: 32 passed, 1 skipped (CUDA env check).

`notebooks/00_setup.py` wrote a synthetic timeline. `notebooks/02_gate.py` then ran on CPU. Training loss fell from 0.625 to 0.011 over 60 steps. The damaged unit left the fit near a constant around 11–13 (mean squared error about 144 against the sine). The gate accepted `prune-junk` (probe gain −143.9385) and the live fit came back to the sine (mean squared error about 0.011). It rejected `prune-useful` (probe gain +0.3496), so the live network stayed on the repaired fit. Ledger `gate-s0-2337dce6c530-20261019T071507Z`, finding `gate-toy`, provenance `results/figures/02_gate_timeline.provenance.json` with `synthetic` false. This is the toy gate on one seed. It is not a Level 1 result, and it is not pooled with the CUDA machine.

The user's run `gate-s0-2337dce6c530-20261021T021223Z` matches that: repair gain -143.9385 accepted, useful-unit prune gain +0.3496 refused. Lineage contains only the kept prune. Training loss fell from about 0.62 to near 0. The refused point sits near the zero line on the chart because the kept point is about -144; the ledger value is the one that counts.

## 2026-10-20 — Milestone 3, diagnostics

While the sine net trains, the notebook records three of our own measurements: dead-unit fraction, singular-entropy rank of the hidden activations, and gradient norm. CKA is still not implemented. The user's run `diagnostics-s0-2337dce6c530-20261021T021711Z` ended at dead fraction 0.5, rank 2.327, gradient norm 0.029.

## 2026-10-20 — Milestone 4, one permuted-MNIST task

Going forward, a paper-worthy figure gets a ledger finding and a provenance manifest in the same session. Only `claim` figures are copied into `paper/figures/`. Downloads and run outputs stay gitignored (`data/`, `results/`). The user's notebook 04 run reached probe accuracy 0.634 from 0.224. Loss fell from 2.32 to 1.53. The permuted digits are scrambled. Method check only. Notebook 05 then put a planted damaged unit through the gate: accuracy fell to 0.084 and returned to 0.630 when that unit was removed. The useful-unit prune was refused.

## 2026-10-20 — Milestone 6, proposals from silent units

The controller now looks at the current batch. If a hidden unit is silent, it proposes pruning that unit. The existing gate still keeps the edit only when probe loss falls. The user's run `l0-mnist-s0-2337dce6c530-20261021T033055Z` proposed 2 and accepted 0. Gains were about +1.1e-5 and -7.6e-6, both inside the refusal band. Final probe accuracy was 0.634 with and without those proposals, so the two curves sit on top of each other. Lineage is empty. CPU, synthetic false, seed 0. Removing a silent unit does not lower probe loss, so this gate refuses the cleanup. The static gain chart no longer draws a line through unrelated decisions, and each point is labeled with its gain.

## 2026-10-20 — Milestone 7, widen with matched shadow training

A function-preserving widen has immediate gain 0, so the gate that only looks at the edit itself will refuse it. `train_with_widen_trials` proposes `widen` of hidden unit 0 every 10 steps. The gate then trains the widened copy and an unedited copy for 8 full-batch cross-entropy steps and keeps the widen only when the widened copy has lower probe loss. Shadow training for classification uses cross-entropy. The sine gate still uses mean squared error.

The user's run `widen-mnist-s0-2337dce6c530-20261021T033931Z` proposed 3 and accepted 1. `widen-10` gain +0.00436 and `widen-20` gain +0.00103 were refused. `widen-30` gain -0.000589 was kept, lineage `u0` split to `u42`. Final probe accuracy was 0.634 with no edits and 0.668 with the proposals. CPU, synthetic false, seed 0. The accuracy curves lie on top of each other until the kept step. That keep also adopted the 8 extra steps, so the 0.034 accuracy gap is not a matched-step comparison.

The follow-up stays in notebook 07. A third copy now receives the same extra steps when a widen is kept, and does not widen. The kept step is a dotted line. The extra-steps curve is dashed so it stays visible when it matches the widen curve. A local check of that revision ended at 0.634, 0.668, and 0.668. The widen curve and the extra-steps curve differed by at most 0.004 (steps 31, 36, 37, 38) and finished equal. That revision is commit `dacc942`.

The later user run `widen-mnist-s0-2337dce6c530-20261021T035219Z` has the same three decisions and the same 0.634 / 0.668 accuracies. It has no `step_metrics` table, and the finding sentence is the two-curve sentence. The saved accuracy figure has two lines. Jupyter executed the notebook that was already open, not the file on disk.

The reloaded run is `widen-mnist-s0-2337dce6c530-20261021T035630Z`, git `4a89b86`, CPU, synthetic false, seed 0. Same gate decisions. Final probe accuracy was 0.634 with no edits, 0.668 with extra steps only, and 0.668 with the kept widen. The widen curve and the extra-steps curve differed at steps 31, 36, 37, and 38, by at most 0.004, and finished equal. Provenance `results/figures/07_widen_accuracy.provenance.json`. On this seed the kept unit does not move probe accuracy beyond the 8 extra steps the gate adopted with it. Method finding only.

## 2026-10-20 — Random-edit control

The pre-registered control uses the same proposal times and the same gate. Only the edit is random: prune or widen of one unit in the first hidden layer. `notebooks/08_random_mnist.ipynb` plots that run against no edits and against the unit-0 widen rule. A dashed curve takes the extra shadow steps when a random edit is kept.

The user's run is `random-mnist-s0-2337dce6c530-20261021T040323Z`, git `6ea3afd9-dirty`, CPU, synthetic false, seed 0. Random proposals: prune `u15` at step 10 (gain +0.00720), widen `u0` at step 20 (gain +0.00103), widen `u27` at step 30 (gain +0.000314). All three were refused. The unit-0 rule refused its first two widens and kept `widen-30` (gain -0.000589, lineage `u0` to `u42`). At step 20 both controllers widened `u0`, and the gains match. Probe accuracy was 0.634 for no edits, random edits, and the random extra-steps copy, equal at every step. The widen rule ended at 0.668, and that series separates only from step 30. Provenance `results/figures/08_random_accuracy.provenance.json`. On this seed the gate refuses the random edits, and the only kept edit is the one notebook 07 already matched to extra training steps. Method finding only. Two proposals share steps 20 and 30, so those timeline labels sit on the same point. The ledger values are the record.

The next step from that output is the adoption rule. A kept edit was copying the shadow's extra training into the live network, which is why the accuracy curve moved. `consider_edit` now applies the edit to the weights from before that shadow training. The dashed curve in notebook 07 still takes the extra steps, so the chart separates the unit from the old rule. A local check of that revision kept the same `widen-30` (gain -0.000589). Final probe accuracy was 0.634 with no edits, 0.668 on the dashed extra-steps curve, and 0.634 on the live widen curve.

The user's re-run is `widen-mnist-s0-2337dce6c530-20261021T042444Z`, git `2b9b23e`, CPU, synthetic false, seed 0. Same three decisions and the same lineage, `u0` to `u42`. Final probe accuracy was 0.634, 0.668, and 0.634. The live widen curve differed from no edits at steps 32, 34, 35, and 38, by at most 0.002, and finished equal. Provenance `results/figures/07_widen_accuracy.provenance.json`. On this seed the kept unit does not move probe accuracy. The rise on the dashed curve is the extra steps the live network no longer receives.

## 2026-10-21 — Magnitude prune

A function-preserving widen did not move accuracy, so the next proposal is an edit that can change the function. Every 10 steps the controller offers to prune the hidden unit with the smallest incoming L2 norm, weights and bias together. That definition is ours. The gate is unchanged, and a kept prune is applied at the current weights.

The user's run is `magnitude-mnist-s0-2337dce6c530-20261021T042853Z`, git `46ac3877-dirty`, CPU, synthetic false, seed 0. All three proposals pruned `u2`. Gains were +0.00312, +0.00474, and +0.00578, so probe loss rose and the gate refused each one. There is no lineage table. Probe accuracy was 0.634 for no edits, magnitude prune, and extra steps, equal at every step. Provenance `results/figures/09_magnitude_accuracy.provenance.json`. On this seed the smallest unit is still useful, and a refusal leaves it smallest, so the controller asked for it again.

A refused unit is now skipped for the rest of the run. The user's re-run is `magnitude-mnist-s0-2337dce6c530-20261021T043213Z`, git `3b477ff`, CPU, synthetic false, seed 0. The proposals pruned `u2`, `u3`, and `u24`. Gains were +0.00312, +0.00845, and +0.000030. All refused. No lineage table. Probe accuracy was 0.634 for no edits, magnitude prune, and extra steps, equal at every step. Provenance `results/figures/09_magnitude_accuracy.provenance.json`. On this seed the three smallest units are still useful. Removing them does not lower probe loss.

## 2026-10-21 — Scheduled widen, no gate

Magnitude pruning did not find an edit to keep, and the gated widen did not move accuracy once the shadow steps stayed off the live network. `notebooks/10_scheduled_widen.ipynb` copies hidden unit 0 at steps 10, 20, and 30 and does not ask the gate.

The user's 40-step run is `scheduled-widen-s0-2337dce6c530-20261021T043703Z`, git `922ad864-dirty`, CPU, synthetic false, seed 0. Hidden width ended at 35. Final probe accuracy was 0.634 with no edits and 0.636 with the schedule. The series differ at 24 steps, first at step 10. The largest gap is 0.020 at step 25, where the schedule is ahead, 0.616 against 0.596. Provenance `results/figures/10_scheduled_accuracy.provenance.json`. On this seed the schedule moves the middle of the curve and finishes one probe example ahead. That is not a 2 percentage point final gap.

The right edge of that plot was not a plateau. The same three widens now train through step 80. A local check reproduces step 39 at 0.634 and 0.636. Step 79 is 0.752 and 0.754. The largest gap after step 40 in that check is 0.010.

The later user runs `scheduled-widen-s0-2337dce6c530-20261021T044319Z` and `scheduled-widen-s0-2337dce6c530-20261021T044559Z` both repeat the 40-step result. The 80-step run is `scheduled-widen-s0-2337dce6c530-20261021T044713Z`, git `3207268e-dirty`, CPU, synthetic false, seed 0. Config `steps` is 80 and `until` is 30. Hidden width 35. Step 39 is 0.634 and 0.636. Step 79 is 0.752 with no edits and 0.754 with the schedule. The largest gap remains 0.020 at step 25. After step 40 the largest gap is 0.010, at step 40. Provenance `results/figures/10_scheduled_accuracy.provenance.json`. On this seed the schedule's lead does not grow while both curves climb from about 0.63 to about 0.75.

## 2026-10-21 — Gated widen over 80 steps

Step 40 was mid-training, so the gate is asked again across 80 steps. Every 10 steps the proposal is a widen of unit 0, with 8 matched shadow steps. A kept edit is applied at the current weights.

The user's run is `widen-long-s0-2337dce6c530-20261022T194419Z`, git `a24fd503`, CPU, synthetic false, seed 0. An earlier run the same minute, `20261022T194248Z`, has the same decisions and the same finals. Seven proposals. Steps 10 and 20 were refused (gains +0.00436 and +0.00103). Steps 30, 40, 50, 60, and 70 were kept, gains -0.000589, -0.002145, -0.000955, -0.000393, and -0.000328. Each kept edit splits `u0`, into `u42` through `u46`. Step 39 matches notebook 07: no edits and the live widen are 0.634, and the extra-steps curve is 0.668. Step 79 is 0.752, 0.750, and 0.806. The live widen differs from no edits by at most 0.008 and is behind at 26 steps. Provenance `results/figures/11_widen_long_accuracy.provenance.json`. On this seed the gate keeps the later widens, and the live accuracy does not rise. The 0.054 final gap belongs to the accumulated shadow steps.

## 2026-10-22 — Fixed width from the same start

Five kept widens end at hidden width 37 and do not beat width 32. `notebooks/12_fixed_width.ipynb` copies unit 0 five times before training, so the wider net starts as the same function, and also repeats the schedule that widens at steps 10, 20, and 30.

The user's run is `fixed-width-s0-2337dce6c530-20261022T195241Z`, git `896c3e97-dirty`, CPU, synthetic false, seed 0. Step 0 is 0.224 for all three. Step 39 is 0.634, 0.636, and 0.634. Final probe accuracy is 0.752 at width 32, 0.754 for the schedule that grows to 35, and 0.750 after five copies made before training. The widest gap between width 37 and width 32 is 0.022 at step 14, with the wider net behind. The schedule's widest lead is 0.020 at step 25, the same step as notebook 10. Provenance `results/figures/12_fixed_width_accuracy.provenance.json`. On this seed, starting with the extra units does not beat width 32, and neither does growing into them.

## 2026-10-22 — Two tasks, one pass

One permutation gives the editor nothing old to protect. `notebooks/13_two_tasks.ipynb` trains width 32 for 80 steps on the same permutation as the earlier notebooks, then 80 steps on a second permutation. Both probes are scored every step. A copy with the same starting weights trains only on the second task, and that curve is drawn over the second half. No edits. The user's run is `two-task-s0-2337dce6c530-20261022T200014Z`, git `ba9d4c42-dirty`, CPU, synthetic false, seed 0. Task A is 0.752 at the switch, dips to 0.712 at step 85, and is 0.774 after 80 steps of task B. Task B is 0.808. The fresh net on task B is 0.754, and that dashed curve is still rising. Provenance `results/figures/13_two_task_accuracy.provenance.json` for that run.

The same notebook now trains task B for 160 steps, because the fresh curve had not leveled off. The user's re-run is `two-task-s0-2337dce6c530-20261022T200410Z`, git `ba8be9b3-dirty`, CPU, synthetic false, seed 0. Config `steps` is `[80, 160]`. It matches the local check at every step. Task A is 0.752 at the switch, dips to 0.712 at step 85, and finishes at 0.770. Task B is 0.808 at step 159 and 0.852 at step 239. The fresh net is 0.754 at its step 80 and 0.836 at its step 160. The gap shrinks from 0.054 to 0.016. Provenance `results/figures/13_two_task_accuracy.provenance.json`. On this seed the old task does not stay down, and most of the lead on the new task was a head start.

## 2026-10-22 — A third permutation

Two permutations left the old task above its switch accuracy. `notebooks/14_three_tasks.ipynb` trains three permutations, 160 steps each, width 32, no edits. A fresh copy trains only on the third. A local check, `three-task-s0-2337dce6c530-20261022T200723Z`, leaves task A at 0.840 after its own training, 0.832 after task B, and 0.806 at the end. Task B goes from 0.858 to 0.826. Task C ends at 0.870. The fresh net on task C ends at 0.846.

The user's run is `three-task-s0-2337dce6c530-20261022T205509Z`, git `48b4ea7d-dirty`, CPU, synthetic false, seed 0. It matches that check at every step. Provenance `results/figures/14_three_task_accuracy.provenance.json`. On this seed the third permutation leaves both earlier tasks about 3 points below the accuracy they had at the end of their own training.

## 2026-10-22 — Gate during the three permutations

The plain network's drop is the first place an edit has something old to protect. `notebooks/15_gate_tasks.py` trains the same three tasks. Every 20 steps it proposes a widen of unit 0. The gate scores the probes of every task seen so far, and the shadow trains for 8 steps on the current task only. The run is `gate-tasks-s0-2337dce6c530-20261022T205948Z`, git `9fed6c05-dirty`, CPU, synthetic false, seed 0. The no-edit curves match notebook 14. Twenty-three proposals. Steps 40 and 60 were kept, gains -0.001586 and -0.000503, splits `u0` to `u42` and `u43`. Every proposal from step 80 on was refused, including the ones during the third task. Final probe accuracy is 0.806 and 0.802 on task A, 0.826 and 0.824 on task B, 0.870 and 0.872 on task C. The largest gap on any curve is 0.020. Provenance `results/figures/15_gate_tasks_accuracy.provenance.json`. On this seed the gate keeps two early splits and does not change the drop. Short CPU notebooks are run in the session from here.

## 2026-10-22 — L1 bandit

ADR-011 chooses L1: a two-arm bandit over prune and widen. The unit is a uniform draw. Untried arms go first, then every fourth choice retries the less-used arm, and the rest follow the lower mean validated gain. The random controller draws from the same arms.

`notebooks/16_bandit_tasks.py` trains the three permutations from notebook 14. The run is `bandit-tasks-s0-2337dce6c530-20261022T211018Z`, git `fe292ca4-dirty`, CPU, synthetic false, seed 0. The no-edit curves match notebook 14: task A 0.806, task B 0.826, task C 0.870. The bandit kept five widens, at steps 40, 80, 200, 360, and 460, each with a gain between -0.000150 and -0.000428. It ends at 0.804, 0.824, and 0.868. The random controller kept four edits, including a prune at step 440, and ends at 0.796, 0.826, and 0.868. The bandit pulled prune 6 times and widen 17 times. Mean prune gain is +0.010. Mean widen gain is about +0.00004. Provenance `results/figures/16_bandit_task_a.provenance.json`. On this seed the bandit learns that prune raises probe loss, and neither controller moves the task-A curve off the no-edit line.

## 2026-10-22 — Learning-rate arm

ADR-012 adds a third arm. It multiplies the first hidden layer's learning-rate scale by one half. The forward function is unchanged. Shadow steps and later steps use the scale. The output layer stays at 1.

`notebooks/17_lr_tasks.py` trains the same three permutations. The run is `lr-tasks-s0-2337dce6c530-20261022T212758Z`, git `e4d24f6a-dirty`, CPU, synthetic false, seed 0. The no-edit curves match notebook 14. The bandit proposes the halving at steps 60, 180, and 340. Gains are +0.026, +0.028, and +0.014, all refused. It keeps five widens, the same steps as the two-arm run. Hidden scale stays 1. Finals are 0.804, 0.824, and 0.868. Mean gains are prune +0.013, widen about +0.00005, and halving +0.023. The random controller keeps a halving at step 420, gain -0.000804, so its hidden scale ends at 0.5. Its finals are 0.808, 0.828, and 0.852. Provenance `results/figures/17_lr_task_a.provenance.json`. On this seed, slowing the hidden layer raises probe loss during the shadow trial, and the one accepted halving does not lift the old task.

## 2026-10-20 — Milestone 5, gate on permuted MNIST

The gate now scores digit classification by probe cross-entropy. Notebook 05 trains the same small permuted-MNIST net, breaks it with one added unit, then offers a prune that removes that unit and a prune that removes a useful unit. A local smoke restored accuracy from 0.084 to 0.630 and refused the second prune. The user runs the notebook. Still CPU.
