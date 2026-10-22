# %% [markdown]
# # 15 — The gate across three permutations
#
# Notebook 14 trained three permutations with no edits. Task A went from
# 0.840 after its own training to 0.806 at the end. Task B went from 0.858
# to 0.826. This notebook trains that same sequence again and, every 20
# steps, proposes a widen of unit 0. The gate scores the probes of every
# task seen so far. Shadow training uses only the current task, for 8 steps.
# A kept widen is applied at the current weights.
#
# The solid curves are the no-edit network. The dashed curves are the gate.
# One seed. CPU. Not Level 1.

# %%
import json
import sys

import torch

from valmorph.benchmarks.classify import train_on_tasks
from valmorph.benchmarks.mnist import (
    apply_permutation,
    download_mnist,
    load_split,
    pixel_permutation,
)
from valmorph.controllers import train_across_tasks
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import LEDGER_VERSION, Finding, Ledger, RunMeta, StepMetric, make_run_id
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.gate import clone_net
from valmorph.models import ValmorphMLP
from valmorph.viz import (
    plot_diagnostic_traces,
    plot_gate_timeline,
    save_diagnostic_traces_png,
    save_gate_timeline_png,
    write_provenance,
)

root = repo_root()
cfg = load_run_config(root / "configs" / "setup.yaml")
torch.manual_seed(cfg.seed)
mnist_dir = download_mnist(root / "data" / "mnist")
train_x, train_y = load_split(mnist_dir, "train")
test_x, test_y = load_split(mnist_dir, "test")
train_rows = train_x[:2000]
train_labels = train_y[:2000]
probe_rows = test_x[:500]
probe_labels = test_y[:500]
tasks = []
probes = []
for permutation_seed in (cfg.seed, cfg.seed + 1, cfg.seed + 2):
    permutation = pixel_permutation(permutation_seed)
    tasks.append((apply_permutation(train_rows, permutation), train_labels))
    probes.append((apply_permutation(probe_rows, permutation), probe_labels))
task_steps = 160
plain = ValmorphMLP([784, 32, 10]).cpu()
gated = clone_net(plain)

# %%
run_id = make_run_id("gate-tasks", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="gate-tasks",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "gate-tasks", "tasks": 3, "steps": task_steps, "every": 20},
        git_commit=git_commit(),
        device_type="cpu",
        device_name="cpu",
        device_index=None,
        capability=None,
        torch_version=library_versions()["torch"],
        cuda_version=None,
        python_version=sys.version,
        platform=platform_string(),
        libraries=library_versions(),
    ),
)
gated_a, gated_b, gated_c = train_across_tasks(
    gated,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    ledger=ledger,
    every=20,
    shadow_steps=8,
)
plain_a, plain_b, plain_c = train_on_tasks(
    plain,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
)
accuracy_series = {
    "task A": plain_a,
    "task A, gate": gated_a,
    "task B": plain_b,
    "task B, gate": gated_b,
    "task C": plain_c,
    "task C, gate": gated_c,
}
gate_dashes = {"task A, gate": "dash", "task B, gate": "dash", "task C, gate": "dash"}
switch = [task_steps, 2 * task_steps]
curves = plot_diagnostic_traces(
    accuracy_series,
    title="probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=gate_dashes,
)
decisions = ledger.read("gate_decisions")
timeline = plot_gate_timeline(decisions, synthetic=False)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curves.write_html(figure_dir / "15_gate_tasks_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "15_gate_tasks_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=gate_dashes,
)
timeline.write_html(figure_dir / "15_gate_tasks_timeline.html")
save_gate_timeline_png(decisions, figure_dir / "15_gate_tasks_timeline.png", synthetic=False)
end_c = 3 * task_steps - 1
kept = sum(1 for flag in decisions.column("accepted").to_pylist() if flag)
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_task_a", plain_a),
            ("accuracy_task_b", plain_b),
            ("accuracy_task_c", plain_c),
            ("accuracy_task_a_gate", gated_a),
            ("accuracy_task_b_gate", gated_b),
            ("accuracy_task_c_gate", gated_c),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="gate-tasks",
            kind="method",
            statement=(
                f"Three tasks, 160 steps, widen every 20. Kept {kept} of "
                f"{decisions.num_rows}. "
                f"Task A ends at {plain_a[end_c]:.3f} with no edits and "
                f"{gated_a[end_c]:.3f} with the gate. "
                f"Task B ends at {plain_b[end_c]:.3f} and {gated_b[end_c]:.3f}. "
                f"Task C ends at {plain_c[end_c]:.3f} and {gated_c[end_c]:.3f}."
            ),
            evidence_ref="results/figures/15_gate_tasks_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "15_gate_tasks_accuracy.provenance.json",
    figure="15_gate_tasks_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
write_provenance(
    figure_dir / "15_gate_tasks_timeline.provenance.json",
    figure="15_gate_tasks_timeline",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "kept": kept,
            "proposed": decisions.num_rows,
            "plain": [plain_a[end_c], plain_b[end_c], plain_c[end_c]],
            "gate": [gated_a[end_c], gated_b[end_c], gated_c[end_c]],
            "width": int(gated.linear(0).out_features),
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
