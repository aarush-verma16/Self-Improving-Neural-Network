# %% [markdown]
# # 14 — A third permutation
#
# Notebook 13 trained two permutations, 80 steps then 160. Task A dipped from
# 0.752 to 0.712 and finished at 0.770. Task B finished at 0.852. A fresh net
# on task B finished at 0.836. The early lead was a head start, and the old
# task did not stay down.
#
# This notebook adds a third permutation. Each task trains for 160 steps.
# Width 32, no edits. A fresh copy trains only on the third task, and its
# curve starts at the second switch.
#
# Three tasks is the next smoke. It is not a published task count. Not Level 1.

# %%
import json
import sys

import torch

from valmorph.benchmarks.classify import train_classifier, train_on_tasks
from valmorph.benchmarks.mnist import (
    apply_permutation,
    download_mnist,
    load_split,
    pixel_permutation,
)
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import LEDGER_VERSION, Finding, Ledger, RunMeta, StepMetric, make_run_id
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.gate import clone_net
from valmorph.models import ValmorphMLP
from valmorph.viz import plot_diagnostic_traces, save_diagnostic_traces_png, write_provenance

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
net = ValmorphMLP([784, 32, 10]).cpu()
fresh = clone_net(net)

# %%
run_id = make_run_id("three-task", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="three-task",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "three-task", "tasks": 3, "steps": task_steps},
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
on_a, on_b, on_c = train_on_tasks(
    net,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
)
_loss, fresh_c = train_classifier(
    fresh,
    tasks[2][0],
    tasks[2][1],
    steps=task_steps,
    batch_size=64,
    lr=0.1,
    probe_images=probes[2][0],
    probe_labels=probes[2][1],
)
fresh_aligned = [float("nan")] * (2 * task_steps) + fresh_c
accuracy_series = {
    "task A": on_a,
    "task B": on_b,
    "task C": on_c,
    "task C, fresh start": fresh_aligned,
}
switch = [task_steps, 2 * task_steps]
curves = plot_diagnostic_traces(
    accuracy_series,
    title="probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes={"task C, fresh start": "dash"},
)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curves.write_html(figure_dir / "14_three_task_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "14_three_task_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes={"task C, fresh start": "dash"},
)
end_a = task_steps - 1
end_b = 2 * task_steps - 1
end_c = 3 * task_steps - 1
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_task_a", on_a),
            ("accuracy_task_b", on_b),
            ("accuracy_task_c", on_c),
        )
        for step, value in enumerate(series)
    ]
    + [
        StepMetric(
            step=2 * task_steps + step,
            split="probe",
            name="accuracy_task_c_fresh",
            value=float(value),
        )
        for step, value in enumerate(fresh_c)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="three-task",
            kind="method",
            statement=(
                "Three tasks, 160 steps each, no edits. "
                f"Task A is {on_a[end_a]:.3f} after its own training, "
                f"{on_a[end_b]:.3f} after task B, and {on_a[end_c]:.3f} at the end. "
                f"Task B is {on_b[end_b]:.3f} after its own training "
                f"and {on_b[end_c]:.3f} at the end. "
                f"Task C ends at {on_c[end_c]:.3f}. "
                f"A fresh net on task C ends at {fresh_c[-1]:.3f}."
            ),
            evidence_ref="results/figures/14_three_task_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "14_three_task_accuracy.provenance.json",
    figure="14_three_task_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "task_a_own": on_a[end_a],
            "task_a_final": on_a[end_c],
            "task_b_own": on_b[end_b],
            "task_b_final": on_b[end_c],
            "task_c_final": on_c[end_c],
            "fresh_c_final": fresh_c[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
