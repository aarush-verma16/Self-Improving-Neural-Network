# %% [markdown]
# # 13 — Two tasks, one pass
#
# Notebooks 07 through 12 trained one permutation. Extra units, a growth
# schedule, and the gate's kept widens all finished within a few probe
# examples of width 32. One task does not ask the network to keep an old input.
#
# This is the two-task smoke from the protocol: one pass, width 32, no edits.
# Task A uses the same pixel permutation as the earlier notebooks. Task B uses
# a second permutation. The network trains 80 steps on A, then 80 on B, and
# both probes are scored every step. A second copy starts from the same
# weights and trains only on B. Its curve is drawn across the second half.
#
# Two tasks is the smoke size. It is not a published task count. Not Level 1.

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
for permutation_seed in (cfg.seed, cfg.seed + 1):
    permutation = pixel_permutation(permutation_seed)
    tasks.append((apply_permutation(train_rows, permutation), train_labels))
    probes.append((apply_permutation(probe_rows, permutation), probe_labels))
steps_per_task = 80
net = ValmorphMLP([784, 32, 10]).cpu()
fresh = clone_net(net)

# %%
run_id = make_run_id("two-task", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="two-task",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "two-task", "tasks": 2, "steps_per_task": steps_per_task},
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
on_a, on_b = train_on_tasks(
    net,
    tasks,
    probes,
    steps_per_task=steps_per_task,
    batch_size=64,
    lr=0.1,
)
_loss, fresh_b = train_classifier(
    fresh,
    tasks[1][0],
    tasks[1][1],
    steps=steps_per_task,
    batch_size=64,
    lr=0.1,
    probe_images=probes[1][0],
    probe_labels=probes[1][1],
)
fresh_aligned = [float("nan")] * steps_per_task + fresh_b
accuracy_series = {
    "task A": on_a,
    "task B": on_b,
    "task B, fresh start": fresh_aligned,
}
curves = plot_diagnostic_traces(
    accuracy_series,
    title="probe accuracy",
    y_title="accuracy",
    marks=[steps_per_task],
    mark_label="task 2",
    dashes={"task B, fresh start": "dash"},
)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curves.write_html(figure_dir / "13_two_task_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "13_two_task_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
    marks=[steps_per_task],
    mark_label="task 2",
    dashes={"task B, fresh start": "dash"},
)
end_a = steps_per_task - 1
end_b = 2 * steps_per_task - 1
dip = min(on_a[steps_per_task:])
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (("accuracy_task_a", on_a), ("accuracy_task_b", on_b))
        for step, value in enumerate(series)
    ]
    + [
        StepMetric(
            step=steps_per_task + step,
            split="probe",
            name="accuracy_task_b_fresh",
            value=float(value),
        )
        for step, value in enumerate(fresh_b)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="two-task",
            kind="method",
            statement=(
                "Two tasks, 80 steps each, no edits. "
                f"Task A is {on_a[end_a]:.3f} at the switch, dips to {dip:.3f}, "
                f"and finishes at {on_a[end_b]:.3f}. "
                f"Task B finishes at {on_b[end_b]:.3f}. "
                f"A fresh net on task B finishes at {fresh_b[-1]:.3f}."
            ),
            evidence_ref="results/figures/13_two_task_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "13_two_task_accuracy.provenance.json",
    figure="13_two_task_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "task_a_end": on_a[end_a],
            "task_a_final": on_a[end_b],
            "task_b_final": on_b[end_b],
            "fresh_b_final": fresh_b[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
