# %% [markdown]
# # 18 — Freeze the hidden layer
#
# The bandit refused every learning-rate halving. Those trials lasted 8
# shadow steps. This notebook applies a freeze for the rest of training,
# with no gate. One copy stops updating the first hidden layer at step 160,
# when task B starts. Another stops at step 320, when task C starts. The
# output layer keeps learning.
#
# Same three permutations, 160 steps each, width 32. The solid line is no
# edits. One seed. CPU. Not a Level 1 claim.

# %%
import json
import sys

import torch

from valmorph.benchmarks.classify import train_on_tasks, train_with_frozen_layer
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
plain = ValmorphMLP([784, 32, 10]).cpu()
early = clone_net(plain)
late = clone_net(plain)

# %%
run_id = make_run_id("freeze-tasks", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="freeze-tasks",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "freeze-tasks",
            "tasks": 3,
            "steps": task_steps,
            "freeze_at": [160, 320],
        },
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
plain_curves = train_on_tasks(
    plain,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
)
early_curves = train_with_frozen_layer(
    early,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    freeze_at=task_steps,
)
late_curves = train_with_frozen_layer(
    late,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    freeze_at=2 * task_steps,
)
task_a = {
    "no edits": plain_curves[0],
    "freeze at task B": early_curves[0],
    "freeze at task C": late_curves[0],
}
task_c = {
    "no edits": plain_curves[2],
    "freeze at task B": early_curves[2],
    "freeze at task C": late_curves[2],
}
style = {"freeze at task B": "dash", "freeze at task C": "dot"}
switch = [task_steps, 2 * task_steps]
curve_a = plot_diagnostic_traces(
    task_a,
    title="task A probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
curve_c = plot_diagnostic_traces(
    task_c,
    title="task C probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curve_a.write_html(figure_dir / "18_freeze_task_a.html")
save_diagnostic_traces_png(
    task_a,
    figure_dir / "18_freeze_task_a.png",
    title="task A probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
curve_c.write_html(figure_dir / "18_freeze_task_c.html")
save_diagnostic_traces_png(
    task_c,
    figure_dir / "18_freeze_task_c.png",
    title="task C probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
end = 3 * task_steps - 1
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_task_a", plain_curves[0]),
            ("accuracy_task_b", plain_curves[1]),
            ("accuracy_task_c", plain_curves[2]),
            ("accuracy_task_a_freeze_b", early_curves[0]),
            ("accuracy_task_b_freeze_b", early_curves[1]),
            ("accuracy_task_c_freeze_b", early_curves[2]),
            ("accuracy_task_a_freeze_c", late_curves[0]),
            ("accuracy_task_b_freeze_c", late_curves[1]),
            ("accuracy_task_c_freeze_c", late_curves[2]),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="freeze-tasks",
            kind="method",
            statement=(
                "Three tasks, 160 steps. Hidden layer frozen at step 160 or 320. "
                f"Task A ends at {plain_curves[0][end]:.3f}, "
                f"{early_curves[0][end]:.3f}, {late_curves[0][end]:.3f}. "
                f"Task B ends at {plain_curves[1][end]:.3f}, "
                f"{early_curves[1][end]:.3f}, {late_curves[1][end]:.3f}. "
                f"Task C ends at {plain_curves[2][end]:.3f}, "
                f"{early_curves[2][end]:.3f}, {late_curves[2][end]:.3f}."
            ),
            evidence_ref="results/figures/18_freeze_task_a.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "18_freeze_task_a.provenance.json",
    figure="18_freeze_task_a",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
write_provenance(
    figure_dir / "18_freeze_task_c.provenance.json",
    figure="18_freeze_task_c",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "plain": [plain_curves[i][end] for i in range(3)],
            "freeze_b": [early_curves[i][end] for i in range(3)],
            "freeze_c": [late_curves[i][end] for i in range(3)],
            "scales": [early.lr_scale[0], late.lr_scale[0]],
            "synthetic": manifest["synthetic"],
        }
    )
)
curve_a
