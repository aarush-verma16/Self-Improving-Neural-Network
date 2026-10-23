# %% [markdown]
# # 19 — Old rows in the minibatch
#
# Freezing the hidden layer made task A fall further. The output layer kept
# training on the new permutation only. This notebook keeps earlier training
# rows in later minibatches. Half of each later batch comes from those rows.
# One copy stores every previous training row. One copy stores the first 256
# rows of each finished task, in their existing order.
#
# This mix is ours. It is not a published replay method. Same three
# permutations, 160 steps each, width 32, no gate. One seed. CPU.

# %%
import json
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch

from valmorph.benchmarks.classify import train_on_tasks, train_with_rehearsal
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
from valmorph.viz.theme import COLOR_CYCLE, apply_matplotlib_style

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
full = clone_net(plain)
short = clone_net(plain)

# %%
run_id = make_run_id("rehearsal-tasks", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="rehearsal-tasks",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "rehearsal-tasks",
            "tasks": 3,
            "steps": task_steps,
            "memory": [None, 256],
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
full_curves = train_with_rehearsal(
    full,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    memory=None,
)
short_curves = train_with_rehearsal(
    short,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    memory=256,
)
task_a = {
    "no edits": plain_curves[0],
    "all previous rows": full_curves[0],
    "256 rows per task": short_curves[0],
}
task_c = {
    "no edits": plain_curves[2],
    "all previous rows": full_curves[2],
    "256 rows per task": short_curves[2],
}
style = {"all previous rows": "dash", "256 rows per task": "dot"}
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
curve_a.write_html(figure_dir / "19_rehearsal_task_a.html")
save_diagnostic_traces_png(
    task_a,
    figure_dir / "19_rehearsal_task_a.png",
    title="task A probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
curve_c.write_html(figure_dir / "19_rehearsal_task_c.html")
save_diagnostic_traces_png(
    task_c,
    figure_dir / "19_rehearsal_task_c.png",
    title="task C probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
end = 3 * task_steps - 1
own = task_steps - 1
seed_rows = []
for seed in range(5):
    torch.manual_seed(seed)
    seeded_tasks = []
    for offset in range(3):
        permutation = pixel_permutation(seed + offset)
        seeded_tasks.append((apply_permutation(train_rows, permutation), train_labels))
    seeded_probes = []
    for offset in range(3):
        permutation = pixel_permutation(seed + offset)
        seeded_probes.append((apply_permutation(probe_rows, permutation), probe_labels))
    base = ValmorphMLP([784, 32, 10]).cpu()
    copies = [base, clone_net(base), clone_net(base)]
    trained = [
        train_on_tasks(
            copies[0],
            seeded_tasks,
            seeded_probes,
            steps_per_task=task_steps,
            batch_size=64,
            lr=0.1,
        ),
        train_with_rehearsal(
            copies[1],
            seeded_tasks,
            seeded_probes,
            steps_per_task=task_steps,
            batch_size=64,
            lr=0.1,
            memory=None,
        ),
        train_with_rehearsal(
            copies[2],
            seeded_tasks,
            seeded_probes,
            steps_per_task=task_steps,
            batch_size=64,
            lr=0.1,
            memory=256,
        ),
    ]
    seed_rows.append([curve[task][end] for curve in trained for task in range(3)])
if seed_rows[0] != [
    plain_curves[0][end],
    plain_curves[1][end],
    plain_curves[2][end],
    full_curves[0][end],
    full_curves[1][end],
    full_curves[2][end],
    short_curves[0][end],
    short_curves[1][end],
    short_curves[2][end],
]:
    raise RuntimeError("seed 0 summary does not match the plotted run")
labels = ("a_plain", "b_plain", "c_plain", "a_full", "b_full", "c_full", "a_256", "b_256", "c_256")
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_task_a", plain_curves[0]),
            ("accuracy_task_b", plain_curves[1]),
            ("accuracy_task_c", plain_curves[2]),
            ("accuracy_task_a_full", full_curves[0]),
            ("accuracy_task_b_full", full_curves[1]),
            ("accuracy_task_c_full", full_curves[2]),
            ("accuracy_task_a_256", short_curves[0]),
            ("accuracy_task_b_256", short_curves[1]),
            ("accuracy_task_c_256", short_curves[2]),
        )
        for step, value in enumerate(series)
    ]
    + [
        StepMetric(step=seed, split="probe", name=f"final_{label}", value=float(value))
        for seed, row in enumerate(seed_rows)
        for label, value in zip(labels, row, strict=True)
    ]
)
full_a = [row[3] - row[0] for row in seed_rows]
full_c = [row[5] - row[2] for row in seed_rows]
short_a = [row[6] - row[0] for row in seed_rows]
short_c = [row[8] - row[2] for row in seed_rows]
ledger.append_findings(
    [
        Finding(
            finding_id="rehearsal-tasks",
            kind="method",
            statement=(
                "Three tasks, 160 steps. Half of each later minibatch is old rows. "
                f"Seed 0 task A ends at {plain_curves[0][end]:.3f}, "
                f"{full_curves[0][end]:.3f}, {short_curves[0][end]:.3f}. "
                f"Task B ends at {plain_curves[1][end]:.3f}, "
                f"{full_curves[1][end]:.3f}, {short_curves[1][end]:.3f}. "
                f"Task C ends at {plain_curves[2][end]:.3f}, "
                f"{full_curves[2][end]:.3f}, {short_curves[2][end]:.3f}. "
                f"Task A after its own training is {plain_curves[0][own]:.3f}. "
                "Seeds 0-4, full-memory minus plain, task A "
                + ", ".join(f"{gap:+.3f}" for gap in full_a)
                + "; task C "
                + ", ".join(f"{gap:+.3f}" for gap in full_c)
                + ". Memory 256 minus plain, task A "
                + ", ".join(f"{gap:+.3f}" for gap in short_a)
                + "; task C "
                + ", ".join(f"{gap:+.3f}" for gap in short_c)
                + "."
            ),
            evidence_ref="results/figures/19_rehearsal_gaps.provenance.json",
        )
    ]
)
seeds = list(range(5))
gap_figure = go.Figure(
    data=[
        go.Bar(name="all rows, task A", x=seeds, y=full_a, marker_color=COLOR_CYCLE[0]),
        go.Bar(name="all rows, task C", x=seeds, y=full_c, marker_color=COLOR_CYCLE[1]),
        go.Bar(name="256 rows, task A", x=seeds, y=short_a, marker_color=COLOR_CYCLE[2]),
        go.Bar(name="256 rows, task C", x=seeds, y=short_c, marker_color=COLOR_CYCLE[3]),
    ]
)
gap_figure.update_layout(
    barmode="group",
    title="final accuracy minus the no-edit run",
    xaxis_title="seed",
    yaxis_title="accuracy gap",
    template="plotly_white",
)
gap_figure.write_html(figure_dir / "19_rehearsal_gaps.html")
apply_matplotlib_style()
axes = plt.subplots()[1]
width = 0.18
for index, (name, values, color) in enumerate(
    (
        ("all rows, task A", full_a, COLOR_CYCLE[0]),
        ("all rows, task C", full_c, COLOR_CYCLE[1]),
        ("256 rows, task A", short_a, COLOR_CYCLE[2]),
        ("256 rows, task C", short_c, COLOR_CYCLE[3]),
    )
):
    axes.bar(
        [seed + (index - 1.5) * width for seed in seeds],
        values,
        width=width,
        label=name,
        color=color,
    )
axes.axhline(0.0, color="black", linewidth=0.8)
axes.set_title("final accuracy minus the no-edit run")
axes.set_xlabel("seed")
axes.set_ylabel("accuracy gap")
axes.legend()
plt.tight_layout()
plt.savefig(figure_dir / "19_rehearsal_gaps.png")
plt.close()
manifest = write_provenance(
    figure_dir / "19_rehearsal_task_a.provenance.json",
    figure="19_rehearsal_task_a",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
write_provenance(
    figure_dir / "19_rehearsal_task_c.provenance.json",
    figure="19_rehearsal_task_c",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
write_provenance(
    figure_dir / "19_rehearsal_gaps.provenance.json",
    figure="19_rehearsal_gaps",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=list(range(5)),
)
print(
    json.dumps(
        {
            "plain": [plain_curves[i][end] for i in range(3)],
            "full": [full_curves[i][end] for i in range(3)],
            "memory_256": [short_curves[i][end] for i in range(3)],
            "task_a_own": plain_curves[0][own],
            "prefix_match": plain_curves[0][:task_steps] == full_curves[0][:task_steps],
            "full_minus_plain_a": full_a,
            "full_minus_plain_c": full_c,
            "memory256_minus_plain_a": short_a,
            "memory256_minus_plain_c": short_c,
            "synthetic": manifest["synthetic"],
        }
    )
)
curve_a
