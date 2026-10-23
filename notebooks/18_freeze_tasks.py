# %% [markdown]
# # 18 — Freeze the hidden layer
#
# The bandit refused every learning-rate halving. Those trials lasted 8
# shadow steps. This notebook applies a freeze for the rest of training,
# with no gate. One pair stops the first hidden layer, at step 160 or 320.
# The other pair stops the output layer at those same steps. The layer that
# is not frozen keeps learning.
#
# Same three permutations, 160 steps each, width 32. The solid line is no
# edits. Seeds 0-4 also freeze each layer at step 160. The bars are that
# final gap. CPU. Not a Level 1 claim.

# %%
import json
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
early = clone_net(plain)
late = clone_net(plain)
output_early = clone_net(plain)
output_late = clone_net(plain)

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
            "layers": [0, 1],
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
output_early_curves = train_with_frozen_layer(
    output_early,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    freeze_at=task_steps,
    layer=1,
)
output_late_curves = train_with_frozen_layer(
    output_late,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    freeze_at=2 * task_steps,
    layer=1,
)
task_a = {
    "no edits": plain_curves[0],
    "hidden at task B": early_curves[0],
    "hidden at task C": late_curves[0],
    "output at task B": output_early_curves[0],
    "output at task C": output_late_curves[0],
}
task_c = {
    "no edits": plain_curves[2],
    "hidden at task B": early_curves[2],
    "hidden at task C": late_curves[2],
    "output at task B": output_early_curves[2],
    "output at task C": output_late_curves[2],
}
style = {
    "hidden at task B": "dash",
    "hidden at task C": "dot",
    "output at task B": "dashdot",
    "output at task C": "dash",
}
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
seed_rows = []
for seed in range(5):
    torch.manual_seed(seed)
    seeded_tasks = []
    seeded_probes = []
    for offset in range(3):
        permutation = pixel_permutation(seed + offset)
        seeded_tasks.append((apply_permutation(train_rows, permutation), train_labels))
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
        train_with_frozen_layer(
            copies[1],
            seeded_tasks,
            seeded_probes,
            steps_per_task=task_steps,
            batch_size=64,
            lr=0.1,
            freeze_at=task_steps,
        ),
        train_with_frozen_layer(
            copies[2],
            seeded_tasks,
            seeded_probes,
            steps_per_task=task_steps,
            batch_size=64,
            lr=0.1,
            freeze_at=task_steps,
            layer=1,
        ),
    ]
    seed_rows.append([curve[task][end] for curve in trained for task in range(3)])
if seed_rows[0] != [
    plain_curves[0][end],
    plain_curves[1][end],
    plain_curves[2][end],
    early_curves[0][end],
    early_curves[1][end],
    early_curves[2][end],
    output_early_curves[0][end],
    output_early_curves[1][end],
    output_early_curves[2][end],
]:
    raise RuntimeError("seed 0 summary does not match the plotted run")
hidden_a = [row[3] - row[0] for row in seed_rows]
hidden_c = [row[5] - row[2] for row in seed_rows]
output_a = [row[6] - row[0] for row in seed_rows]
output_c = [row[8] - row[2] for row in seed_rows]
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
            ("accuracy_task_a_output_b", output_early_curves[0]),
            ("accuracy_task_b_output_b", output_early_curves[1]),
            ("accuracy_task_c_output_b", output_early_curves[2]),
            ("accuracy_task_a_output_c", output_late_curves[0]),
            ("accuracy_task_b_output_c", output_late_curves[1]),
            ("accuracy_task_c_output_c", output_late_curves[2]),
        )
        for step, value in enumerate(series)
    ]
    + [
        StepMetric(step=seed, split="probe", name=f"final_{label}", value=float(value))
        for seed, row in enumerate(seed_rows)
        for label, value in zip(
            (
                "a_plain",
                "b_plain",
                "c_plain",
                "a_hidden",
                "b_hidden",
                "c_hidden",
                "a_output",
                "b_output",
                "c_output",
            ),
            row,
            strict=True,
        )
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="freeze-tasks",
            kind="method",
            statement=(
                "Three tasks, 160 steps. Hidden or output frozen at step 160 or 320. "
                f"Task A ends at {plain_curves[0][end]:.3f}, "
                f"{early_curves[0][end]:.3f}, {late_curves[0][end]:.3f}, "
                f"{output_early_curves[0][end]:.3f}, {output_late_curves[0][end]:.3f}. "
                f"Task B ends at {plain_curves[1][end]:.3f}, "
                f"{early_curves[1][end]:.3f}, {late_curves[1][end]:.3f}, "
                f"{output_early_curves[1][end]:.3f}, {output_late_curves[1][end]:.3f}. "
                f"Task C ends at {plain_curves[2][end]:.3f}, "
                f"{early_curves[2][end]:.3f}, {late_curves[2][end]:.3f}, "
                f"{output_early_curves[2][end]:.3f}, {output_late_curves[2][end]:.3f}. "
                "Seeds 0-4, freeze at step 160 minus plain. Hidden, task A "
                + ", ".join(f"{gap:+.3f}" for gap in hidden_a)
                + "; task C "
                + ", ".join(f"{gap:+.3f}" for gap in hidden_c)
                + ". Output, task A "
                + ", ".join(f"{gap:+.3f}" for gap in output_a)
                + "; task C "
                + ", ".join(f"{gap:+.3f}" for gap in output_c)
                + "."
            ),
            evidence_ref="results/figures/18_freeze_gaps.provenance.json",
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
seeds = list(range(5))
gap_figure = go.Figure(
    data=[
        go.Bar(name="hidden, task A", x=seeds, y=hidden_a, marker_color=COLOR_CYCLE[0]),
        go.Bar(name="hidden, task C", x=seeds, y=hidden_c, marker_color=COLOR_CYCLE[1]),
        go.Bar(name="output, task A", x=seeds, y=output_a, marker_color=COLOR_CYCLE[2]),
        go.Bar(name="output, task C", x=seeds, y=output_c, marker_color=COLOR_CYCLE[3]),
    ]
)
gap_figure.update_layout(
    barmode="group",
    title="final accuracy minus the no-edit run",
    xaxis_title="seed",
    yaxis_title="accuracy gap",
    template="plotly_white",
)
gap_figure.write_html(figure_dir / "18_freeze_gaps.html")
apply_matplotlib_style()
axes = plt.subplots()[1]
width = 0.18
for index, (name, values, color) in enumerate(
    (
        ("hidden, task A", hidden_a, COLOR_CYCLE[0]),
        ("hidden, task C", hidden_c, COLOR_CYCLE[1]),
        ("output, task A", output_a, COLOR_CYCLE[2]),
        ("output, task C", output_c, COLOR_CYCLE[3]),
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
plt.savefig(figure_dir / "18_freeze_gaps.png")
plt.close()
write_provenance(
    figure_dir / "18_freeze_gaps.provenance.json",
    figure="18_freeze_gaps",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=seeds,
)
print(
    json.dumps(
        {
            "plain": [plain_curves[i][end] for i in range(3)],
            "hidden_b": [early_curves[i][end] for i in range(3)],
            "hidden_c": [late_curves[i][end] for i in range(3)],
            "output_b": [output_early_curves[i][end] for i in range(3)],
            "output_c": [output_late_curves[i][end] for i in range(3)],
            "scales": [early.lr_scale[0], late.lr_scale[0], output_early.lr_scale[1]],
            "hidden_minus_plain_a": hidden_a,
            "hidden_minus_plain_c": hidden_c,
            "output_minus_plain_a": output_a,
            "output_minus_plain_c": output_c,
            "synthetic": manifest["synthetic"],
        }
    )
)
curve_a
