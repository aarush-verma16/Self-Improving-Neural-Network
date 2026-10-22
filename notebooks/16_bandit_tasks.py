# %% [markdown]
# # 16 — The bandit chooses the edit
#
# ADR-011 makes L1 a two-arm bandit over prune and widen. The unit is a
# uniform draw. The reward is the gate's validated gain. The random
# controller uses the same two arms and draws units the same way. Only the
# arm choice differs.
#
# Three permutations, 160 steps each, the same slice as notebook 14. The
# gate scores every task probe seen so far. A proposal is made every 20
# steps. Solid lines have no edits. Dashed lines are the bandit. Dotted
# lines are the random controller.
#
# One seed. CPU. Not a Level 1 claim.

# %%
import json
import sys

import pyarrow as pa
import torch

from valmorph.benchmarks.classify import train_on_tasks
from valmorph.benchmarks.mnist import (
    apply_permutation,
    download_mnist,
    load_split,
    pixel_permutation,
)
from valmorph.controllers import EditBandit, choose_edit, train_across_tasks
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import (
    LEDGER_VERSION,
    Finding,
    GateDecision,
    Ledger,
    RunMeta,
    StepMetric,
    make_run_id,
)
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
bandit_net = clone_net(plain)
random_net = clone_net(plain)
bandit = EditBandit()
bandit_generator = torch.Generator().manual_seed(cfg.seed)
random_generator = torch.Generator().manual_seed(cfg.seed)

# %%
run_id = make_run_id("bandit-tasks", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="bandit-tasks",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "bandit-tasks", "tasks": 3, "steps": task_steps, "every": 20},
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


def record_gain(kind: str, decision: GateDecision) -> None:
    bandit.observe(kind, decision.validated_gain)


bandit_curves = train_across_tasks(
    bandit_net,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    ledger=ledger,
    every=20,
    shadow_steps=8,
    choose=bandit.choose,
    controller="L1",
    generator=bandit_generator,
    on_decision=record_gain,
)
random_curves = train_across_tasks(
    random_net,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
    ledger=ledger,
    every=20,
    shadow_steps=8,
    choose=choose_edit,
    controller="random",
    generator=random_generator,
)
plain_curves = train_on_tasks(
    plain,
    tasks,
    probes,
    steps_per_task=task_steps,
    batch_size=64,
    lr=0.1,
)
task_a = {
    "no edits": plain_curves[0],
    "bandit": bandit_curves[0],
    "random": random_curves[0],
}
task_c = {
    "no edits": plain_curves[2],
    "bandit": bandit_curves[2],
    "random": random_curves[2],
}
style = {"bandit": "dash", "random": "dot"}
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
decisions = ledger.read("gate_decisions")
bandit_rows = pa.array(
    [str(value).startswith("l1-") for value in decisions.column("proposal_id").to_pylist()]
)
timeline = plot_gate_timeline(decisions.filter(bandit_rows), synthetic=False)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curve_a.write_html(figure_dir / "16_bandit_task_a.html")
save_diagnostic_traces_png(
    task_a,
    figure_dir / "16_bandit_task_a.png",
    title="task A probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
curve_c.write_html(figure_dir / "16_bandit_task_c.html")
save_diagnostic_traces_png(
    task_c,
    figure_dir / "16_bandit_task_c.png",
    title="task C probe accuracy",
    y_title="accuracy",
    marks=switch,
    mark_label="switch",
    dashes=style,
)
timeline.write_html(figure_dir / "16_bandit_timeline.html")
save_gate_timeline_png(
    decisions.filter(bandit_rows),
    figure_dir / "16_bandit_timeline.png",
    synthetic=False,
)
end = 3 * task_steps - 1
proposals = ledger.read("edit_proposals")
kinds = proposals.column("edit_type").to_pylist()
controllers = proposals.column("controller").to_pylist()
accepted = decisions.column("accepted").to_pylist()
proposal_ids = decisions.column("proposal_id").to_pylist()
bandit_kept = sum(
    1
    for flag, pid in zip(accepted, proposal_ids, strict=True)
    if flag and str(pid).startswith("l1-")
)
random_kept = sum(
    1
    for flag, pid in zip(accepted, proposal_ids, strict=True)
    if flag and str(pid).startswith("random-")
)
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_task_a", plain_curves[0]),
            ("accuracy_task_b", plain_curves[1]),
            ("accuracy_task_c", plain_curves[2]),
            ("accuracy_task_a_bandit", bandit_curves[0]),
            ("accuracy_task_b_bandit", bandit_curves[1]),
            ("accuracy_task_c_bandit", bandit_curves[2]),
            ("accuracy_task_a_random", random_curves[0]),
            ("accuracy_task_b_random", random_curves[1]),
            ("accuracy_task_c_random", random_curves[2]),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="bandit-tasks",
            kind="method",
            statement=(
                "Three tasks, 160 steps. "
                f"Bandit kept {bandit_kept}, random kept {random_kept}. "
                f"Task A ends at {plain_curves[0][end]:.3f}, "
                f"{bandit_curves[0][end]:.3f}, {random_curves[0][end]:.3f}. "
                f"Task B ends at {plain_curves[1][end]:.3f}, "
                f"{bandit_curves[1][end]:.3f}, {random_curves[1][end]:.3f}. "
                f"Task C ends at {plain_curves[2][end]:.3f}, "
                f"{bandit_curves[2][end]:.3f}, {random_curves[2][end]:.3f}."
            ),
            evidence_ref="results/figures/16_bandit_task_a.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "16_bandit_task_a.provenance.json",
    figure="16_bandit_task_a",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
write_provenance(
    figure_dir / "16_bandit_task_c.provenance.json",
    figure="16_bandit_task_c",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
write_provenance(
    figure_dir / "16_bandit_timeline.provenance.json",
    figure="16_bandit_timeline",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "bandit_kept": bandit_kept,
            "random_kept": random_kept,
            "plain": [plain_curves[i][end] for i in range(3)],
            "bandit": [bandit_curves[i][end] for i in range(3)],
            "random": [random_curves[i][end] for i in range(3)],
            "means": {"prune": bandit.mean("prune"), "widen": bandit.mean("widen")},
            "counts": bandit.count,
            "arms": [
                f"{controller}:{kind}" for controller, kind in zip(controllers, kinds, strict=True)
            ],
            "synthetic": manifest["synthetic"],
        }
    )
)
curve_a
