# %% [markdown]
# # 20 — GEM's 20-task permutation protocol
#
# ADR-013. Lopez-Paz and Ranzato, NIPS 2017, section 4.1: 20 tasks, the
# first 1000 MNIST training images under a fixed pixel permutation, one
# pass, two hidden layers of 100 ReLU units, plain SGD, minibatch 10.
# Their appendix selects learning rate 0.03 for the single model on this
# dataset. Each task is scored on the full MNIST test set after every task.
# Average accuracy is the mean of the last row. Backward transfer is the
# mean of last-row minus end-of-task accuracy, over the earlier tasks.
# One seed. CPU. Not a reproduction of their table, and not a Level 1 claim.

# %%
import json
import sys

import torch

from valmorph.benchmarks.classify import (
    batch_accuracy,
    train_rehearsal_end_scores,
    train_sequence_end_scores,
)
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
    assert_same_machine,
    make_run_id,
)
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
train_rows = train_x[:1000]
train_labels = train_y[:1000]
task_count = 20
tasks = []
tests = []
for task in range(task_count):
    permutation = pixel_permutation(cfg.seed + task)
    tasks.append((apply_permutation(train_rows, permutation), train_labels))
    tests.append((apply_permutation(test_x, permutation), test_y))
net = ValmorphMLP([784, 100, 100, 10]).cpu()
with torch.no_grad():
    initial = [batch_accuracy(net(images), labels) for images, labels in tests]

# %%
run_id = make_run_id("gem-protocol", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="gem-protocol",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "gem-protocol",
            "tasks": task_count,
            "examples": 1000,
            "width": [784, 100, 100, 10],
            "batch_size": 10,
            "lr": 0.03,
            "passes": 1,
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
rows = train_sequence_end_scores(net, tasks, tests, batch_size=10, lr=0.03)
last = rows[-1]
average = sum(last) / task_count
backward = sum(last[task] - rows[task][task] for task in range(task_count - 1)) / (task_count - 1)
first_task = [row[0] for row in rows]
curve = plot_diagnostic_traces(
    {"task 0": first_task},
    title="task 0 test accuracy after each task",
    y_title="accuracy",
    x_title="task index",
)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curve.write_html(figure_dir / "20_gem_task0.html")
save_diagnostic_traces_png(
    {"task 0": first_task},
    figure_dir / "20_gem_task0.png",
    title="task 0 test accuracy after each task",
    y_title="accuracy",
    x_title="task index",
)
ledger.append_step_metrics(
    [
        StepMetric(step=task, split="test", name="accuracy_task_0", value=float(value))
        for task, value in enumerate(first_task)
    ]
    + [
        StepMetric(step=task_count - 1, split="test", name=f"final_task_{task}", value=float(value))
        for task, value in enumerate(last)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="gem-protocol",
            kind="method",
            statement=(
                "GEM section 4.1 sizes, one pass, lr 0.03, seed 0. "
                f"Mean of the last row is {average:.3f}. "
                f"Backward transfer is {backward:.3f}. "
                f"Task 0 ends at {first_task[-1]:.3f} "
                f"after its own training at {first_task[0]:.3f}. "
                f"Initial test accuracy mean is {sum(initial) / task_count:.3f}."
            ),
            evidence_ref="results/figures/20_gem_task0.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "20_gem_task0.provenance.json",
    figure="20_gem_task0",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "average": average,
            "backward": backward,
            "task0_end": first_task[-1],
            "task0_after_own": first_task[0],
            "final_min": min(last),
            "final_max": max(last),
            "classes_in_1000": sorted(set(int(label) for label in train_labels)),
            "synthetic": manifest["synthetic"],
        }
    )
)

# %% [markdown]
# # The gate on the same twenty tasks
#
# Training rows 1000 through 1499 are the probe. They are disjoint from the
# 1000 training rows and from the test set. The gate sees only that probe.
# The test matrix is scored after each task. L1 is the bandit. Random draws
# the same three arms. The no-edit copy uses the same one-step update and
# proposes nothing. One seed. CPU. Not Level 1.

# %%
torch.manual_seed(cfg.seed)
base = ValmorphMLP([784, 100, 100, 10]).cpu()
plain_steps = clone_net(base)
bandit_net = clone_net(base)
random_net = clone_net(base)
probe_images = train_x[1000:1500]
probe_labels = train_y[1000:1500]
probes = []
for task in range(task_count):
    permutation = pixel_permutation(cfg.seed + task)
    probes.append((apply_permutation(probe_images, permutation), probe_labels))
gate_id = make_run_id("gem-gate", cfg.seed, config_hash(cfg))
gate_ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=gate_id,
        created_utc=gate_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="gem-gate",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "gem-gate",
            "tasks": task_count,
            "examples": 1000,
            "probe": 500,
            "width": [784, 100, 100, 10],
            "batch_size": 10,
            "lr": 0.03,
            "every": 20,
            "shadow_steps": 8,
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


def test_row(model: ValmorphMLP) -> list[float]:
    with torch.no_grad():
        return [batch_accuracy(model(images), labels) for images, labels in tests]


plain_rows: list[list[float]] = []
bandit_rows: list[list[float]] = []
random_rows: list[list[float]] = []
train_across_tasks(
    plain_steps,
    tasks,
    probes,
    steps_per_task=100,
    batch_size=10,
    lr=0.03,
    ledger=gate_ledger,
    every=0,
    on_task_end=lambda index, model: plain_rows.append(test_row(model)),
)
bandit = EditBandit()
bandit_generator = torch.Generator().manual_seed(cfg.seed)
train_across_tasks(
    bandit_net,
    tasks,
    probes,
    steps_per_task=100,
    batch_size=10,
    lr=0.03,
    ledger=gate_ledger,
    every=20,
    shadow_steps=8,
    choose=bandit.choose,
    controller="L1",
    generator=bandit_generator,
    on_decision=lambda kind, decision: bandit.observe(kind, decision.validated_gain),
    on_task_end=lambda index, model: bandit_rows.append(test_row(model)),
)
random_generator = torch.Generator().manual_seed(cfg.seed)
train_across_tasks(
    random_net,
    tasks,
    probes,
    steps_per_task=100,
    batch_size=10,
    lr=0.03,
    ledger=gate_ledger,
    every=20,
    shadow_steps=8,
    choose=choose_edit,
    controller="random",
    generator=random_generator,
    on_task_end=lambda index, model: random_rows.append(test_row(model)),
)


def row_summary(rows: list[list[float]]) -> tuple[float, float, float, float]:
    final = rows[-1]
    mean = sum(final) / task_count
    transfer = sum(final[i] - rows[i][i] for i in range(task_count - 1)) / (task_count - 1)
    return mean, transfer, rows[0][0], final[0]


plain_summary = row_summary(plain_rows)
bandit_summary = row_summary(bandit_rows)
random_summary = row_summary(random_rows)
task0_curves = {
    "no edits": [row[0] for row in plain_rows],
    "L1": [row[0] for row in bandit_rows],
    "random": [row[0] for row in random_rows],
}
gate_curve = plot_diagnostic_traces(
    task0_curves,
    title="task 0 test accuracy after each task",
    y_title="accuracy",
    x_title="task index",
    dashes={"L1": "dash", "random": "dot"},
)
gate_curve.write_html(figure_dir / "20_gem_gate_task0.html")
save_diagnostic_traces_png(
    task0_curves,
    figure_dir / "20_gem_gate_task0.png",
    title="task 0 test accuracy after each task",
    y_title="accuracy",
    x_title="task index",
    dashes={"L1": "dash", "random": "dot"},
)
decision_rows = gate_ledger.read("gate_decisions").to_pylist()
kept_l1 = sum(
    row["accepted"] and str(row["proposal_id"]).startswith("l1-") for row in decision_rows
)
kept_random = sum(
    row["accepted"] and str(row["proposal_id"]).startswith("random-") for row in decision_rows
)
gate_ledger.append_step_metrics(
    [
        StepMetric(step=task, split="test", name=name, value=float(value))
        for name, series in (
            ("accuracy_task_0_plain", task0_curves["no edits"]),
            ("accuracy_task_0_l1", task0_curves["L1"]),
            ("accuracy_task_0_random", task0_curves["random"]),
        )
        for task, value in enumerate(series)
    ]
)
gate_ledger.append_findings(
    [
        Finding(
            finding_id="gem-gate",
            kind="method",
            statement=(
                "Same 20 tasks. Probe is 500 held-out training rows. "
                f"No edits, last-row mean {plain_summary[0]:.3f}, "
                f"backward {plain_summary[1]:.3f}, "
                f"task 0 {plain_summary[2]:.3f} then {plain_summary[3]:.3f}. "
                f"L1, last-row mean {bandit_summary[0]:.3f}, "
                f"backward {bandit_summary[1]:.3f}, "
                f"task 0 {bandit_summary[2]:.3f} then {bandit_summary[3]:.3f}. "
                f"Random, last-row mean {random_summary[0]:.3f}, "
                f"backward {random_summary[1]:.3f}, "
                f"task 0 {random_summary[2]:.3f} then {random_summary[3]:.3f}. "
                f"Kept L1 edits: {kept_l1}. "
                f"Kept random edits: {kept_random}."
            ),
            evidence_ref="results/figures/20_gem_gate_task0.provenance.json",
        )
    ]
)
gate_manifest = write_provenance(
    figure_dir / "20_gem_gate_task0.provenance.json",
    figure="20_gem_gate_task0",
    ledgers=[gate_ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "plain": plain_summary,
            "l1": bandit_summary,
            "random": random_summary,
            "kept_l1": kept_l1,
            "kept_random": kept_random,
            "proposals": len(decision_rows),
            "synthetic": gate_manifest["synthetic"],
        }
    )
)
gate_curve

# %% [markdown]
# # Five seeds of the same comparison
#
# Seed 0 repeats the cell above and must match it. Seeds 1 through 4 reroll
# the initialization and the permutations. The test set is still only a
# report. This is the robustness of the smoke, not Level 1.


# %%
def open_seed_ledger(seed: int) -> Ledger:
    run_id = make_run_id(f"gem-seed{seed}", seed, config_hash(cfg))
    return Ledger.create(
        root / "results" / "ledgers",
        RunMeta(
            ledger_version=LEDGER_VERSION,
            run_id=run_id,
            created_utc=run_id.rsplit("-", 1)[-1],
            synthetic=False,
            experiment="gem-gate-seeds",
            seed=seed,
            config_hash=config_hash(cfg),
            config={
                "experiment": "gem-gate-seeds",
                "tasks": task_count,
                "examples": 1000,
                "probe": 500,
                "width": [784, 100, 100, 10],
                "batch_size": 10,
                "lr": 0.03,
                "every": 20,
                "shadow_steps": 8,
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


Summary = tuple[float, float, float, float]


def run_seed(seed: int) -> tuple[Ledger, Summary, Summary, Summary, int, int]:
    torch.manual_seed(seed)
    seed_base = ValmorphMLP([784, 100, 100, 10]).cpu()
    seed_plain = clone_net(seed_base)
    seed_bandit_net = clone_net(seed_base)
    seed_random_net = clone_net(seed_base)
    seed_tasks = []
    seed_probes = []
    seed_tests = []
    for task in range(task_count):
        order = pixel_permutation(seed + task)
        seed_tasks.append((apply_permutation(train_x[:1000], order), train_y[:1000]))
        seed_probes.append((apply_permutation(train_x[1000:1500], order), train_y[1000:1500]))
        seed_tests.append((apply_permutation(test_x, order), test_y))

    def score(model: ValmorphMLP) -> list[float]:
        with torch.no_grad():
            return [batch_accuracy(model(images), labels) for images, labels in seed_tests]

    seed_plain_rows: list[list[float]] = []
    seed_bandit_rows: list[list[float]] = []
    seed_random_rows: list[list[float]] = []
    seed_ledger = open_seed_ledger(seed)
    train_across_tasks(
        seed_plain,
        seed_tasks,
        seed_probes,
        steps_per_task=100,
        batch_size=10,
        lr=0.03,
        ledger=seed_ledger,
        every=0,
        on_task_end=lambda index, model: seed_plain_rows.append(score(model)),
    )
    seed_bandit = EditBandit()

    def observe_bandit(kind: str, decision: GateDecision) -> None:
        seed_bandit.observe(kind, decision.validated_gain)
        if decision.accepted:
            seed_bandit_kept[0] += 1

    seed_bandit_kept = [0]
    train_across_tasks(
        seed_bandit_net,
        seed_tasks,
        seed_probes,
        steps_per_task=100,
        batch_size=10,
        lr=0.03,
        ledger=seed_ledger,
        every=20,
        shadow_steps=8,
        choose=seed_bandit.choose,
        controller="L1",
        generator=torch.Generator().manual_seed(seed),
        on_decision=observe_bandit,
        on_task_end=lambda index, model: seed_bandit_rows.append(score(model)),
    )
    seed_random_kept = [0]

    def observe_random(kind: str, decision: GateDecision) -> None:
        if decision.accepted:
            seed_random_kept[0] += 1

    train_across_tasks(
        seed_random_net,
        seed_tasks,
        seed_probes,
        steps_per_task=100,
        batch_size=10,
        lr=0.03,
        ledger=seed_ledger,
        every=20,
        shadow_steps=8,
        choose=choose_edit,
        controller="random",
        generator=torch.Generator().manual_seed(seed),
        on_task_end=lambda index, model: seed_random_rows.append(score(model)),
        on_decision=observe_random,
    )
    return (
        seed_ledger,
        row_summary(seed_plain_rows),
        row_summary(seed_bandit_rows),
        row_summary(seed_random_rows),
        seed_bandit_kept[0],
        seed_random_kept[0],
    )


seed_ledgers: list[Ledger] = []
seed_plain: list[Summary] = []
seed_l1: list[Summary] = []
seed_random: list[Summary] = []
seed_kept_l1: list[int] = []
seed_kept_random: list[int] = []
for seed in range(5):
    seed_ledger, plain_one, l1_one, random_one, kept_one_l1, kept_one_random = run_seed(seed)
    if seed == 0:
        if abs(l1_one[0] - bandit_summary[0]) > 1e-6:
            raise RuntimeError("seed 0 does not match the detailed L1 run")
        if abs(plain_one[0] - plain_summary[0]) > 1e-6:
            raise RuntimeError("seed 0 does not match the detailed no-edit run")
    seed_ledgers.append(seed_ledger)
    seed_plain.append(plain_one)
    seed_l1.append(l1_one)
    seed_random.append(random_one)
    seed_kept_l1.append(kept_one_l1)
    seed_kept_random.append(kept_one_random)

assert_same_machine(seed_ledgers)
mean_gap = {
    "L1 minus no edits": [
        l1_one[0] - plain_one[0] for plain_one, l1_one in zip(seed_plain, seed_l1, strict=True)
    ],
    "random minus no edits": [
        random_one[0] - plain_one[0]
        for plain_one, random_one in zip(seed_plain, seed_random, strict=True)
    ],
}
task0_gap = {
    "L1 task 0 minus no edits": [
        l1_one[3] - plain_one[3] for plain_one, l1_one in zip(seed_plain, seed_l1, strict=True)
    ],
    "random task 0 minus no edits": [
        random_one[3] - plain_one[3]
        for plain_one, random_one in zip(seed_plain, seed_random, strict=True)
    ],
}
seed_curves = {**mean_gap, **task0_gap}
seed_curve = plot_diagnostic_traces(
    seed_curves,
    title="test accuracy minus the no-edit run",
    y_title="accuracy gap",
    x_title="seed",
    dashes={
        "random minus no edits": "dot",
        "L1 task 0 minus no edits": "dash",
        "random task 0 minus no edits": "dashdot",
    },
)
seed_curve.write_html(figure_dir / "20_gem_gate_seeds.html")
save_diagnostic_traces_png(
    seed_curves,
    figure_dir / "20_gem_gate_seeds.png",
    title="test accuracy minus the no-edit run",
    y_title="accuracy gap",
    x_title="seed",
    dashes={
        "random minus no edits": "dot",
        "L1 task 0 minus no edits": "dash",
        "random task 0 minus no edits": "dashdot",
    },
)
triples = zip(seed_plain, seed_l1, seed_random, strict=True)
for seed, (plain_one, l1_one, random_one) in enumerate(triples):
    seed_ledgers[seed].append_step_metrics(
        [
            StepMetric(step=seed, split="test", name="plain_mean", value=plain_one[0]),
            StepMetric(step=seed, split="test", name="l1_mean", value=l1_one[0]),
            StepMetric(step=seed, split="test", name="random_mean", value=random_one[0]),
            StepMetric(step=seed, split="test", name="plain_backward", value=plain_one[1]),
            StepMetric(step=seed, split="test", name="l1_backward", value=l1_one[1]),
            StepMetric(step=seed, split="test", name="random_backward", value=random_one[1]),
            StepMetric(step=seed, split="test", name="plain_task0_end", value=plain_one[3]),
            StepMetric(step=seed, split="test", name="l1_task0_end", value=l1_one[3]),
            StepMetric(step=seed, split="test", name="random_task0_end", value=random_one[3]),
            StepMetric(step=seed, split="test", name="kept_l1", value=float(seed_kept_l1[seed])),
            StepMetric(
                step=seed, split="test", name="kept_random", value=float(seed_kept_random[seed])
            ),
        ]
    )
gap_text = "; ".join(
    f"seed {seed}: L1 mean {mean_gap['L1 minus no edits'][seed]:+.3f} "
    f"({seed_kept_l1[seed]} kept), "
    f"random mean {mean_gap['random minus no edits'][seed]:+.3f} "
    f"({seed_kept_random[seed]} kept), "
    f"L1 task 0 {task0_gap['L1 task 0 minus no edits'][seed]:+.3f}, "
    f"random task 0 {task0_gap['random task 0 minus no edits'][seed]:+.3f}"
    for seed in range(5)
)
seed_ledgers[0].append_findings(
    [
        Finding(
            finding_id="gem-gate-seeds",
            kind="method",
            statement=(
                "Five seeds of the 20-task gate comparison. "
                "Gaps are last-row mean and task 0 at the end, minus the no-edit run. "
                f"{gap_text}."
            ),
            evidence_ref="results/figures/20_gem_gate_seeds.provenance.json",
        )
    ]
)
seed_manifest = write_provenance(
    figure_dir / "20_gem_gate_seeds.provenance.json",
    figure="20_gem_gate_seeds",
    ledgers=seed_ledgers,
    config_hash=config_hash(cfg),
    seeds=list(range(5)),
)
print(json.dumps({"gaps": gap_text, "synthetic": seed_manifest["synthetic"]}))
seed_curve

# %% [markdown]
# # One pass with stored rows
#
# Each new example is still seen once. On later tasks, half of a full batch
# is rows kept from finished tasks, spaced through the whole store, and the
# rest is the next new rows. Every finished task is kept. A task has 1000
# rows, so this is the same store as the 1024-row mix. Five seeds. CPU.
# Not Level 1.


# %%
def open_rehearsal_ledger(seed: int) -> Ledger:
    run_id = make_run_id(f"gem-rehearsal{seed}", seed, config_hash(cfg))
    return Ledger.create(
        root / "results" / "ledgers",
        RunMeta(
            ledger_version=LEDGER_VERSION,
            run_id=run_id,
            created_utc=run_id.rsplit("-", 1)[-1],
            synthetic=False,
            experiment="gem-rehearsal",
            seed=seed,
            config_hash=config_hash(cfg),
            config={
                "experiment": "gem-rehearsal",
                "tasks": task_count,
                "examples": 1000,
                "width": [784, 100, 100, 10],
                "batch_size": 10,
                "lr": 0.03,
                "passes": 1,
                "memory": "all",
                "old_fraction": 0.5,
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


def rehearsal_for_seed(
    seed: int,
) -> tuple[Ledger, Summary, Summary, list[float], list[float], float, float]:
    torch.manual_seed(seed)
    rehearsal_base = ValmorphMLP([784, 100, 100, 10]).cpu()
    plain_net = clone_net(rehearsal_base)
    mix_net = clone_net(rehearsal_base)
    seed_tasks = []
    seed_tests = []
    for task in range(task_count):
        order = pixel_permutation(seed + task)
        seed_tasks.append((apply_permutation(train_x[:1000], order), train_y[:1000]))
        seed_tests.append((apply_permutation(test_x, order), test_y))
    rehearsal_ledger = open_rehearsal_ledger(seed)
    plain_matrix = train_sequence_end_scores(
        plain_net, seed_tasks, seed_tests, batch_size=10, lr=0.03
    )
    mix_matrix = train_rehearsal_end_scores(
        mix_net, seed_tasks, seed_tests, batch_size=10, lr=0.03, memory=None
    )
    return (
        rehearsal_ledger,
        row_summary(plain_matrix),
        row_summary(mix_matrix),
        [row[0] for row in plain_matrix],
        [row[0] for row in mix_matrix],
        plain_matrix[-1][-1],
        mix_matrix[-1][-1],
    )


rehearsal_ledgers: list[Ledger] = []
rehearsal_plain: list[Summary] = []
rehearsal_mix: list[Summary] = []
rehearsal_newest: list[tuple[float, float]] = []
rehearsal_task0: dict[str, list[float]] = {}
for seed in range(5):
    (
        rehearsal_ledger,
        plain_one,
        mix_one,
        plain_task0,
        mix_task0,
        plain_newest,
        mix_newest,
    ) = rehearsal_for_seed(seed)
    if seed == 0:
        if abs(plain_one[0] - average) > 1e-6 or abs(plain_one[3] - first_task[-1]) > 1e-6:
            raise RuntimeError("seed 0 plain pass does not match the first run")
        rehearsal_task0 = {"no stored rows": plain_task0, "stored rows": mix_task0}
    rehearsal_ledgers.append(rehearsal_ledger)
    rehearsal_plain.append(plain_one)
    rehearsal_mix.append(mix_one)
    rehearsal_newest.append((plain_newest, mix_newest))

assert_same_machine(rehearsal_ledgers)
rehearsal_curve = plot_diagnostic_traces(
    rehearsal_task0,
    title="task 0 test accuracy after each task",
    y_title="accuracy",
    x_title="task index",
    dashes={"stored rows": "dash"},
)
rehearsal_curve.write_html(figure_dir / "20_gem_rehearsal_task0.html")
save_diagnostic_traces_png(
    rehearsal_task0,
    figure_dir / "20_gem_rehearsal_task0.png",
    title="task 0 test accuracy after each task",
    y_title="accuracy",
    x_title="task index",
    dashes={"stored rows": "dash"},
)
rehearsal_gaps = {
    "stored rows minus plain": [
        mix_one[0] - plain_one[0]
        for plain_one, mix_one in zip(rehearsal_plain, rehearsal_mix, strict=True)
    ],
    "stored rows task 0 minus plain": [
        mix_one[3] - plain_one[3]
        for plain_one, mix_one in zip(rehearsal_plain, rehearsal_mix, strict=True)
    ],
    "stored rows task 19 minus plain": [
        mix_newest - plain_newest for plain_newest, mix_newest in rehearsal_newest
    ],
}
rehearsal_gap_curve = plot_diagnostic_traces(
    rehearsal_gaps,
    title="test accuracy minus the plain pass",
    y_title="accuracy gap",
    x_title="seed",
    dashes={
        "stored rows task 0 minus plain": "dash",
        "stored rows task 19 minus plain": "dot",
    },
)
rehearsal_gap_curve.write_html(figure_dir / "20_gem_rehearsal_seeds.html")
save_diagnostic_traces_png(
    rehearsal_gaps,
    figure_dir / "20_gem_rehearsal_seeds.png",
    title="test accuracy minus the plain pass",
    y_title="accuracy gap",
    x_title="seed",
    dashes={
        "stored rows task 0 minus plain": "dash",
        "stored rows task 19 minus plain": "dot",
    },
)
for seed, (plain_one, mix_one) in enumerate(zip(rehearsal_plain, rehearsal_mix, strict=True)):
    rehearsal_ledgers[seed].append_step_metrics(
        [
            StepMetric(step=seed, split="test", name="plain_mean", value=plain_one[0]),
            StepMetric(step=seed, split="test", name="mix_mean", value=mix_one[0]),
            StepMetric(step=seed, split="test", name="plain_backward", value=plain_one[1]),
            StepMetric(step=seed, split="test", name="mix_backward", value=mix_one[1]),
            StepMetric(step=seed, split="test", name="plain_task0_end", value=plain_one[3]),
            StepMetric(step=seed, split="test", name="mix_task0_end", value=mix_one[3]),
            StepMetric(
                step=seed,
                split="test",
                name="plain_task19_end",
                value=rehearsal_newest[seed][0],
            ),
            StepMetric(
                step=seed,
                split="test",
                name="mix_task19_end",
                value=rehearsal_newest[seed][1],
            ),
        ]
    )
rehearsal_text = "; ".join(
    f"seed {seed}: mean {rehearsal_gaps['stored rows minus plain'][seed]:+.3f}, "
    f"task 0 {rehearsal_gaps['stored rows task 0 minus plain'][seed]:+.3f}, "
    f"task 19 {rehearsal_gaps['stored rows task 19 minus plain'][seed]:+.3f}"
    for seed in range(5)
)
rehearsal_ledgers[0].append_findings(
    [
        Finding(
            finding_id="gem-rehearsal-task0",
            kind="method",
            statement=(
                "Seed 0, one pass. "
                f"Plain task 0 ends at {rehearsal_plain[0][3]:.3f}. "
                f"With stored rows it ends at {rehearsal_mix[0][3]:.3f}."
            ),
            evidence_ref="results/figures/20_gem_rehearsal_task0.provenance.json",
        ),
        Finding(
            finding_id="gem-rehearsal",
            kind="method",
            statement=(
                "Five seeds. One pass, half of each later batch from stored rows. "
                f"Seed 0 plain mean {rehearsal_plain[0][0]:.3f}, "
                f"stored rows {rehearsal_mix[0][0]:.3f}. "
                f"Task 0 ends at {rehearsal_plain[0][3]:.3f} and {rehearsal_mix[0][3]:.3f}. "
                f"Gaps, {rehearsal_text}."
            ),
            evidence_ref="results/figures/20_gem_rehearsal_seeds.provenance.json",
        ),
    ]
)
write_provenance(
    figure_dir / "20_gem_rehearsal_task0.provenance.json",
    figure="20_gem_rehearsal_task0",
    ledgers=[rehearsal_ledgers[0]],
    config_hash=config_hash(cfg),
    seeds=[0],
)
rehearsal_manifest = write_provenance(
    figure_dir / "20_gem_rehearsal_seeds.provenance.json",
    figure="20_gem_rehearsal_seeds",
    ledgers=rehearsal_ledgers,
    config_hash=config_hash(cfg),
    seeds=list(range(5)),
)
print(json.dumps({"gaps": rehearsal_text, "synthetic": rehearsal_manifest["synthetic"]}))
rehearsal_gap_curve
