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

from valmorph.benchmarks.classify import batch_accuracy, train_sequence_end_scores
from valmorph.benchmarks.mnist import (
    apply_permutation,
    download_mnist,
    load_split,
    pixel_permutation,
)
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import LEDGER_VERSION, Finding, Ledger, RunMeta, StepMetric, make_run_id
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
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
curve
