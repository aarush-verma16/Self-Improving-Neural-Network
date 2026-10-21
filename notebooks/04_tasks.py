# %% [markdown]
# # 04 — Permuted MNIST, one short task
#
# Each image is flattened to 784 pixels and then shuffled with one fixed
# permutation. The labels stay 0-9. The network is a small MLP. This notebook
# trains for a few dozen steps on a subset and plots loss and probe accuracy.
#
# The digit grid shows what the permutation does to the pixels. This is a
# method check, not a self-improvement result. The files land in `data/mnist`
# and are not committed.
#
# Run this yourself. It is a short CPU run. It downloads MNIST the first time,
# about 10 MB.

# %%
import json
import sys

import torch

from valmorph.benchmarks.classify import train_classifier
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
from valmorph.viz import (
    plot_diagnostic_traces,
    save_diagnostic_traces_png,
    save_digit_row,
    write_provenance,
)

root = repo_root()
cfg = load_run_config(root / "configs" / "setup.yaml")
torch.manual_seed(cfg.seed)
mnist_dir = download_mnist(root / "data" / "mnist")
train_x, train_y = load_split(mnist_dir, "train")
test_x, test_y = load_split(mnist_dir, "test")
permutation = pixel_permutation(cfg.seed)
train_x = apply_permutation(train_x[:2000], permutation)
train_y = train_y[:2000]
probe_x = apply_permutation(test_x[:500], permutation)
probe_y = test_y[:500]
net = ValmorphMLP([784, 32, 10]).cpu()
losses, accuracies = train_classifier(
    net,
    train_x,
    train_y,
    steps=40,
    batch_size=64,
    lr=0.1,
    probe_images=probe_x,
    probe_labels=probe_y,
)
traces = plot_diagnostic_traces({"train loss": losses, "probe accuracy": accuracies})
original = test_x[:8]
permuted = apply_permutation(original, permutation)

# %%
run_id = make_run_id("permuted-mnist-smoke", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="permuted-mnist-smoke",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "permuted-mnist-smoke",
            "widths": [784, 32, 10],
            "steps": 40,
            "train_rows": 2000,
            "probe_rows": 500,
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
ledger.append_step_metrics(
    [StepMetric(step=i, split="train", name="loss", value=value) for i, value in enumerate(losses)]
)
ledger.append_step_metrics(
    [
        StepMetric(step=i, split="probe", name="accuracy", value=value)
        for i, value in enumerate(accuracies)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="permuted-mnist-smoke",
            kind="method",
            statement=(
                f"One permutation, 2000 train rows, 40 steps on CPU. "
                f"Final probe accuracy={accuracies[-1]:.3f}, final loss={losses[-1]:.3f}. "
                "Not a self-improvement result."
            ),
            evidence_ref="results/figures/04_permuted_mnist.provenance.json",
        )
    ]
)
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
save_digit_row(original, figure_dir / "04_digits_original.png", title="original digits")
save_digit_row(permuted, figure_dir / "04_digits_permuted.png", title="one fixed pixel permutation")
traces.write_html(figure_dir / "04_permuted_mnist.html")
save_diagnostic_traces_png(
    {"train loss": losses, "probe accuracy": accuracies},
    figure_dir / "04_permuted_mnist.png",
)
manifest = write_provenance(
    figure_dir / "04_permuted_mnist.provenance.json",
    figure="04_permuted_mnist",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "final_loss": losses[-1],
            "final_probe_accuracy": accuracies[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
traces
