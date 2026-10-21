# %% [markdown]
# # 10 — Widen on a schedule
#
# Notebook 09 proposed the three smallest units, `u2`, `u3`, and `u24`. The
# gate refused all three. Probe accuracy stayed on the no-edit curve at 0.634.
# Pruning a small unit did not lower probe loss.
#
# The one edit the gate has kept on this task is a widen, and after the live
# network stopped copying the shadow's extra steps that widen also ended at
# 0.634. This notebook applies the widen anyway, on a schedule, and does not
# ask the gate. Hidden unit 0 of the first layer is copied at steps 10, 20,
# and 30. Each copy leaves the function unchanged at that moment. Later
# minibatch steps can use the new units.
#
# One seed. Short CPU run. Not a Level 1 comparison.

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
from valmorph.controllers import train_with_scheduled_widen
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
permutation = pixel_permutation(cfg.seed)
train_x = apply_permutation(train_x[:2000], permutation)
train_y = train_y[:2000]
probe_x = apply_permutation(test_x[:500], permutation)
probe_y = test_y[:500]
grown = ValmorphMLP([784, 32, 10]).cpu()
plain = clone_net(grown)
_plain_loss, plain_accuracy = train_classifier(
    plain,
    train_x,
    train_y,
    steps=40,
    batch_size=64,
    lr=0.1,
    probe_images=probe_x,
    probe_labels=probe_y,
)

# %%
run_id = make_run_id("scheduled-widen", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="scheduled-widen",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "scheduled-widen",
            "widths": [784, 32, 10],
            "steps": 40,
            "every": 10,
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
grown_accuracy = train_with_scheduled_widen(
    grown,
    train_x,
    train_y,
    probe_images=probe_x,
    probe_labels=probe_y,
    steps=40,
    batch_size=64,
    lr=0.1,
    every=10,
)
width = int(grown.linear(0).out_features)
accuracy_series = {"no edits": plain_accuracy, "scheduled widen": grown_accuracy}
curves = plot_diagnostic_traces(accuracy_series, title="probe accuracy", y_title="accuracy")

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curves.write_html(figure_dir / "10_scheduled_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "10_scheduled_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
)
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_no_edits", plain_accuracy),
            ("accuracy_scheduled_widen", grown_accuracy),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="scheduled-widen",
            kind="method",
            statement=(
                f"Widened unit 0 at steps 10, 20, and 30 with no gate. "
                f"Hidden width ended at {width}. Final probe accuracy "
                f"no-edit={plain_accuracy[-1]:.3f}, "
                f"scheduled={grown_accuracy[-1]:.3f}."
            ),
            evidence_ref="results/figures/10_scheduled_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "10_scheduled_accuracy.provenance.json",
    figure="10_scheduled_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "width": width,
            "plain_accuracy": plain_accuracy[-1],
            "grown_accuracy": grown_accuracy[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
