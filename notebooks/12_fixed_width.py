# %% [markdown]
# # 12 — Start at the final width
#
# Notebook 11 kept five widens of unit 0, at steps 30, 40, 50, 60, and 70.
# The live network ended at probe accuracy 0.750. The no-edit network ended
# at 0.752. The dashed curve, which took 8 extra steps on each keep, ended
# at 0.806. The kept units did not raise accuracy. The extra steps did.
#
# Those five keeps leave a hidden layer of width 37. This notebook makes
# that width before any training, by copying unit 0 five times, so the wider
# net starts as the same function. A third copy widens on the schedule from
# notebook 10, at steps 10, 20, and 30 only. No gate. All three start from
# the same weights.
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
from valmorph.models import ValmorphMLP, widen
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
small = ValmorphMLP([784, 32, 10]).cpu()
grown = clone_net(small)
large = clone_net(small)
for _ in range(5):
    widen(large, 0, 0)
_loss, small_accuracy = train_classifier(
    small,
    train_x,
    train_y,
    steps=80,
    batch_size=64,
    lr=0.1,
    probe_images=probe_x,
    probe_labels=probe_y,
)
_loss, large_accuracy = train_classifier(
    large,
    train_x,
    train_y,
    steps=80,
    batch_size=64,
    lr=0.1,
    probe_images=probe_x,
    probe_labels=probe_y,
)

# %%
run_id = make_run_id("fixed-width", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="fixed-width",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "fixed-width", "steps": 80, "widths": [32, 37]},
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
    steps=80,
    batch_size=64,
    lr=0.1,
    every=10,
    until=30,
)
accuracy_series = {
    "width 32": small_accuracy,
    "grow to 35": grown_accuracy,
    "widen five times, then train": large_accuracy,
}
curves = plot_diagnostic_traces(accuracy_series, title="probe accuracy", y_title="accuracy")

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curves.write_html(figure_dir / "12_fixed_width_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "12_fixed_width_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
)
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_width_32", small_accuracy),
            ("accuracy_grow_35", grown_accuracy),
            ("accuracy_width_37", large_accuracy),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="fixed-width",
            kind="method",
            statement=(
                "Eighty steps, no gate. Final probe accuracy "
                f"width-32={small_accuracy[-1]:.3f}, "
                f"grow-to-35={grown_accuracy[-1]:.3f}, "
                f"width-37={large_accuracy[-1]:.3f}."
            ),
            evidence_ref="results/figures/12_fixed_width_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "12_fixed_width_accuracy.provenance.json",
    figure="12_fixed_width_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "width_32": small_accuracy[-1],
            "grow_35": grown_accuracy[-1],
            "width_37": large_accuracy[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
