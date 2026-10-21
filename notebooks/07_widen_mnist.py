# %% [markdown]
# # 07 — Widen, then let the copy train
#
# Notebook 06 proposed pruning silent units. Those prunes did not lower the
# probe loss, so the gate refused them and the accuracy matched the no-edit
# network.
#
# This notebook proposes a widen instead. A widen does not change the function
# at the moment it is applied. The gate therefore trains the widened copy and
# an unedited copy for 8 steps on the same data, and keeps the widen only if
# the widened copy has lower probe loss.
#
# The first run kept one widen, `widen-30`, with gain -0.000589. Probe accuracy
# then moved from 0.634 with no edits to 0.668 with the proposal. That keep
# had copied the shadow's 8 extra training steps into the live network. The
# dashed curve still takes those steps when a widen is kept. The live network
# no longer does: a kept widen is applied at the current weights, and training
# continues from there. The gain chart is the gate's decision. The accuracy
# chart asks whether the extra unit changes probe accuracy once those steps
# stay on the dashed curve.
#
# Run this yourself. Short CPU run. Not yet a claim that self-edits help.

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
from valmorph.controllers import train_with_widen_trials
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import (
    LEDGER_VERSION,
    Finding,
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
permutation = pixel_permutation(cfg.seed)
train_x = apply_permutation(train_x[:2000], permutation)
train_y = train_y[:2000]
probe_x = apply_permutation(test_x[:500], permutation)
probe_y = test_y[:500]
editor = ValmorphMLP([784, 32, 10]).cpu()
plain = clone_net(editor)
matched = clone_net(editor)
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
run_id = make_run_id("widen-mnist", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="widen-mnist",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "widen-mnist",
            "widths": [784, 32, 10],
            "steps": 40,
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
editor_accuracy, matched_accuracy = train_with_widen_trials(
    editor,
    train_x,
    train_y,
    probe_images=probe_x,
    probe_labels=probe_y,
    steps=40,
    batch_size=64,
    lr=0.1,
    ledger=ledger,
    matched=matched,
    every=10,
    shadow_steps=8,
)
decisions = ledger.read("gate_decisions")
accepted_flags = [bool(value) for value in decisions.column("accepted").to_pylist()]
accepted = int(sum(accepted_flags))
proposed = int(decisions.num_rows)
kept_steps = [
    int(step)
    for step, flag in zip(decisions.column("step").to_pylist(), accepted_flags, strict=True)
    if flag
]
accuracy_series = {
    "no edits": plain_accuracy,
    "widen proposals": editor_accuracy,
    "extra steps only": matched_accuracy,
}
accuracy_dashes = {"extra steps only": "dash"}
curves = plot_diagnostic_traces(
    accuracy_series,
    title="probe accuracy",
    y_title="accuracy",
    marks=kept_steps,
    dashes=accuracy_dashes,
)
timeline = plot_gate_timeline(decisions, synthetic=False)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
curves.write_html(figure_dir / "07_widen_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "07_widen_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
    marks=kept_steps,
    dashes=accuracy_dashes,
)
timeline.write_html(figure_dir / "07_widen_timeline.html")
save_gate_timeline_png(decisions, figure_dir / "07_widen_timeline.png", synthetic=False)
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_no_edits", plain_accuracy),
            ("accuracy_extra_steps", matched_accuracy),
            ("accuracy_widen", editor_accuracy),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="widen-mnist",
            kind="method",
            statement=(
                f"Proposed {proposed} widens with 8 matched shadow steps, "
                f"accepted {accepted}. Final probe accuracy no-edit="
                f"{plain_accuracy[-1]:.3f}, extra-steps={matched_accuracy[-1]:.3f}, "
                f"with widens={editor_accuracy[-1]:.3f}."
            ),
            evidence_ref="results/figures/07_widen_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "07_widen_accuracy.provenance.json",
    figure="07_widen_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "proposed": proposed,
            "accepted": accepted,
            "plain_accuracy": plain_accuracy[-1],
            "matched_accuracy": matched_accuracy[-1],
            "editor_accuracy": editor_accuracy[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
# %%
timeline
