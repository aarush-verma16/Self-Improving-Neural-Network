# %% [markdown]
# # 08 — Random edits, same gate
#
# Notebook 06 proposed pruning silent units. The gate refused them. Notebook 07
# proposed widening unit 0. It kept one widen, and the accuracy matched a copy
# that took the extra training steps without the new unit.
#
# This notebook is the control those rules are compared with. At the same
# steps, it proposes either a prune or a widen of a random unit in the first
# hidden layer. The gate is the same: 8 matched training steps, keep the edit
# only when probe loss falls. A dashed curve takes those extra steps when a
# random edit is kept, and does not take the edit. The unit-0 widen rule from
# notebook 07 is on the same chart.
#
# One seed. Short CPU run. Not the Level 1 comparison.

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
from valmorph.controllers import train_with_random_edits, train_with_widen_trials
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
starter = ValmorphMLP([784, 32, 10]).cpu()
plain = clone_net(starter)
random_net = clone_net(starter)
random_matched = clone_net(starter)
widen_net = clone_net(starter)
widen_matched = clone_net(starter)
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
run_id = make_run_id("random-mnist", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="random-mnist",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={
            "experiment": "random-mnist",
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
random_accuracy, random_extra = train_with_random_edits(
    random_net,
    train_x,
    train_y,
    probe_images=probe_x,
    probe_labels=probe_y,
    steps=40,
    batch_size=64,
    lr=0.1,
    ledger=ledger,
    matched=random_matched,
    generator=torch.Generator().manual_seed(cfg.seed),
    every=10,
    shadow_steps=8,
)
widen_accuracy, _widen_extra = train_with_widen_trials(
    widen_net,
    train_x,
    train_y,
    probe_images=probe_x,
    probe_labels=probe_y,
    steps=40,
    batch_size=64,
    lr=0.1,
    ledger=ledger,
    matched=widen_matched,
    every=10,
    shadow_steps=8,
)
decisions = ledger.read("gate_decisions")
proposal_ids = [str(value) for value in decisions.column("proposal_id").to_pylist()]
accepted_flags = [bool(value) for value in decisions.column("accepted").to_pylist()]
random_accepted = sum(
    flag
    for name, flag in zip(proposal_ids, accepted_flags, strict=True)
    if name.startswith("random")
)
widen_accepted = sum(
    flag
    for name, flag in zip(proposal_ids, accepted_flags, strict=True)
    if name.startswith("widen")
)
kept_steps = [
    int(step)
    for step, flag in zip(decisions.column("step").to_pylist(), accepted_flags, strict=True)
    if flag
]
accuracy_series = {
    "no edits": plain_accuracy,
    "random edits": random_accuracy,
    "extra steps only": random_extra,
    "widen unit 0": widen_accuracy,
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
curves.write_html(figure_dir / "08_random_accuracy.html")
save_diagnostic_traces_png(
    accuracy_series,
    figure_dir / "08_random_accuracy.png",
    title="probe accuracy",
    y_title="accuracy",
    marks=kept_steps,
    dashes=accuracy_dashes,
)
timeline.write_html(figure_dir / "08_random_timeline.html")
save_gate_timeline_png(decisions, figure_dir / "08_random_timeline.png", synthetic=False)
ledger.append_step_metrics(
    [
        StepMetric(step=step, split="probe", name=name, value=float(value))
        for name, series in (
            ("accuracy_no_edits", plain_accuracy),
            ("accuracy_random", random_accuracy),
            ("accuracy_random_extra_steps", random_extra),
            ("accuracy_widen", widen_accuracy),
        )
        for step, value in enumerate(series)
    ]
)
ledger.append_findings(
    [
        Finding(
            finding_id="random-mnist",
            kind="method",
            statement=(
                f"Random edits accepted {random_accepted} of "
                f"{sum(name.startswith('random') for name in proposal_ids)}. "
                f"Unit-0 widens accepted {widen_accepted}. Final probe accuracy "
                f"no-edit={plain_accuracy[-1]:.3f}, random={random_accuracy[-1]:.3f}, "
                f"random-extra-steps={random_extra[-1]:.3f}, "
                f"widen={widen_accuracy[-1]:.3f}."
            ),
            evidence_ref="results/figures/08_random_accuracy.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "08_random_accuracy.provenance.json",
    figure="08_random_accuracy",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "random_accepted": random_accepted,
            "widen_accepted": widen_accepted,
            "plain_accuracy": plain_accuracy[-1],
            "random_accuracy": random_accuracy[-1],
            "random_extra": random_extra[-1],
            "widen_accuracy": widen_accuracy[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
curves
# %%
timeline
