# %% [markdown]
# # 02 — Acceptance gate
#
# The network trains on a sine wave. A copied unit is then given a large outgoing
# weight so the fit breaks. The gate is offered two prunes:
#
# 1. Remove that damaged unit. The probe loss should fall, so the edit is kept.
# 2. Remove a unit the fit is using. The probe loss should rise, so the edit is refused
#    and the live network stays as it was.
#
# On the gain chart, **below zero means the probe loss went down**. The kept edit is
# the marker below zero. The refused edit is the marker above zero.
#
# Run this notebook yourself. It is a short CPU run.

# %%
import json
import sys

import torch

from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import LEDGER_VERSION, Finding, Ledger, RunMeta, make_run_id
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.gate import consider_edit, train_steps
from valmorph.gate.toy import prediction_grid, sine_split
from valmorph.models import ValmorphMLP, prune, widen
from valmorph.viz import (
    plot_fit,
    plot_gate_timeline,
    plot_loss_curve,
    save_gate_timeline_png,
    save_loss_curve_png,
    write_provenance,
)

root = repo_root()
cfg = load_run_config(root / "configs" / "setup.yaml")
torch.manual_seed(cfg.seed)
train_x, train_y, probe_x, probe_y = sine_split(seed=cfg.seed)
net = ValmorphMLP([1, 4, 1]).cpu()
losses = train_steps(net, train_x, train_y, steps=60, lr=0.05)
loss_figure = plot_loss_curve(losses)
widen(net, 0, 0)
with torch.no_grad():
    net.linear(0).bias[-1] = 3.0
    net.linear(1).weight[0, -1] = 4.0
grid, before_fit = prediction_grid(net)
truth = torch.sin(torch.linspace(-2, 2, 200)).tolist()

# %%
run_id = make_run_id("gate", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="gate",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "gate", "widths": [1, 4, 1], "train_steps": 60},
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
junk = int(net.linear(0).out_features) - 1
repair = consider_edit(
    net,
    lambda shadow: prune(shadow, 0, int(shadow.linear(0).out_features) - 1),
    probe_x=probe_x,
    probe_y=probe_y,
    ledger=ledger,
    step=60,
    proposal_id="prune-junk",
    edit_type="prune",
    target="layer0",
    params={"unit": junk},
)
useful = int(torch.argmax(net.linear(1).weight.detach().abs()))
hurt = consider_edit(
    net,
    lambda shadow, unit=useful: prune(shadow, 0, unit),
    probe_x=probe_x,
    probe_y=probe_y,
    ledger=ledger,
    step=61,
    proposal_id="prune-useful",
    edit_type="prune",
    target="layer0",
    params={"unit": useful},
)
_, after_fit = prediction_grid(net)
timeline = plot_gate_timeline(ledger.read("gate_decisions"), synthetic=False)
fit = plot_fit(grid, truth, before_fit, after_fit)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
loss_figure.write_html(figure_dir / "02_gate_loss.html")
save_loss_curve_png(losses, figure_dir / "02_gate_loss.png")
timeline.write_html(figure_dir / "02_gate_timeline.html")
decisions = ledger.read("gate_decisions")
save_gate_timeline_png(
    decisions,
    figure_dir / "02_gate_timeline.png",
    synthetic=False,
)
fit.write_html(figure_dir / "02_gate_fit.html")
ledger.append_findings(
    [
        Finding(
            finding_id="gate-toy",
            kind="method",
            statement=(
                "Remove junk accepted="
                f"{repair.accepted} gain={repair.validated_gain:.4f}. "
                "Remove useful accepted="
                f"{hurt.accepted} gain={hurt.validated_gain:.4f}."
            ),
            evidence_ref="results/figures/02_gate_timeline.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "02_gate_timeline.provenance.json",
    figure="02_gate_timeline",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "repair_accepted": repair.accepted,
            "repair_gain": repair.validated_gain,
            "hurt_accepted": hurt.accepted,
            "hurt_gain": hurt.validated_gain,
            "synthetic": manifest["synthetic"],
        }
    )
)
loss_figure
# %%
timeline
# %%
fit
