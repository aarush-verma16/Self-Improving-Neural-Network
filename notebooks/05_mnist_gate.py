# %% [markdown]
# # 05 — Gate on permuted MNIST
#
# The network from notebook 04 trains again on one pixel permutation. A new
# hidden unit is then added and turned into pure damage: it pushes every
# example toward class 0. The parent's weights are put back, so deleting that
# new unit returns the trained network. The gate is offered two prunes, scored
# by probe cross-entropy:
#
# 1. Remove the damaged unit. The loss should fall and accuracy should return.
# 2. Remove a unit the classifier is using. The loss should rise, so the edit
#    is refused and accuracy stays at the repaired value.
#
# Below zero on the gain chart means the probe loss went down. The accuracy
# chart has four points: trained, damaged, after the kept prune, after the
# refused prune. This is a method check on one short CPU run.

# %%
import json
import sys

import torch

from valmorph.benchmarks.classify import batch_accuracy, probe_cross_entropy, train_classifier
from valmorph.benchmarks.mnist import (
    apply_permutation,
    download_mnist,
    load_split,
    pixel_permutation,
)
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import LEDGER_VERSION, Finding, Ledger, RunMeta, make_run_id
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.gate import consider_edit
from valmorph.models import ValmorphMLP, prune, widen
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
net = ValmorphMLP([784, 32, 10]).cpu()
_losses, accuracies = train_classifier(
    net,
    train_x,
    train_y,
    steps=30,
    batch_size=64,
    lr=0.1,
    probe_images=probe_x,
    probe_labels=probe_y,
)
trained_accuracy = accuracies[-1]
widen(net, 0, 0)
with torch.no_grad():
    net.linear(1).weight[:, 0] *= 2
    net.linear(0).bias[-1] = 5.0
    net.linear(1).weight[:, -1] = 0.0
    net.linear(1).weight[0, -1] = 20.0
with torch.no_grad():
    damaged_accuracy = batch_accuracy(net(probe_x), probe_y)

# %%
run_id = make_run_id("mnist-gate", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="mnist-gate",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "mnist-gate", "widths": [784, 32, 10], "steps": 30},
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
repair = consider_edit(
    net,
    lambda shadow: prune(shadow, 0, int(shadow.linear(0).out_features) - 1),
    probe_x=probe_x,
    probe_y=probe_y,
    score=probe_cross_entropy,
    ledger=ledger,
    step=30,
    proposal_id="prune-junk",
    edit_type="prune",
    target="layer0",
)
with torch.no_grad():
    repaired_accuracy = batch_accuracy(net(probe_x), probe_y)
useful = int(torch.argmax(net.linear(1).weight.detach().abs().sum(dim=0)))
hurt = consider_edit(
    net,
    lambda shadow, unit=useful: prune(shadow, 0, unit),
    probe_x=probe_x,
    probe_y=probe_y,
    score=probe_cross_entropy,
    ledger=ledger,
    step=31,
    proposal_id="prune-useful",
    edit_type="prune",
    target="layer0",
    params={"unit": useful},
)
with torch.no_grad():
    final_accuracy = batch_accuracy(net(probe_x), probe_y)
moments = {
    "probe accuracy": [
        trained_accuracy,
        damaged_accuracy,
        repaired_accuracy,
        final_accuracy,
    ]
}
accuracy_figure = plot_diagnostic_traces(moments)
accuracy_figure.update_xaxes(
    tickvals=[0, 1, 2, 3],
    ticktext=["trained", "damaged", "kept prune", "refused prune"],
)
timeline = plot_gate_timeline(ledger.read("gate_decisions"), synthetic=False)

# %%
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
accuracy_figure.write_html(figure_dir / "05_mnist_gate_accuracy.html")
save_diagnostic_traces_png(moments, figure_dir / "05_mnist_gate_accuracy.png")
timeline.write_html(figure_dir / "05_mnist_gate_timeline.html")
save_gate_timeline_png(
    ledger.read("gate_decisions"),
    figure_dir / "05_mnist_gate_timeline.png",
    synthetic=False,
)
ledger.append_findings(
    [
        Finding(
            finding_id="mnist-gate",
            kind="method",
            statement=(
                f"Trained accuracy={trained_accuracy:.3f}, damaged={damaged_accuracy:.3f}, "
                f"after kept prune={repaired_accuracy:.3f}, "
                f"after refused prune={final_accuracy:.3f}. "
                f"Junk accepted={repair.accepted} gain={repair.validated_gain:.3f}. "
                f"Useful accepted={hurt.accepted} gain={hurt.validated_gain:.3f}."
            ),
            evidence_ref="results/figures/05_mnist_gate_timeline.provenance.json",
        )
    ]
)
manifest = write_provenance(
    figure_dir / "05_mnist_gate_timeline.provenance.json",
    figure="05_mnist_gate_timeline",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "trained": trained_accuracy,
            "damaged": damaged_accuracy,
            "repaired": repaired_accuracy,
            "final": final_accuracy,
            "repair_accepted": repair.accepted,
            "repair_gain": repair.validated_gain,
            "hurt_accepted": hurt.accepted,
            "hurt_gain": hurt.validated_gain,
            "synthetic": manifest["synthetic"],
        }
    )
)
accuracy_figure
# %%
timeline
