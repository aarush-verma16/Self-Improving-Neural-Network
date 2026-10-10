# %% [markdown]
# # 01 — Substrate
#
# A small MLP on CPU. Widen and identity-deepen should leave the logits unchanged.
# This is a method check, not a learning result.

# %%
import json
import sys

import torch

from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import (
    LEDGER_VERSION,
    Finding,
    Ledger,
    RunMeta,
    make_run_id,
)
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.models import ValmorphMLP, identity_deepen, widen
from valmorph.viz import plot_morphism_error, save_morphism_error_png, write_provenance

root = repo_root()
cfg = load_run_config(root / "configs" / "setup.yaml")
torch.manual_seed(cfg.seed)
net = ValmorphMLP([8, 16, 16, 4]).cpu()
batch = torch.randn(32, 8)
labels: list[str] = []
errors: list[float] = []

before = net(batch).detach()
widen(net, 0, 3)
labels.append("widen layer 0")
errors.append(float((net(batch).detach() - before).abs().max()))
before = net(batch).detach()
identity_deepen(net, 0)
labels.append("identity deepen")
errors.append(float((net(batch).detach() - before).abs().max()))
before = net(batch).detach()
widen(net, 1, 0)
labels.append("widen layer 1")
errors.append(float((net(batch).detach() - before).abs().max()))

run_id = make_run_id("substrate", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="substrate",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "substrate", "widths": [8, 16, 16, 4]},
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
ledger.append_findings(
    [
        Finding(
            finding_id="morphism-error",
            kind="method",
            statement=(
                "On CPU, function-preserving widen and identity deepen changed "
                f"the MLP logits by at most {max(errors):.3e}."
            ),
            evidence_ref="results/figures/01_substrate_morphism.provenance.json",
        )
    ]
)
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
figure = plot_morphism_error(labels, errors)
html_path = figure_dir / "01_substrate_morphism.html"
png_path = figure_dir / "01_substrate_morphism.png"
manifest_path = figure_dir / "01_substrate_morphism.provenance.json"
figure.write_html(html_path)
save_morphism_error_png(labels, errors, png_path)
manifest = write_provenance(
    manifest_path,
    figure="01_substrate_morphism",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(json.dumps({"errors": errors, "max": max(errors), "synthetic": manifest["synthetic"]}))
