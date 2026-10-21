# %% [markdown]
# # 03 — Diagnostics
#
# The same small sine-fitting network, watched while it trains. Three measurements,
# all computed by us from the hidden units:
#
# - **Dead-unit fraction.** Share of ReLU units that output nothing on this batch.
# - **Singular entropy rank.** How many directions the hidden activations actually use.
#   A value near 1 means the units are repeating one pattern. A higher value means
#   they are doing different things. This is not a published CKA number.
# - **Gradient norm.** How large the weight update is on this batch.
#
# The heatmap is the covariance of the hidden units at the last step. Bright
# off-diagonal cells mean two units move together.
#
# Run this notebook yourself. It is a short CPU run.

# %%
import json
import sys

import torch

from valmorph.core.config import config_hash, load_run_config
from valmorph.core.ledger import LEDGER_VERSION, Diagnostic, Finding, Ledger, RunMeta, make_run_id
from valmorph.core.runtime import git_commit, library_versions, platform_string, repo_root
from valmorph.gate.toy import sine_split
from valmorph.introspect import (
    activation_covariance,
    dead_unit_fraction,
    hidden_activations,
    layer_gradient_norms,
    mean_off_diagonal,
    singular_entropy_rank,
)
from valmorph.models import ValmorphMLP
from valmorph.viz import (
    plot_covariance_heatmap,
    plot_diagnostic_traces,
    save_diagnostic_traces_png,
    write_provenance,
)

root = repo_root()
cfg = load_run_config(root / "configs" / "setup.yaml")
torch.manual_seed(cfg.seed)
train_x, train_y, _probe_x, _probe_y = sine_split(seed=cfg.seed)
net = ValmorphMLP([1, 8, 1]).cpu()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
dead: list[float] = []
rank: list[float] = []
grad_norm: list[float] = []
rows: list[Diagnostic] = []
for step in range(60):
    norms = layer_gradient_norms(net, train_x, train_y)
    activations = hidden_activations(net, train_x)
    dead_value = dead_unit_fraction(activations[0])
    rank_value = singular_entropy_rank(activations[0])
    dead.append(dead_value)
    rank.append(rank_value)
    grad_norm.append(norms[0])
    rows.append(Diagnostic(step=step, layer="hidden0", name="dead_unit_fraction", value=dead_value))
    rows.append(
        Diagnostic(step=step, layer="hidden0", name="singular_entropy_rank", value=rank_value)
    )
    rows.append(Diagnostic(step=step, layer="layer0", name="gradient_norm", value=norms[0]))
    optimizer.zero_grad()
    loss = torch.mean((net(train_x) - train_y) ** 2)
    loss.backward()
    optimizer.step()

final_covariance = activation_covariance(hidden_activations(net, train_x)[0])
traces = plot_diagnostic_traces(
    {
        "dead unit fraction": dead,
        "singular entropy rank": rank,
        "gradient norm": grad_norm,
    }
)
heatmap = plot_covariance_heatmap(final_covariance.tolist())

# %%
run_id = make_run_id("diagnostics", cfg.seed, config_hash(cfg))
ledger = Ledger.create(
    root / "results" / "ledgers",
    RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=False,
        experiment="diagnostics",
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        config={"experiment": "diagnostics", "widths": [1, 8, 1], "steps": 60},
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
ledger.append_diagnostics(rows)
ledger.append_findings(
    [
        Finding(
            finding_id="diagnostics-sine",
            kind="method",
            statement=(
                f"Final dead fraction={dead[-1]:.3f}, "
                f"singular entropy rank={rank[-1]:.3f}, "
                f"gradient norm={grad_norm[-1]:.3f}, "
                f"mean off-diagonal covariance={mean_off_diagonal(final_covariance):.3f}."
            ),
            evidence_ref="results/figures/03_diagnostics_traces.provenance.json",
        )
    ]
)
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
traces.write_html(figure_dir / "03_diagnostics_traces.html")
save_diagnostic_traces_png(
    {
        "dead unit fraction": dead,
        "singular entropy rank": rank,
        "gradient norm": grad_norm,
    },
    figure_dir / "03_diagnostics_traces.png",
)
heatmap.write_html(figure_dir / "03_diagnostics_covariance.html")
manifest = write_provenance(
    figure_dir / "03_diagnostics_traces.provenance.json",
    figure="03_diagnostics_traces",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(
    json.dumps(
        {
            "final_dead_fraction": dead[-1],
            "final_rank": rank[-1],
            "final_gradient_norm": grad_norm[-1],
            "synthetic": manifest["synthetic"],
        }
    )
)
traces
# %%
heatmap
