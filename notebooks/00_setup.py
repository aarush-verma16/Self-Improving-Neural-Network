# %% [markdown]
# # 00 — Setup
#
# Checks this machine and draws one interactive figure from a **synthetic**
# ledger. The figure is a workflow check. It is not a result.

# %%
from valmorph.core.config import config_hash, load_run_config
from valmorph.core.envcheck import collect_environment, write_environment
from valmorph.core.runtime import repo_root
from valmorph.core.synthetic import write_synthetic_demo
from valmorph.viz import plot_gate_timeline, save_gate_timeline_png, write_provenance

root = repo_root()
cfg = load_run_config(root / "configs" / "setup.yaml")
record = collect_environment(
    cfg.device,
    memory_fraction=cfg.compute.memory_fraction,
    allow_high_memory=cfg.compute.allow_high_memory,
    seed=cfg.seed,
)
env_path = root / "results" / "env" / "notebook_00_setup.json"
write_environment(record, env_path)
print(record["device_type"], record["device_name"], record["capability"], record["matmul_ok"])
if record["matmul_ok"] is not True:
    raise RuntimeError(record["error"])

# %%
ledger = write_synthetic_demo(root / "results" / "ledgers", seed=cfg.seed)
decisions = ledger.read("gate_decisions")
figure = plot_gate_timeline(decisions, synthetic=True)
figure_dir = root / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
html_path = figure_dir / "00_setup_gate_timeline.html"
png_path = figure_dir / "00_setup_gate_timeline.png"
manifest_path = figure_dir / "00_setup_gate_timeline.provenance.json"
figure.write_html(html_path)
save_gate_timeline_png(decisions, png_path, synthetic=True)
manifest = write_provenance(
    manifest_path,
    figure="00_setup_gate_timeline",
    ledgers=[ledger],
    config_hash=config_hash(cfg),
    seeds=[cfg.seed],
)
print(html_path)
print(png_path)
print("synthetic", manifest["synthetic"], "ledger", manifest["ledger_hashes"])
# Jupyter displays the last assigned figure when this cell runs.
displayed_figure = figure
