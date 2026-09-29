"""Concatenate every results/<name>/*.md table into results/<name>/ALL_RESULTS.md."""
import sys

from _common import load, results_dir

ORDER = ["e1_baselines", "e2_retrieval", "e3_ranking", "e4_e6_bandits", "e8_agent", "latency"]

if __name__ == "__main__":
    cfg, ds = load(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
    rd = results_dir(cfg)
    parts = [f"# All results: `{cfg['name']}`\n",
             "**SYNTHETIC DATA: not MovieLens results.**\n" if cfg["name"] == "synthetic" else ""]
    stats = rd / "data_stats.json"
    if stats.exists():
        parts.append("## Data\n\n```json\n" + stats.read_text() + "\n```\n")
    for name in ORDER:
        p = rd / f"{name}.md"
        parts.append(p.read_text() if p.exists() else f"_{name}: not run_\n")
    for fig in ("fig_retrieval_recall.png", "fig_regret.png", "fig_adaptation.png"):
        if (rd / fig).exists():
            parts.append(f"![{fig}]({fig})\n")
    (rd / "ALL_RESULTS.md").write_text("\n\n".join(parts))
    print("wrote", rd / "ALL_RESULTS.md")
