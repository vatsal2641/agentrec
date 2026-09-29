"""Figures for README/EXPERIMENTS from results/<name>/*.json (static PNGs).

Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green),
assigned per series in fixed order. Several hues are < 3:1 contrast on white, so
every line is direct-labelled and every figure has a matching markdown table.
usage: python scripts/make_plots.py configs/ml1m.yaml
"""
import json
import sys

from _common import load, results_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def label_ends(ax, ends):
    """ends: list of (x, y, text, color). Direct labels, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.055 * (hi - lo)
    placed = []
    for x, y, text, color in sorted(ends, key=lambda e: e[1]):
        ly = max(y, placed[-1] + gap) if placed else y
        placed.append(ly)
        ax.plot([x], [y], "o", color=color, markersize=4)
        ax.annotate(text, (x, y), xytext=(x, ly), textcoords="data", va="center", fontsize=8, color=INK,
                    xycoords="data", annotation_clip=False)


if __name__ == "__main__":
    cfg, _ = load(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
    rd = results_dir(cfg)
    tag = " [SYNTHETIC data]" if cfg["name"] == "synthetic" else f" [{cfg['name']}]"

    b = json.loads((rd / "e4_e6_bandits.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
    T = b["rounds"]
    ends = []
    for c, (name, r) in zip(PAL, b["e4_policies"].items()):
        ys = r["regret_curve"]
        xs = [int(x) + 1 for x in __import__("numpy").linspace(0, T - 1, len(ys))]   # matches run_bandits sampling
        ax.plot(xs, ys, color=c, linewidth=2, label=name)
        ends.append((xs[-1], ys[-1], f"  {name}", c))
    label_ends(ax, ends)
    style(ax, "Cumulative regret (simulated users)" + tag, "round", "cumulative regret (expected clicks)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(right=T * 1.3)
    fig.tight_layout(); fig.savefig(rd / "fig_regret.png", dpi=150); plt.close(fig)

    e5 = b["e5_new_user_adaptation"]
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
    ends = []
    for c, name in zip(PAL, ("no_adaptation", "user_state_update", "user_state_and_policy_update")):
        ys = e5[name]["expected_clicks_by_session"]
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=c, linewidth=2, label=name)
        ends.append((xs[-1], ys[-1], "  " + name.replace("_", " "), c))
    ax.set_ylim(0, max(e[1] for e in ends) * 1.25)
    label_ends(ax, ends)
    style(ax, "New users: expected clicks per slate by session" + tag, "session", "expected clicks per slate")
    ax.set_xlim(0.5, len(ys) + 6.5); ax.set_xticks(range(1, len(ys) + 1, 2))
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout(); fig.savefig(rd / "fig_adaptation.png", dpi=150); plt.close(fig)

    e2 = json.loads((rd / "e2_retrieval.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURF)
    ns = [int(n) for n in e2["recall_at_N"]]
    ax.set_ylim(0, 1.0)
    ends = []
    for c, key in zip(PAL, ("embedding_only", "union_90_10", "popular_only")):
        ys = [e2["recall_at_N"][str(n)][key] for n in ns]
        ax.plot(ns, ys, color=c, linewidth=2, marker="o", markersize=4, label=key)
        ends.append((ns[-1], ys[-1], f"  {key.replace('_', ' ')} {ys[-1]:.2f}", c))
    ax.set_xscale("log"); ax.set_xlim(ns[0] * 0.8, ns[-1] * 3.5)
    label_ends(ax, ends)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    style(ax, "Candidate recall vs candidate-set size" + tag, "N (candidates)", "Recall@N (test positives)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(rd / "fig_retrieval_recall.png", dpi=150); plt.close(fig)
    print("wrote", [p.name for p in rd.glob("fig_*.png")])
