"""E1 — Do learned models beat popularity? (full-ranking, temporal split)

Protocol:
  1. small hyper-parameter grid, each config fit on TRAIN, scored on VAL
  2. best config per model re-fit on TRAIN+VAL, scored once on TEST
usage: python scripts/run_baselines.py configs/ml1m.yaml
"""
import sys
import time

from _common import load, results_dir

from agentrec.evaluation.evaluator import evaluate_model
from agentrec.models.baselines import ContentBased, ItemKNN, Popularity, RecentPopularity
from agentrec.models.mf_bpr import BPRMF
from agentrec.models.two_tower import TwoTower
from agentrec.utils.common import get_logger, save_json, set_seed

log = get_logger("e1")


def grid(cfg: dict) -> dict[str, list]:
    b, t, k = cfg["models"]["bpr"], cfg["models"]["two_tower"], cfg["models"]["itemknn"]
    return {
        "popularity": [lambda: Popularity()],
        "recent_popularity": [lambda h=h: RecentPopularity(half_life_days=h) for h in (7, 30, 90)],
        "content": [lambda w=w: ContentBased(decade_weight=w) for w in (0.0, 0.5)],
        "itemknn": [lambda n=n: ItemKNN(k_neighbors=n, shrink=k["shrink"]) for n in (50, 100, 200)],
        "bpr_mf": [lambda r=r: BPRMF(**{**b, "reg": r}, seed=cfg["seed"]) for r in (1e-4, 5e-4, 2e-3)],
        "two_tower": [lambda: TwoTower(**t, seed=cfg["seed"])],
    }


def describe(m) -> str:
    keys = ("half_life_days", "decade_weight", "k", "reg", "dim", "tau")
    return ", ".join(f"{k}={getattr(m, k)}" for k in keys if hasattr(m, k)) or "-"


if __name__ == "__main__":
    cfg, ds = load(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
    set_seed(cfg["seed"])
    ks = cfg["eval"]["ks"]
    out = {"dataset": cfg["name"], "protocol": "fit train -> select on val; refit train+val -> test", "models": {}}
    for name, makers in grid(cfg).items():
        best, best_val, tuning = None, -1.0, []
        for make in makers:
            m = make()
            t0 = time.time()
            m.fit(ds, ds.fit_period("val"))
            v = evaluate_model(m.score, ds, "val", ks)["all"]["ndcg@10"]
            tuning.append({"config": describe(m), "val_ndcg@10": v, "fit_s": time.time() - t0})
            log.info(f"{name} [{describe(m)}] val ndcg@10={v:.4f}")
            if v > best_val:
                best, best_val = make, v
        m = best()
        t0 = time.time()
        m.fit(ds, ds.fit_period("test"))
        fit_s = time.time() - t0
        t0 = time.time()
        res = evaluate_model(m.score, ds, "test", ks)
        res["fit_s"], res["eval_s"], res["config"], res["tuning"] = fit_s, time.time() - t0, describe(m), tuning
        out["models"][name] = res
        log.info(f"{name} TEST ndcg@10={res['all']['ndcg@10']:.4f} recall@50={res['all']['recall@50']:.4f}")

    rd = results_dir(cfg)
    save_json(out, rd / "e1_baselines.json")
    cols = ["ndcg@10", "recall@10", "precision@10", "mrr@10", "map@10", "recall@50", "ndcg@50"]
    lines = [f"# E1 — baselines on `{cfg['name']}` (TEST, full ranking)\n",
             f"users evaluated: {int(next(iter(out['models'].values()))['all']['n_users'])}\n",
             "| model | config | " + " | ".join(cols) + " | warm ndcg@10 | sparse ndcg@10 | cold ndcg@10 | tail recall@10 | coverage@10 | mean pop pct@10 | fit s |",
             "|" + "---|" * (len(cols) + 9)]
    for name, r in out["models"].items():
        a = r["all"]
        g = lambda s, k="ndcg@10": f"{r[s][k]:.4f}" if s in r else "n/a"  # noqa: E731
        lines.append(f"| {name} | {r['config']} | " + " | ".join(f"{a[c]:.4f}" for c in cols)
                     + f" | {g('warm_users')} | {g('sparse_users')} | {g('cold_users')} | {r['tail']['recall@10']:.4f}"
                     + f" | {r['diversity']['coverage@10']:.3f} | {r['diversity']['mean_pop_percentile@10']:.3f} | {r['fit_s']:.1f} |")
    (rd / "e1_baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
