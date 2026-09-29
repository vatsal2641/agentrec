"""E3 — Is a second-stage ranker worth it?  + candidate-size and no-retrieval ablations.

  stage-1 fit on TRAIN -> candidates for VAL users -> labels from VAL -> train rankers
  (ranker selection on a held-out 20% of VAL users)
  stage-1 re-fit on TRAIN+VAL -> candidates for TEST users -> each ranker -> metrics on TEST
usage: python scripts/run_ranking.py configs/ml1m.yaml
"""
import pickle
import sys
import time

import numpy as np

from _common import ROOT, load, results_dir

from agentrec.evaluation.metrics import evaluate_lists, ndcg_at_k
from agentrec.pipeline import FeedPipeline, fit_stage1
from agentrec.ranking.ranker import FEATURES, Ranker
from agentrec.retrieval.retriever import Candidates
from agentrec.utils.common import LatencyStats, get_logger, save_json, set_seed

log = get_logger("e3")


def build_groups(pipe: FeedPipeline, targets: dict, users: list[int], n: int):
    groups, lists = [], {}
    for u in users:
        c = pipe.candidates(u, n)
        items, _, F = pipe.rank(u, c)           # identity ranker -> retrieval order
        y = np.array([int(i in targets[u]) for i in items])
        groups.append((u, items, F, y))
    return groups


if __name__ == "__main__":
    cfg, ds = load(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
    rng = set_seed(cfg["seed"])
    cache = ROOT / "data" / "processed"
    N = cfg["ranking"]["n_candidates"]
    ks = cfg["eval"]["ks"]

    # ---------------- train rankers on the VAL window ----------------
    s1v = fit_stage1(ds, "val", cfg, cache_dir=cache, seed=cfg["seed"])
    pv = FeedPipeline(ds, "val", s1v)
    tv = ds.val.targets()
    vusers = [u for u in sorted(tv) if s1v.tower.user_hist.get(u) is not None]
    groups = build_groups(pv, tv, vusers, N)
    perm = rng.permutation(len(groups))
    cut = int(0.8 * len(groups))
    tr = [groups[i] for i in perm[:cut]]
    ho = [groups[i] for i in perm[cut:]]
    pos_rate = float(np.mean(np.concatenate([g[3] for g in groups])))
    log.info(f"ranker training: {len(tr)} users, holdout {len(ho)}, candidate positive rate {pos_rate:.4f}")

    rankers, sel = {}, {}
    for kind in ("identity", "pointwise", "pairwise", "gbdt"):
        r = Ranker(kind, seed=cfg["seed"]).fit([(F, y) for _, _, F, y in tr if y.sum() > 0])
        rankers[kind] = r
        sel[kind] = float(np.mean([ndcg_at_k(items[np.argsort(-r.score(F), kind="stable")].tolist(), tv[u], 10)
                                   for u, items, F, y in ho]))
        log.info(f"{kind}: holdout-val ndcg@10 (within candidates) = {sel[kind]:.4f}")
    chosen = max(sel, key=sel.get)

    # ---------------- evaluate on TEST ----------------
    s1t = fit_stage1(ds, "test", cfg, cache_dir=cache, seed=cfg["seed"])
    tt = ds.test.targets()
    tusers = [u for u in sorted(tt) if s1t.tower.user_hist.get(u) is not None]
    test_res = {}
    for kind, r in rankers.items():
        pt = FeedPipeline(ds, "test", s1t, ranker=r)
        lat = LatencyStats()
        lists = {}
        for u in tusers:
            t0 = time.perf_counter()
            c = pt.candidates(u, N)
            items, _, _ = pt.rank(u, c)
            lat.add(time.perf_counter() - t0)
            lists[u] = items[:max(ks)].tolist()
        test_res[kind] = {**evaluate_lists(lists, tt, ks), "latency": lat.summary()}
        log.info(f"TEST {kind}: ndcg@10={test_res[kind]['ndcg@10']:.4f}")

    # ---------------- ablation: candidate set size ----------------
    best = rankers[chosen]
    pt = FeedPipeline(ds, "test", s1t, ranker=best)
    size_res = {}
    for n in cfg["retrieval"]["n_candidates"]:
        lat, lists = LatencyStats(), {}
        for u in tusers:
            t0 = time.perf_counter()
            items, _, _ = pt.rank(u, pt.candidates(u, n))
            lat.add(time.perf_counter() - t0)
            lists[u] = items[:max(ks)].tolist()
        size_res[n] = {**evaluate_lists(lists, tt, ks), "latency": lat.summary()}
        log.info(f"N={n}: ndcg@10={size_res[n]['ndcg@10']:.4f} p50={lat.summary()['p50_ms']:.1f}ms")

    # ---------------- ablation: no retrieval (rank the whole catalogue) ----------------
    sub = [tusers[i] for i in rng.permutation(len(tusers))[:300]]
    lat, lists = LatencyStats(), {}
    for u in sub:
        t0 = time.perf_counter()
        hist = pt.history(u)
        uv = pt.user_vec(hist)
        seen = pt._seen[u] | set(hist.tolist())
        allitems = np.array([i for i in np.argsort(-(s1t.tower.V @ uv)) if i not in seen])
        c = Candidates(allitems, s1t.tower.V[allitems] @ uv, ["embedding"] * len(allitems))
        items, _, _ = pt.rank(u, c)
        lat.add(time.perf_counter() - t0)
        lists[u] = items[:max(ks)].tolist()
    with_retr = evaluate_lists({u: pt.rank(u, pt.candidates(u, N))[0][:max(ks)].tolist() for u in sub}, tt, ks)
    no_retr = {"no_retrieval": {**evaluate_lists(lists, tt, ks), "latency": lat.summary()},
               "with_retrieval_same_users": with_retr, "n_users": len(sub)}
    log.info(f"no retrieval: ndcg@10={no_retr['no_retrieval']['ndcg@10']:.4f} vs {with_retr['ndcg@10']:.4f}")

    with open(cache / f"ranker_{cfg['name']}.pkl", "wb") as f:
        pickle.dump(best, f)
    out = {"dataset": cfg["name"], "n_candidates": N, "candidate_positive_rate": pos_rate,
           "selection_holdout_val_ndcg@10": sel, "chosen": chosen, "test": test_res,
           "coefficients": {k: r.coefficients() for k, r in rankers.items()}, "features": FEATURES,
           "ablation_candidate_size": size_res, "ablation_no_retrieval": no_retr}
    save_json(out, results_dir(cfg) / "e3_ranking.json")
    lines = [f"# E3 — ranking on `{cfg['name']}` (TEST, warm users={len(tusers)}, N={N} candidates)\n",
             f"Chosen on held-out VAL users: **{chosen}**\n",
             "| ranker | holdout-VAL ndcg@10 | TEST ndcg@10 | recall@10 | mrr@10 | map@10 | recall@50 | p50 ms/request |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in test_res.items():
        lines.append(f"| {k} | {sel[k]:.4f} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {r['mrr@10']:.4f} | "
                     f"{r['map@10']:.4f} | {r['recall@50']:.4f} | {r['latency']['p50_ms']:.1f} |")
    lines += ["", f"Candidate-size ablation ({chosen} ranker)", "", "| N | ndcg@10 | recall@50 | p50 ms | p95 ms |", "|---|---|---|---|---|"]
    for n, r in size_res.items():
        lines.append(f"| {n} | {r['ndcg@10']:.4f} | {r['recall@50']:.4f} | {r['latency']['p50_ms']:.1f} | {r['latency']['p95_ms']:.1f} |")
    nr = no_retr["no_retrieval"]
    lines += ["", f"No-retrieval ablation ({len(sub)} users): rank whole catalogue ndcg@10={nr['ndcg@10']:.4f}, "
              f"p50={nr['latency']['p50_ms']:.1f} ms  vs  retrieval N={N}: ndcg@10={with_retr['ndcg@10']:.4f}"]
    if rankers["pointwise"].coefficients():
        lines += ["", "Pointwise LR coefficients (standardised features): " + str(rankers["pointwise"].coefficients())]
    (results_dir(cfg) / "e3_ranking.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
