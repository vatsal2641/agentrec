"""E2 — How much recall does candidate retrieval keep, and what does ANN cost/buy?

  * Recall@N of the candidate set (embedding only / popularity only / union) vs TEST positives
  * exact vs IVF (numpy) vs FAISS (if installed): ANN-recall against exact search and per-query latency
usage: python scripts/run_retrieval.py configs/ml1m.yaml
"""
import sys
import time

import numpy as np

from _common import ROOT, load, results_dir

from agentrec.pipeline import fit_stage1
from agentrec.retrieval.index import ExactIndex, IVFIndex
from agentrec.retrieval.retriever import CandidateRetriever
from agentrec.utils.common import LatencyStats, get_logger, save_json, set_seed

log = get_logger("e2")

if __name__ == "__main__":
    cfg, ds = load(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
    set_seed(cfg["seed"])
    s1 = fit_stage1(ds, "test", cfg, cache_dir=ROOT / "data" / "processed", seed=cfg["seed"])
    tt = s1.tower
    V = tt.item_embeddings()
    targets = ds.test.targets()
    seen = ds.matrix(ds.fit_period("test"), positive_only=False)
    warm = [u for u in sorted(targets) if tt.user_hist.get(u) is not None]
    cold = [u for u in sorted(targets) if tt.user_hist.get(u) is None]
    popular = np.argsort(-s1.recent.s)
    exact = ExactIndex(V)
    U = {u: tt.embed_history(tt.user_hist[u]) for u in warm}
    out = {"dataset": cfg["name"], "n_items": ds.n_items, "dim": int(V.shape[1]),
           "n_warm_users": len(warm), "n_cold_users": len(cold), "recall_at_N": {}}

    def recall(retr, users, n, vec):
        r = []
        for u in users:
            c = retr.retrieve(vec(u), n, exclude=set(seen[u].indices.tolist()))
            r.append(len(set(c.items.tolist()) & targets[u]) / len(targets[u]))
        return float(np.mean(r))

    for n in cfg["retrieval"]["n_candidates"]:
        row = {}
        for name, share in [("embedding_only", 0.0), ("popular_only", 1.0), ("union_90_10", 0.1)]:
            retr = CandidateRetriever(exact, V, popular, pop_share=share)
            row[name] = recall(retr, warm, n, lambda u: U[u])
        row["cold_users_popular"] = recall(CandidateRetriever(exact, V, popular), cold, n, lambda u: None) if cold else None
        out["recall_at_N"][n] = row
        log.info(f"N={n}: {row}")

    # ---- ANN: speed vs accuracy --------------------------------------
    Q = np.array([U[u] for u in warm[:300]])
    k = 100
    truth, _ = exact.search(Q, k)
    lat = LatencyStats()
    for q in Q:
        t0 = time.perf_counter(); exact.search(q, k); lat.add(time.perf_counter() - t0)
    ann = {"exact": {"ann_recall@100": 1.0, **lat.summary()}}
    n_lists = cfg["retrieval"]["ivf"]["n_lists"]
    t0 = time.perf_counter()
    ivf = IVFIndex(V, n_lists=n_lists, n_probe=1, seed=cfg["seed"])
    build_s = time.perf_counter() - t0
    for p in (1, 2, 4, 8, 16, 32):
        if p > n_lists:
            continue
        ivf.n_probe = p
        lat = LatencyStats()
        got = []
        for q in Q:
            t0 = time.perf_counter(); i, _ = ivf.search(q, k); lat.add(time.perf_counter() - t0); got.append(i[0])
        rec = float(np.mean([len(set(g.tolist()) & set(t.tolist())) / k for g, t in zip(got, truth)]))
        ann[f"ivf_nlist{n_lists}_nprobe{p}"] = {"ann_recall@100": rec, "frac_items_scanned": p / n_lists, **lat.summary()}
        log.info(f"IVF nprobe={p}: recall={rec:.3f} p50={lat.summary()['p50_ms']:.3f}ms")
    ann["ivf_build_s"] = build_s
    try:
        from agentrec.retrieval.index import FaissIndex
        fx = FaissIndex(V)
        lat = LatencyStats()
        for q in Q:
            t0 = time.perf_counter(); fx.search(q, k); lat.add(time.perf_counter() - t0)
        ann["faiss_flat_ip"] = {"ann_recall@100": 1.0, **lat.summary()}
    except ImportError:
        ann["faiss_flat_ip"] = "not installed (pip install faiss-cpu to compare)"
    out["ann"] = ann

    # ---- raw dot-product throughput -> extrapolation (clearly labelled) ----
    big = np.random.default_rng(0).standard_normal((200_000, V.shape[1])).astype(np.float32)
    q = big[0]
    t0 = time.perf_counter()
    for _ in range(20):
        big @ q
    per_item_ns = (time.perf_counter() - t0) / 20 / len(big) * 1e9
    out["bruteforce_extrapolation"] = {
        "measured_ns_per_item_dot": per_item_ns,
        "EXTRAPOLATED_ms_per_query_10M_items": per_item_ns * 1e7 / 1e6,
        "note": "single-thread numpy on this machine; extrapolation, not a measurement at 10M items",
    }
    save_json(out, results_dir(cfg) / "e2_retrieval.json")
    lines = [f"# E2 — retrieval on `{cfg['name']}` (TEST positives, warm users={len(warm)}, cold={len(cold)})\n",
             "| N | embedding only | popular only | union (90/10) | cold users (popular) |", "|---|---|---|---|---|"]
    for n, r in out["recall_at_N"].items():
        c = f"{r['cold_users_popular']:.4f}" if r["cold_users_popular"] is not None else "n/a"
        lines.append(f"| {n} | {r['embedding_only']:.4f} | {r['popular_only']:.4f} | {r['union_90_10']:.4f} | {c} |")
    lines += ["", "| index | ANN recall@100 vs exact | items scanned | p50 ms | p95 ms |", "|---|---|---|---|---|"]
    for name, r in ann.items():
        if isinstance(r, dict):
            lines.append(f"| {name} | {r['ann_recall@100']:.3f} | {r.get('frac_items_scanned', 1.0):.3f} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} |")
    e = out["bruteforce_extrapolation"]
    lines.append(f"\nBrute-force dot product: {e['measured_ns_per_item_dot']:.2f} ns/item measured -> "
                 f"~{e['EXTRAPOLATED_ms_per_query_10M_items']:.0f} ms/query at 10M items (EXTRAPOLATED).")
    (results_dir(cfg) / "e2_retrieval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
