"""Systems numbers: per-stage latency, throughput, memory footprint (single process, CPU).

usage: python scripts/run_latency.py configs/ml1m.yaml
"""
import pickle
import sys
import time

import numpy as np

from _common import ROOT, load, results_dir

from agentrec.agents.agent import RecAgent
from agentrec.agents.tools import RecTools
from agentrec.bandits.env import BanditEnv
from agentrec.bandits.linear import LinUCB
from agentrec.bandits.simulator import Simulator, TruthModel
from agentrec.llm.rule_based import RuleBasedPlanner
from agentrec.memory.store import MemoryStore
from agentrec.pipeline import FeedPipeline, fit_stage1
from agentrec.utils.common import LatencyStats, save_json, set_seed

if __name__ == "__main__":
    cfg, ds = load(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
    rng = set_seed(cfg["seed"])
    cache = ROOT / "data" / "processed"
    s1 = fit_stage1(ds, "test", cfg, cache_dir=cache, seed=cfg["seed"])
    rp = cache / f"ranker_{cfg['name']}.pkl"
    ranker = pickle.load(open(rp, "rb")) if rp.exists() else None
    out = {"dataset": cfg["name"], "n_items": ds.n_items}
    users = [u for u in range(ds.n_users) if s1.tower.user_hist.get(u) is not None]
    users = [int(u) for u in rng.choice(users, size=min(300, len(users)), replace=False)]
    N = cfg["ranking"]["n_candidates"]

    for index in ("exact", "ivf"):
        pipe = FeedPipeline(ds, "test", s1, ranker=ranker, index=index, ivf={**cfg["retrieval"]["ivf"], "seed": 0})
        dummy = TruthModel(np.zeros((ds.n_users, 2)), np.zeros((ds.n_items, 2)))
        env = BanditEnv(pipe, Simulator(dummy), n_candidates=cfg["bandit"]["n_candidates"], slate_size=5)
        pol = LinUCB(env.d, alpha=1.0, theta0=env.theta0())
        st = {k: LatencyStats() for k in ("user_embedding", "retrieval", "features", "ranker", "bandit", "total")}
        for u in users:
            t_all = time.perf_counter()
            t0 = time.perf_counter(); hist = pipe.history(u); uv = pipe.user_vec(hist); st["user_embedding"].add(time.perf_counter() - t0)
            t0 = time.perf_counter(); c = pipe.retriever.retrieve(uv, N, exclude=pipe._seen[u] | set(hist.tolist())); st["retrieval"].add(time.perf_counter() - t0)
            t0 = time.perf_counter(); F = pipe.fb.build(u, c.items, c.scores, c.sources, hist_items=hist, user_vec=uv); st["features"].add(time.perf_counter() - t0)
            t0 = time.perf_counter(); s = pipe.ranker.score(F); order = np.argsort(-s); st["ranker"].add(time.perf_counter() - t0)
            st["total"].add(time.perf_counter() - t_all)   # retrieve + rank (bandit timed separately)
            ctx = env.context(u)                            # untimed: builds the bandit's context matrix
            t0 = time.perf_counter(); pol.select(ctx.X, 5, rng); st["bandit"].add(time.perf_counter() - t0)
        out[f"feed_{index}"] = {k: v.summary() for k, v in st.items()}
        tot = out[f"feed_{index}"]["total"]["mean_ms"] + out[f"feed_{index}"]["bandit"]["mean_ms"]
        out[f"feed_{index}"]["throughput_req_per_s_sequential"] = 1000.0 / tot

    tools = RecTools(ds, FeedPipeline(ds, "test", s1, ranker=ranker), MemoryStore())
    lat, calls = LatencyStats(), []
    for u in users[:100]:
        t0 = time.perf_counter()
        r = RecAgent(RuleBasedPlanner(ds.titles), tools).run(u, "Comedy movies from the 90s but no Horror")
        lat.add(time.perf_counter() - t0)
        calls.append(r.n_tool_calls)
    out["agent_rule_based"] = {**lat.summary(), "mean_tool_calls": float(np.mean(calls)),
                               "note": "planner is rule-based; a real LLM adds (#llm_calls x LLM latency), typically seconds"}

    mb = lambda a: a.nbytes / 1e6  # noqa: E731
    out["memory_mb"] = {"item_embeddings": mb(s1.tower.V), "two_tower_params": sum(mb(v) for v in s1.tower.p.values()),
                        "itemknn_similarity_sparse": (s1.knn.S.data.nbytes + s1.knn.S.indices.nbytes) / 1e6}
    save_json(out, results_dir(cfg) / "latency.json")
    L = [f"# Latency on `{cfg['name']}` ({ds.n_items} items, N={N} candidates, {len(users)} requests, single process, CPU)\n",
         "| stage (total = embed+retrieve+features+rank) | exact p50 ms | exact p95 ms | IVF p50 ms | IVF p95 ms |", "|---|---|---|---|---|"]
    for k in ("user_embedding", "retrieval", "features", "ranker", "bandit", "total"):
        e, i = out["feed_exact"][k], out["feed_ivf"][k]
        L.append(f"| {k} | {e['p50_ms']:.3f} | {e['p95_ms']:.3f} | {i['p50_ms']:.3f} | {i['p95_ms']:.3f} |")
    L.append(f"\nThroughput estimate (1000 / mean latency of total + bandit, sequential, exact): "
             f"{out['feed_exact']['throughput_req_per_s_sequential']:.0f} req/s. Not a load test. "
             "`bandit` times only `select()` on a prebuilt context matrix; building the context re-uses retrieval + ranking.")
    a = out["agent_rule_based"]
    L.append(f"\nAgent request (rule-based planner): p50 {a['p50_ms']:.1f} ms, p95 {a['p95_ms']:.1f} ms, "
             f"{a['mean_tool_calls']:.1f} tool calls. {a['note']}.")
    L.append("\nMemory (MB): " + ", ".join(f"{k}={v:.2f}" for k, v in out["memory_mb"].items()))
    (results_dir(cfg) / "latency.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
