"""E8 — Does the agent loop help on free-text requests? Are its guardrails effective?

Requests are generated from templates with KNOWN constraints (ground truth comes
from the template, not from any parser), so parser mistakes count as failures.

Systems
  feed         : ignores the request text (the plain recommender)
  fixed        : regex-parse -> one filtered retrieval -> rank. No replanning, no memory.
  agent_rule   : agent loop driven by the RULE-BASED planner (tools, memory, relaxation)
  agent_halluc : same, but the planner injects a non-existent id at the first `recommend` (guardrail stress)
  agent_llm    : optional, a real LLM via an OpenAI-compatible endpoint:
                 --llm http://localhost:11434/v1 qwen2.5:7b-instruct
usage: python scripts/run_agent_eval.py configs/ml1m.yaml [--n 150] [--llm URL MODEL]
"""
import pickle
import sys
import time

import numpy as np

from _common import ROOT, load, results_dir

from agentrec.agents.agent import RecAgent
from agentrec.agents.intent import parse_request
from agentrec.agents.tools import RecTools, Session
from agentrec.llm.rule_based import HallucinatingPlanner, RuleBasedPlanner
from agentrec.memory.store import MemoryStore
from agentrec.pipeline import FeedPipeline, fit_stage1
from agentrec.utils.common import LatencyStats, get_logger, save_json, set_seed

log = get_logger("e8")
K = 5


def make_requests(ds, pipe, users, rng):
    G = ds.genre_names
    reqs = []
    for u in users:
        hist = pipe.history(u)
        seed = int(hist[rng.integers(len(hist))])
        g_user = [G[k] for k in np.argsort(-ds.item_genres[hist].sum(0))[:3]]
        g1 = g_user[0]
        g2 = str(rng.choice([g for g in G if g != g1]))
        dec = int(rng.choice([1940, 1950, 1960, 1970, 1980, 1990]))
        seed_title = ds.titles[seed].rsplit(" (", 1)[0]
        reqs += [
            {"u": u, "type": "genre", "text": f"Recommend some {g1} movies", "include": [g1]},
            {"u": u, "type": "genre_exclude", "text": f"{g1} movies but no {g2}", "include": [g1], "exclude": [g2]},
            {"u": u, "type": "genre_decade", "text": f"{g1} movies from the {dec % 100}s", "include": [g1],
             "years": (dec, dec + 9)},
            {"u": u, "type": "similar", "text": f"Something like {seed_title}", "seed": seed},
            {"u": u, "type": "similar_exclude", "text": f"Something like {seed_title} but not {g2}", "seed": seed,
             "exclude": [g2]},
            {"u": u, "type": "hidden_gems", "text": f"Hidden gems in {g1}", "include": [g1], "tail": True},
            {"u": u, "type": "memory", "text": f"I hate {g2}. Recommend me {K} movies", "exclude": [g2],
             "followup": "Recommend me something", "followup_exclude": [g2]},
        ]
    return reqs


def check(ds, tools, u, items, req, exclude_key="exclude"):
    gi = {g: k for k, g in enumerate(ds.genre_names)}
    m = {"filled": float(len(items) == K)}
    if not items:
        return {**m, "hard_violation": 0.0, "soft_ok": 0.0, "grounded": 1.0}
    items = np.array(items)
    valid = (items >= 0) & (items < ds.n_items)
    seen = tools.pipe._seen.get(u, set()) | set(tools.pipe.history(u).tolist())
    m["grounded"] = float(valid.all() and not (set(items.tolist()) & seen))
    items = items[valid]
    ex = req.get(exclude_key, [])
    m["hard_violation"] = float(np.mean([any(ds.item_genres[i, gi[g]] > 0 for g in ex) for i in items])) if ex else 0.0
    soft = []
    for i in items:
        ok = True
        if req.get("include"):
            ok &= any(ds.item_genres[i, gi[g]] > 0 for g in req["include"])
        if req.get("years"):
            ok &= req["years"][0] <= ds.item_year[i] <= req["years"][1]
        if req.get("tail"):
            ok &= not tools.head[i]
        soft.append(ok)
    m["soft_ok"] = float(np.mean(soft))
    V = tools.pipe.s1.tower.V
    if "seed" in req:
        m["seed_similarity"] = float(np.mean(V[items] @ V[req["seed"]]))
    uv = tools.pipe.user_vec(tools.pipe.history(u))
    m["taste_similarity"] = float(np.mean(V[items] @ uv))
    return m


def fixed_pipeline(tools, u, text):
    c = parse_request(text, tools.ds.titles)
    s = Session(u)
    args = {"user_id": u, "n": 200}
    if c.exclude_genres:
        args["exclude_genres"] = c.exclude_genres
    if c.include_genres:
        args["include_genres"] = c.include_genres
    if c.year_min is not None:
        args["year_min"] = c.year_min
    if c.year_max is not None:
        args["year_max"] = c.year_max
    if c.popularity:
        args["popularity"] = c.popularity
    if c.similar_to:
        r = tools.call(s, "search_items", {"query": c.similar_to, "limit": 1})
        if r.get("results"):
            args["similar_to_item_ids"] = [r["results"][0]["item_id"]]
    tools.call(s, "retrieve_candidates", args)
    if not s.last_candidates:
        return []
    r = tools.call(s, "rank_candidates", {"user_id": u, "top_k": c.k})
    return [x["item_id"] for x in r.get("ranked", [])]


if __name__ == "__main__":
    argv = sys.argv[1:]
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 150
    llm_spec = argv[argv.index("--llm") + 1: argv.index("--llm") + 3] if "--llm" in argv else None
    cfg_path = next((a for a in argv if a.endswith(".yaml")), "configs/synthetic.yaml")
    cfg, ds = load(cfg_path)
    rng = set_seed(cfg["seed"])
    cache = ROOT / "data" / "processed"
    s1 = fit_stage1(ds, "test", cfg, cache_dir=cache, seed=cfg["seed"])
    rp = cache / f"ranker_{cfg['name']}.pkl"
    ranker = pickle.load(open(rp, "rb")) if rp.exists() else None
    pipe = FeedPipeline(ds, "test", s1, ranker=ranker)
    tusers = [u for u in sorted(ds.test.targets()) if s1.tower.user_hist.get(u) is not None and len(pipe.history(u)) >= 3]
    users = [int(u) for u in rng.choice(tusers, size=min(n, len(tusers)), replace=False)]
    reqs = make_requests(ds, pipe, users, rng)
    log.info(f"{len(reqs)} requests from {len(users)} users")

    systems = {"feed": None, "fixed": None, "agent_rule": lambda: RuleBasedPlanner(ds.titles),
               "agent_halluc": lambda: HallucinatingPlanner(ds.titles, ds.n_items, p=1.0, seed=cfg["seed"])}
    if llm_spec:
        from agentrec.llm.client import OpenAICompatClient
        systems["agent_llm"] = lambda: OpenAICompatClient(llm_spec[0], llm_spec[1])

    results = {}
    for name, mk in systems.items():
        tools = RecTools(ds, pipe, MemoryStore())      # fresh memory per system
        rows, lat = [], LatencyStats()
        stats = {"tool_calls": [], "llm_calls": [], "fallbacks": 0, "violations_caught": 0, "relaxed": 0,
                 "schema_errors": 0, "tool_errors": 0}
        for req in reqs:
            u = req["u"]
            texts = [(req["text"], "exclude")] + ([(req["followup"], "followup_exclude")] if "followup" in req else [])
            for text, key in texts:
                t0 = time.perf_counter()
                if name == "feed":
                    items = pipe.recommend(u, K)
                elif name == "fixed":
                    items = fixed_pipeline(tools, u, text)
                else:
                    res = RecAgent(mk(), tools).run(u, text)
                    items = res.items
                    stats["tool_calls"].append(res.n_tool_calls)
                    stats["llm_calls"].append(res.n_llm_calls)
                    stats["fallbacks"] += int(res.fallback_used)
                    stats["violations_caught"] += res.grounding_violations
                    stats["relaxed"] += int(len(res.relaxations) > 0)
                    stats["schema_errors"] += res.schema_errors
                    stats["tool_errors"] += res.tool_errors
                lat.add(time.perf_counter() - t0)
                r = check(ds, tools, u, items, req, key)
                r["type"] = req["type"] + ("_followup" if key == "followup_exclude" else "")
                rows.append(r)
        agg = {}
        for t in sorted({r["type"] for r in rows}) + ["ALL"]:
            sub = [r for r in rows if t == "ALL" or r["type"] == t]
            agg[t] = {k: float(np.mean([r[k] for r in sub if k in r])) for k in
                      ("filled", "hard_violation", "soft_ok", "grounded", "taste_similarity", "seed_similarity")
                      if any(k in r for r in sub)}
            agg[t]["n"] = len(sub)
        results[name] = {"by_type": agg, "latency": lat.summary(),
                         "mean_tool_calls": float(np.mean(stats["tool_calls"])) if stats["tool_calls"] else 0.0,
                         "mean_llm_calls": float(np.mean(stats["llm_calls"])) if stats["llm_calls"] else 0.0,
                         "fallbacks": stats["fallbacks"], "grounding_violations_caught": stats["violations_caught"],
                         "requests_with_replanning": stats["relaxed"], "schema_errors": stats["schema_errors"],
                         "tool_errors": stats["tool_errors"], "n_turns": len(rows)}
        a = agg["ALL"]
        log.info(f"{name}: filled={a['filled']:.3f} hard_viol={a['hard_violation']:.3f} soft={a['soft_ok']:.3f} "
                 f"grounded={a['grounded']:.3f} p50={lat.summary()['p50_ms']:.1f}ms")

    out = {"dataset": cfg["name"], "k": K, "n_users": len(users), "n_requests": len(reqs), "systems": results,
           "note": "agent_rule uses a deterministic rule-based planner, NOT an LLM"}
    save_json(out, results_dir(cfg) / "e8_agent.json")
    n_turns = results["feed"]["n_turns"]
    L = [f"# E8 — request mode on `{cfg['name']}` ({len(reqs)} templated requests + {n_turns - len(reqs)} memory follow-ups "
         f"= {n_turns} turns per system, {len(users)} users, k={K})\n",
         "`agent_rule` = agent loop + tools + memory + guardrails, driven by a RULE-BASED planner (not an LLM). "
         "`agent_halluc` = same, but the first `recommend` call of every turn contains one out-of-catalogue item id "
         "injected by the test harness.\n",
         "Columns: *valid & unseen* = every returned id exists in the catalogue and was not already seen by the user "
         "(membership in the retrieved set is enforced separately by the `recommend` tool). "
         "Rates are per returned item, except *filled k* (per turn).\n",
         "| system | filled k | hard-constraint violations | soft constraints met | valid & unseen | taste sim. | p50 ms | tool calls | turns replanned | fallbacks | ungrounded ids rejected | schema errors |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["by_type"]["ALL"]
        L.append(f"| {name} | {a['filled']:.3f} | {a['hard_violation']:.3f} | {a['soft_ok']:.3f} | {a['grounded']:.3f} | "
                 f"{a['taste_similarity']:.3f} | {r['latency']['p50_ms']:.1f} | {r['mean_tool_calls']:.1f} | "
                 f"{r['requests_with_replanning']} | {r['fallbacks']} | {r['grounding_violations_caught']} | {r['schema_errors']} |")
    types = sorted(results["feed"]["by_type"])
    L += ["", "Per request type — soft constraints met / hard violations / filled:", "",
          "| type | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for t in types:
        if t == "ALL":
            continue
        cells = []
        for name in results:
            a = results[name]["by_type"][t]
            cells.append(f"{a['soft_ok']:.2f} / {a['hard_violation']:.2f} / {a['filled']:.2f}")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    (results_dir(cfg) / "e8_agent.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
