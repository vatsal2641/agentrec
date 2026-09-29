"""Interactive demo: feed mode + request mode + feedback, in the terminal.

  python scripts/demo.py configs/ml1m.yaml                     # rule-based planner
  python scripts/demo.py configs/ml1m.yaml --llm http://localhost:11434/v1 qwen2.5:7b-instruct

Commands:  feed | ask <free text> | click <n> | skip <n> | rate <n> <1-5> | user <id> | profile | quit
"""
import pickle
import sys

import numpy as np

from _common import ROOT, load

from agentrec.agents.agent import RecAgent
from agentrec.bandits.env import BanditEnv, serve_slate
from agentrec.bandits.linear import make_policy
from agentrec.bandits.simulator import Simulator, TruthModel
from agentrec.agents.tools import RecTools, Session
from agentrec.feedback.events import Event, EventType, RewardMapper, new_request_id
from agentrec.llm.rule_based import RuleBasedPlanner
from agentrec.memory.store import MemoryStore
from agentrec.pipeline import FeedPipeline, fit_stage1

if __name__ == "__main__":
    argv = sys.argv[1:]
    llm = None
    if "--llm" in argv:
        from agentrec.llm.client import OpenAICompatClient
        i = argv.index("--llm")
        llm = OpenAICompatClient(argv[i + 1], argv[i + 2])
    cfg, ds = load(next((a for a in argv if a.endswith(".yaml")), "configs/synthetic.yaml"))
    cache = ROOT / "data" / "processed"
    s1 = fit_stage1(ds, "test", cfg, cache_dir=cache, seed=cfg["seed"])
    rp = cache / f"ranker_{cfg['name']}.pkl"
    pipe = FeedPipeline(ds, "test", s1, ranker=pickle.load(open(rp, "rb")) if rp.exists() else None)
    store = MemoryStore(cache / "demo_memory.sqlite")
    tools = RecTools(ds, pipe, store)
    user = int(next(iter(sorted(ds.test.targets()))))
    shown: list[int] = []
    shown_X = None
    # feed path = retrieve -> rank -> LinUCB slate; the simulator is unused here (dummy truth)
    env = BanditEnv(pipe, Simulator(TruthModel(np.zeros((ds.n_users, 1)), np.zeros((ds.n_items, 1)))),
                    n_candidates=cfg["bandit"]["n_candidates"], slate_size=10)
    policy = make_policy("linucb", env.d, env.theta0(), {**cfg["bandit"], "lam": 100.0})
    rng = np.random.default_rng(0)
    print(__doc__)
    while True:
        try:
            line = input(f"[user {user}]> ").strip()
        except EOFError:
            break
        cmd, _, rest = line.partition(" ")
        st = store.load_state(user, len(ds.genre_names))
        if cmd == "quit":
            break
        if cmd == "user":
            user, shown = int(rest), []
        elif cmd == "profile":
            print(tools.call(Session(user), "get_user_profile", {"user_id": user}))
        elif cmd == "feed":
            items, shown_X = serve_slate(env, policy, st, rng)
            shown = items.tolist()
            for k, i in enumerate(shown):
                print(f"  {k}. {ds.titles[i]}")
        elif cmd == "ask":
            r = RecAgent(llm or RuleBasedPlanner(ds.titles), tools).run(user, rest)
            shown, shown_X = r.items, None
            print("  tool calls:", " -> ".join(t["name"] for t in r.trajectory if t["type"] == "tool"),
                  "| fallback" if r.fallback_used else "")
            for k, i in enumerate(shown):
                print(f"  {k}. {ds.titles[i]}  — {r.explanations.get(i, '')}")
        elif cmd in ("click", "skip", "rate"):
            parts = rest.split()
            i = shown[int(parts[0])]
            et = {"click": EventType.CLICK, "skip": EventType.SKIP, "rate": EventType.RATING}[cmd]
            e = Event(user, i, et, 0.0, new_request_id(), int(parts[0]), float(parts[1]) if cmd == "rate" else None)
            store.log_events([e])
            st.apply(e, ds.item_genres)
            store.save_state(st)
            if shown_X is not None:                       # feed item -> also update the bandit policy
                # same reward definition as feedback/events.py::RewardMapper (one update per event)
                policy.update(shown_X[int(parts[0])], RewardMapper().reward([e]))
            print(f"  recorded {cmd} on {ds.titles[i]}; online history {len(st.history)} items; "
                  f"policy updates so far {policy.state.n_updates}")
        else:
            print(__doc__)
