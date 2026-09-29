"""E4/E5/E6 — exploration, feedback adaptation, noise robustness (SIMULATED users).

E4a  toy multi-armed bandit: eps-greedy vs UCB1 vs Thompson (pure maths check)
E4   warm users, slate of K from the ranker's top-N: frozen / greedy / eps-greedy / LinUCB / LinTS / random
E4b  position-bias handling: naive vs position-as-feature vs oracle (examined-only)
E4c  LinUCB alpha sensitivity
E5   NEW users (history hidden) over S sessions: no adaptation vs user-state update vs user-state + policy update
E6   noisy feedback: flip observed click labels with prob p
Every number here is measured inside the simulator (bandits/simulator.py) — not on real users.
usage: python scripts/run_bandits.py configs/ml1m.yaml [--quick]
"""
import json
import pickle
import sys
import time

import numpy as np

from _common import ROOT, load, results_dir

from agentrec.bandits.env import BanditEnv, run_policy
from agentrec.bandits.linear import make_policy
from agentrec.bandits.mab import EpsilonGreedy, ThompsonBeta, UCB1, run_mab
from agentrec.bandits.simulator import Simulator, build_truth
from agentrec.feedback.loop import FeedbackLoop
from agentrec.memory.store import MemoryStore
from agentrec.memory.user_state import UserState
from agentrec.pipeline import FeedPipeline, fit_stage1
from agentrec.utils.common import get_logger, save_json, set_seed

log = get_logger("e4")


def summarize(runs: list[dict], T: int) -> dict:
    last = slice(int(0.75 * T), T)
    return {
        "final_cum_regret_mean": float(np.mean([r["cum_regret"][-1] for r in runs])),
        "final_cum_regret_std": float(np.std([r["cum_regret"][-1] for r in runs])),
        "expected_clicks_per_slate": float(np.mean([r["expected_clicks"].mean() for r in runs])),
        "expected_clicks_last_quarter": float(np.mean([r["expected_clicks"][last].mean() for r in runs])),
        "observed_clicks_per_slate": float(np.mean([r["clicks"].mean() for r in runs])),
        "exploration_rate": float(np.mean([r["exploration_rate"].mean() for r in runs])),
        "regret_curve": np.mean([r["cum_regret"][np.linspace(0, T - 1, 51).astype(int)] for r in runs], axis=0).round(3).tolist(),
    }


def write_md(out: dict, cfg: dict, rd) -> None:
    bc = cfg["bandit"]
    seeds, T = out["seeds"], out["rounds"]
    s2 = seeds[:2]
    toy, e4, e5, e6 = out["e4a_toy_mab"], out["e4_policies"], out["e5_new_user_adaptation"], out["e6_noise"]
    sim = out["simulator"]
    L = [f"# E4–E6 — bandits & feedback on `{out['dataset']}` — ALL NUMBERS FROM THE SIMULATOR\n",
         f"truth: {sim['truth']}; {sim['n_users']} warm users; slate K={bc['slate_size']} from top-{bc['n_candidates']} "
         f"ranked candidates; T={T} rounds; examination probs {sim['exam_probs']}.\n",
         f"**Seeds:** E4 and E4b use {len(seeds)} seeds {seeds}; E4c, E4c', E5 and E6 use {len(s2)} seeds {s2}, "
         "so the same configuration can show slightly different numbers across sections.\n",
         "## E4a toy 10-armed Bernoulli bandit (T=5000, 20 seeds)", "", "| policy | final regret (mean ± sd) |", "|---|---|"]
    for k, v in toy.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['final_regret_mean']:.1f} ± {v['final_regret_std']:.1f} |")
    L += ["", f"## E4 slate policies (position-as-feature updates, {len(seeds)} seeds)", "",
          "| policy | cum. regret (mean ± sd) | exp. clicks/slate | exp. clicks/slate (last 25%) | exploration rate |",
          "|---|---|---|---|---|"]
    for k, v in e4.items():
        L.append(f"| {k} | {v['final_cum_regret_mean']:.1f} ± {v['final_cum_regret_std']:.1f} | "
                 f"{v['expected_clicks_per_slate']:.4f} | {v['expected_clicks_last_quarter']:.4f} | {v['exploration_rate']:.3f} |")
    L += ["", f"## E4b position-bias handling (LinUCB, {len(seeds)} seeds)", "",
          "| update mode | cum. regret (mean ± sd) | exp. clicks/slate |", "|---|---|---|"]
    for k, v in out["e4b_position_bias"].items():
        L.append(f"| {k} | {v['final_cum_regret_mean']:.1f} ± {v['final_cum_regret_std']:.1f} | {v['expected_clicks_per_slate']:.4f} |")
    L += ["", f"## E4c LinUCB alpha ({len(s2)} seeds)", "", "| alpha | cum. regret (mean ± sd) | exploration rate |", "|---|---|---|"]
    for k, v in out["e4c_linucb_alpha"].items():
        L.append(f"| {k} | {v['final_cum_regret_mean']:.1f} ± {v['final_cum_regret_std']:.1f} | {v['exploration_rate']:.3f} |")
    L += ["", f"## E4c' LinTS posterior scale v ({len(s2)} seeds)", "", "| v | cum. regret (mean ± sd) | exploration rate |", "|---|---|---|"]
    for k, v in out["e4c_lints_v"].items():
        L.append(f"| {k} | {v['final_cum_regret_mean']:.1f} ± {v['final_cum_regret_std']:.1f} | {v['exploration_rate']:.3f} |")
    L += ["", f"## E6 noisy feedback (observed click label flipped with prob p; {len(s2)} seeds; mean ± sd)", "",
          "| p | frozen | greedy | linucb (λ=1) | linucb (λ=100, stronger prior) |", "|---|---|---|---|---|"]
    f = lambda v: f"{v['final_cum_regret_mean']:.1f} ± {v['final_cum_regret_std']:.1f}"  # noqa: E731
    for p, v in e6.items():
        L.append(f"| {p} | {f(v['frozen'])} | {f(v['greedy'])} | {f(v['linucb'])} | {f(v['linucb_strong_prior'])} |")
    L += ["", f"## E5 new users, {e5['sessions']} sessions each ({e5['n_users']} users, history hidden; {len(s2)} seeds)", "",
          "Arms: `no_adaptation` = ranker top-K, state never updated; `user_state_update` = ranker top-K + user-state "
          "updates; `user_state_and_policy_update` = **LinUCB slate selection** + user-state + policy updates "
          "(two changes vs the previous arm).", "",
          "| condition | exp. clicks session 1 | session last | mean over sessions |", "|---|---|---|---|"]
    for k in ("no_adaptation", "user_state_update", "user_state_and_policy_update"):
        v = e5[k]
        L.append(f"| {k} | {v['first_session']:.4f} | {v['last_session']:.4f} | {v['mean']:.4f} |")
    (rd / "e4_e6_bandits.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg, ds = load(args[0] if args else "configs/synthetic.yaml")
    if "--md-only" in sys.argv:   # regenerate the markdown from the saved JSON without re-running
        rd = results_dir(cfg)
        write_md(json.loads((rd / "e4_e6_bandits.json").read_text()), cfg, rd)
        sys.exit(0)
    bc = cfg["bandit"]
    rng = set_seed(cfg["seed"])
    T = 3000 if quick else bc["n_rounds"]
    seeds = [0, 1] if quick else [0, 1, 2]
    out: dict = {"dataset": cfg["name"], "rounds": T, "seeds": seeds, "SIMULATED": True}

    # ---------------- E4a: toy MAB ----------------
    mu = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    toy = {}
    for name, mk in [("eps_greedy_0.1", lambda: EpsilonGreedy(10, 0.1)), ("ucb1", lambda: UCB1(10)),
                     ("thompson", lambda: ThompsonBeta(10))]:
        regs = [run_mab(mk(), mu, 5000, np.random.default_rng(s))[-1] for s in range(20)]
        toy[name] = {"final_regret_mean": float(np.mean(regs)), "final_regret_std": float(np.std(regs))}
    out["e4a_toy_mab"] = {"arms": mu.tolist(), "T": 5000, "seeds": 20, **toy}
    log.info(f"toy MAB: {toy}")

    # ---------------- environment ----------------
    cache = ROOT / "data" / "processed"
    s1 = fit_stage1(ds, "test", cfg, cache_dir=cache, seed=cfg["seed"])
    rp = cache / f"ranker_{cfg['name']}.pkl"
    ranker = pickle.load(open(rp, "rb")) if rp.exists() else None
    if ranker is None:
        log.warning("no trained ranker found (run run_ranking.py first); using identity ranker")
    pipe = FeedPipeline(ds, "test", s1, ranker=ranker)
    truth = build_truth(ds, ROOT / cfg["data"]["raw_dir"], cfg["models"]["bpr"], seed=cfg["seed"] + 1)
    sim = Simulator(truth, seed=cfg["seed"])
    env = BanditEnv(pipe, sim, n_candidates=bc["n_candidates"], slate_size=bc["slate_size"])
    warm_all = np.array([u for u in range(ds.n_users) if s1.tower.user_hist.get(u) is not None])
    users = rng.choice(warm_all, size=min(300 if quick else 500, len(warm_all)), replace=False)
    t0 = time.time()
    contexts = {int(u): env.context(int(u)) for u in users}
    z0 = sim.calibrate([(u, c.items) for u, c in contexts.items()], target_mean=0.1)
    out["simulator"] = {"truth": truth.source, "z0": z0, "target_mean_click_prob_over_candidates": 0.1,
                        "exam_probs": sim.exam_probs(env.K).round(3).tolist(), "context_dim": env.d,
                        "n_users": len(users), "context_build_s": time.time() - t0}
    log.info(f"simulator: {out['simulator']}")

    def run(name: str, mode: str = "position_feature", flip: float = 0.0, extra: dict | None = None, s_list=seeds):
        runs = []
        for s in s_list:
            sim.rng = np.random.default_rng(1000 + s)
            sim.flip = flip
            pol = make_policy(name, env.d, env.theta0(), {**bc, **(extra or {})})
            runs.append(run_policy(env, pol, users, contexts, T, np.random.default_rng(s), mode=mode))
        sim.flip = 0.0
        return summarize(runs, T)

    # ---------------- E4: policies ----------------
    e4 = {}
    for name in ("random", "frozen", "greedy", "eps_greedy", "linucb", "lints"):
        e4[name] = run(name)
        log.info(f"E4 {name}: regret={e4[name]['final_cum_regret_mean']:.1f} "
                 f"clicks/slate={e4[name]['expected_clicks_per_slate']:.4f} explore={e4[name]['exploration_rate']:.3f}")
    out["e4_policies"] = e4

    # ---------------- E4b: position bias ----------------
    out["e4b_position_bias"] = {m: run("linucb", mode=m) for m in ("naive", "position_feature", "oracle_examined")}
    log.info("E4b " + str({k: round(v["final_cum_regret_mean"], 1) for k, v in out["e4b_position_bias"].items()}))

    # ---------------- E4c: alpha ----------------
    out["e4c_linucb_alpha"] = {a: run("linucb", extra={"alpha_ucb": a}, s_list=seeds[:2]) for a in (0.1, 0.5, 1.0, 2.0)}
    out["e4c_lints_v"] = {v: run("lints", extra={"ts_v": v}, s_list=seeds[:2]) for v in (0.05, 0.1, 0.2, 0.5)}
    log.info("E4c lints v: " + str({k: round(v["final_cum_regret_mean"], 1) for k, v in out["e4c_lints_v"].items()}))

    # ---------------- E6: noise ----------------
    e6 = {}
    for p in bc["noise_levels"]:
        e6[p] = {name: run(name, flip=p, s_list=seeds[:2]) for name in ("frozen", "greedy", "linucb")}
        e6[p]["linucb_strong_prior"] = run("linucb", flip=p, extra={"lam": 100.0}, s_list=seeds[:2])
        log.info(f"E6 flip={p}: " + str({k: round(v['final_cum_regret_mean'], 1) for k, v in e6[p].items()}))
    out["e6_noise"] = e6

    # ---------------- E5: new-user adaptation through the feedback loop ----------------
    S = 10 if quick else 15
    new_users = rng.choice(warm_all, size=100 if quick else 200, replace=False)
    e5 = {}
    for cond in ("no_adaptation", "user_state_update", "user_state_and_policy_update"):
        curves = []
        for s in seeds[:2]:
            sim.rng = np.random.default_rng(2000 + s)
            prng = np.random.default_rng(s)
            store = MemoryStore()
            loop = FeedbackLoop(store, ds.item_genres, update_user=cond != "no_adaptation",
                                update_policy=cond == "user_state_and_policy_update")
            pol = make_policy("linucb", env.d, env.theta0(), bc)
            curve = np.zeros(S)
            for u in new_users:
                st = UserState(int(u), len(ds.genre_names))          # history hidden: a brand-new user
                for sess in range(S):
                    c = env.context(int(u), hist=np.array(st.history, dtype=np.int64), exclude=st.blocked_items(),
                                    new_user=True)
                    sel = pol.select(c.X, env.K, prng) if cond == "user_state_and_policy_update" else np.arange(env.K)
                    slate = c.items[sel]
                    curve[sess] += sim.expected_clicks(int(u), slate)
                    fb = sim.respond(int(u), slate)
                    loop.process(st, slate, fb, policy=pol, X_update=env.with_position(c.X[sel], np.arange(env.K)))
            curves.append(curve / len(new_users))
        c = np.mean(curves, axis=0)
        e5[cond] = {"expected_clicks_by_session": c.round(4).tolist(), "first_session": float(c[0]),
                    "last_session": float(c[-1]), "mean": float(c.mean())}
        log.info(f"E5 {cond}: session1={c[0]:.4f} last={c[-1]:.4f} mean={c.mean():.4f}")
    out["e5_new_user_adaptation"] = {"sessions": S, "n_users": len(new_users), **e5}

    rd = results_dir(cfg)
    save_json(out, rd / "e4_e6_bandits.json")
    write_md(json.loads((rd / "e4_e6_bandits.json").read_text()), cfg, rd)
