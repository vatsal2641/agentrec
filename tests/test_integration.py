"""End-to-end on the tiny synthetic dataset: data -> stage1 -> pipeline -> bandit -> feedback -> agent."""
import numpy as np

from helpers import tiny_dataset, tiny_raw_dir, tiny_stage1
from agentrec.bandits.env import BanditEnv, run_policy
from agentrec.bandits.linear import LinUCB
from agentrec.bandits.simulator import Simulator, build_truth
from agentrec.evaluation.evaluator import evaluate_model
from agentrec.feedback.loop import FeedbackLoop
from agentrec.memory.store import MemoryStore
from agentrec.memory.user_state import UserState
from agentrec.pipeline import FeedPipeline


def test_pipeline_returns_k_unseen_items():
    ds = tiny_dataset()
    pipe = FeedPipeline(ds, "test", tiny_stage1())
    u = int(ds.test.df["u"].iloc[0])
    recs = pipe.recommend(u, k=10, n_candidates=50)
    seen = set(ds.fit_period("test").df.query("u == @u")["i"])
    assert len(recs) == 10 and len(set(recs)) == 10 and not seen & set(recs)


def test_offline_evaluation_runs_on_stage1_model():
    ds = tiny_dataset()
    res = evaluate_model(tiny_stage1().tower.score, ds, "test", ks=[10])
    assert 0.0 <= res["all"]["ndcg@10"] <= 1.0 and res["all"]["n_users"] > 0


def test_bandit_loop_and_feedback_adaptation_run():
    ds = tiny_dataset()
    pipe = FeedPipeline(ds, "test", tiny_stage1())
    sim = Simulator(build_truth(ds, tiny_raw_dir()), seed=0)
    env = BanditEnv(pipe, sim, n_candidates=20, slate_size=3, retrieve_n=40)
    users = np.array(sorted(set(ds.test.df["u"])))[:10]
    ctx = {int(u): env.context(int(u)) for u in users}
    out = run_policy(env, LinUCB(env.d, alpha=0.5, theta0=env.theta0()), users, ctx, 50, np.random.default_rng(0))
    assert out["cum_regret"][-1] >= 0 and len(out["clicks"]) == 50

    # new user: empty history -> popularity; after feedback -> history-based embedding
    loop, st = FeedbackLoop(MemoryStore(), ds.item_genres), UserState(int(users[0]), len(ds.genre_names))
    first = env.context(st.user_id, hist=np.array(st.history, dtype=np.int64))
    for _ in range(5):
        c = env.context(st.user_id, hist=np.array(st.history, dtype=np.int64), exclude=st.blocked_items())
        fb = sim.respond(st.user_id, c.items[:3])
        fb.clicked[0] = True; fb.skipped[0] = False              # force at least one click
        loop.process(st, c.items[:3], fb)
    assert len(st.history) >= 1
    after = env.context(st.user_id, hist=np.array(st.history, dtype=np.int64))
    assert not np.array_equal(first.items, after.items)


def test_feed_serving_path_uses_bandit_and_respects_stated_excludes():
    from agentrec.bandits.env import serve_slate
    ds = tiny_dataset()
    pipe = FeedPipeline(ds, "test", tiny_stage1())
    env = BanditEnv(pipe, Simulator(build_truth(ds, tiny_raw_dir())), n_candidates=20, slate_size=4, retrieve_n=40)
    pol = LinUCB(env.d, alpha=0.5, theta0=env.theta0())
    u = int(ds.test.df["u"].iloc[0])
    st = UserState(u, len(ds.genre_names), stated={"exclude_genres": ["Drama"]})
    items, X = serve_slate(env, pol, st, np.random.default_rng(0))
    g = ds.genre_names.index("Drama")
    assert len(items) == 4 and all(ds.item_genres[i, g] == 0 for i in items)
    assert X.shape == (4, env.d) and np.allclose(X[:, -env.K:], np.eye(4))
    pol.update(X[0], 1.0)
    assert pol.state.n_updates == 1
