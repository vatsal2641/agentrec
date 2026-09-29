"""Glue between the feed pipeline, the bandit policies and the simulator.

CONTEXT VECTOR for (user u, candidate i)            dimension d = 11 + D + K
    [ 1,                                   intercept
      ranker score (z-scored within list), the offline system's opinion
      9 ranking features (standardised),   popularity, genre match, recency, ...
      sqrt(D) * (z_u ⊙ v_i),               per-dimension user-item agreement of
                                           the two-tower embeddings (lets theta
                                           re-weight embedding dims online)
      one-hot(position) (K dims) ]         only non-zero when learning with the
                                           "position as a feature" debiasing trick
PRIOR theta0: intercept = base CTR guess, weight on ranker score > 0, rest 0.
    At t=0 the prior MEAN reproduces the ranker's order, so greedy / eps-greedy (and
    frozen) start exactly at the ranker. LinUCB and LinTS do NOT: their bonus
    alpha*sqrt(x^T A^-1 x) = alpha*||x||/sqrt(lambda) at t=0 varies across candidates
    far more than the small prior term, so their first slates are mostly
    exploration. Larger lambda shrinks that bonus (1/sqrt(lambda)) — see E6's
    lambda=100 runs. A warm-start prior mean does not by itself tame UCB.

POSITION-BIAS MODES for the policy update
    naive           : learn from every shown item, not-clicked = 0.
                      Items at low positions look bad just because nobody looked.
    position_feature: include position one-hot while learning; at scoring time all
                      position dims are 0 (a neutral reference, same for every
                      candidate). The position effect is absorbed by the position
                      weights instead of polluting item weights.
    oracle_examined : learn only from items the user actually examined.
                      Impossible in reality (examination is hidden) -> upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentrec.bandits.linear import LinearPolicy
from agentrec.bandits.simulator import Simulator
from agentrec.pipeline import FeedPipeline


@dataclass
class Context:
    items: np.ndarray     # (N,) candidate items (ranker order)
    X: np.ndarray         # (N, d) contexts with position dims = 0


class BanditEnv:
    def __init__(self, pipe: FeedPipeline, sim: Simulator, n_candidates: int = 50, slate_size: int = 5,
                 retrieve_n: int = 200) -> None:
        self.pipe, self.sim = pipe, sim
        self.N, self.K, self.retrieve_n = n_candidates, slate_size, retrieve_n
        self.D = pipe.s1.tower.dim
        self.d = 11 + self.D + self.K

    def context(self, user: int, hist: np.ndarray | None = None, exclude: set[int] | None = None,
                new_user: bool = False, allowed: np.ndarray | None = None) -> Context:
        hist = self.pipe.history(user) if hist is None else np.asarray(hist, dtype=np.int64)
        c = self.pipe.candidates(user, self.retrieve_n, hist=hist, exclude=exclude, ignore_logged_seen=new_user,
                                 allowed=allowed)
        items, s, F = self.pipe.rank(user, c, hist)
        items, s, F = items[: self.N], s[: self.N], F[: self.N]
        sz = (s - s.mean()) / (s.std() + 1e-9)
        Fs = self.pipe.ranker.scaler.transform(F) if hasattr(self.pipe.ranker.scaler, "mean_") else F
        uv = self.pipe.user_vec(hist)
        emb = (self.pipe.s1.tower.V[items] * uv) * np.sqrt(self.D) if uv is not None else np.zeros((len(items), self.D))
        X = np.hstack([np.ones((len(items), 1)), sz[:, None], Fs, emb, np.zeros((len(items), self.K))])
        return Context(items, X)

    def theta0(self, base_ctr: float = 0.1, rank_weight: float = 0.05) -> np.ndarray:
        t = np.zeros(self.d)
        t[0], t[1] = base_ctr, rank_weight
        return t

    def with_position(self, X: np.ndarray, positions: np.ndarray) -> np.ndarray:
        Xp = X.copy()
        Xp[:, -self.K:] = 0.0
        Xp[np.arange(len(X)), self.d - self.K + positions] = 1.0
        return Xp


def run_policy(env: BanditEnv, policy: LinearPolicy, users: np.ndarray, contexts: dict[int, Context],
               T: int, rng: np.random.Generator, mode: str = "position_feature") -> dict[str, np.ndarray]:
    """Warm-user experiment: user histories are fixed; each round a random user arrives."""
    if mode not in {"naive", "position_feature", "oracle_examined"}:
        raise ValueError(mode)
    K = env.K
    regret = np.empty(T)
    clicks = np.empty(T)
    exp_clicks = np.empty(T)
    explored = np.empty(T)
    for t in range(T):
        u = int(users[rng.integers(len(users))])
        ctx = contexts[u]
        sel = policy.select(ctx.X, K, rng)
        greedy = policy.greedy(ctx.X, K)
        explored[t] = np.mean(sel != greedy)
        slate = ctx.items[sel]
        fb = env.sim.respond(u, slate)
        r = fb.clicked.astype(float)
        Xs = ctx.X[sel]
        if mode == "position_feature":
            Xs = env.with_position(Xs, np.arange(K))
        for k in range(K):
            if mode == "oracle_examined" and not fb.examined[k]:
                continue
            policy.update(Xs[k], r[k])
        opt = env.sim.optimal_expected_clicks(u, ctx.items, K)
        exp_clicks[t] = env.sim.expected_clicks(u, slate)
        regret[t] = opt - exp_clicks[t]
        clicks[t] = r.sum()
    return {"cum_regret": np.cumsum(regret), "clicks": clicks, "expected_clicks": exp_clicks,
            "exploration_rate": explored}


def serve_slate(env: BanditEnv, policy: LinearPolicy, state, rng: np.random.Generator,
                base_hist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The FEED serving path with the bandit in it (used by scripts/demo.py):
    user state -> retrieve (stated excludes + blocked items applied in code) -> rank -> bandit slate.
    Returns (slate items, contexts for the shown items with their positions, for policy updates)."""
    user = int(state.user_id)
    hist = list(env.pipe.history(user) if base_hist is None else base_hist) + list(state.history)
    ctx = env.context(user, hist=np.array(hist, dtype=np.int64), exclude=state.blocked_items(),
                      allowed=state.allowed_mask(env.pipe.ds.item_genres, env.pipe.ds.genre_names))
    k = min(env.K, len(ctx.items))
    sel = policy.select(ctx.X, k, rng)
    return ctx.items[sel], env.with_position(ctx.X[sel], np.arange(k))
