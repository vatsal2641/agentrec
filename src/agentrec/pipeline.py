"""The deterministic feed pipeline:  retrieve -> rank -> (bandit) -> filter -> serve.

This is the part of AgentRec that runs for every "home feed" request. It has no
LLM in it on purpose: a feed request carries no intent to interpret, the steps
never change, and it must answer in milliseconds.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentrec.data.dataset import Dataset
from agentrec.models.baselines import ItemKNN, RecentPopularity
from agentrec.models.two_tower import TwoTower
from agentrec.ranking.ranker import FeatureBuilder, Ranker
from agentrec.retrieval.index import ExactIndex, IVFIndex
from agentrec.retrieval.retriever import Candidates, CandidateRetriever


@dataclass
class Stage1:
    """All models fit on one period (phase 'val' -> train, phase 'test' -> train+val)."""

    tower: TwoTower
    knn: ItemKNN
    recent: RecentPopularity


def fit_stage1(ds: Dataset, phase: str, cfg: dict, cache_dir: str | Path | None = None, seed: int = 0) -> Stage1:
    path = Path(cache_dir) / f"stage1_{cfg['name']}_{phase}.pkl" if cache_dir else None
    if path and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    period = ds.fit_period(phase)
    tt = TwoTower(**cfg["models"]["two_tower"], seed=seed).fit(ds, period)
    knn = ItemKNN(**cfg["models"]["itemknn"]).fit(ds, period)
    rp = RecentPopularity().fit(ds, period)
    s1 = Stage1(tt, knn, rp)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(s1, f)
    return s1


class FeedPipeline:
    def __init__(self, ds: Dataset, phase: str, stage1: Stage1, ranker: Ranker | None = None,
                 index: str = "exact", ivf: dict | None = None) -> None:
        self.ds, self.phase, self.s1 = ds, phase, stage1
        V = stage1.tower.item_embeddings()
        self.index = ExactIndex(V) if index == "exact" else IVFIndex(V, **(ivf or {}))
        popular = np.argsort(-stage1.recent.s)
        self.retriever = CandidateRetriever(self.index, V, popular)
        self.fb = FeatureBuilder(ds, ds.fit_period(phase), V, stage1.knn.S)
        self.ranker = ranker or Ranker("identity")
        seen = ds.matrix(ds.fit_period(phase), positive_only=False)
        self._seen = {u: set(seen[u].indices.tolist()) for u in range(ds.n_users)}

    # history in the fit period (time-ordered) unless overridden by online state
    def history(self, user: int) -> np.ndarray:
        h = self.s1.tower.user_hist.get(int(user))
        return h if h is not None else np.array([], dtype=np.int64)

    def user_vec(self, hist: np.ndarray) -> np.ndarray | None:
        return self.s1.tower.embed_history(hist) if len(hist) else None

    def candidates(self, user: int, n: int, hist: np.ndarray | None = None, exclude: set[int] | None = None,
                   allowed: np.ndarray | None = None, ignore_logged_seen: bool = False) -> Candidates:
        """ignore_logged_seen=True treats `user` as brand new (simulation of new users)."""
        hist = self.history(user) if hist is None else hist
        logged = set() if ignore_logged_seen else set(self._seen.get(user, set()))
        ex = logged | set(exclude or ()) | set(np.asarray(hist).tolist())
        return self.retriever.retrieve(self.user_vec(hist), n, exclude=ex, allowed=allowed)

    def rank(self, user: int, c: Candidates, hist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (items sorted best-first, ranker scores, feature matrix in that order)."""
        hist = self.history(user) if hist is None else hist
        uv = self.user_vec(hist)
        F = self.fb.build(user, c.items, c.scores, c.sources, hist_items=np.asarray(hist, dtype=np.int64), user_vec=uv)
        s = self.ranker.score(F)
        order = np.argsort(-s, kind="stable")
        return c.items[order], s[order], F[order]

    def recommend(self, user: int, k: int = 10, n_candidates: int = 200, hist: np.ndarray | None = None,
                  exclude: set[int] | None = None, allowed: np.ndarray | None = None) -> list[int]:
        c = self.candidates(user, n_candidates, hist, exclude, allowed)
        items, _, _ = self.rank(user, c, hist)
        return items[:k].tolist()
