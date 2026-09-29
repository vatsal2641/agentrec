"""Non-learned / shallow baselines. Read these before the learned models.

Popularity       score(u, i) = f(#positives of i)                      (same list for everyone)
RecentPopularity score(u, i) = sum_{interactions of i} exp(-age / half_life)
ContentBased     score(u, i) = cos(profile_u, features_i)              (metadata only)
ItemKNN          score(u, i) = sum_{j in hist(u)} sim(j, i)            (collaborative, memory-based)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from agentrec.data.dataset import Dataset, Period
from agentrec.models.base import Recommender


class Popularity(Recommender):
    """Most-liked items overall. Training O(|interactions|), inference O(|I|).

    Strength: very hard to beat on sparse data; no cold-start-user problem.
    Weakness: zero personalization; reinforces popularity bias.
    """

    name = "popularity"

    def _fit(self, ds: Dataset, period: Period) -> None:
        self.s = self.pop_scores

    def _score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.s, (len(users), self.n_items)).copy()


class RecentPopularity(Recommender):
    """Popularity with exponential time decay: recent interactions count more.

    This matters under a temporal split: what was popular *last month* predicts
    next month better than all-time counts.
    """

    name = "recent_popularity"

    def __init__(self, half_life_days: float = 30.0) -> None:
        self.half_life_days = half_life_days

    def _fit(self, ds: Dataset, period: Period) -> None:
        p = period.positives
        t_end = period.df["timestamp"].max()
        age_days = (t_end - p["timestamp"].to_numpy()) / 86400.0
        w = np.power(0.5, age_days / self.half_life_days)
        self.s = np.bincount(p["i"].to_numpy(), weights=w, minlength=self.n_items).astype(np.float32)
        self.pop_scores = self.s  # cold users also get the recency-weighted list

    def _score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.s, (len(users), self.n_items)).copy()


class ContentBased(Recommender):
    """User profile = mean feature vector of liked items; score = cosine similarity.

    Item features: TF-IDF-weighted genres + decade one-hot (both L2-normalised).
    Strength: can score items nobody has interacted with (cold items).
    Weakness: only as good as metadata; recommends "more of the same"
    (over-specialisation); ignores what similar users liked.
    """

    name = "content"

    def __init__(self, decade_weight: float = 0.5) -> None:
        self.decade_weight = decade_weight

    def _item_features(self, ds: Dataset) -> np.ndarray:
        g = ds.item_genres.astype(np.float64)
        idf = np.log((1 + ds.n_items) / (1 + g.sum(0))) + 1.0
        g = g * idf
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        years = np.where(ds.item_year > 0, ds.item_year, 1990)
        dec = np.clip((years - 1920) // 10, 0, 8)
        d = np.eye(9)[dec] * self.decade_weight
        f = np.hstack([g, d])
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)

    def _fit(self, ds: Dataset, period: Period) -> None:
        self.F = self._item_features(ds)                              # (I, f)
        prof = self.X @ self.F                                        # (U, f) sum of liked items
        self.P = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self.P[users] @ self.F.T


class ItemKNN(Recommender):
    """Item-item collaborative filtering.

    sim(i, j) = |U_i ∩ U_j| / (sqrt(|U_i|) sqrt(|U_j|) + shrink)   (shrunk cosine on binary data)
    keep top-k neighbours per item, score(u, i) = sum_{j in hist(u)} sim(j, i).

    Training: X^T X is O(sum_u |hist_u|^2) -> fine for ML-1M, not for 10M items.
    Strength: strong, explainable ("because you liked j"). Weakness: cannot score
    cold items; quadratic similarity cost; popular items dominate neighbourhoods.
    """

    name = "itemknn"

    def __init__(self, k_neighbors: int = 100, shrink: float = 10.0) -> None:
        self.k = k_neighbors
        self.shrink = shrink

    def _fit(self, ds: Dataset, period: Period) -> None:
        X = self.X.astype(np.float32)
        co = (X.T @ X).toarray()                                      # (I, I) co-occurrence
        n = np.diag(co).copy()
        np.fill_diagonal(co, 0.0)
        denom = np.sqrt(n)[:, None] * np.sqrt(n)[None, :] + self.shrink
        S = co / denom
        if self.k < S.shape[1]:                                       # keep top-k per column
            thresh = -np.partition(-S, self.k - 1, axis=0)[self.k - 1]
            S[S < thresh[None, :]] = 0.0
        self.S = sp.csr_matrix(S.astype(np.float32))

    def _score(self, users: np.ndarray) -> np.ndarray:
        return (self.X[users] @ self.S).toarray()
