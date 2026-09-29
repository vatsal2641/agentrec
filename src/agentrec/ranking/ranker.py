"""Second-stage ranking over retrieved candidates.

WHY A SEPARATE RANKER
  Retrieval must be a dot product (so it can use an ANN index) -> it cannot
  use cross features like "does this item's genre match the user's recent
  taste" or "how many of this user's liked items are neighbours of this item".
  The ranker only sees ~N=200 candidates, so it can afford richer features and
  a non-factorised model.

TRAINING DATA (important, easy to get wrong)
  Stage-1 models are fit on window A (train). The ranker is trained on
  candidates for window B (val) with labels from window B. If you trained the
  ranker on window A, stage-1 scores would be *memorised* (overconfident) and
  the ranker would learn to trust them too much.

RANKERS
  identity  : keep retrieval order (baseline: "is the ranker worth it?")
  pointwise : logistic regression on (features -> clicked?)      loss = binary cross-entropy
  pairwise  : logistic regression on (f_pos - f_neg -> 1)         loss = RankNet/BPR-style
  gbdt      : gradient-boosted trees, pointwise                    non-linear feature interactions
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from agentrec.data.dataset import Dataset, Period

FEATURES = [
    "retrieval_score", "retrieval_rank", "itemknn_score", "log_pop", "log_recent_pop",
    "genre_match", "item_age", "log_user_activity", "from_popular_source",
]


@dataclass
class FeatureBuilder:
    """Computes ranking features for (user, candidate list). One per phase."""

    ds: Dataset
    period: Period
    item_vecs: np.ndarray            # retrieval item embeddings (for scores of popular-source items)
    itemknn_S: sp.csr_matrix         # item-item similarity from ItemKNN fit on the same period

    def __post_init__(self) -> None:
        ds, p = self.ds, self.period
        self.X = ds.matrix(p, positive_only=True)
        cnt = np.asarray(self.X.sum(0)).ravel()
        self.log_pop = np.log1p(cnt)
        pos = p.positives
        age_days = (p.df["timestamp"].max() - pos["timestamp"].to_numpy()) / 86400
        self.log_recent = np.log1p(np.bincount(pos["i"], weights=0.5 ** (age_days / 30), minlength=ds.n_items))
        g = ds.item_genres
        self.Gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        yr = ds.item_year.astype(float)
        yr[yr <= 0] = np.median(yr[yr > 0])
        self.item_age = (yr.max() - yr) / 50.0
        self.user_act = np.log1p(np.asarray(self.X.sum(1)).ravel())

    def user_genre_profile(self, hist_items: np.ndarray) -> np.ndarray:
        if len(hist_items) == 0:
            return np.zeros(self.Gn.shape[1])
        v = self.ds.item_genres[hist_items].sum(0)
        return v / (np.linalg.norm(v) + 1e-12)

    def build(self, user: int, cand_items: np.ndarray, cand_scores: np.ndarray, sources: list[str],
              hist_items: np.ndarray | None = None, user_vec: np.ndarray | None = None) -> np.ndarray:
        """hist_items / user_vec override the stored history (used for online users)."""
        if hist_items is None:
            hist_items = self.X[user].indices
        n = len(cand_items)
        rs = np.where(np.isnan(cand_scores), 0.0, cand_scores)
        if user_vec is not None:
            rs = self.item_vecs[cand_items] @ user_vec
        knn = np.asarray(self.itemknn_S[hist_items][:, cand_items].sum(0)).ravel() if len(hist_items) else np.zeros(n)
        gm = self.Gn[cand_items] @ self.user_genre_profile(hist_items)
        return np.column_stack([
            rs,
            np.arange(n) / max(n, 1),
            np.log1p(knn),
            self.log_pop[cand_items],
            self.log_recent[cand_items],
            gm,
            self.item_age[cand_items],
            np.full(n, np.log1p(len(hist_items))),
            np.array([s == "popular" for s in sources], dtype=float),
        ])


class Ranker:
    def __init__(self, kind: str = "pointwise", seed: int = 0, neg_per_pos: int = 10) -> None:
        if kind not in {"identity", "pointwise", "pairwise", "gbdt"}:
            raise ValueError(f"unknown ranker {kind!r}")
        self.kind, self.seed, self.neg_per_pos = kind, seed, neg_per_pos
        self.scaler = StandardScaler()

    def fit(self, groups: list[tuple[np.ndarray, np.ndarray]]) -> "Ranker":
        """groups: list of (features (n, f), labels (n,)) per user request."""
        if self.kind == "identity":
            return self
        Xall = np.vstack([f for f, _ in groups])
        self.scaler.fit(Xall)
        rng = np.random.default_rng(self.seed)
        if self.kind == "pairwise":
            diffs = []
            for f, y in groups:
                f = self.scaler.transform(f)
                P, N = np.where(y == 1)[0], np.where(y == 0)[0]
                if len(P) == 0 or len(N) == 0:
                    continue
                for p in P:
                    for n in rng.choice(N, size=min(self.neg_per_pos, len(N)), replace=False):
                        diffs.append(f[p] - f[n])
            D = np.array(diffs)
            X = np.vstack([D, -D])                  # symmetric -> balanced classes, no intercept needed
            y = np.r_[np.ones(len(D)), np.zeros(len(D))]
            self.model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000).fit(X, y)
        else:
            Y = np.concatenate([y for _, y in groups])
            Xs = self.scaler.transform(Xall)
            if self.kind == "pointwise":
                self.model = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, Y)
            else:
                self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                            max_leaf_nodes=31, random_state=self.seed).fit(Xs, Y)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        n = len(feats)
        if self.kind == "identity":
            return -np.arange(n, dtype=float)       # keep retrieval order
        Xs = self.scaler.transform(feats)
        if self.kind == "pairwise":
            return Xs @ self.model.coef_.ravel()
        return self.model.predict_proba(Xs)[:, 1]

    def coefficients(self) -> dict[str, float] | None:
        if self.kind in {"pointwise", "pairwise"}:
            return dict(zip(FEATURES, self.model.coef_.ravel().round(4).tolist()))
        return None
