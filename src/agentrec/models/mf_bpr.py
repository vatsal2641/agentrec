"""Matrix factorisation trained with BPR (Rendle et al., UAI 2009), in numpy.

MODEL      s(u, i) = p_u . q_i + b_i          p_u, q_i in R^d
OBJECTIVE  for triples (u, i, j): i is a positive of u, j a sampled item
           x_uij = s(u, i) - s(u, j)
           L = - sum log sigmoid(x_uij) + reg * (||p_u||^2 + ||q_i||^2 + ||q_j||^2 + b_i^2 + b_j^2)
           "a positive should score higher than a random item" -> pairwise ranking loss.
GRADIENT   let g = sigmoid(-x_uij)   (large when the pair is mis-ordered)
           dL/dp_u = -g (q_i - q_j) + 2 reg p_u
           dL/dq_i = -g p_u         + 2 reg q_i
           dL/dq_j = +g p_u         + 2 reg q_j
           dL/db_i = -g + 2 reg b_i ; dL/db_j = +g + 2 reg b_j
TRAINING   mini-batch SGD, uniform negative sampling. Cost per epoch O(|pos| * d).
INFERENCE  one matrix product: O(|I| * d) per user -> retrieval via nearest neighbours.
WHY BPR    implicit data has no negatives; we only know "i observed" > "j not observed".
           BPR optimises exactly that ordering (a smooth proxy for AUC).
"""
from __future__ import annotations

import numpy as np

from agentrec.data.dataset import Dataset, Period
from agentrec.models.base import Recommender


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable


class BPRMF(Recommender):
    name = "bpr_mf"

    def __init__(self, dim: int = 64, epochs: int = 30, lr: float = 0.05, reg: float = 5e-4,
                 batch_size: int = 4096, seed: int = 0, verbose: bool = False) -> None:
        self.dim, self.epochs, self.lr, self.reg = dim, epochs, lr, reg
        self.batch_size, self.seed, self.verbose = batch_size, seed, verbose
        self.history: list[float] = []

    def _fit(self, ds: Dataset, period: Period) -> None:
        rng = np.random.default_rng(self.seed)
        U, I, d = self.n_users, self.n_items, self.dim
        self.P = rng.normal(0, 0.1, (U, d)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (I, d)).astype(np.float32)
        self.b = np.zeros(I, dtype=np.float32)
        coo = self.X.tocoo()
        users, items = coo.row.astype(np.int64), coo.col.astype(np.int64)
        pos_keys = np.sort(users * I + items)       # for O(log n) "is j a positive?" checks
        n = len(users)
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                u, i = users[idx], items[idx]
                j = rng.integers(0, I, len(idx))
                # resample negatives that are actually positives (once; residual collisions are rare)
                hit = np.isin(u * I + j, pos_keys, assume_unique=False)
                j[hit] = rng.integers(0, I, hit.sum())
                pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
                x = (pu * (qi - qj)).sum(1) + self.b[i] - self.b[j]
                g = _sigmoid(-x)[:, None]                          # (B, 1)
                tot += float(-np.log(_sigmoid(x) + 1e-10).sum())
                lr, r = self.lr, self.reg
                gp = g * (qi - qj) - r * pu
                gi = g * pu - r * qi
                gj = -g * pu - r * qj
                # np.add.at handles repeated indices in a batch correctly (plain += would drop them)
                np.add.at(self.P, u, lr * gp)
                np.add.at(self.Q, i, lr * gi)
                np.add.at(self.Q, j, lr * gj)
                np.add.at(self.b, i, lr * (g[:, 0] - r * self.b[i]))
                np.add.at(self.b, j, lr * (-g[:, 0] - r * self.b[j]))
            self.history.append(tot / n)
            if self.verbose:
                print(f"[bpr] epoch {ep + 1}/{self.epochs} loss={tot / n:.4f}")

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self.P[users] @ self.Q.T + self.b[None, :]

    # embeddings for retrieval: fold the bias in as an extra dimension so a pure
    # dot product reproduces the full score: [p_u, 1] . [q_i, b_i]
    def user_embeddings(self, users: np.ndarray) -> np.ndarray:
        return np.hstack([self.P[users], np.ones((len(users), 1), np.float32)])

    def item_embeddings(self) -> np.ndarray:
        return np.hstack([self.Q, self.b[:, None]])
