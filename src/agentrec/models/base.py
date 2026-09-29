"""Common interface for every scorer in the repo.

A Recommender is anything that, after `fit`, can produce a score for every
(user, item) pair: `score(users) -> (len(users), n_items)` array.
Higher = better. Evaluation and retrieval are built on this single contract.

Cold-start users (no history in the fit period) are routed to a popularity
fallback here, explicitly, so no model silently scores them with random
embeddings. Evaluation reports these users as a separate slice.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from agentrec.data.dataset import Dataset, Period


class Recommender(ABC):
    name: str = "base"

    def fit(self, ds: Dataset, period: Period) -> "Recommender":
        self.n_users, self.n_items = ds.n_users, ds.n_items
        self.X: sp.csr_matrix = ds.matrix(period, positive_only=True)  # positives
        counts = np.asarray(self.X.sum(0)).ravel()
        self.pop_scores = np.log1p(counts).astype(np.float32)
        self.has_history = np.asarray(self.X.sum(1)).ravel() > 0
        self._fit(ds, period)
        return self

    @abstractmethod
    def _fit(self, ds: Dataset, period: Period) -> None: ...

    @abstractmethod
    def _score(self, users: np.ndarray) -> np.ndarray: ...

    def score(self, users: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        s = np.asarray(self._score(users), dtype=np.float32)
        cold = ~self.has_history[users]
        if cold.any():
            s[cold] = self.pop_scores  # explicit popularity fallback
        return s
