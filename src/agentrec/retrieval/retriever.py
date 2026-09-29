"""Candidate generation = union of several cheap sources.

Real systems rarely trust one retriever. Each source covers a different failure:
  * embedding (two-tower / MF) : personalised, but weak for sparse users
  * recent popularity          : safe fallback for cold users, captures trends
Candidates are merged with de-duplication; the source of every candidate is
kept so we can measure each source's contribution (Recall@N per source).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidates:
    items: np.ndarray                 # (N,) item indices, retrieval order
    scores: np.ndarray                # (N,) embedding score (nan if from another source)
    sources: list[str] = field(default_factory=list)


class CandidateRetriever:
    def __init__(self, index, item_vecs: np.ndarray, popular_items: np.ndarray, pop_share: float = 0.1) -> None:
        """index: ExactIndex/IVFIndex over item_vecs; popular_items: items sorted by (recent) popularity."""
        self.index = index
        self.V = item_vecs
        self.popular = np.asarray(popular_items)
        self.pop_share = pop_share

    def retrieve(self, user_vec: np.ndarray | None, n: int, exclude: set[int] | None = None,
                 allowed: np.ndarray | None = None) -> Candidates:
        """user_vec None -> cold user -> popularity only.
        allowed: optional boolean mask over items (hard filters such as genre/year),
        applied *inside* retrieval so filtered-out items do not waste candidate slots."""
        exclude = set(exclude or ())
        if allowed is not None:
            exclude |= set(np.where(~allowed)[0].tolist())
        items: list[int] = []
        scores: list[float] = []
        sources: list[str] = []
        n_pop = n if user_vec is None or not np.any(user_vec) else int(round(n * self.pop_share))
        n_emb = n - n_pop
        if n_emb > 0:
            idx, sc = self.index.search(user_vec, n_emb, exclude=[exclude])
            for i, s in zip(idx[0], sc[0]):
                if i >= 0 and np.isfinite(s):
                    items.append(int(i)); scores.append(float(s)); sources.append("embedding")
        taken = set(items) | exclude
        for i in self.popular:
            if len(items) >= n:
                break
            if int(i) not in taken:
                items.append(int(i)); taken.add(int(i))
                scores.append(float(self.V[i] @ user_vec) if user_vec is not None and np.any(user_vec) else float("nan"))
                sources.append("popular")
        return Candidates(np.array(items, dtype=np.int64), np.array(scores), sources)
