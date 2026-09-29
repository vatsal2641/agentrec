"""Top-K ranking metrics, written to be read.

Notation for one user:
    L = [l_1, ..., l_K]  ranked recommendation list (best first)
    R                    set of relevant (held-out positive) items
    rel_k = 1 if l_k in R else 0

    Precision@K = (1/K)   * sum_k rel_k
    Recall@K    = (1/|R|) * sum_k rel_k
    HitRate@K   = 1 if any rel_k else 0
    MRR@K       = 1 / (position of first relevant item)   (0 if none in top K)
    DCG@K       = sum_k rel_k / log2(k + 1)
    NDCG@K      = DCG@K / IDCG@K,  IDCG = DCG of a perfect list = sum_{k<=min(K,|R|)} 1/log2(k+1)
    AP@K        = (1/min(K,|R|)) * sum_k Precision@k * rel_k        (MAP = mean over users)

All metrics are averaged over users that have >= 1 relevant item.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _rel(ranked: Sequence[int], relevant: set[int], k: int) -> np.ndarray:
    return np.fromiter((1.0 if x in relevant else 0.0 for x in ranked[:k]), dtype=float, count=min(k, len(ranked)))


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return float(_rel(ranked, relevant, k).sum() / k)


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        raise ValueError("recall undefined for empty relevant set")
    return float(_rel(ranked, relevant, k).sum() / len(relevant))


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return float(_rel(ranked, relevant, k).any())


def mrr_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    for pos, x in enumerate(ranked[:k], start=1):
        if x in relevant:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    rel = _rel(ranked, relevant, k)
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    n_ideal = min(k, len(relevant))
    idcg = float((1.0 / np.log2(np.arange(2, n_ideal + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    rel = _rel(ranked, relevant, k)
    if rel.sum() == 0:
        return 0.0
    precisions = np.cumsum(rel) / np.arange(1, rel.size + 1)
    return float((precisions * rel).sum() / min(k, len(relevant)))


METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "hit": hit_rate_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
}


def evaluate_lists(
    lists: dict[int, Sequence[int]],
    targets: dict[int, set[int]],
    ks: Iterable[int] = (10, 20, 50),
) -> dict[str, float]:
    """Average every metric@k over users present in both dicts."""
    users = [u for u in lists if targets.get(u)]
    if not users:
        raise ValueError("no users with both a list and a non-empty target set")
    out: dict[str, float] = {"n_users": float(len(users))}
    for k in ks:
        for name, fn in METRICS.items():
            out[f"{name}@{k}"] = float(np.mean([fn(lists[u], targets[u], k) for u in users]))
    return out
