"""Full-ranking offline evaluation with slices.

PROTOCOL
  * Rank the ENTIRE catalog for each user (no sampled negatives: sampled metrics
    can reorder models, Krichene & Rendle, KDD 2020).
  * Mask everything the user interacted with in the fit period (any rating).
  * Ground truth = positives (rating >= threshold) in the evaluation window.
  * Slices:
        cold users   : 0 positives in fit period (popularity fallback)
        sparse users : 1..9 positives
        warm users   : >= 10 positives
        tail recall  : recall computed only on targets outside the top-20% most
                       popular items (measures long-tail ability)
  * Also reports catalog coverage and mean popularity percentile of recs
    (popularity bias).
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sp

from agentrec.data.dataset import Dataset
from agentrec.evaluation.metrics import evaluate_lists, recall_at_k


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise top-k indices, sorted best-first. O(n_items) per row via argpartition."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1, kind="stable")
    return part[rows, order]


def recommend_all(
    score_fn: Callable[[np.ndarray], np.ndarray],
    users: np.ndarray,
    seen: sp.csr_matrix,
    k: int,
    batch: int = 512,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = np.array(score_fn(ub), dtype=np.float32, copy=True)
        sub = seen[ub]
        sc[sub.nonzero()] = -np.inf           # never recommend already-seen items
        for u, row in zip(ub, topk_from_scores(sc, k)):
            out[int(u)] = row.tolist()
    return out


def evaluate_model(
    score_fn: Callable[[np.ndarray], np.ndarray],
    ds: Dataset,
    phase: str,
    ks: Iterable[int] = (10, 20, 50),
) -> dict[str, dict[str, float]]:
    ks = list(ks)
    fit = ds.fit_period(phase)
    seen = ds.matrix(fit, positive_only=False)
    pos_fit = ds.matrix(fit, positive_only=True)
    n_hist = np.asarray(pos_fit.sum(1)).ravel()
    targets = ds.eval_period(phase).targets()
    users = np.array(sorted(targets), dtype=np.int64)
    lists = recommend_all(score_fn, users, seen, max(ks))

    res: dict[str, dict[str, float]] = {"all": evaluate_lists(lists, targets, ks)}
    slices = {
        "cold_users": [u for u in users if n_hist[u] == 0],
        "sparse_users": [u for u in users if 0 < n_hist[u] < 10],
        "warm_users": [u for u in users if n_hist[u] >= 10],
    }
    for name, us in slices.items():
        if us:
            res[name] = evaluate_lists({u: lists[u] for u in us}, {u: targets[u] for u in us}, ks)

    # long-tail: targets outside the top-20% most popular items (by fit-period positives)
    pop = np.asarray(pos_fit.sum(0)).ravel()
    head = set(np.argsort(-pop)[: int(0.2 * ds.n_items)].tolist())
    k = ks[0]
    tail_rec = [recall_at_k(lists[u], targets[u] - head, k) for u in users if targets[u] - head]
    res["tail"] = {"n_users": float(len(tail_rec)), f"recall@{k}": float(np.mean(tail_rec)) if tail_rec else float("nan")}

    # popularity bias / coverage of the top-k lists
    rank_pct = np.empty(ds.n_items)
    rank_pct[np.argsort(-pop)] = np.arange(ds.n_items) / ds.n_items  # 0 = most popular
    rec_items = np.array([i for u in users for i in lists[u][:k]])
    res["diversity"] = {
        f"coverage@{k}": float(len(set(rec_items.tolist())) / ds.n_items),
        f"mean_pop_percentile@{k}": float(rank_pct[rec_items].mean()),
    }
    return res
