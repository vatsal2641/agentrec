"""Nearest-neighbour indexes for maximum-inner-product search (MIPS).

WHY RETRIEVAL EXISTS
  Scoring every item with an expensive ranker costs O(|I| * cost_ranker) per
  request. With |I| = 10^7 and a 1 µs ranker that is 10 s per request: impossible.
  A retrieval stage uses a *cheap, factorised* score (dot product of two
  embeddings) to shortlist N ≈ 100–1000 items; only those reach the ranker.

ExactIndex  brute force  q . V^T  -> O(|I| d) per query. Exact, simple, the baseline.
IVFIndex    inverted file (the idea behind FAISS IndexIVF):
    build : k-means the item vectors into `n_lists` clusters (centroids c_1..c_C)
    query : score the C centroids, visit only the `n_probe` best clusters,
            brute-force the items inside them.
    cost  : O(C d + (n_probe / C) |I| d)   vs  O(|I| d)
    trade : recall@N < 1 (true neighbours in unvisited clusters are missed);
            n_probe controls the speed/recall trade-off.
FaissIndex  optional wrapper to compare against the real library when installed.
"""
from __future__ import annotations

import numpy as np


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[-1])
    part = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(scores, part, -1), axis=-1, kind="stable")
    return np.take_along_axis(part, order, -1)


class ExactIndex:
    def __init__(self, item_vecs: np.ndarray) -> None:
        self.V = np.ascontiguousarray(item_vecs, dtype=np.float32)

    def search(self, q: np.ndarray, k: int, exclude: list[set[int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(q).astype(np.float32)
        s = q @ self.V.T
        if exclude:
            for r, ex in enumerate(exclude):
                if ex:
                    s[r, list(ex)] = -np.inf
        idx = _topk(s, k)
        sc = np.take_along_axis(s, idx, 1)
        idx[~np.isfinite(sc)] = -1        # fewer than k allowed items: never return an excluded one
        return idx, sc


def kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd's k-means with k-means++ init. Returns (centroids, assignment)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):                                    # k-means++ seeding
        C.append(X[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        dist = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        a = dist.argmin(1)
        for j in range(k):
            m = a == j
            C[j] = X[m].mean(0) if m.any() else X[rng.integers(n)]
    return C, a


class IVFIndex:
    def __init__(self, item_vecs: np.ndarray, n_lists: int = 64, n_probe: int = 8, seed: int = 0) -> None:
        self.V = np.ascontiguousarray(item_vecs, dtype=np.float32)
        self.n_probe = n_probe
        self.C, assign = kmeans(self.V.astype(np.float64), n_lists, seed=seed)
        self.C = self.C.astype(np.float32)
        self.lists = [np.where(assign == j)[0] for j in range(n_lists)]

    def search(self, q: np.ndarray, k: int, exclude: list[set[int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(q).astype(np.float32)
        probes = _topk(q @ self.C.T, self.n_probe)
        out_i = np.full((len(q), k), -1, dtype=np.int64)
        out_s = np.full((len(q), k), -np.inf, dtype=np.float32)
        for r in range(len(q)):
            cand = np.concatenate([self.lists[j] for j in probes[r]])
            if exclude and exclude[r]:
                cand = cand[~np.isin(cand, list(exclude[r]))]
            if len(cand) == 0:
                continue
            s = self.V[cand] @ q[r]
            top = _topk(s, k)
            out_i[r, :len(top)] = cand[top]
            out_s[r, :len(top)] = s[top]
        return out_i, out_s


class FaissIndex:
    """Optional: exact inner-product FAISS index (pip install faiss-cpu)."""

    def __init__(self, item_vecs: np.ndarray) -> None:
        import faiss  # noqa: PLC0415  (optional dependency)
        self.index = faiss.IndexFlatIP(item_vecs.shape[1])
        self.index.add(np.ascontiguousarray(item_vecs, dtype=np.float32))

    def search(self, q: np.ndarray, k: int, exclude: list[set[int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
        extra = max((len(e) for e in exclude), default=0) if exclude else 0
        s, i = self.index.search(np.atleast_2d(q).astype(np.float32), k + extra)
        if exclude:  # over-fetch, then drop excluded ids
            rows_i, rows_s = [], []
            for r in range(len(i)):
                keep = [c for c in range(i.shape[1]) if i[r, c] not in exclude[r]][:k]
                rows_i.append(i[r, keep])
                rows_s.append(s[r, keep])
            return np.array(rows_i), np.array(rows_s)
        return i, s
