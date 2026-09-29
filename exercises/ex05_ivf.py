"""Exercise 05: IVF search. Reference: retrieval/index.py::IVFIndex."""
import numpy as np


def build_ivf(V: np.ndarray, centroids: np.ndarray) -> list[np.ndarray]:
    """Assign each item to its nearest centroid (by inner product). Return lists of item ids per centroid."""
    raise NotImplementedError


def ivf_search(q: np.ndarray, V: np.ndarray, centroids: np.ndarray, lists: list[np.ndarray], k: int, n_probe: int) -> np.ndarray:
    """Probe the n_probe best centroids, brute-force their items, return top-k ids (best first)."""
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    V = rng.normal(size=(1000, 8)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    C = V[rng.choice(1000, 16, replace=False)]
    lists = build_ivf(V, C)
    assert sum(len(l) for l in lists) == 1000
    q = rng.normal(size=8)
    exact = np.argsort(-(V @ q))[:10]
    assert list(ivf_search(q, V, C, lists, 10, n_probe=16)) == list(exact)
    approx = ivf_search(q, V, C, lists, 10, n_probe=2)
    print("recall@10 with 2/16 probes:", len(set(approx) & set(exact)) / 10)
    print("all tests passed. Question: what is the cost of a query as a function of n_probe?")
