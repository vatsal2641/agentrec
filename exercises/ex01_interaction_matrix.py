"""Exercise 01 — interaction matrix + popularity top-K.

Rules: plain Python + numpy only (no pandas, no recsys libraries).
Fill in the three functions; run `python exercises/ex01_interaction_matrix.py`.
All asserts must pass. Then answer the written questions in chat.
"""
from __future__ import annotations

import numpy as np

# (user, item, rating) — the toy matrix from the lesson
EVENTS: list[tuple[str, str, int]] = [
    ("A", "i1", 5), ("A", "i2", 3), ("A", "i4", 1),
    ("B", "i1", 4), ("B", "i3", 5),
    ("C", "i2", 4), ("C", "i3", 4), ("C", "i5", 2),
    ("D", "i1", 5),
]


def build_matrix(
    events: list[tuple[str, str, int]],
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Return (R, user_index, item_index).

    R[u, i] = rating if observed, else np.nan  (NOT 0 — think about why).
    user_index / item_index map raw IDs -> contiguous row/col ints, sorted by raw ID.
    """
    raise NotImplementedError


def density(R: np.ndarray) -> float:
    """Fraction of observed cells."""
    raise NotImplementedError


def popular_topk(
    R: np.ndarray, user_row: int, k: int, item_index: dict[str, int]
) -> list[str]:
    """Top-k raw item IDs by interaction COUNT, excluding items the user already rated.

    Tie-break: smaller raw item ID first. Document why a deterministic tie-break matters.
    """
    raise NotImplementedError


if __name__ == "__main__":
    R, u_idx, i_idx = build_matrix(EVENTS)
    assert R.shape == (4, 5), R.shape
    assert np.isnan(R[u_idx["D"], i_idx["i3"]])
    assert R[u_idx["B"], i_idx["i3"]] == 5
    assert abs(density(R) - 0.45) < 1e-9
    assert popular_topk(R, u_idx["D"], 2, i_idx) == ["i2", "i3"]
    assert popular_topk(R, u_idx["A"], 1, i_idx) == ["i3"]
    print("all tests passed")
