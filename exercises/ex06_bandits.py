"""Exercise 06: bandit maths. References: bandits/mab.py, bandits/linear.py."""
import numpy as np


def ucb1_select(means: np.ndarray, counts: np.ndarray, t: int, c: float = 1.0) -> int:
    """Play any unplayed arm first; else argmax mean + c*sqrt(2 ln t / n)."""
    raise NotImplementedError


def beta_update(alpha: np.ndarray, beta: np.ndarray, arm: int, reward: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def sherman_morrison(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return (A + x x^T)^{-1} given A^{-1}, in O(d^2)."""
    raise NotImplementedError


def linucb_scores(X: np.ndarray, A_inv: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    assert ucb1_select(np.array([0.6, 0.55]), np.array([100, 4]), 104) == 1        # LEARNING_NOTES quiz 6.1
    assert ucb1_select(np.array([0.6, 0.0]), np.array([10, 0]), 10) == 1
    a, b = beta_update(np.ones(2), np.ones(2), 0, 1)
    assert a.tolist() == [2, 1] and b.tolist() == [1, 1]
    rng = np.random.default_rng(0)
    A = np.eye(3) * 2.0
    A_inv = np.linalg.inv(A)
    for _ in range(10):
        x = rng.normal(size=3)
        A += np.outer(x, x)
        A_inv = sherman_morrison(A_inv, x)
    assert np.allclose(A_inv, np.linalg.inv(A))
    s = linucb_scores(np.eye(3), np.eye(3), np.array([1.0, 0, 0]), alpha=0.5)
    assert np.allclose(s, [1.5, 0.5, 0.5])
    print("all tests passed. Question: why is the UCB bonus sqrt(x^T A^-1 x) and not x^T A^-1 x?")
