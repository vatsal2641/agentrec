"""Contextual linear bandits for slate selection.

MODEL      expected reward of showing item i to user u:  E[r | x] = theta . x_{u,i}
           x_{u,i} = context features of the (user, item) pair (built in bandits/env.py)
           One theta shared by all items ("shared/hybrid" LinUCB) -> generalises to
           items never shown before, unlike per-arm ("disjoint") LinUCB.
ESTIMATE   ridge regression, updated online:
               A = lambda I + sum x x^T        b = lambda theta0 + sum r x
               theta_hat = A^{-1} b
           theta0 = prior mean (warm start from the offline ranker).
           A^{-1} is updated in O(d^2) with Sherman–Morrison instead of re-inverting:
               A^{-1} <- A^{-1} - (A^{-1} x)(A^{-1} x)^T / (1 + x^T A^{-1} x)
UNCERTAINTY  width(x) = sqrt(x^T A^{-1} x)  — large for feature directions rarely seen.

POLICIES (all score each candidate, then take the top-k as the slate)
  Frozen     : score = prior (ranker) only, never learns.           (the offline system)
  Greedy     : score = theta_hat . x                                   (learns, never explores)
  EpsGreedy  : each slot random with prob eps, else greedy
  LinUCB     : score = theta_hat . x + alpha * width(x)                (Li et al., WWW 2010)
  LinTS      : theta~ ~ N(theta_hat, v^2 A^{-1});  score = theta~ . x (Agrawal & Goyal, ICML 2013)
  Random     : uniform slate                                           (lower bound)
"""
from __future__ import annotations

import numpy as np


class RidgeState:
    def __init__(self, d: int, lam: float = 1.0, theta0: np.ndarray | None = None) -> None:
        self.d, self.lam = d, lam
        self.A_inv = np.eye(d) / lam
        self.b = lam * (np.zeros(d) if theta0 is None else np.asarray(theta0, float).copy())
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def update(self, x: np.ndarray, r: float, w: float = 1.0) -> None:
        Ax = self.A_inv @ x
        self.A_inv -= w * np.outer(Ax, Ax) / (1.0 + w * x @ Ax)
        self.b += w * r * x
        self.n_updates += 1

    def width(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(np.einsum("nd,de,ne->n", X, self.A_inv, X), 0.0))


class LinearPolicy:
    name = "base"
    learns = True

    def __init__(self, d: int, lam: float = 1.0, theta0: np.ndarray | None = None) -> None:
        self.state = RidgeState(d, lam, theta0)

    def scores(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return X @ self.state.theta

    def select(self, X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        s = self.scores(X, rng)
        return np.argsort(-s, kind="stable")[:k]

    def greedy(self, X: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(-(X @ self.state.theta), kind="stable")[:k]

    def update(self, x: np.ndarray, r: float, w: float = 1.0) -> None:
        if self.learns:
            self.state.update(x, r, w)


class Frozen(LinearPolicy):
    name, learns = "frozen", False


class Greedy(LinearPolicy):
    name = "greedy"


class RandomPolicy(LinearPolicy):
    name, learns = "random", False

    def select(self, X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(X), size=k, replace=False)


class EpsGreedy(LinearPolicy):
    name = "eps_greedy"

    def __init__(self, d: int, eps: float = 0.1, **kw) -> None:
        super().__init__(d, **kw)
        self.eps = eps

    def select(self, X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        order = list(np.argsort(-(X @ self.state.theta), kind="stable"))
        out: list[int] = []
        for _ in range(k):
            if rng.random() < self.eps:
                rest = [i for i in range(len(X)) if i not in out]
                pick = int(rng.choice(rest))
            else:
                pick = next(i for i in order if i not in out)
            out.append(pick)
        return np.array(out)


class LinUCB(LinearPolicy):
    name = "linucb"

    def __init__(self, d: int, alpha: float = 1.0, **kw) -> None:
        super().__init__(d, **kw)
        self.alpha = alpha

    def scores(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return X @ self.state.theta + self.alpha * self.state.width(X)


class LinTS(LinearPolicy):
    name = "lints"

    def __init__(self, d: int, v: float = 0.5, **kw) -> None:
        super().__init__(d, **kw)
        self.v = v

    def scores(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cov = self.v ** 2 * (self.state.A_inv + self.state.A_inv.T) / 2
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))
        theta = self.state.theta + L @ rng.standard_normal(len(cov))
        return X @ theta


def make_policy(name: str, d: int, theta0: np.ndarray, cfg: dict) -> LinearPolicy:
    kw = {"lam": cfg.get("lam", 1.0), "theta0": theta0}
    if name == "frozen":
        return Frozen(d, **kw)
    if name == "greedy":
        return Greedy(d, **kw)
    if name == "random":
        return RandomPolicy(d, **kw)
    if name == "eps_greedy":
        return EpsGreedy(d, eps=cfg.get("epsilon", 0.1), **kw)
    if name == "linucb":
        return LinUCB(d, alpha=cfg.get("alpha_ucb", 1.0), **kw)
    if name == "lints":
        return LinTS(d, v=cfg.get("ts_v", 0.5), **kw)
    raise ValueError(f"unknown policy {name!r}")
