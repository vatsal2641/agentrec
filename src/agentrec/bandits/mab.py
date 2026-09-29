"""Non-contextual multi-armed bandits (for learning the ideas; see LEARNING_NOTES §6).

Setting: K arms, arm a pays reward ~ Bernoulli(mu_a), mu unknown.
Each round pick one arm, observe its reward only (bandit feedback).
Regret after T rounds:  R_T = T * mu*  -  sum_t mu_{a_t}      (mu* = best arm's mean)

EpsilonGreedy   explore uniformly with prob eps, else play the best empirical mean.
                Linear regret if eps is constant (it never stops exploring).
UCB1            play argmax  mean_a + c * sqrt(2 ln t / n_a)
                "optimism in the face of uncertainty": the bonus shrinks as n_a grows.
                Regret O(sum_a ln T / gap_a)  (Auer et al., 2002).
Thompson        keep a Beta(alpha_a, beta_a) posterior per arm; sample theta_a from each,
                play argmax theta_a; update alpha += r, beta += 1 - r.
                Explores in proportion to the probability an arm is best.
"""
from __future__ import annotations

import numpy as np


class EpsilonGreedy:
    def __init__(self, k: int, eps: float = 0.1) -> None:
        self.eps, self.n, self.s = eps, np.zeros(k), np.zeros(k)

    def select(self, rng: np.random.Generator) -> int:
        if rng.random() < self.eps or self.n.min() == 0:
            return int(rng.integers(len(self.n))) if self.n.min() > 0 else int(np.argmin(self.n))
        return int(np.argmax(self.s / self.n))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.s[a] += r


class UCB1:
    def __init__(self, k: int, c: float = 1.0) -> None:
        self.c, self.n, self.s, self.t = c, np.zeros(k), np.zeros(k), 0

    def select(self, rng: np.random.Generator) -> int:
        self.t += 1
        if self.n.min() == 0:
            return int(np.argmin(self.n))       # play every arm once
        ucb = self.s / self.n + self.c * np.sqrt(2 * np.log(self.t) / self.n)
        return int(np.argmax(ucb))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.s[a] += r


class ThompsonBeta:
    def __init__(self, k: int, a0: float = 1.0, b0: float = 1.0) -> None:
        self.a, self.b = np.full(k, a0), np.full(k, b0)

    def select(self, rng: np.random.Generator) -> int:
        return int(np.argmax(rng.beta(self.a, self.b)))

    def update(self, a: int, r: float) -> None:
        self.a[a] += r
        self.b[a] += 1 - r


def run_mab(policy, mu: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """Returns cumulative (pseudo-)regret per round."""
    best = mu.max()
    reg = np.empty(T)
    for t in range(T):
        a = policy.select(rng)
        policy.update(a, float(rng.random() < mu[a]))
        reg[t] = best - mu[a]
    return np.cumsum(reg)
