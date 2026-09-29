"""Exercise 04: one BPR SGD step. Reference: models/mf_bpr.py (derivation in its docstring).

x = p_u.(q_i - q_j);  loss = -log sigmoid(x) + reg/2 (|p_u|^2 + |q_i|^2 + |q_j|^2)
"""
import numpy as np


def bpr_loss(p, qi, qj, reg):
    raise NotImplementedError


def bpr_grads(p, qi, qj, reg):
    """Return (dL/dp, dL/dqi, dL/dqj)."""
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p, qi, qj, reg = rng.normal(size=4), rng.normal(size=4), rng.normal(size=4), 0.1
    g = bpr_grads(p, qi, qj, reg)
    for k, v in enumerate((p, qi, qj)):
        num = np.zeros_like(v)
        for d in range(len(v)):
            v[d] += 1e-6; lp = bpr_loss(p, qi, qj, reg)
            v[d] -= 2e-6; lm = bpr_loss(p, qi, qj, reg)
            v[d] += 1e-6
            num[d] = (lp - lm) / 2e-6
        assert np.allclose(num, g[k], atol=1e-6), (k, num, g[k])
    # one step must reduce the loss
    l0 = bpr_loss(p, qi, qj, reg)
    l1 = bpr_loss(p - 0.1 * g[0], qi - 0.1 * g[1], qj - 0.1 * g[2], reg)
    assert l1 < l0
    print("all tests passed. Question: when is sigmoid(-x) close to 0, and what does that mean for learning?")
