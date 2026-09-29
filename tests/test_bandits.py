import numpy as np

from helpers import *  # noqa: F401,F403
from agentrec.bandits.linear import EpsGreedy, Frozen, Greedy, LinTS, LinUCB, RandomPolicy, RidgeState
from agentrec.bandits.mab import EpsilonGreedy, ThompsonBeta, UCB1, run_mab


def test_sherman_morrison_matches_direct_inverse():
    rng = np.random.default_rng(0)
    s = RidgeState(4, lam=2.0)
    A = 2.0 * np.eye(4)
    for _ in range(20):
        x, w = rng.normal(size=4), rng.random() + 0.5
        s.update(x, 1.0, w)
        A += w * np.outer(x, x)
    assert np.allclose(s.A_inv, np.linalg.inv(A), atol=1e-8)


def test_prior_mean_is_theta_before_any_update():
    th0 = np.array([0.1, 0.5, 0, 0])
    assert np.allclose(RidgeState(4, lam=3.0, theta0=th0).theta, th0)


def test_ucb_and_thompson_beat_uniform_exploration():
    mu = np.array([0.1, 0.2, 0.3, 0.35, 0.5])
    T = 4000
    res = {}
    for name, p in [("eps1", EpsilonGreedy(5, eps=1.0)), ("ucb", UCB1(5)), ("ts", ThompsonBeta(5))]:
        res[name] = run_mab(p, mu, T, np.random.default_rng(1))[-1]
    assert res["ucb"] < res["eps1"] and res["ts"] < res["eps1"]
    assert res["ts"] < 0.25 * res["eps1"]


def _linear_env_regret(policy, T=1500, seed=0):
    rng = np.random.default_rng(seed)
    d, n, k = 6, 20, 1
    theta = rng.normal(size=d); theta /= np.linalg.norm(theta)
    total = 0.0
    for _ in range(T):
        X = rng.normal(size=(n, d))
        p = 1 / (1 + np.exp(-3 * X @ theta))
        a = policy.select(X, k, rng)[0]
        policy.update(X[a], float(rng.random() < p[a]))
        total += p.max() - p[a]
    return total


def test_learning_policies_beat_random_on_linear_env():
    d = 6
    rnd = _linear_env_regret(RandomPolicy(d))
    for pol in (Greedy(d), LinUCB(d, alpha=0.5), LinTS(d, v=0.3), EpsGreedy(d, eps=0.1)):
        assert _linear_env_regret(pol) < 0.6 * rnd, pol.name


def test_frozen_policy_never_changes():
    f = Frozen(3, theta0=np.array([1.0, 0, 0]))
    f.update(np.ones(3), 1.0)
    assert np.allclose(f.state.theta, [1, 0, 0])


def test_linucb_width_shrinks_in_observed_directions():
    p = LinUCB(3, alpha=1.0)
    x = np.array([1.0, 0, 0])
    w0 = p.state.width(x[None])[0]
    for _ in range(50):
        p.update(x, 0.0)
    assert p.state.width(x[None])[0] < 0.2 * w0
    assert np.isclose(p.state.width(np.array([[0, 1.0, 0]]))[0], w0)
