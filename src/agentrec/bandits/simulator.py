"""Semi-synthetic user simulator for online (bandit / feedback-loop) experiments.

WHY A SIMULATOR
  Logged data only has feedback for items the *old* system showed. If a bandit
  picks something never shown, its reward is unknown -> you cannot evaluate a
  new exploration policy on MovieLens logs directly. Options: (a) a fully
  observed dataset, (b) replay on uniformly-random logs, (c) a simulator.
  We use (c) and say so everywhere: results describe behaviour *in this
  simulator*, not real users.

GROUND TRUTH  z(u, i): standardised "true" affinity
  * synthetic data : the generator's latent factors (truth.npz) — exact.
  * MovieLens      : a BPR model fit on ALL periods (train+val+test), which the
                     recommender never sees. A proxy for truth, not truth.
USER MODEL for a slate [i_1..i_K]  (position-based model, PBM)
  examined_k ~ Bernoulli(e_k),  e_k = 1 / k^eta          (position bias)
  click_k    ~ Bernoulli(p(u, i_k)) if examined,  p = sigmoid(beta (z - z0))
  skip_k     = examined and not clicked                   (explicit "swipe away")
  rating_k   : after a click, with prob p_rate, rating = clip(round(3.5 + 1.2 (z - z0) + noise), 1, 5)
  noise      : each observed click label is flipped with prob `flip` (accidental
               clicks / missed engagement) — the learner sees the noisy label,
               regret is computed on true probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentrec.data.dataset import Dataset


@dataclass
class SlateFeedback:
    examined: np.ndarray   # (K,) bool (hidden from the learner in reality)
    clicked: np.ndarray    # (K,) bool, possibly noisy
    skipped: np.ndarray    # (K,) bool
    ratings: np.ndarray    # (K,) int, 0 = no rating


class TruthModel:
    def __init__(self, U: np.ndarray, V: np.ndarray, bias: np.ndarray | None = None, source: str = "") -> None:
        self.U, self.V, self.bias, self.source = U, V, bias, source
        sample = (U[:200] @ V.T).ravel() + (0 if bias is None else np.tile(bias, min(200, len(U))))
        self.mu, self.sd = float(sample.mean()), float(sample.std())

    def z(self, u: int, items: np.ndarray) -> np.ndarray:
        s = self.V[items] @ self.U[u] + (0 if self.bias is None else self.bias[items])
        return (s - self.mu) / self.sd


def build_truth(ds: Dataset, raw_dir: str | Path, bpr_cfg: dict | None = None, seed: int = 123) -> TruthModel:
    t = Path(raw_dir) / "truth.npz"
    if t.exists():   # synthetic data: exact generative truth
        z = np.load(t)
        W, V = z["W"], z["V"]
        order_u = ds.user_raw - 1     # raw ids are 1..n in the generator
        order_i = ds.item_raw - 1
        return TruthModel(W[order_u] / np.sqrt(W.shape[1]), V[order_i], source="synthetic generator latents")
    # real data: proxy truth = BPR on everything (never used by the recommender)
    import pandas as pd
    from agentrec.data.dataset import Period
    from agentrec.models.mf_bpr import BPRMF
    allp = Period(pd.concat([ds.train.df, ds.val.df, ds.test.df], ignore_index=True), ds.pos_threshold)
    m = BPRMF(**(bpr_cfg or {}), seed=seed).fit(ds, allp)
    return TruthModel(m.P, m.Q, m.b, source="BPR fit on all periods (proxy truth)")


class Simulator:
    def __init__(self, truth: TruthModel, beta: float = 1.5, z0: float = 2.0, eta: float = 0.7,
                 p_rate: float = 0.3, flip: float = 0.0, seed: int = 0) -> None:
        self.truth, self.beta, self.z0, self.eta = truth, beta, z0, eta
        self.p_rate, self.flip = p_rate, flip
        self.rng = np.random.default_rng(seed)

    def exam_probs(self, k: int) -> np.ndarray:
        return 1.0 / np.arange(1, k + 1) ** self.eta

    def click_probs(self, u: int, items: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.beta * (self.truth.z(u, items) - self.z0)))

    def calibrate(self, pairs: list[tuple[int, np.ndarray]], target_mean: float) -> float:
        """Choose z0 so the mean click prob over the given candidate sets equals target_mean."""
        zs = np.concatenate([self.truth.z(u, it) for u, it in pairs])
        lo, hi = -5.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            m = (1 / (1 + np.exp(-self.beta * (zs - mid)))).mean()
            lo, hi = (mid, hi) if m > target_mean else (lo, mid)
        self.z0 = (lo + hi) / 2
        return self.z0

    def expected_clicks(self, u: int, slate: np.ndarray) -> float:
        return float((self.exam_probs(len(slate)) * self.click_probs(u, slate)).sum())

    def optimal_expected_clicks(self, u: int, candidates: np.ndarray, k: int) -> float:
        p = np.sort(self.click_probs(u, candidates))[::-1][:k]
        return float((self.exam_probs(k) * p).sum())   # best items in best positions

    def respond(self, u: int, slate: np.ndarray) -> SlateFeedback:
        k = len(slate)
        ex = self.rng.random(k) < self.exam_probs(k)
        p = self.click_probs(u, slate)
        cl = ex & (self.rng.random(k) < p)
        if self.flip > 0:
            flip = self.rng.random(k) < self.flip
            cl = np.where(flip, ~cl, cl)
        sk = ex & ~cl
        ratings = np.zeros(k, dtype=int)
        z = self.truth.z(u, slate)
        rate = cl & (self.rng.random(k) < self.p_rate)
        ratings[rate] = np.clip(np.rint(3.5 + 1.2 * (z[rate] - self.z0) + self.rng.normal(0, 0.5, rate.sum())), 1, 5)
        return SlateFeedback(ex, cl, sk, ratings)
