"""Two-tower retrieval model with hand-written backprop (numpy only).

ARCHITECTURE
  user tower  (input = the user's last L liked items, NOT a user-ID embedding)
      hbar_u = mean_{j in hist(u)} H[j]                         (d)
      z_u    = hbar_u + relu(hbar_u W1 + b1) W2                 (residual MLP, d)
  item tower
      v_i    = E[i] + genres_i Wg                               (ID embedding + metadata)
  score      s(u, i) = cos(z_u, v_i) / tau

  Why history-based user tower? (1) a user's embedding can be recomputed the
  moment they click something -> instant feedback adaptation without retraining;
  (2) new users with a few clicks get a meaningful embedding (partial cold start);
  (3) no per-user parameters -> scales with #items, not #users.
  Why genres in the item tower? New items get a non-random vector (cold items).

LOSS: in-batch sampled softmax with logQ correction
  batch of B (history, positive item) pairs. Every other positive in the batch is
  used as a negative for this row:
      S_bc = cos(z_b, v_c) / tau - log q_c         (c = columns = batch items)
      L    = -(1/B) sum_b log softmax(S_b)_b
  In-batch negatives are sampled ∝ popularity, so popular items are over-used as
  negatives; subtracting log q_c (their sampling probability) corrects that bias
  (Yi et al., RecSys 2019). Duplicate positives in a batch are masked out
  (otherwise an item would be its own negative).

BACKPROP (all derived by hand; verified by finite differences in tests/)
  P = softmax(S)                  dS   = (P - Y) / B
  dzn = dS vn / tau               dvn  = dS^T zn / tau
  x -> x/||x||  backward:         dx   = (dxn - xn (xn . dxn)) / ||x||
  z = hbar + r W2, r = relu(a), a = hbar W1 + b1
      dW2 = r^T dz ; dr = dz W2^T ; da = dr * [a>0] ; dW1 = hbar^T da ; db1 = sum da
      dhbar = dz + da W1^T
  hbar = mean H[hist]  ->  dH[j] += dhbar / n_hist   (scatter-add)
  v = E[i] + g Wg      ->  dE[i] += dv ; dWg = g^T dv
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentrec.data.dataset import Dataset, Period
from agentrec.models.base import Recommender


@dataclass
class Batch:
    hist: np.ndarray      # (B, L) item indices, -1 = padding
    pos: np.ndarray       # (B,) positive item


def _l2n(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n, n


class TwoTower(Recommender):
    name = "two_tower"
    PARAMS = ("H", "W1", "b1", "W2", "E", "Wg")

    def __init__(self, dim: int = 64, hidden: int | None = None, epochs: int = 20, lr: float = 0.05,
                 batch_size: int = 1024, temperature: float = 0.1, reg: float = 1e-4,
                 history_len: int = 50, seed: int = 0, verbose: bool = False) -> None:
        self.dim, self.hidden = dim, hidden or dim
        self.epochs, self.lr, self.bs, self.tau = epochs, lr, batch_size, temperature
        self.reg, self.L, self.seed, self.verbose = reg, history_len, seed, verbose
        self.history: list[float] = []

    # ------------------------------------------------------------------ init
    def init_params(self, n_items: int, n_genres: int, rng: np.random.Generator) -> None:
        d, h = self.dim, self.hidden
        self.p = {
            "H": rng.normal(0, 0.1, (n_items, d)),
            "W1": rng.normal(0, 1 / np.sqrt(d), (d, h)),
            "b1": np.zeros(h),
            "W2": rng.normal(0, 0.01, (h, d)),
            "E": rng.normal(0, 0.1, (n_items, d)),
            "Wg": rng.normal(0, 0.1, (n_genres, d)),
        }

    # --------------------------------------------------------------- towers
    def _user_forward(self, hist: np.ndarray) -> tuple[np.ndarray, dict]:
        p = self.p
        mask = (hist >= 0).astype(np.float64)                        # (B, L)
        cnt = np.maximum(mask.sum(1, keepdims=True), 1.0)
        hbar = (p["H"][np.maximum(hist, 0)] * mask[..., None]).sum(1) / cnt
        a = hbar @ p["W1"] + p["b1"]
        r = np.maximum(a, 0.0)
        z = hbar + r @ p["W2"]
        return z, {"hist": hist, "mask": mask, "cnt": cnt, "hbar": hbar, "a": a, "r": r}

    def _item_forward(self, items: np.ndarray) -> np.ndarray:
        return self.p["E"][items] + self.G[items] @ self.p["Wg"]

    # ------------------------------------------------------ loss + gradients
    def loss_and_grads(self, b: Batch) -> tuple[float, dict[str, np.ndarray]]:
        p, tau = self.p, self.tau
        B = len(b.pos)
        z, c = self._user_forward(b.hist)
        v = self._item_forward(b.pos)
        zn, zno = _l2n(z)
        vn, vno = _l2n(v)
        S = zn @ vn.T / tau - self.logq[b.pos][None, :]
        dup = (b.pos[:, None] == b.pos[None, :]) & ~np.eye(B, dtype=bool)
        S = np.where(dup, -1e9, S)
        S = S - S.max(1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(1, keepdims=True)
        loss = float(-np.log(P[np.arange(B), np.arange(B)] + 1e-12).mean())

        dS = P.copy()
        dS[np.arange(B), np.arange(B)] -= 1.0
        dS /= B
        dzn = dS @ vn / tau
        dvn = dS.T @ zn / tau
        dz = (dzn - zn * (zn * dzn).sum(1, keepdims=True)) / zno
        dv = (dvn - vn * (vn * dvn).sum(1, keepdims=True)) / vno

        g = {k: np.zeros_like(val) for k, val in p.items()}
        g["W2"] = c["r"].T @ dz
        da = (dz @ p["W2"].T) * (c["a"] > 0)
        g["W1"] = c["hbar"].T @ da
        g["b1"] = da.sum(0)
        dhbar = dz + da @ p["W1"].T
        w = (c["mask"] / c["cnt"])[..., None] * dhbar[:, None, :]       # (B, L, d), 0 on padding
        np.add.at(g["H"], np.maximum(c["hist"], 0).ravel(), w.reshape(-1, self.dim))
        np.add.at(g["E"], b.pos, dv)
        g["Wg"] = self.G[b.pos].T @ dv

        # L2 on the rows touched in this batch + dense weights
        if self.reg > 0:
            for k in ("W1", "W2", "Wg"):
                g[k] += self.reg * p[k]
            touched_h = np.unique(c["hist"][c["hist"] >= 0])
            g["H"][touched_h] += self.reg * p["H"][touched_h]
            g["E"][np.unique(b.pos)] += self.reg * p["E"][np.unique(b.pos)]
        return loss, g

    # --------------------------------------------------------------- training
    def _make_samples(self, period: Period) -> None:
        """Every (user, k) with k >= 1 is a sample: target = k-th liked item (time order),
        history = the up-to-L items liked strictly BEFORE it (no future leakage)."""
        pos = period.positives.sort_values(["u", "timestamp"], kind="stable")
        self.user_hist: dict[int, np.ndarray] = {int(u): g.to_numpy() for u, g in pos.groupby("u")["i"]}
        self.flat = pos["i"].to_numpy()                                  # items grouped by user
        users, counts = np.unique(pos["u"].to_numpy(), return_counts=True)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        k = np.concatenate([np.arange(1, c) for c in counts]) if len(counts) else np.array([], int)
        self.sample_off = np.repeat(offsets, counts - 1)                   # start of this user's block
        self.sample_pos = self.sample_off + k                              # flat index of the target

    def _batch(self, idx: np.ndarray) -> Batch:
        tgt = self.sample_pos[idx]
        win = tgt[:, None] - self.L + np.arange(self.L)[None, :]          # (B, L) flat indices
        valid = win >= self.sample_off[idx][:, None]
        hist = np.where(valid, self.flat[np.maximum(win, 0)], -1)
        return Batch(hist.astype(np.int64), self.flat[tgt].astype(np.int64))

    def _fit(self, ds: Dataset, period: Period) -> None:
        rng = np.random.default_rng(self.seed)
        g = ds.item_genres.astype(np.float64)
        self.G = g / np.maximum(g.sum(1, keepdims=True), 1.0)
        self.init_params(ds.n_items, g.shape[1], rng)
        self._make_samples(period)
        # sampling probability of each item as an in-batch negative ~ its frequency as a target
        freq = np.bincount(self.flat[self.sample_pos], minlength=ds.n_items) + 1.0
        self.logq = np.log(freq / freq.sum())
        # Adam state
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        s2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        n = len(self.sample_pos)
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            tot, nb = 0.0, 0
            for s in range(0, n - self.bs + 1, self.bs):
                loss, grads = self.loss_and_grads(self._batch(perm[s:s + self.bs]))
                t += 1
                for k in self.p:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    s2[k] = b2 * s2[k] + (1 - b2) * grads[k] ** 2
                    self.p[k] -= self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(s2[k] / (1 - b2 ** t)) + eps)
                tot += loss
                nb += 1
            self.history.append(tot / max(nb, 1))
            if self.verbose:
                print(f"[two_tower] epoch {ep + 1}/{self.epochs} loss={tot / max(nb, 1):.4f}")
        self._cache_items()

    def _cache_items(self) -> None:
        self.V, _ = _l2n(self._item_forward(np.arange(self.n_items)))

    # --------------------------------------------------------------- inference
    def embed_history(self, hist_items: list[int] | np.ndarray) -> np.ndarray:
        """User embedding from an arbitrary item list (used for online updates)."""
        h = np.full((1, self.L), -1, dtype=np.int64)
        items = np.asarray(hist_items, dtype=np.int64)[-self.L:]
        h[0, :len(items)] = items
        z, _ = self._user_forward(h)
        return _l2n(z)[0][0]

    def user_embeddings(self, users: np.ndarray) -> np.ndarray:
        out = np.zeros((len(users), self.dim))
        for r, u in enumerate(users):
            items = self.user_hist.get(int(u))
            if items is not None and len(items):
                out[r] = self.embed_history(items)
        return out

    def item_embeddings(self) -> np.ndarray:
        return self.V

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self.user_embeddings(users) @ self.V.T
