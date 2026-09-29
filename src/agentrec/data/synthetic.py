"""Synthetic MovieLens-1M-format data generator.

WHY THIS EXISTS
    The build environment had no internet, so real MovieLens could not be
    downloaded. Rather than fake results, we generate a dataset whose *mechanics*
    resemble MovieLens and write it in the exact ML-1M file format. All code is
    tested on it; real results come from running the same scripts on ML-1M.
    Anything measured on this data must be labelled "synthetic".

GENERATIVE STORY (so you can reason about what models *should* find)
    * 18 genres, each with a latent centroid G_g in R^d.
    * item i: 1–3 genres; latent v_i = mean(G_genres) + noise; popularity
      bias b_i with a heavy (power-law-like) tail; an entry time (some items are
      released *during* the log, creating cold items in later periods).
    * user u: 1–3 favourite genres; latent w_u = weighted G + noise.
    * affinity a_ui = w_u . v_i / sqrt(d)
    * Exposure (which items a user rates) ∝ exp(alpha*b_i + beta*a_ui):
      popular items and liked items are more likely to be *seen*
      -> missing-not-at-random (MNAR), like real logs.
    * rating = clip(round(mu + gamma*z(a_ui) + noise), 1, 5).
    * timestamps: each user arrives at a random time, rates in bursts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentrec.data.movielens import GENRES

# rough genre frequency prior (Drama/Comedy dominate, as in MovieLens)
_GENRE_W = np.array([8, 5, 2, 3, 12, 4, 2, 15, 2, 1, 4, 2, 2, 6, 4, 6, 2, 1], dtype=float)

_ADJ = ("Silent Crimson Hidden Broken Golden Last Lost Midnight Northern Burning Quiet Iron "
        "Paper Glass Wild Distant Electric Frozen Hollow Velvet Savage Gentle Neon Bitter "
        "Secret Endless Fallen Rising Stolen Twisted").split()
_NOUN = ("River Empire Garden Signal Harbor Mirror Frontier Orchard Machine Kingdom Voyage "
         "Station Promise Shadow Letter Horizon Engine Island Summer Winter Circus Canyon "
         "Detective Symphony Protocol Crown Lantern Ghost Planet Harvest").split()


@dataclass
class SyntheticConfig:
    n_users: int = 3000
    n_items: int = 2000
    dim: int = 12
    alpha_pop: float = 1.0       # exposure weight on popularity
    beta_aff: float = 1.4        # exposure weight on affinity
    frac_new_items: float = 0.15 # items released during the log window
    mean_extra_ratings: float = 3.6  # log-mean of ratings beyond the 20 minimum
    seed: int = 7
    t0: int = 956_700_000        # ~ April 2000 (MovieLens-1M starts here)
    span_days: int = 900


def _titles(n: int, years: np.ndarray, rng: np.random.Generator) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i in range(n):
        base = f"{_ADJ[rng.integers(len(_ADJ))]} {_NOUN[rng.integers(len(_NOUN))]}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        name = base if k == 0 else f"{base} {k + 1}"
        out.append(f"{name} ({years[i]})")
    return out


def generate(cfg: SyntheticConfig, out_dir: str | Path) -> dict[str, int]:
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_g, d = len(GENRES), cfg.dim

    # ---- genres & latents -------------------------------------------------
    G = rng.normal(size=(n_g, d))
    gp = _GENRE_W / _GENRE_W.sum()
    item_genres: list[list[int]] = []
    V = np.zeros((cfg.n_items, d))
    for i in range(cfg.n_items):
        k = rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15])
        gs = list(rng.choice(n_g, size=k, replace=False, p=gp))
        item_genres.append(gs)
        V[i] = G[gs].mean(0) + 0.6 * rng.normal(size=d)
    W = np.zeros((cfg.n_users, d))
    for u in range(cfg.n_users):
        k = rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])
        gs = rng.choice(n_g, size=k, replace=False, p=gp)
        w = rng.dirichlet(np.ones(k))
        W[u] = (w[:, None] * G[gs]).sum(0) * 1.5 + 0.7 * rng.normal(size=d)

    # popularity: heavy tail (log-normal), standardised
    b = rng.lognormal(0, 1.0, size=cfg.n_items)
    b = (np.log(b) - np.log(b).mean()) / np.log(b).std()

    years = np.clip((2000 - rng.gamma(2.0, 7.0, size=cfg.n_items)).astype(int), 1925, 2000)

    # item entry times: most items exist from t0; some are released later
    span = cfg.span_days * 86400
    entry = np.full(cfg.n_items, cfg.t0, dtype=np.int64)
    new = rng.random(cfg.n_items) < cfg.frac_new_items
    entry[new] = cfg.t0 + (rng.random(new.sum()) * 0.9 * span).astype(np.int64)
    years[new] = 2000

    A = W @ V.T / np.sqrt(d)                      # true affinity (n_users, n_items)
    A_z = (A - A.mean()) / A.std()

    # ---- interactions -----------------------------------------------------
    rows = []
    for u in range(cfg.n_users):
        n_u = int(min(20 + rng.lognormal(cfg.mean_extra_ratings, 1.0), cfg.n_items * 0.5))
        # arrival skewed early (like ML-1M, most activity in the first months)
        start = cfg.t0 + int(rng.beta(0.9, 1.6) * 0.95 * span)
        gaps = np.where(rng.random(n_u) < 0.8,
                        rng.exponential(300, n_u),          # within-session: minutes
                        rng.exponential(20 * 86400, n_u))   # between sessions: weeks
        times = start + np.cumsum(gaps).astype(np.int64)
        times = times[times <= cfg.t0 + span]  # drop events past the log end (no pile-up)
        logits = cfg.alpha_pop * b + cfg.beta_aff * A_z[u]
        keys = logits + rng.gumbel(size=cfg.n_items)   # Gumbel-max sampling w/o replacement
        used = np.zeros(cfg.n_items, bool)
        for t in times:
            avail = (entry <= t) & ~used
            if not avail.any():
                break
            i = int(np.argmax(np.where(avail, keys, -np.inf)))
            used[i] = True
            r = 2.45 + 0.95 * A_z[u, i] + 0.15 * b[i] + rng.normal(0, 0.75)
            rows.append((u + 1, i + 1, int(np.clip(np.rint(r), 1, 5)), int(t)))

    # ---- write ML-1M format ----------------------------------------------
    titles = _titles(cfg.n_items, years, rng)
    with open(out / "ratings.dat", "w", encoding="latin-1") as f:
        for u, i, r, t in rows:
            f.write(f"{u}::{i}::{r}::{t}\n")
    with open(out / "movies.dat", "w", encoding="latin-1") as f:
        for i in range(cfg.n_items):
            f.write(f"{i + 1}::{titles[i]}::{'|'.join(GENRES[g] for g in item_genres[i])}\n")
    ages = [1, 18, 25, 35, 45, 50, 56]
    with open(out / "users.dat", "w", encoding="latin-1") as f:
        for u in range(cfg.n_users):
            f.write(f"{u + 1}::{rng.choice(['M', 'F'])}::{rng.choice(ages)}::"
                    f"{rng.integers(21)}::{rng.integers(10000, 99999)}\n")
    np.savez_compressed(out / "truth.npz", W=W, V=V, b=b, entry=entry)
    (out / "SYNTHETIC_README.txt").write_text(
        "SYNTHETIC DATA generated by agentrec.data.synthetic. NOT MovieLens.\n"
        f"config: {cfg}\n")
    return {"users": cfg.n_users, "items": cfg.n_items, "ratings": len(rows)}
