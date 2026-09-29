"""Turn raw logs into model-ready data with a leakage-free temporal split.

KEY DECISIONS (see DESIGN_DECISIONS.md for the long form)
1. Implicit feedback: a rating >= `pos_threshold` (default 4) is a positive.
   Ratings below it are "seen but not liked": excluded from positives, but still
   masked at evaluation time (you would not re-recommend a movie already watched).
2. GLOBAL temporal split: pick two timestamps t_val < t_test.
      train = [.., t_val)   val = [t_val, t_test)   test = [t_test, ..]
   Every model only ever sees the past relative to what it predicts.
   A random split would put a user's 2003 ratings in train while testing on
   their 2000 ratings -> future leaks into the past -> inflated metrics.
3. Two phases:
      tuning : fit on train,        evaluate on val
      final  : fit on train + val,  evaluate on test   (hyper-params frozen)
4. IDs: raw IDs -> contiguous 0..N-1 over the FULL catalog (users.dat, movies.dat)
   so items that only appear in test still have metadata (content model can score
   them; collaborative models cannot -> the cold-item problem, made visible).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from agentrec.data.movielens import GENRES, RawData


@dataclass
class Period:
    """Interactions for one time window, in index space."""

    df: pd.DataFrame  # columns: u, i, rating, timestamp (sorted by timestamp)
    pos_threshold: int

    @property
    def positives(self) -> pd.DataFrame:
        return self.df[self.df["rating"] >= self.pos_threshold]

    def targets(self) -> dict[int, set[int]]:
        """user -> set of positive items in this window (the ground truth)."""
        p = self.positives
        return {int(u): set(map(int, g)) for u, g in p.groupby("u")["i"]}


@dataclass
class Dataset:
    n_users: int
    n_items: int
    user_raw: np.ndarray            # idx -> raw user id
    item_raw: np.ndarray            # idx -> raw item id
    titles: list[str]
    item_genres: np.ndarray         # (n_items, n_genres) multi-hot float32
    item_year: np.ndarray           # (n_items,) int, -1 unknown
    genre_names: list[str]
    train: Period
    val: Period
    test: Period
    t_val: int
    t_test: int
    pos_threshold: int
    meta: dict = field(default_factory=dict)

    # ---- views used by models -------------------------------------------
    def fit_period(self, phase: str) -> Period:
        """History available to a model in `phase` ('val' or 'test')."""
        if phase == "val":
            return self.train
        if phase == "test":
            return Period(pd.concat([self.train.df, self.val.df], ignore_index=True), self.pos_threshold)
        raise ValueError(f"phase must be 'val' or 'test', got {phase!r}")

    def eval_period(self, phase: str) -> Period:
        return {"val": self.val, "test": self.test}[phase]

    def matrix(self, period: Period, positive_only: bool = True) -> sp.csr_matrix:
        """Binary user x item CSR matrix. positive_only=False -> everything seen."""
        df = period.positives if positive_only else period.df
        m = sp.csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df["u"].to_numpy(), df["i"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )
        m.data[:] = 1.0  # duplicates -> 1
        return m

    def title_to_index(self) -> dict[str, int]:
        return {t.lower(): i for i, t in enumerate(self.titles)}


def build_dataset(
    raw: RawData,
    pos_threshold: int = 4,
    val_quantile: float = 0.8,
    test_quantile: float = 0.9,
) -> Dataset:
    if not 0 < val_quantile < test_quantile < 1:
        raise ValueError("need 0 < val_quantile < test_quantile < 1")

    users = np.sort(raw.users["user_id"].unique())
    items = np.sort(raw.movies["item_id"].unique())
    u_map = pd.Series(np.arange(len(users)), index=users)
    i_map = pd.Series(np.arange(len(items)), index=items)

    df = pd.DataFrame({
        "u": u_map.loc[raw.ratings["user_id"]].to_numpy(),
        "i": i_map.loc[raw.ratings["item_id"]].to_numpy(),
        "rating": raw.ratings["rating"].to_numpy(),
        "timestamp": raw.ratings["timestamp"].to_numpy(),
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)

    t_val = int(df["timestamp"].quantile(val_quantile))
    t_test = int(df["timestamp"].quantile(test_quantile))
    tr = df[df["timestamp"] < t_val]
    va = df[(df["timestamp"] >= t_val) & (df["timestamp"] < t_test)]
    te = df[df["timestamp"] >= t_test]

    movies = raw.movies.set_index("item_id").loc[items]
    g_idx = {g: k for k, g in enumerate(GENRES)}
    genres = np.zeros((len(items), len(GENRES)), dtype=np.float32)
    for row, gs in enumerate(movies["genres"]):
        for g in gs:
            if g in g_idx:
                genres[row, g_idx[g]] = 1.0

    return Dataset(
        n_users=len(users), n_items=len(items),
        user_raw=users, item_raw=items,
        titles=list(movies["title"].astype(str)),
        item_genres=genres, item_year=movies["year"].to_numpy(),
        genre_names=list(GENRES),
        train=Period(tr.reset_index(drop=True), pos_threshold),
        val=Period(va.reset_index(drop=True), pos_threshold),
        test=Period(te.reset_index(drop=True), pos_threshold),
        t_val=t_val, t_test=t_test, pos_threshold=pos_threshold,
    )


def describe(ds: Dataset) -> dict:
    """Summary statistics; printed by scripts/prepare_data.py."""
    out = {"n_users": ds.n_users, "n_items": ds.n_items}
    all_df = pd.concat([ds.train.df, ds.val.df, ds.test.df])
    out["n_ratings"] = len(all_df)
    out["density"] = len(all_df) / (ds.n_users * ds.n_items)
    out["frac_positive"] = float((all_df["rating"] >= ds.pos_threshold).mean())
    for name, p in [("train", ds.train), ("val", ds.val), ("test", ds.test)]:
        out[f"{name}_ratings"] = len(p.df)
        out[f"{name}_users"] = int(p.df["u"].nunique())
    train_users = set(ds.train.df["u"])
    tu = set(ds.test.df["u"])
    fit_users = train_users | set(ds.val.df["u"])
    out["test_users_cold"] = len(tu - fit_users)
    fit_items = set(ds.train.df["i"]) | set(ds.val.df["i"])
    te_pos = ds.test.positives
    out["test_positive_pairs_on_cold_items"] = int((~te_pos["i"].isin(fit_items)).sum())
    return out
