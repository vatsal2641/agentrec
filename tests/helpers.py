"""Shared tiny fixtures (plain functions + caching so tests stay fast and pytest-agnostic)."""
from __future__ import annotations

import sys
import tempfile
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentrec.data.dataset import build_dataset  # noqa: E402
from agentrec.data.movielens import load_ml1m  # noqa: E402
from agentrec.data.synthetic import SyntheticConfig, generate  # noqa: E402

TINY_CFG = {
    "name": "tiny",
    "models": {
        "itemknn": {"k_neighbors": 20, "shrink": 5.0},
        "two_tower": {"dim": 8, "epochs": 3, "lr": 0.05, "batch_size": 128, "temperature": 0.2,
                      "reg": 1e-4, "history_len": 10},
    },
}


@lru_cache(maxsize=1)
def tiny_raw_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="agentrec_tiny_"))
    generate(SyntheticConfig(n_users=200, n_items=150, dim=6, seed=1, mean_extra_ratings=2.5), d)
    return d


@lru_cache(maxsize=1)
def tiny_dataset():
    return build_dataset(load_ml1m(tiny_raw_dir()))


@lru_cache(maxsize=1)
def tiny_stage1():
    from agentrec.pipeline import fit_stage1
    return fit_stage1(tiny_dataset(), "test", TINY_CFG)
