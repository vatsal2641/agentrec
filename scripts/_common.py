"""Shared helpers for scripts: load config + dataset (cached)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrec.data.dataset import Dataset, build_dataset  # noqa: E402
from agentrec.data.movielens import load_ml1m  # noqa: E402
from agentrec.utils.common import load_config  # noqa: E402


def load(config_path: str) -> tuple[dict, Dataset]:
    cfg = load_config(ROOT / config_path if not Path(config_path).is_absolute() else config_path)
    cache = ROOT / "data" / "processed" / f"{cfg['name']}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return cfg, pickle.load(f)
    d = cfg["data"]
    ds = build_dataset(load_ml1m(ROOT / d["raw_dir"]), d["pos_threshold"], d["val_quantile"], d["test_quantile"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(ds, f)
    return cfg, ds


def results_dir(cfg: dict) -> Path:
    p = ROOT / "results" / cfg["name"]
    p.mkdir(parents=True, exist_ok=True)
    return p
