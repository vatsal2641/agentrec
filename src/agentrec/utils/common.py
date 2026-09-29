"""Small shared helpers: seeding, logging, timing, config loading."""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml


def set_seed(seed: int) -> np.random.Generator:
    """Seed python + numpy global RNGs and return a fresh Generator.

    We pass Generators explicitly everywhere (never rely on global state inside
    library code) so that two components cannot silently share a random stream.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def get_logger(name: str = "agentrec", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Usage: `with timer() as t: ...; t["s"]` -> elapsed seconds."""
    out: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["s"] = time.perf_counter() - t0


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} is not a mapping")
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    p.write_text(json.dumps(obj, indent=2, default=default))


@dataclass
class LatencyStats:
    """Collects per-call latencies (seconds) and reports percentiles in ms."""

    samples: list[float] = field(default_factory=list)

    def add(self, s: float) -> None:
        self.samples.append(s)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"n": 0}
        a = np.asarray(self.samples) * 1000.0
        return {
            "n": int(a.size),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()),
        }
