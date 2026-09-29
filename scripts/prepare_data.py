"""Load raw data, apply implicit conversion + temporal split, cache, print stats.

usage: python scripts/prepare_data.py configs/ml1m.yaml
"""
import json
import sys

from _common import ROOT, load, results_dir

from agentrec.data.dataset import describe
from agentrec.utils.common import save_json

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml"
    cache = ROOT / "data" / "processed"
    for p in cache.glob("*.pkl"):
        if p.stem in cfg_path:
            p.unlink()  # always rebuild when this script is run explicitly
    cfg, ds = load(cfg_path)
    stats = describe(ds)
    print(json.dumps(stats, indent=2))
    save_json(stats, results_dir(cfg) / "data_stats.json")
