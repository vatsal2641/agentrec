"""Generate the synthetic ML-1M-format dataset into data/raw/synthetic/."""
from _common import ROOT

from agentrec.data.synthetic import SyntheticConfig, generate

if __name__ == "__main__":
    stats = generate(SyntheticConfig(), ROOT / "data" / "raw" / "synthetic")
    print("wrote synthetic data:", stats)
