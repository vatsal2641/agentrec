"""Exercise 03: global temporal split + leakage detector. Reference: data/dataset.py::build_dataset."""
import pandas as pd


def temporal_split(df: pd.DataFrame, val_q: float = 0.8, test_q: float = 0.9):
    """Return (train, val, test) by timestamp quantiles. Every row in exactly one part."""
    raise NotImplementedError


def leaks(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """True if ANY training interaction happened at or after the earliest test interaction."""
    raise NotImplementedError


if __name__ == "__main__":
    df = pd.DataFrame({"u": [1, 1, 2, 2, 3, 3, 1, 2, 3, 1], "i": range(10), "timestamp": range(100, 110)})
    tr, va, te = temporal_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["timestamp"].max() < va["timestamp"].min() <= va["timestamp"].max() < te["timestamp"].min()
    assert not leaks(tr, te)
    rnd = df.sample(frac=1.0, random_state=0)
    assert leaks(rnd.iloc[:8], rnd.iloc[8:]), "a random split of this data should leak"
    print("all tests passed. Question: can a global time split still leak through FEATURES? give an example.")
