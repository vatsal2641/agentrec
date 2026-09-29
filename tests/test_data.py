import numpy as np
import pandas as pd
import pytest

from helpers import tiny_dataset, tiny_raw_dir
from agentrec.data.dataset import build_dataset
from agentrec.data.movielens import load_ml1m


def test_temporal_split_has_no_time_overlap():
    ds = tiny_dataset()
    assert ds.train.df["timestamp"].max() < ds.t_val <= ds.val.df["timestamp"].min()
    assert ds.val.df["timestamp"].max() < ds.t_test <= ds.test.df["timestamp"].min()


def test_every_rating_lands_in_exactly_one_period():
    ds = tiny_dataset()
    raw = load_ml1m(tiny_raw_dir())
    assert len(ds.train.df) + len(ds.val.df) + len(ds.test.df) == len(raw.ratings)


def test_positives_respect_threshold():
    ds = tiny_dataset()
    assert (ds.train.positives["rating"] >= ds.pos_threshold).all()
    assert all(len(v) > 0 for v in ds.test.targets().values())


def test_ids_are_contiguous_and_matrix_shape():
    ds = tiny_dataset()
    for p in (ds.train, ds.val, ds.test):
        assert p.df["u"].between(0, ds.n_users - 1).all()
        assert p.df["i"].between(0, ds.n_items - 1).all()
    m = ds.matrix(ds.train)
    assert m.shape == (ds.n_users, ds.n_items)
    assert set(np.unique(m.data)) <= {1.0}


def test_fit_period_for_test_includes_val():
    ds = tiny_dataset()
    assert len(ds.fit_period("test").df) == len(ds.train.df) + len(ds.val.df)
    with pytest.raises(ValueError):
        ds.fit_period("train")


def test_loader_rejects_bad_ratings(tmp_path=None):
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    src = tiny_raw_dir()
    for f in ("movies.dat", "users.dat"):
        (d / f).write_bytes((src / f).read_bytes())
    (d / "ratings.dat").write_text("1::1::7::956700000\n")
    with pytest.raises(ValueError):
        load_ml1m(d)


def test_bad_quantiles_rejected():
    with pytest.raises(ValueError):
        build_dataset(load_ml1m(tiny_raw_dir()), val_quantile=0.9, test_quantile=0.8)
