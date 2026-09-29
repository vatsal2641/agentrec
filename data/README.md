# Data

## MovieLens-1M (the real dataset)
1. Download `ml-1m.zip` from https://grouplens.org/datasets/movielens/1m/. Read its license: it is for
   research use, and redistribution is not allowed. **Do not commit the data to GitHub.**
2. Unzip it so that you have `data/raw/ml-1m/ratings.dat`, `movies.dat` and `users.dat`.
3. Run `python scripts/prepare_data.py configs/ml1m.yaml`. This prints the split statistics and
   writes `results/ml1m/data_stats.json`.

## Synthetic data (used during development)
Run `python scripts/make_synthetic_data.py`. It writes to `data/raw/synthetic/` in the ML-1M format,
plus `truth.npz` (the true latent factors, used only by the bandit simulator). See the docstring of
`src/agentrec/data/synthetic.py` for how it's generated. **Results on this data are not MovieLens
results.**

## Processed cache
`data/processed/` holds pickled `Dataset` objects and fitted stage-1 models and rankers. Delete it to
force a rebuild. It is git-ignored.
