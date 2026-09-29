"""Load MovieLens-1M style `.dat` files.

File format (from the dataset README, `::` separated, latin-1 encoded):
    ratings.dat : UserID::MovieID::Rating::Timestamp
    movies.dat  : MovieID::Title (Year)::Genre1|Genre2|...
    users.dat   : UserID::Gender::Age::Occupation::Zip-code

The synthetic generator (`agentrec.data.synthetic`) writes the *same* format, so
every downstream component is exercised identically on synthetic and real data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class RawData:
    ratings: pd.DataFrame  # user_id, item_id, rating, timestamp (raw IDs)
    movies: pd.DataFrame   # item_id, title, genres (list[str]), year (int | -1)
    users: pd.DataFrame    # user_id, gender, age, occupation, zip


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for how to obtain MovieLens-1M "
            f"or run scripts/make_synthetic_data.py."
        )
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


def load_ml1m(root: str | Path) -> RawData:
    root = Path(root)
    ratings = _read(root / "ratings.dat", ["user_id", "item_id", "rating", "timestamp"])
    movies = _read(root / "movies.dat", ["item_id", "title", "genres"])
    users = _read(root / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])

    movies["genres"] = movies["genres"].fillna("").map(lambda s: [g for g in s.split("|") if g])
    movies["year"] = movies["title"].map(
        lambda t: int(m.group(1)) if (m := _YEAR_RE.search(str(t))) else -1
    )

    # Basic validation: fail loudly rather than train on garbage.
    if ratings[["user_id", "item_id", "timestamp"]].isna().any().any():
        raise ValueError("ratings.dat contains missing values")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("ratings outside 1..5")
    unknown_items = set(ratings["item_id"]) - set(movies["item_id"])
    if unknown_items:
        raise ValueError(f"{len(unknown_items)} rated items missing from movies.dat")
    return RawData(ratings=ratings, movies=movies, users=users)
