"""Per-user recommendation state — the thing feedback actually changes.

This is "user preference memory" in the resume sense. It is deliberately a
small structured object, not a vector database:
  history      : liked/clicked items in time order -> fed to the two-tower user
                 tower, so the user embedding changes immediately after a click
  disliked     : items rated <= 2 -> never recommended again
  skip_counts  : repeated skips suppress an item (stops "recommending the same
                 thing forever", a classic feedback-loop annoyance)
  impressions  : shown-but-ignored counts (fatigue)
  genre_pref   : exponential moving average of liked(+)/skipped(-) item genres;
                 a cheap, interpretable preference signal (also used by the agent
                 to explain recommendations)
  stated       : explicit preferences from conversation ("no horror") — hard filters
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from agentrec.feedback.events import Event, EventType


@dataclass
class UserState:
    user_id: int
    n_genres: int
    history: list[int] = field(default_factory=list)
    disliked: set[int] = field(default_factory=set)
    skip_counts: dict[int, int] = field(default_factory=dict)
    impressions: dict[int, int] = field(default_factory=dict)
    genre_pref: np.ndarray | None = None
    stated: dict[str, list[str]] = field(default_factory=dict)   # e.g. {"exclude_genres": ["Horror"]}
    n_events: int = 0

    def __post_init__(self) -> None:
        if self.genre_pref is None:
            self.genre_pref = np.zeros(self.n_genres)

    # ------------------------------------------------------------------ update
    def apply(self, e: Event, item_genres: np.ndarray, lr: float = 0.2) -> None:
        if e.user_id != self.user_id:
            raise ValueError(f"event for user {e.user_id} applied to state of {self.user_id}")
        i = int(e.item_id)
        g = item_genres[i] / max(item_genres[i].sum(), 1.0)
        self.n_events += 1
        t = e.event_type
        if t == EventType.IMPRESSION:
            self.impressions[i] = self.impressions.get(i, 0) + 1
        elif t in (EventType.CLICK, EventType.LIKE) or (t == EventType.RATING and e.value is not None and e.value >= 4):
            if i not in self.history:
                self.history.append(i)
            self.disliked.discard(i)
            self.genre_pref = (1 - lr) * self.genre_pref + lr * g
        elif t == EventType.RATING and e.value is not None and e.value <= 2:
            if i in self.history:
                self.history.remove(i)
            self.disliked.add(i)
            self.genre_pref = (1 - lr) * self.genre_pref - lr * g
        elif t == EventType.SKIP:
            self.skip_counts[i] = self.skip_counts.get(i, 0) + 1
            self.genre_pref = (1 - lr / 4) * self.genre_pref - (lr / 4) * g

    def blocked_items(self, skip_limit: int = 2, impression_limit: int = 3) -> set[int]:
        """Items that must not be shown again."""
        out = set(self.disliked)
        out |= {i for i, c in self.skip_counts.items() if c >= skip_limit}
        out |= {i for i, c in self.impressions.items() if c >= impression_limit and i not in self.history}
        return out

    def allowed_mask(self, item_genres: np.ndarray, genre_names: list[str]) -> np.ndarray:
        """Boolean mask over items that satisfy the user's STATED hard constraints
        (applied in feed mode too, not only in agent requests)."""
        ok = np.ones(len(item_genres), bool)
        for g in self.stated.get("exclude_genres", []):
            if g in genre_names:
                ok &= item_genres[:, genre_names.index(g)] == 0
        return ok

    # --------------------------------------------------------------- persistence
    def to_json(self) -> str:
        return json.dumps({
            "user_id": self.user_id, "n_genres": self.n_genres, "history": self.history,
            "disliked": sorted(self.disliked), "skip_counts": {str(k): v for k, v in self.skip_counts.items()},
            "impressions": {str(k): v for k, v in self.impressions.items()},
            "genre_pref": self.genre_pref.tolist(), "stated": self.stated, "n_events": self.n_events,
        })

    @classmethod
    def from_json(cls, s: str) -> "UserState":
        d = json.loads(s)
        return cls(
            user_id=d["user_id"], n_genres=d["n_genres"], history=list(d["history"]),
            disliked=set(d["disliked"]), skip_counts={int(k): v for k, v in d["skip_counts"].items()},
            impressions={int(k): v for k, v in d["impressions"].items()},
            genre_pref=np.array(d["genre_pref"]), stated=d.get("stated", {}), n_events=d["n_events"],
        )
