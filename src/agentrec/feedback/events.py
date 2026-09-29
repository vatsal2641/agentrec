"""Interaction event schema and reward mapping.

Every user action becomes one immutable Event row. Rewards are *derived* from
events by a RewardMapper, never stored as the source of truth, so the reward
definition can change without losing data (a common production lesson).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    IMPRESSION = "impression"   # item was shown (says nothing about attention)
    CLICK = "click"             # positive, implicit
    SKIP = "skip"               # explicitly passed over (examined, not clicked)
    LIKE = "like"               # positive, explicit
    RATING = "rating"           # explicit 1..5 (value in `value`)


@dataclass(frozen=True)
class Event:
    user_id: int
    item_id: int
    event_type: EventType
    timestamp: float
    request_id: str
    position: int | None = None
    value: float | None = None            # rating value for RATING events
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type == EventType.RATING and (self.value is None or not 1 <= self.value <= 5):
            raise ValueError(f"rating event needs value in 1..5, got {self.value}")
        if self.position is not None and self.position < 0:
            raise ValueError("position must be >= 0")

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class RewardMapper:
    """reward(item) = sum of weights of that item's events in one request.

    Defaults: click=1, like=1, skip=0 (a skip is informative for the *user state*
    but we do not punish the policy below "not clicked"), rating r adds
    rating_weight * (r - 3) / 2  in [-rating_weight, +rating_weight].
    """

    click: float = 1.0
    like: float = 1.0
    skip: float = 0.0
    impression: float = 0.0
    rating_weight: float = 0.5

    def reward(self, events: list[Event]) -> float:
        r = 0.0
        for e in events:
            if e.event_type == EventType.CLICK:
                r += self.click
            elif e.event_type == EventType.LIKE:
                r += self.like
            elif e.event_type == EventType.SKIP:
                r += self.skip
            elif e.event_type == EventType.IMPRESSION:
                r += self.impression
            elif e.event_type == EventType.RATING:
                r += self.rating_weight * (float(e.value) - 3.0) / 2.0
        return r
