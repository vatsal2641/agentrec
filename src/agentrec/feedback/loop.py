"""The feedback loop:  recommendation -> interaction -> events -> reward
                         -> user-state update -> policy update -> next recommendation.

Two different things learn from feedback, on different time scales:
  * USER STATE (per user, instant): history/dislikes/skips change, so the
    two-tower user embedding and hard filters change on the very next request.
  * POLICY (global, online): the contextual bandit's ridge statistics (A, b)
    are updated with (context, reward) pairs, so the *scoring rule* shared by all
    users changes.
Everything is logged first; state is a function of the log (see MemoryStore.rebuild_state).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from agentrec.bandits.simulator import SlateFeedback
from agentrec.feedback.events import Event, EventType, RewardMapper, new_request_id
from agentrec.memory.store import MemoryStore
from agentrec.memory.user_state import UserState


def feedback_to_events(user: int, slate: np.ndarray, fb: SlateFeedback, request_id: str,
                       now: float | None = None) -> list[Event]:
    now = time.time() if now is None else now
    ev: list[Event] = []
    for k, item in enumerate(slate):
        item = int(item)
        ev.append(Event(user, item, EventType.IMPRESSION, now, request_id, k))
        if fb.clicked[k]:
            ev.append(Event(user, item, EventType.CLICK, now, request_id, k))
        if fb.skipped[k]:
            ev.append(Event(user, item, EventType.SKIP, now, request_id, k))
        if fb.ratings[k] > 0:
            ev.append(Event(user, item, EventType.RATING, now, request_id, k, value=float(fb.ratings[k])))
    return ev


@dataclass
class FeedbackLoop:
    store: MemoryStore
    item_genres: np.ndarray
    mapper: RewardMapper = field(default_factory=RewardMapper)
    update_user: bool = True
    update_policy: bool = True

    def process(self, state: UserState, slate: np.ndarray, fb: SlateFeedback, policy=None,
                X_update: np.ndarray | None = None, update_mask: np.ndarray | None = None,
                request_id: str | None = None) -> np.ndarray:
        """Log events, compute per-item rewards, update user state and (optionally) the policy.

        X_update    : (K, d) contexts to feed the policy for the shown items
        update_mask : (K,) which shown items the policy should learn from
        Returns the per-item rewards (K,).
        """
        rid = request_id or new_request_id()
        events = feedback_to_events(state.user_id, slate, fb, rid)
        self.store.log_events(events)
        by_item: dict[int, list[Event]] = {}
        for e in events:
            by_item.setdefault(e.item_id, []).append(e)
        rewards = np.array([self.mapper.reward(by_item[int(i)]) for i in slate])
        if self.update_user:
            for e in events:
                state.apply(e, self.item_genres)
            self.store.save_state(state)
        if self.update_policy and policy is not None and X_update is not None:
            mask = np.ones(len(slate), bool) if update_mask is None else update_mask
            for k in np.where(mask)[0]:
                policy.update(X_update[k], float(rewards[k]))
        return rewards
