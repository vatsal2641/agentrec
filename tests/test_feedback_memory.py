import numpy as np
import pytest

from helpers import *  # noqa: F401,F403
from agentrec.bandits.simulator import SlateFeedback
from agentrec.feedback.events import Event, EventType, RewardMapper
from agentrec.feedback.loop import FeedbackLoop, feedback_to_events
from agentrec.memory.store import MemoryStore
from agentrec.memory.user_state import UserState

G = np.eye(4)[[0, 1, 2, 3, 0, 1]].astype(float)   # 6 items, 4 genres


def ev(t, item, value=None):
    return Event(7, item, t, 0.0, "r1", 0, value)


def test_event_validation():
    with pytest.raises(ValueError):
        Event(1, 1, EventType.RATING, 0.0, "r", 0, value=None)
    with pytest.raises(ValueError):
        Event(1, 1, EventType.CLICK, 0.0, "r", -1)


def test_reward_mapping():
    m = RewardMapper()
    assert m.reward([ev(EventType.IMPRESSION, 1), ev(EventType.CLICK, 1)]) == 1.0
    assert m.reward([ev(EventType.CLICK, 1), ev(EventType.RATING, 1, 5)]) == pytest.approx(1.5)
    assert m.reward([ev(EventType.SKIP, 1)]) == 0.0


def test_user_state_updates():
    s = UserState(7, 4)
    s.apply(ev(EventType.CLICK, 2), G)
    assert s.history == [2] and s.genre_pref[2] > 0
    s.apply(ev(EventType.RATING, 2, 1), G)          # clicked then hated it
    assert 2 not in s.history and 2 in s.disliked
    s.apply(ev(EventType.SKIP, 3), G); s.apply(ev(EventType.SKIP, 3), G)
    assert {2, 3} <= s.blocked_items()
    with pytest.raises(ValueError):
        s.apply(Event(8, 1, EventType.CLICK, 0.0, "r"), G)


def test_state_persists_and_rebuilds_from_log():
    store = MemoryStore()
    s = UserState(7, 4)
    events = [ev(EventType.CLICK, 1), ev(EventType.SKIP, 4), ev(EventType.RATING, 5, 5), ev(EventType.RATING, 1, 2)]
    store.log_events(events)
    for e in events:
        s.apply(e, G)
    store.save_state(s)
    loaded = store.load_state(7, 4)
    rebuilt = store.rebuild_state(7, 4, G)
    for x in (loaded, rebuilt):
        assert x.history == s.history and x.disliked == s.disliked and np.allclose(x.genre_pref, s.genre_pref)
    assert store.count_events()["rating"] == 2


def test_feedback_loop_updates_user_and_policy():
    class Recorder:
        def __init__(self): self.calls = []
        def update(self, x, r, w=1.0): self.calls.append(r)

    store, pol = MemoryStore(), Recorder()
    loop = FeedbackLoop(store, G)
    st = UserState(7, 4)
    fb = SlateFeedback(examined=np.array([1, 1, 0], bool), clicked=np.array([1, 0, 0], bool),
                       skipped=np.array([0, 1, 0], bool), ratings=np.array([4, 0, 0]))
    r = loop.process(st, np.array([0, 1, 2]), fb, policy=pol, X_update=np.eye(3))
    assert r.tolist() == [1.25, 0.0, 0.0]
    assert st.history == [0] and st.skip_counts == {1: 1}
    assert pol.calls == [1.25, 0.0, 0.0]
    assert store.count_events() == {"click": 1, "impression": 3, "rating": 1, "skip": 1}
    frozen = FeedbackLoop(MemoryStore(), G, update_user=False, update_policy=False)
    st2, pol2 = UserState(7, 4), Recorder()
    frozen.process(st2, np.array([0, 1, 2]), fb, policy=pol2, X_update=np.eye(3))
    assert st2.history == [] and pol2.calls == []


def test_feedback_to_events_positions():
    fb = SlateFeedback(np.array([1, 0], bool), np.array([1, 0], bool), np.array([0, 0], bool), np.array([0, 0]))
    e = feedback_to_events(3, np.array([10, 11]), fb, "rid", now=1.0)
    assert [(x.item_id, x.event_type.value, x.position) for x in e] == [(10, "impression", 0), (10, "click", 0), (11, "impression", 1)]


def test_stated_preferences_become_a_feed_mask():
    s = UserState(7, 4, stated={"exclude_genres": ["B"]})
    mask = s.allowed_mask(G, ["A", "B", "C", "D"])
    assert mask.tolist() == [True, False, True, True, True, False]
