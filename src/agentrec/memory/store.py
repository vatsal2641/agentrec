"""SQLite-backed memory: an append-only event log + a per-user state row.

WHY SQLITE AND NOT A VECTOR DATABASE
  What we store is structured: events (user, item, type, time) and a small
  per-user state. Access pattern = "get everything for user u" and "append".
  That is a keyed lookup, which any relational DB does perfectly. A vector DB
  is for *similarity search over unstructured content* (e.g. retrieving past
  conversation snippets by meaning). We have no such need; item similarity
  search already lives in the retrieval index. In production the same roles
  are played by a log stream (Kafka) + a feature store / key-value store (Redis,
  Bigtable).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from agentrec.feedback.events import Event, EventType
from agentrec.memory.user_state import UserState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL, item_id INTEGER NOT NULL, event_type TEXT NOT NULL,
    timestamp REAL NOT NULL, request_id TEXT NOT NULL, position INTEGER, value REAL,
    context TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, timestamp);
CREATE TABLE IF NOT EXISTS user_state (
    user_id INTEGER PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL
);
"""


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SCHEMA)

    # ---------------------------------------------------------------- events
    def log_events(self, events: list[Event]) -> None:
        self.conn.executemany(
            "INSERT INTO events (user_id,item_id,event_type,timestamp,request_id,position,value,context)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [(e.user_id, e.item_id, e.event_type.value, e.timestamp, e.request_id, e.position, e.value,
              json.dumps(e.context)) for e in events],
        )
        self.conn.commit()

    def events_for(self, user_id: int, limit: int = 1000) -> list[Event]:
        rows = self.conn.execute(
            "SELECT user_id,item_id,event_type,timestamp,request_id,position,value,context FROM events"
            " WHERE user_id=? ORDER BY timestamp, id LIMIT ?", (user_id, limit)).fetchall()
        return [Event(r[0], r[1], EventType(r[2]), r[3], r[4], r[5], r[6], json.loads(r[7] or "{}")) for r in rows]

    def count_events(self) -> dict[str, int]:
        return dict(self.conn.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall())

    # ----------------------------------------------------------------- state
    def load_state(self, user_id: int, n_genres: int) -> UserState:
        row = self.conn.execute("SELECT state FROM user_state WHERE user_id=?", (user_id,)).fetchone()
        return UserState.from_json(row[0]) if row else UserState(user_id, n_genres)

    def save_state(self, s: UserState) -> None:
        self.conn.execute("INSERT OR REPLACE INTO user_state VALUES (?,?,?)", (s.user_id, s.to_json(), time.time()))
        self.conn.commit()

    def rebuild_state(self, user_id: int, n_genres: int, item_genres) -> UserState:
        """Recompute state from the event log (proves the log is the source of truth)."""
        s = UserState(user_id, n_genres)
        for e in self.events_for(user_id, limit=10**9):
            s.apply(e, item_genres)
        return s
