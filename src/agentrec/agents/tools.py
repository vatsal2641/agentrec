"""Tools the agent can call. Each tool = a python function + a JSON schema.

DESIGN RULES (these are the guardrails that do not depend on the LLM behaving)
  * Tools wrap the SAME retrieval/ranking code as the feed pipeline — the agent
    orchestrates, it does not re-implement recommendation.
  * Every item id the agent is allowed to output must have been returned by
    `retrieve_candidates` in THIS session (grounding). `recommend` enforces it.
  * Hard constraints from memory (stated excludes, dislikes, already-seen) are
    applied inside `retrieve_candidates` in code, whether or not the LLM asks.
  * Tool outputs are truncated before going back to the LLM (context budget).
  * Arguments are validated against the schema; errors are returned to the LLM
    as tool results so it can correct itself (instead of crashing the loop).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from agentrec.agents.intent import GENRE_SYNONYMS, match_title
from agentrec.data.dataset import Dataset
from agentrec.memory.store import MemoryStore
from agentrec.pipeline import FeedPipeline

GENRES = list(GENRE_SYNONYMS)


@dataclass
class Session:
    user_id: int
    retrieved: set[int] = field(default_factory=set)    # grounding set
    last_candidates: list[int] = field(default_factory=list)
    last_ranked: list[int] = field(default_factory=list)
    final: list[int] | None = None
    explanations: dict[int, str] = field(default_factory=dict)
    grounding_violations: int = 0
    relaxations: list[str] = field(default_factory=list)


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_GENRE_LIST = {"type": "array", "items": {"type": "string", "enum": GENRES}}

SCHEMAS = [
    _schema("get_user_profile", "Summary of the user's taste: top genres, recent liked titles, stored preferences.",
            {"user_id": {"type": "integer"}}, ["user_id"]),
    _schema("search_items", "Find catalog items by (fuzzy) title. Use when the user names a specific movie.",
            {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
    _schema("retrieve_candidates",
            "Personalised candidate retrieval with optional filters. Returns how many candidates matched. "
            "Items already seen / disliked / excluded by stored preferences are removed automatically.",
            {"user_id": {"type": "integer"}, "n": {"type": "integer"},
             "include_genres": _GENRE_LIST, "exclude_genres": _GENRE_LIST,
             "year_min": {"type": "integer"}, "year_max": {"type": "integer"},
             "similar_to_item_ids": {"type": "array", "items": {"type": "integer"}},
             "popularity": {"type": "string", "enum": ["head", "tail"]}},
            ["user_id"]),
    _schema("rank_candidates", "Rank the most recently retrieved candidates for the user with the learned ranker.",
            {"user_id": {"type": "integer"}, "top_k": {"type": "integer"}}, ["user_id"]),
    _schema("get_item_metadata", "Title, genres and year for item ids.",
            {"item_ids": {"type": "array", "items": {"type": "integer"}}}, ["item_ids"]),
    _schema("save_preference", "Persist a lasting user preference (e.g. the user says they hate a genre).",
            {"user_id": {"type": "integer"}, "exclude_genres": _GENRE_LIST}, ["user_id", "exclude_genres"]),
    _schema("recommend", "FINAL step. Return the chosen item ids (must come from retrieved candidates) "
            "with a one-line explanation each.",
            {"user_id": {"type": "integer"}, "item_ids": {"type": "array", "items": {"type": "integer"}},
             "explanations": {"type": "array", "items": {"type": "string"}}},
            ["user_id", "item_ids"]),
]

_TYPES = {"integer": int, "string": str, "array": list, "object": dict}


def validate_args(schema: dict, args: dict) -> str | None:
    """Minimal JSON-schema check: required keys, types, enums. Returns error text or None."""
    params = schema["function"]["parameters"]
    for r in params["required"]:
        if r not in args:
            return f"missing required argument '{r}'"
    for k, v in args.items():
        spec = params["properties"].get(k)
        if spec is None:
            return f"unknown argument '{k}'"
        if v is None:
            continue
        t = _TYPES[spec["type"]]
        if t is int and isinstance(v, float) and v.is_integer():
            v = int(v); args[k] = v
        if not isinstance(v, t) or (t is int and isinstance(v, bool)):
            return f"argument '{k}' must be {spec['type']}"
        if spec["type"] == "array":
            it = _TYPES[spec["items"]["type"]]
            bad_t = [x for x in v if not isinstance(x, it) or (it is int and isinstance(x, bool))]
            if bad_t:
                return f"argument '{k}' must be an array of {spec['items']['type']}; bad values {bad_t[:5]}"
        if spec["type"] == "array" and "enum" in spec["items"]:
            bad = [x for x in v if x not in spec["items"]["enum"]]
            if bad:
                return f"argument '{k}' has invalid values {bad}; allowed: {spec['items']['enum']}"
        if "enum" in spec and v not in spec["enum"]:
            return f"argument '{k}' must be one of {spec['enum']}"
    return None


class RecTools:
    def __init__(self, ds: Dataset, pipe: FeedPipeline, store: MemoryStore, max_list: int = 20) -> None:
        self.ds, self.pipe, self.store, self.max_list = ds, pipe, store, max_list
        self.gidx = {g: k for k, g in enumerate(ds.genre_names)}
        pop = np.asarray(ds.matrix(ds.fit_period(pipe.phase)).sum(0)).ravel()
        order = np.argsort(-pop)
        self.head = np.zeros(ds.n_items, bool)
        self.head[order[: int(0.2 * ds.n_items)]] = True
        self.fns: dict[str, Callable[..., dict]] = {
            "get_user_profile": self.get_user_profile, "search_items": self.search_items,
            "retrieve_candidates": self.retrieve_candidates, "rank_candidates": self.rank_candidates,
            "get_item_metadata": self.get_item_metadata, "save_preference": self.save_preference,
            "recommend": self.recommend,
        }
        self.schemas = {s["function"]["name"]: s for s in SCHEMAS}

    # ------------------------------------------------------------ helpers
    def _meta(self, i: int) -> dict:
        return {"item_id": int(i), "title": self.ds.titles[i],
                "genres": [g for g, k in self.gidx.items() if self.ds.item_genres[i, k] > 0],
                "year": int(self.ds.item_year[i])}

    def _state(self, user_id: int):
        return self.store.load_state(user_id, len(self.gidx))

    def _history(self, user_id: int) -> np.ndarray:
        st = self._state(user_id)
        base = list(self.pipe.history(user_id))
        return np.array(base + [i for i in st.history if i not in base], dtype=np.int64)

    def _check_user(self, s: Session, user_id: int) -> None:
        if user_id != s.user_id:
            raise ValueError(f"this session is for user {s.user_id}; refusing to act on user {user_id}")

    # ------------------------------------------------------------ dispatcher
    def call(self, s: Session, name: str, args: dict) -> dict:
        """Every failure becomes {"error": ..., "kind": ...} returned to the LLM; nothing raises."""
        if name not in self.fns:
            return {"error": f"unknown tool '{name}'. Available: {list(self.fns)}", "kind": "unknown_tool"}
        err = validate_args(self.schemas[name], args)
        if err:
            return {"error": err, "kind": "schema"}
        try:
            out = self.fns[name](s, **args)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            return {"error": str(e), "kind": "execution"}
        if "error" in out and "kind" not in out:
            out["kind"] = "grounding" if name == "recommend" and "not returned" in out["error"] else "execution"
        return out

    # ------------------------------------------------------------ tools
    def get_user_profile(self, s: Session, user_id: int) -> dict:
        self._check_user(s, user_id)
        h = self._history(user_id)
        st = self._state(user_id)
        counts = self.ds.item_genres[h].sum(0) if len(h) else np.zeros(len(self.gidx))
        top = [self.ds.genre_names[k] for k in np.argsort(-counts)[:3] if counts[k] > 0]
        return {"user_id": user_id, "n_liked": int(len(h)), "top_genres": top,
                "recent_likes": [self.ds.titles[i] for i in h[-5:]][::-1],
                "stored_preferences": st.stated, "cold_start": bool(len(h) == 0)}

    def search_items(self, s: Session, query: str, limit: int = 5) -> dict:
        t = match_title(query, self.ds.titles)
        hits = [self.ds.titles.index(t)] if t else []
        ql = query.lower()
        for i, title in enumerate(self.ds.titles):       # substring matches as a fallback
            if len(hits) >= limit:
                break
            if ql in title.lower() and i not in hits:
                hits.append(i)
        return {"results": [self._meta(i) for i in hits[:limit]]}

    def retrieve_candidates(self, s: Session, user_id: int, n: int = 100, include_genres: list[str] | None = None,
                            exclude_genres: list[str] | None = None, year_min: int | None = None,
                            year_max: int | None = None, similar_to_item_ids: list[int] | None = None,
                            popularity: str | None = None) -> dict:
        self._check_user(s, user_id)
        n = int(np.clip(n, 1, 500))
        st = self._state(user_id)
        excl = set(exclude_genres or []) | set(st.stated.get("exclude_genres", []))  # memory is enforced
        allowed = np.ones(self.ds.n_items, bool)
        for g in excl:
            allowed &= self.ds.item_genres[:, self.gidx[g]] == 0
        if include_genres:
            cols = [self.gidx[g] for g in include_genres]
            allowed &= self.ds.item_genres[:, cols].sum(1) > 0
        yr = self.ds.item_year
        if year_min is not None:
            allowed &= yr >= year_min
        if year_max is not None:
            allowed &= yr <= year_max
        if popularity == "tail":
            allowed &= ~self.head
        elif popularity == "head":
            allowed &= self.head
        hist = self._history(user_id)
        exclude = st.blocked_items() | set(int(i) for i in (similar_to_item_ids or []))
        uvec = self.pipe.user_vec(hist)
        if similar_to_item_ids:
            bad = [i for i in similar_to_item_ids if not 0 <= i < self.ds.n_items]
            if bad:
                return {"error": f"unknown item ids {bad}"}
            seed = self.pipe.s1.tower.V[similar_to_item_ids].mean(0)
            q = seed if uvec is None else 0.7 * seed / np.linalg.norm(seed) + 0.3 * uvec
            uvec = q / (np.linalg.norm(q) + 1e-12)
        ex = set(self.pipe._seen.get(user_id, set())) | exclude | set(hist.tolist())
        c = self.pipe.retriever.retrieve(uvec, n, exclude=ex, allowed=allowed)
        s.last_candidates = c.items.tolist()
        s.retrieved |= set(s.last_candidates)
        self._last_c, self._last_hist = c, hist
        return {"count": len(s.last_candidates), "requested": n,
                "applied_filters": {"exclude_genres": sorted(excl), "include_genres": include_genres or [],
                                    "year_min": year_min, "year_max": year_max, "popularity": popularity},
                "preview": [self.ds.titles[i] for i in s.last_candidates[:5]]}

    def rank_candidates(self, s: Session, user_id: int, top_k: int = 10) -> dict:
        self._check_user(s, user_id)
        if not s.last_candidates:
            return {"error": "no candidates retrieved yet; call retrieve_candidates first"}
        items, scores, F = self.pipe.rank(user_id, self._last_c, self._last_hist)
        top_k = int(np.clip(top_k, 1, self.max_list))
        s.last_ranked = items[:top_k].tolist()
        out = []
        for i, sc, f in zip(items[:top_k], scores[:top_k], F[:top_k]):
            m = self._meta(int(i))
            m["score"] = round(float(sc), 4)
            m["genre_match"] = round(float(f[5]), 2)       # FEATURES[5]
            m["is_long_tail"] = bool(not self.head[int(i)])
            out.append(m)
        return {"ranked": out}

    def get_item_metadata(self, s: Session, item_ids: list[int]) -> dict:
        bad = [i for i in item_ids if not 0 <= i < self.ds.n_items]
        if bad:
            return {"error": f"unknown item ids {bad}"}
        return {"items": [self._meta(i) for i in item_ids[: self.max_list]]}

    def save_preference(self, s: Session, user_id: int, exclude_genres: list[str]) -> dict:
        self._check_user(s, user_id)
        st = self._state(user_id)
        cur = set(st.stated.get("exclude_genres", []))
        st.stated["exclude_genres"] = sorted(cur | set(exclude_genres))
        self.store.save_state(st)
        return {"saved": st.stated}

    def recommend(self, s: Session, user_id: int, item_ids: list[int], explanations: list[str] | None = None) -> dict:
        self._check_user(s, user_id)
        invalid = [i for i in item_ids if i not in s.retrieved]
        if invalid:
            s.grounding_violations += len(invalid)
            return {"error": f"item ids {invalid} were not returned by retrieve_candidates in this session. "
                             "Only recommend retrieved items."}
        if not item_ids:
            return {"error": "item_ids is empty"}
        dedup = list(dict.fromkeys(item_ids))
        s.final = dedup
        ex = explanations or []
        s.explanations = {i: (ex[k] if k < len(ex) else "") for k, i in enumerate(dedup)}
        return {"ok": True, "n": len(dedup)}


def truncate(obj: dict, max_chars: int = 2000) -> str:
    txt = json.dumps(obj)
    return txt if len(txt) <= max_chars else txt[:max_chars] + '..."truncated"'
