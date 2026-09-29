"""The bounded agent loop (request mode).

    messages = [system, user request]
    for step in 1..max_steps:
        response = llm.chat(messages, tool_schemas)          # model PROPOSES
        if response has tool calls:
            for call in calls:
                result = tools.call(session, call)           # our code EXECUTES + validates
                messages += [tool result]
            if `recommend` succeeded: stop
        else:                                                # model answered in text only
            nudge once to call `recommend`, then give up -> fallback
    fallback: deterministic pipeline (never return nothing, never return ungrounded ids)

What makes this an AGENT rather than a pipeline: the sequence of tool calls is
decided at run time from the request and from intermediate results (e.g. "only
3 candidates matched -> relax the year filter and retrieve again"). What keeps
it safe: step budget, schema validation, grounding check, code-enforced hard
filters, and a deterministic fallback.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agentrec.agents.intent import parse_request
from agentrec.agents.tools import RecTools, Session, truncate
from agentrec.llm.client import LLMClient

SYSTEM_PROMPT = """You are the planning component of a movie recommender.
You cannot see the catalog directly; you act only through tools.
Rules:
1. Always call get_user_profile first.
2. If the user names a movie, call search_items to get its item_id.
3. Call retrieve_candidates with filters that match the request. Genre values must be from the allowed list.
4. If retrieve_candidates returns fewer candidates than needed, relax the SOFT constraints (popularity, then year, then
   include_genres) and retrieve again. Never relax exclude_genres.
5. If the user states a lasting dislike ("I hate X"), call save_preference.
6. Call rank_candidates, then call recommend with item_ids taken ONLY from the ranked list, plus a one-line reason each.
Do not invent item ids or titles."""


@dataclass
class AgentResult:
    items: list[int]
    explanations: dict[int, str]
    trajectory: list[dict[str, Any]]
    n_tool_calls: int
    n_llm_calls: int
    fallback_used: bool
    grounding_violations: int
    schema_errors: int          # invalid arguments / unknown tool / bad JSON (the LLM's formatting mistakes)
    tool_errors: int            # every error returned by a tool, incl. grounding rejections
    relaxations: list[str]
    latency_s: float
    llm_latency_s: float
    usage: dict[str, int] = field(default_factory=dict)


class RecAgent:
    def __init__(self, llm: LLMClient, tools: RecTools, max_steps: int = 10) -> None:
        self.llm, self.tools, self.max_steps = llm, tools, max_steps

    def run(self, user_id: int, request: str) -> AgentResult:
        t0 = time.perf_counter()
        s = Session(user_id)
        schemas = list(self.tools.schemas.values())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"user_id={user_id}\nrequest: {request}"},
        ]
        traj: list[dict[str, Any]] = []
        n_tools = n_llm = schema_err = tool_err = nudges = 0
        llm_lat = 0.0
        usage: dict[str, int] = {}
        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, schemas)
            n_llm += 1
            llm_lat += resp.latency_s
            for k, v in (resp.usage or {}).items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
            if not resp.tool_calls:
                traj.append({"type": "text", "content": resp.content})
                if nudges == 0:
                    nudges += 1
                    messages.append({"role": "assistant", "content": resp.content or ""})
                    messages.append({"role": "user", "content": "Please finish by calling the recommend tool."})
                    continue
                break
            messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in resp.tool_calls]})
            for c in resp.tool_calls:
                n_tools += 1
                result = ({"error": c.parse_error, "kind": "schema"} if c.parse_error
                          else self.tools.call(s, c.name, dict(c.arguments)))
                if "error" in result:
                    tool_err += 1
                    schema_err += int(result.get("kind") in ("schema", "unknown_tool"))
                traj.append({"type": "tool", "name": c.name, "args": c.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": truncate(result)})
            if s.final is not None:
                break

        retrieves = [t["args"] for t in traj if t.get("name") == "retrieve_candidates"]
        s.relaxations = [json.dumps(a, sort_keys=True) for a in retrieves[1:]]   # every re-retrieval = a replan
        fallback = s.final is None
        items = s.final if s.final is not None else self._fallback(user_id, request, s)
        return AgentResult(items, s.explanations, traj, n_tools, n_llm, fallback, s.grounding_violations,
                           schema_err, tool_err, s.relaxations, time.perf_counter() - t0, llm_lat, usage)

    def _fallback(self, user_id: int, request: str, s: Session) -> list[int]:
        """Deterministic safety net: parse constraints with regex, run the filtered pipeline."""
        c = parse_request(request, self.tools.ds.titles)
        r = self.tools.call(s, "retrieve_candidates", {"user_id": user_id, "n": 200,
                                                        "exclude_genres": c.exclude_genres})
        if "error" in r or not s.last_candidates:
            return []
        ranked = self.tools.call(s, "rank_candidates", {"user_id": user_id, "top_k": c.k})
        items = [x["item_id"] for x in ranked.get("ranked", [])]
        s.explanations = {i: "fallback: ranked by the feed pipeline" for i in items}
        return items


def item_ids_from(result: dict) -> list[int]:
    return [int(x["item_id"]) for x in result.get("ranked", [])]


def as_numpy(items: list[int]) -> np.ndarray:
    return np.asarray(items, dtype=np.int64)
