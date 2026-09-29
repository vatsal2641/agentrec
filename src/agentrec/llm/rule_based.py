"""A deterministic, rule-based stand-in for the LLM planner.

IMPORTANT (honesty): this is NOT an LLM and NOT "reasoning". It is a hand-written
state machine that speaks the same tool-calling protocol, so that
  * the agent loop, tools and guardrails can be tested without network access;
  * offline evaluation is reproducible and free;
  * we have a baseline that a real LLM planner must beat.
Plugging a real model in = replacing this class with OpenAICompatClient.

Decision logic (a fixed plan with one data-dependent branch: relaxation):
  profile -> [save_preference] -> [search_items] -> retrieve
     -> if too few candidates: relax popularity -> year -> include_genres -> larger n
     -> rank -> recommend (ids from the ranked list only)
"""
from __future__ import annotations

import json
import random
from typing import Any

from agentrec.agents.intent import parse_request
from agentrec.llm.client import LLMResponse, ToolCall

RELAX_ORDER = ["popularity", "year", "include_genres", "bigger_n"]


class RuleBasedPlanner:
    name = "rule_based_planner"

    def __init__(self, titles: list[str], n_candidates: int = 200, min_candidates: int | None = None) -> None:
        self.titles, self.n, self.min_candidates = titles, n_candidates, min_candidates
        self._i = 0

    def _call(self, name: str, args: dict[str, Any]) -> LLMResponse:
        self._i += 1
        return LLMResponse(None, [ToolCall(f"rb_{self._i}", name, args)])

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        uid = int(user_msg.split("\n")[0].split("=")[1])
        request = user_msg.split("request:", 1)[1].strip()
        c = parse_request(request, self.titles)
        need = self.min_candidates or c.k
        done = [(m["name"], json.loads(m["content"])) for m in messages if m["role"] == "tool"]
        names = [n for n, _ in done]

        if "get_user_profile" not in names:
            return self._call("get_user_profile", {"user_id": uid})
        if c.persistent_excludes and "save_preference" not in names:
            return self._call("save_preference", {"user_id": uid, "exclude_genres": c.persistent_excludes})
        if c.similar_to and "search_items" not in names:
            return self._call("search_items", {"query": c.similar_to, "limit": 1})
        seeds: list[int] = []
        for n, r in done:
            if n == "search_items" and r.get("results"):
                seeds = [r["results"][0]["item_id"]]

        retrieves = [r for n, r in done if n == "retrieve_candidates"]
        last_retrieve_idx = max((k for k, n in enumerate(names) if n == "retrieve_candidates"), default=-1)
        ranked_after = [r for k, (n, r) in enumerate(done) if n == "rank_candidates" and k > last_retrieve_idx]

        if not retrieves or (retrieves[-1].get("count", 0) < need and not ranked_after
                             and len(retrieves) <= len(RELAX_ORDER)):
            level = len(retrieves)                     # 0 = no relaxation
            relaxed = set(RELAX_ORDER[:level])
            args: dict[str, Any] = {"user_id": uid, "n": self.n * (3 if "bigger_n" in relaxed else 1)}
            if c.exclude_genres:
                args["exclude_genres"] = c.exclude_genres            # hard: never relaxed
            if c.include_genres and "include_genres" not in relaxed:
                args["include_genres"] = c.include_genres
            if "year" not in relaxed:
                if c.year_min is not None:
                    args["year_min"] = c.year_min
                if c.year_max is not None:
                    args["year_max"] = c.year_max
            if c.popularity and "popularity" not in relaxed:
                args["popularity"] = c.popularity
            if seeds:
                args["similar_to_item_ids"] = seeds
            return self._call("retrieve_candidates", args)

        if not ranked_after:
            return self._call("rank_candidates", {"user_id": uid, "top_k": c.k})

        ranked = ranked_after[-1].get("ranked", [])
        last_rec = [r for n, r in done if n == "recommend"]
        if last_rec and last_rec[-1].get("ok"):
            return LLMResponse("Done.")
        ids = [x["item_id"] for x in ranked[: c.k]]
        seed_title = next((r["results"][0]["title"] for n, r in done if n == "search_items" and r.get("results")), None)
        relaxed = RELAX_ORDER[: max(0, len(retrieves) - 1)]
        expl = []
        for x in ranked[: c.k]:
            why = [f"genres {', '.join(x['genres']) or 'n/a'}", f"taste match {x['genre_match']}"]
            if seed_title:
                why.append(f"close to {seed_title} in embedding space")
            if x.get("is_long_tail"):
                why.append("less-known pick")
            if relaxed:
                why.append("too few exact matches, relaxed: " + ", ".join(relaxed))
            expl.append("; ".join(why))
        return self._call("recommend", {"user_id": uid, "item_ids": ids, "explanations": expl})


class HallucinatingPlanner(RuleBasedPlanner):
    """Stress test for the grounding guardrail: on the FIRST recommend call, swap one
    id for an id that does not exist in the catalog (what an LLM does when it
    'remembers' a movie that is not in the candidate set / not available)."""

    name = "hallucinating_planner"

    def __init__(self, titles: list[str], n_items: int, p: float = 1.0, seed: int = 0, **kw) -> None:
        super().__init__(titles, **kw)
        self.n_items, self.p, self.rng = n_items, p, random.Random(seed)
        self._corrupted = False

    def chat(self, messages, tools):
        r = super().chat(messages, tools)
        if r.tool_calls and r.tool_calls[0].name == "recommend" and not self._corrupted and self.rng.random() < self.p:
            self._corrupted = True
            fake = self.n_items + self.rng.randrange(1000)     # an id that does not exist in the catalog
            args = dict(r.tool_calls[0].arguments)
            args["item_ids"] = [fake] + list(args["item_ids"][1:])
            r.tool_calls[0].arguments = args
        return r

    def reset(self) -> None:
        self._corrupted = False
