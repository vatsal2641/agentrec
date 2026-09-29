import numpy as np

from helpers import tiny_dataset, tiny_stage1
from agentrec.agents.agent import RecAgent
from agentrec.agents.intent import parse_request
from agentrec.agents.tools import SCHEMAS, RecTools, Session, validate_args
from agentrec.llm.client import LLMResponse, ScriptedLLM, ToolCall
from agentrec.llm.rule_based import HallucinatingPlanner, RuleBasedPlanner
from agentrec.memory.store import MemoryStore
from agentrec.pipeline import FeedPipeline

TITLES = ["Heat (1995)", "Toy Story (1995)", "Alien (1979)"]


def _tools():
    ds = tiny_dataset()
    pipe = FeedPipeline(ds, "test", tiny_stage1())
    return ds, RecTools(ds, pipe, MemoryStore())


def _warm_user(ds):
    return int(ds.test.df["u"].iloc[0])


def test_parse_request_basic_constraints():
    c = parse_request("3 comedy movies from the 90s but no horror, hidden gems please", TITLES)
    assert c.k == 3 and c.include_genres == ["Comedy"] and c.exclude_genres == ["Horror"]
    assert (c.year_min, c.year_max) == (1990, 1999) and c.popularity == "tail"


def test_parse_similar_to_and_persistent_dislike():
    c = parse_request("something like heat but not romance. I hate musicals", TITLES)
    assert c.similar_to == "Heat (1995)"
    assert c.exclude_genres == ["Romance", "Musical"] and c.persistent_excludes == ["Musical"]


def test_schema_validation():
    s = {x["function"]["name"]: x for x in SCHEMAS}
    assert validate_args(s["retrieve_candidates"], {}) is not None                       # missing user_id
    assert "invalid values" in validate_args(s["retrieve_candidates"], {"user_id": 1, "include_genres": ["Jazz"]})
    assert validate_args(s["retrieve_candidates"], {"user_id": "1"}) is not None           # wrong type
    assert validate_args(s["retrieve_candidates"], {"user_id": 1, "foo": 2}) is not None   # unknown arg
    assert validate_args(s["retrieve_candidates"], {"user_id": 1, "include_genres": ["Drama"]}) is None


def test_grounding_guardrail_rejects_unretrieved_ids():
    ds, tools = _tools()
    u = _warm_user(ds)
    agent = RecAgent(HallucinatingPlanner(ds.titles, ds.n_items), tools)
    r = agent.run(u, "Action movies")
    assert r.grounding_violations == 1 and not r.fallback_used
    assert all(0 <= i < ds.n_items for i in r.items)


def test_scripted_llm_that_invents_ids_falls_back_safely():
    ds, tools = _tools()
    u = _warm_user(ds)
    bad = [LLMResponse(None, [ToolCall("1", "recommend", {"user_id": u, "item_ids": [ds.n_items + 5]})])] * 3
    r = RecAgent(ScriptedLLM(bad), tools, max_steps=3).run(u, "anything")
    assert r.fallback_used and r.grounding_violations == 3 and len(r.items) > 0


def test_text_only_llm_gets_one_nudge_then_fallback():
    ds, tools = _tools()
    u = _warm_user(ds)
    llm = ScriptedLLM([LLMResponse("You should watch Heat!"), LLMResponse("Really, Heat.")])
    r = RecAgent(llm, tools).run(u, "anything")
    assert llm.calls == 2 and r.fallback_used


def test_memory_excludes_are_enforced_even_if_llm_forgets():
    ds, tools = _tools()
    u = _warm_user(ds)
    RecAgent(RuleBasedPlanner(ds.titles), tools).run(u, "I hate drama. recommend 5 movies")
    s = Session(u)
    tools.call(s, "retrieve_candidates", {"user_id": u, "n": 50})          # no exclude passed
    g = ds.genre_names.index("Drama")
    assert all(ds.item_genres[i, g] == 0 for i in s.last_candidates)


def test_relaxation_replans_when_constraints_too_tight():
    ds, tools = _tools()
    u = _warm_user(ds)
    agent = RecAgent(RuleBasedPlanner(ds.titles, min_candidates=40), tools)
    r = agent.run(u, "Film-Noir documentaries from the 30s, hidden gems")
    assert len(r.relaxations) >= 1
    names = [t["name"] for t in r.trajectory if t["type"] == "tool"]
    assert names.count("retrieve_candidates") == len(r.relaxations) + 1


def test_agent_refuses_other_users():
    ds, tools = _tools()
    out = tools.call(Session(1), "get_user_profile", {"user_id": 2})
    assert "error" in out


def test_unknown_tool_is_reported_not_crashing():
    ds, tools = _tools()
    assert "error" in tools.call(Session(1), "delete_database", {})


def test_array_item_types_are_validated_and_nothing_raises():
    ds, tools = _tools()
    u = _warm_user(ds)
    out = tools.call(Session(u), "retrieve_candidates", {"user_id": u, "similar_to_item_ids": ["3"]})
    assert out["kind"] == "schema"
    out = tools.call(Session(u), "get_item_metadata", {"item_ids": [1.5]})
    assert out["kind"] == "schema"


def test_error_kinds_are_counted_separately():
    ds, tools = _tools()
    u = _warm_user(ds)
    r = RecAgent(HallucinatingPlanner(ds.titles, ds.n_items), tools).run(u, "Action movies")
    assert r.tool_errors == 1 and r.schema_errors == 0
