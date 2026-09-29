"""Exercise 08: a minimal agent loop with a grounding guardrail. Reference: agents/agent.py, agents/tools.py.

The 'LLM' is a list of scripted actions. Implement run_agent:
  - execute tool actions in order; `retrieve` adds its ids to the session's grounding set
  - `recommend` succeeds ONLY if every id was retrieved; otherwise count a violation and continue
  - stop at the first successful recommend, or after max_steps
  - if no successful recommend: fall back to the first k retrieved ids (or [] if none)
Return (items, n_violations, used_fallback).
"""

CATALOG = {i: f"movie {i}" for i in range(100)}


def retrieve(query: str) -> list[int]:
    return [i for i in CATALOG if i % 7 == len(query) % 7][:10]


def run_agent(actions: list[dict], k: int = 3, max_steps: int = 5) -> tuple[list[int], int, bool]:
    raise NotImplementedError


if __name__ == "__main__":
    good = [{"tool": "retrieve", "query": "abc"}, {"tool": "recommend", "ids": [3, 10]}]
    assert run_agent(good) == ([3, 10], 0, False)
    halluc = [{"tool": "retrieve", "query": "abc"}, {"tool": "recommend", "ids": [99]}, {"tool": "recommend", "ids": [17]}]
    assert run_agent(halluc) == ([17], 1, False)
    never = [{"tool": "retrieve", "query": "abc"}] + [{"tool": "recommend", "ids": [99]}] * 10
    assert run_agent(never, max_steps=4) == ([3, 10, 17], 3, True)
    assert run_agent([], max_steps=2) == ([], 0, True)
    print("all tests passed. Question: why is the guardrail in code instead of in the prompt?")
