# E8 — request mode on `synthetic` (1050 templated requests + 150 memory follow-ups = 1200 turns per system, 150 users, k=5)

`agent_rule` = agent loop + tools + memory + guardrails, driven by a RULE-BASED planner (not an LLM). `agent_halluc` = same, but the first `recommend` call of every turn contains one out-of-catalogue item id injected by the test harness.

Columns: *valid & unseen* = every returned id exists in the catalogue and was not already seen by the user (membership in the retrieved set is enforced separately by the `recommend` tool). Rates are per returned item, except *filled k* (per turn).

| system | filled k | hard-constraint violations | soft constraints met | valid & unseen | taste sim. | p50 ms | tool calls | turns replanned | fallbacks | ungrounded ids rejected | schema errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| feed | 1.000 | 0.007 | 0.715 | 1.000 | 0.523 | 2.8 | 0.0 | 0 | 0 | 0 | 0 |
| fixed | 0.967 | 0.002 | 0.987 | 1.000 | 0.478 | 3.4 | 0.0 | 0 | 0 | 0 | 0 |
| agent_rule | 1.000 | 0.000 | 0.967 | 1.000 | 0.484 | 4.4 | 4.4 | 40 | 0 | 0 | 0 |
| agent_halluc | 1.000 | 0.000 | 0.967 | 1.000 | 0.484 | 4.7 | 5.4 | 40 | 0 | 1200 | 0 |

Per request type — soft constraints met / hard violations / filled:

| type | feed | fixed | agent_rule | agent_halluc |
|---|---|---|---|---|
| genre | 0.75 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| genre_decade | 0.10 / 0.00 / 1.00 | 0.89 / 0.00 / 0.73 | 0.74 / 0.00 / 1.00 | 0.74 / 0.00 / 1.00 |
| genre_exclude | 0.75 / 0.01 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| hidden_gems | 0.11 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| memory | 1.00 / 0.01 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| memory_followup | 1.00 / 0.01 / 1.00 | 1.00 / 0.01 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| similar | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| similar_exclude | 1.00 / 0.01 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
