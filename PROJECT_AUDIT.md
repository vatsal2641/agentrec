# PROJECT_AUDIT.md: AgentRec

_First audit 2026-09-29 (Phase 0: no code existed). Updated 2026-09-29 after the build._

## 1. Current state

| area | status | where |
|---|---|---|
| Data pipeline (ML-1M loader, implicit conversion, global temporal split, ID maps) | ✅ implemented + tested | `src/agentrec/data/` |
| Synthetic ML-1M-format generator (used because the build machine had no internet) | ✅ | `data/synthetic.py` |
| Baselines: popularity, recent popularity, content, item-kNN, BPR-MF | ✅ + E1 | `models/` |
| Two-tower retrieval (numpy, hand-written backprop, gradient-checked) | ✅ + E1/E2 | `models/two_tower.py` |
| ANN: exact, IVF (own), FAISS (optional) | ✅ + E2 | `retrieval/` |
| Ranker: pointwise LR, pairwise LR, GBDT; trained on a later window | ✅ + E3 | `ranking/` |
| Contextual bandits: LinUCB, LinTS, ε-greedy, greedy, frozen, random; position debiasing | ✅ + E4/E6 (simulated) | `bandits/` |
| Feedback loop: event schema, reward mapping, user-state + policy updates | ✅ + E5 (simulated) | `feedback/`, `memory/` |
| Memory: SQLite event log + user state, rebuildable from the log | ✅ + tests | `memory/` |
| LLM client (targets the OpenAI-compatible API used by Ollama etc.) | ✅ code, ⚠️ **never run against a real LLM or live endpoint** (no network) | `llm/client.py` |
| Agent loop: tools, schemas, grounding, replanning, fallback | ✅ + E8 with a **rule-based planner** | `agents/` |
| Tests | ✅ 59 unit + integration tests | `tests/` |
| Real MovieLens-1M results | ❌ **not yet run.** Run `bash scripts/run_all.sh configs/ml1m.yaml` | `results/ml1m/` |

## 2. Evidence against each resume claim
See `RESUME.md` for the sentence-by-sentence table. Summary:
- **Supported:** retrieval, personalisation (vs popularity), the bandit machinery, the feedback loop
  updating both user state and policy.
- **Supported in simulation only:** bandit gains, feedback adaptation, clicks and skips. Say
  "simulated".
- **Partially supported:** "agentic" and "LLM-based reasoning". The architecture is there and tested,
  but no real LLM has been run.

## 3. Findings that must NOT be spun (they're good interview material as they stand)
1. **No exploration policy reliably beats greedy online learning** in our simulator. Every learning
   policy cuts regret about 28–35% against the frozen ranker. The default LinUCB, LinTS and ε-greedy
   are equal to or worse than greedy, and the tuned variants are within greedy's seed spread (E4).
   Credit the gain to *online learning*, not to exploration.
2. **The warm start only holds for greedy.** The UCB/TS bonus dominates the small prior at t = 0
   (an independent review measured LinUCB's first slate matching the ranker's top 5 for 9 of 100
   users at λ = 1, and 30 of 100 at λ = 100). A larger λ mitigates this but doesn't remove it.
3. **For new users, the user-state update does the heavy lifting** (E5: 0.174 → 0.316 mean expected
   clicks). The third arm (LinUCB slate selection + policy updates) has the same mean, so its effect
   can't be separated.
4. **UCB1 has higher finite-horizon regret than ε-greedy** in the toy bandit (E4a), despite its
   better asymptotic guarantee.
5. **The ranker adds modest value.** The ranker chosen on validation (GBDT) gives +7.6% NDCG@10 over
   retrieval order. The best-on-test ranker gives +9%, but quoting it would be cherry-picking (E3).
6. **Our numpy IVF is slower than exact search** at 2,000 items because of Python overhead, even
   though it scans fewer items (E2).
7. The **fixed pipeline under-fills tight requests** (0.73 filled on decade requests). It also doesn't
   know stated preferences, though the measured violation rate is small (0.013 vs 0). The agent fixes
   both, but relaxation lowers soft-constraint satisfaction (E8).
8. **An independent review found a real bug.** Array item types weren't validated, so
   `similar_to_item_ids=["3"]` raised an uncaught TypeError. It's fixed and tested. Keep this as an
   interview story.

## 4. Remaining risks
- Synthetic-data numbers can't be quoted as MovieLens results.
- MovieLens "truth" for the simulator is a BPR proxy, so the bandit results depend on it.
- The LLM planner is unevaluated. Its failure modes (schema errors, loops, latency) are untested
  with a real model.
- An interviewer may ask you to write any of this from scratch. See `SELF_STUDY_PLAN.md`, Gate 5.
