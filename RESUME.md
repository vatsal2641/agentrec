# RESUME.md: Phase 18, resume defensibility

## The original bullets, checked sentence by sentence

> **AgentRec — Agentic Personalized Recommendation System**
> • Built an agentic recommendation system combining candidate retrieval, contextual bandits, and LLM-based reasoning for personalized recommendations.
> • Implemented feedback-driven adaptation from clicks, skips, and ratings to update user preferences and recommendation policies.

| claim | status | evidence in repo | what's missing / risky |
|---|---|---|---|
| "Built … system" | **Supported** (once *you* can explain it, i.e. after Gate 5) | full package + 59 tests + `scripts/run_all.sh` | You must be able to rebuild it (capstone exercise 09) |
| "agentic" | **Partially supported** | bounded tool-calling loop, schemas, memory, replanning, grounding guardrail, fallback (`agents/`), E8 | Offline results use a **rule-based planner**. Fully true only with an LLM planner, and only once you've run `--llm` yourself |
| "candidate retrieval" | **Supported** | two-tower + exact/IVF index + multi-source retriever; E2 (Recall@N, ANN recall and latency) | none |
| "contextual bandits" | **Supported, in simulation** | LinUCB / LinTS / ε-greedy over the ranker's top N with position debiasing; serving path `serve_slate`; E4–E6 | Evaluated on a **simulator**, not on real users. Exploration did **not** beat greedy online learning on synthetic data, so don't credit the regret cut to exploration |
| "LLM-based reasoning" | **Partially supported** | OpenAI-compatible tool-calling client, prompt, structured tool arguments, grounding | No LLM result exists until you run one. There's no evidence the LLM improves accuracy, and the feed path has no LLM at all (by design) |
| "for personalized recommendations" | **Supported** | the two-tower beats popularity on the temporal split (E1) | Quote *your* ML-1M numbers |
| "feedback-driven adaptation from clicks, skips, and ratings" | **Supported, in simulation** | event schema; RewardMapper; UserState rules for clicks, skips and ratings; E5 | The clicks, skips and ratings come from the **simulator** (MovieLens has only ratings) |
| "update user preferences and recommendation policies" | **Supported** | UserState (history, dislikes, skips → the user embedding changes) and bandit (A, b) updates, both in `FeedbackLoop.process`; E5 | In E5, the arm that adds LinUCB selection + policy updates has the same mean clicks as user-state-only. Don't imply the policy update added value |

## Suggested bullets (only after you reproduce the numbers on ML-1M)

Fill in the `<…>` values from **your** `results/ml1m/*.md`. If a number doesn't support the claim,
drop the claim.

**Version A: accurate and strong (recommended)**
> **AgentRec — Recommender with online learning and a tool-calling agent** · Python, NumPy, scikit-learn, SQLite
> • Built a two-stage recommender on MovieLens-1M (two-tower retrieval with hand-derived in-batch-softmax gradients + GBDT/LR re-ranker) evaluated with full-ranking metrics on a temporal split; NDCG@10 <x> vs <y> for popularity.
> • Implemented an event-driven feedback loop (clicks, skips, ratings → per-user state and online contextual-bandit updates with position-bias correction); in a semi-synthetic user simulator, online learning cut cumulative regret <z>% vs the frozen ranker, and I compared greedy, LinUCB and Thompson-sampling exploration.
> • Built a tool-calling agent loop for free-text requests (retrieve / rank / memory tools, JSON-schema validation, grounding check, fallback). Evaluated offline with a rule-based planner on <n> templated requests: 100% of slates filled, 0 hard-constraint violations, and 100% of harness-injected invalid item ids rejected.

Only write "LinUCB cut regret" if *your* ML-1M run shows LinUCB beating greedy. On synthetic data it
didn't (EXPERIMENTS E4). "Semi-synthetic" is correct for the ML-1M simulator (its truth is a model fit
to real ratings). The committed synthetic run is fully synthetic.

**Version B: close to your original wording, made defensible**
> **AgentRec — Agentic Personalized Recommendation System**
> • Built a recommender combining two-tower candidate retrieval, a learned re-ranker, contextual bandits (LinUCB/Thompson sampling), and a tool-calling agent layer for natural-language requests (LLM-pluggable; evaluated offline with a rule-based planner), with outputs restricted to retrieved candidates.
> • Implemented feedback-driven adaptation from (simulated) clicks, skips, and ratings that updates per-user preference state and bandit policy parameters online; evaluated on MovieLens-1M with a temporal split and a user simulator.

If you run the agent with a real LLM (SELF_STUDY_PLAN, week 4) and measure it, you may replace
"LLM-pluggable; evaluated offline with a rule-based planner" with what you actually measured.

## Words to avoid unless you have evidence
"production", "deployed", "real users", "A/B test", "state-of-the-art", "LLM improved accuracy",
"scalable to millions" (say "designed for" and explain how, if asked), and any percentage you
haven't reproduced.

## The 30-second pitch (practise it)
"Feed requests go through a normal two-stage recommender: a two-tower model retrieves a few hundred
candidates and a ranker orders them. A contextual bandit then picks the slate from the ranker's top
50, using the ranker's score as a prior and correcting for position bias. Every click, skip and
rating is logged as an event. The user's embedding is recomputed from their history immediately, and
the bandit's statistics update online. In my simulator, online learning beat the frozen ranker by
about a third in regret, but explicit exploration didn't beat plain greedy. I can explain why. For
free-text requests, an agent calls the same retrieval and ranking as tools, and it can only return
items that retrieval actually produced. The offline evaluation used a rule-based planner; the LLM
plugs into the same loop."
