# DESIGN_DECISIONS.md

Each decision is recorded with the alternatives considered and the trade-off accepted. Anything
marked **(env)** was forced by the build environment: no internet, and only numpy, pandas,
scikit-learn and matplotlib available. Revisit those when you run the project on your own machine.

| # | Decision | Alternatives | Why this one | Cost accepted |
|---|---|---|---|---|
| D1 | **MovieLens-1M** as the target dataset | MIND (news impressions), KuaiRec (fully observed), Amazon reviews | Has timestamps, ratings, genres, titles (needed for text requests) and demographics; small enough for CPU; standard, so interviewers know it | No impressions, so clicks and skips must be simulated. No fully observed matrix, so bandit evaluation needs a simulator |
| D2 | **Synthetic data in ML-1M format** for development **(env)** | Hard-code toy data; fake numbers | Exercises the exact loader and pipeline, and its generative story is known (§ `data/synthetic.py`) | Results on it are *not* MovieLens results and are labelled as such |
| D3 | **Rating ≥ 4 = positive** | ≥ 3; weight by rating; treat every rating as positive | Standard; separates "watched and liked" from "watched" | 3★ is ambiguous; the threshold changes the positive rate |
| D4 | **Global temporal split** (80/90 percentiles) + refit on train+val | Random split; per-user leave-last-out; rolling windows | No future information in training; matches deployment | Few test users with long histories; cold users appear (reported as a slice) |
| D5 | **Full-ranking evaluation** | 1 positive + 100 sampled negatives | Sampled metrics can reorder models (Krichene & Rendle 2020) | Slower (it scores all items), which is fine at this size |
| D6 | **numpy two-tower with hand-written backprop** **(env)** | PyTorch | No torch available; also forces full understanding; gradients verified numerically | Slower training; small architecture; no GPU |
| D7 | **History-based user tower** (no user-ID embedding) | User-ID embedding (MF-style) | A click changes the embedding immediately → feedback adaptation without retraining; handles new users after a few clicks | Can't capture quirks that aren't visible in recent items; history truncated to L |
| D8 | **Genres inside the item tower** | ID-only | New items get meaningful vectors | Genre is a weak feature; plots or text would be better |
| D9 | **In-batch softmax + logQ correction** | BPR with uniform negatives; hard-negative mining | Many negatives for free; standard in industry two-towers | Popularity-skewed negatives need the correction; false negatives inside the batch (duplicates masked) |
| D10 | **Exact search by default; own IVF implementation; FAISS optional** **(env)** | FAISS / HNSW from the start | At 3.7k items exact search takes microseconds; IVF is implemented to *understand* ANN and measure the recall/latency trade-off | Our IVF is Python-looped, so its latency is not representative of FAISS |
| D11 | **Multi-source retrieval** (90% embedding + 10% recent-popular) | One source | Covers sparse users; captures trends | 10% of slots may be irrelevant for strongly personalised users |
| D12 | **Ranker trained on the VAL window** from stage-1 fit on TRAIN; selected on held-out VAL users | Train the ranker on the same window as stage 1 | Avoids learning from overconfident, memorised stage-1 scores | Less ranker training data; distribution shift when stage 1 is refit |
| D13 | **Compare pointwise LR / pairwise LR / GBDT**; pick by holdout | Choose one a priori; LambdaMART | Evidence-driven; shows the loss trade-offs | No listwise model (LightGBM unavailable **(env)**) |
| D14 | **Shared-parameter LinUCB / LinTS over the ranker's top-N**, prior mean θ₀ = ranker order (a true warm start only for greedy; the UCB/TS bonus dominates at small λ) | Per-item (disjoint) LinUCB; explore over the whole catalogue; neural bandits | Generalises to unseen items; explores only among relevant candidates; interpretable maths | Linear reward assumption; slate treated as independent picks |
| D15 | **Position as a feature** for debiasing | Naive; IPS with propensities; cascade model | Simple, needs no propensity estimates | Additive approximation of a multiplicative effect |
| D16 | **Semi-synthetic simulator** for bandit and feedback experiments | Naive replay on non-random logs (invalid); KuaiRec; online A/B | The only valid option available here; assumptions are explicit | Results reflect the simulator's truth model, not real users |
| D17 | **Events are the source of truth; state is derived** | Store only state | Replay, audit, changeable reward definitions (`rebuild_state`) | Extra storage |
| D18 | **SQLite + in-process arrays for memory** | Redis, a vector DB | The access pattern is a keyed lookup; no semantic search is needed | Not a distributed store; fine for one process |
| D19 | **No LLM in feed mode** | LLM re-ranks every feed | Latency (seconds vs ms), cost, no measured gain, hallucination risk | No natural-language explanations in the feed (template explanations are possible) |
| D20 | **Agent only for free-text requests**, over the *same* tools | Agent for everything; no agent | Open-ended input is where dynamic control flow earns its cost | More moving parts; needs its own evaluation |
| D21 | **Guardrails in code**: grounding set, schema validation, memory filters, step budget, fallback | Prompt instructions only | Prompts can be ignored; code can't | Some flexibility lost (e.g. the LLM can't override a stated exclude even if the user changes their mind in the same message) |
| D22 | **Rule-based planner** for tests and offline evaluation **(env)** | Always call a real LLM | Reproducible, free, runs offline; the baseline a real LLM must beat | It is *not* reasoning. The agentic claim needs the `--llm` run |
| D23 | **OpenAI-compatible HTTP client** | Vendor SDKs | Targets the API format of Ollama (free, local), vLLM, Groq and OpenAI | Not tested against a live endpoint here (no network); tool-calling quality varies by model |
| D24 | **Plain functions + dataclasses, no framework** (no LangChain etc.) | Agent frameworks | Every line is inspectable and defensible in an interview | You write the loop yourself (about 100 lines) |

## Things deliberately NOT done (and why)
- **A vector database.** Not needed (D18).
- **A sequence model (SASRec / GRU4Rec).** It would likely help, but it adds scope; it's listed as
  future work. The two-tower already uses recency, through the last L items.
- **An LLM re-ranking the feed.** See D19. If you want to test it, E7 is the place: measure
  NDCG / latency / cost *and* the leakage risk described in LEARNING_NOTES §8.
- **Reflection or self-critique loops.** No failure mode in E8 needed them. The guardrail error
  message plays that role.
