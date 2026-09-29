# Latency on `synthetic` (2000 items, N=200 candidates, 300 requests, single process, CPU)

| stage (total = embed+retrieve+features+rank) | exact p50 ms | exact p95 ms | IVF p50 ms | IVF p95 ms |
|---|---|---|---|---|
| user_embedding | 0.058 | 0.119 | 0.056 | 0.107 |
| retrieval | 0.449 | 0.666 | 0.525 | 0.750 |
| features | 0.482 | 0.728 | 0.472 | 0.626 |
| ranker | 1.740 | 2.521 | 1.778 | 2.981 |
| bandit | 0.432 | 0.606 | 0.438 | 0.564 |
| total | 2.753 | 3.902 | 2.879 | 4.862 |

Throughput estimate (1000 / mean latency of total + bandit, sequential, exact): 293 req/s. Not a load test. `bandit` times only `select()` on a prebuilt context matrix; building the context re-uses retrieval + ranking.

Agent request (rule-based planner): p50 4.4 ms, p95 5.9 ms, 4.0 tool calls. planner is rule-based; a real LLM adds (#llm_calls x LLM latency), typically seconds.

Memory (MB): item_embeddings=1.02, two_tower_params=2.12, itemknn_similarity_sparse=1.51
