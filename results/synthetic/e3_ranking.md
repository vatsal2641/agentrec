# E3 — ranking on `synthetic` (TEST, warm users=742, N=200 candidates)

Chosen on held-out VAL users: **gbdt**

| ranker | holdout-VAL ndcg@10 | TEST ndcg@10 | recall@10 | mrr@10 | map@10 | recall@50 | p50 ms/request |
|---|---|---|---|---|---|---|---|
| identity | 0.1979 | 0.2226 | 0.1422 | 0.4163 | 0.1249 | 0.3883 | 1.3 |
| pointwise | 0.2014 | 0.2426 | 0.1567 | 0.4473 | 0.1395 | 0.3969 | 1.5 |
| pairwise | 0.2000 | 0.2407 | 0.1549 | 0.4461 | 0.1379 | 0.3961 | 1.6 |
| gbdt | 0.2138 | 0.2395 | 0.1535 | 0.4396 | 0.1375 | 0.3970 | 3.1 |

Candidate-size ablation (gbdt ranker)

| N | ndcg@10 | recall@50 | p50 ms | p95 ms |
|---|---|---|---|---|
| 50 | 0.2460 | 0.3825 | 2.3 | 3.6 |
| 100 | 0.2422 | 0.3951 | 2.8 | 4.4 |
| 200 | 0.2395 | 0.3970 | 3.0 | 4.4 |
| 500 | 0.2340 | 0.3909 | 4.9 | 5.9 |

No-retrieval ablation (300 users): rank whole catalogue ndcg@10=0.1922, p50=8.4 ms  vs  retrieval N=200: ndcg@10=0.2168

Pointwise LR coefficients (standardised features): {'retrieval_score': 0.4564, 'retrieval_rank': -0.434, 'itemknn_score': 0.1913, 'log_pop': -0.1938, 'log_recent_pop': 0.4068, 'genre_match': 0.1535, 'item_age': -0.0542, 'log_user_activity': -0.0467, 'from_popular_source': 0.0043}
