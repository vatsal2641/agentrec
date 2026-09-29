# E2 — retrieval on `synthetic` (TEST positives, warm users=742, cold=63)

| N | embedding only | popular only | union (90/10) | cold users (popular) |
|---|---|---|---|---|
| 50 | 0.3883 | 0.1431 | 0.3825 | 0.1780 |
| 100 | 0.5435 | 0.2228 | 0.5305 | 0.2846 |
| 200 | 0.6964 | 0.3436 | 0.6867 | 0.4156 |
| 500 | 0.8753 | 0.5929 | 0.8657 | 0.6187 |

| index | ANN recall@100 vs exact | items scanned | p50 ms | p95 ms |
|---|---|---|---|---|
| exact | 1.000 | 1.000 | 0.031 | 0.047 |
| ivf_nlist64_nprobe1 | 0.354 | 0.016 | 0.033 | 0.056 |
| ivf_nlist64_nprobe2 | 0.543 | 0.031 | 0.036 | 0.060 |
| ivf_nlist64_nprobe4 | 0.722 | 0.062 | 0.038 | 0.072 |
| ivf_nlist64_nprobe8 | 0.866 | 0.125 | 0.042 | 0.079 |
| ivf_nlist64_nprobe16 | 0.958 | 0.250 | 0.086 | 0.166 |
| ivf_nlist64_nprobe32 | 0.995 | 0.500 | 0.140 | 0.229 |

Brute-force dot product: 9.20 ns/item measured -> ~92 ms/query at 10M items (EXTRAPOLATED).
