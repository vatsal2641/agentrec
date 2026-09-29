# Exercises: implement before you read the reference

Each file is a small, self-contained task. Every one has the same format:
- a docstring explaining the task
- function stubs that raise `NotImplementedError`
- asserts at the bottom that must pass

Run one with `python exercises/exNN_*.py`.

Rules:
1. **Do not open the reference file listed at the top of each exercise until your version passes.**
2. After it passes, diff your version against the reference and write down each difference in your
   own words. For each one, say whether it's a bug, a style choice, or a missing edge case.
3. Record anything you got wrong in `INTERVIEW_WEAKNESSES.md`.

| # | topic | reference in repo | LEARNING_NOTES |
|---|---|---|---|
| 01 | interaction matrix + popularity | `data/dataset.py`, `models/baselines.py` | §1 |
| 02 | ranking metrics | `evaluation/metrics.py` | §5.3 |
| 03 | temporal split + leakage check | `data/dataset.py::build_dataset` | §2 |
| 04 | one BPR SGD step (gradients) | `models/mf_bpr.py` | §3, §5.2 |
| 05 | IVF approximate search | `retrieval/index.py::IVFIndex` | §4.2 |
| 06 | UCB1, Thompson, LinUCB + Sherman–Morrison | `bandits/mab.py`, `bandits/linear.py` | §6 |
| 07 | feedback → user state rules | `memory/user_state.py` | §7 |
| 08 | agent loop + grounding guardrail | `agents/agent.py`, `agents/tools.py` | §8–9 |
| 09 | capstone: re-implement the whole pipeline in one file, from memory | whole repo | §11 |

For exercise 09, write `exercises/ex09_capstone.py` from scratch, with no copying. It must:
load the data, do a temporal split, fit BPR, retrieve top-100, re-rank with 3 features, run LinUCB
over 2,000 simulated rounds, and print NDCG@10 and cumulative regret. Aim for under 300 lines. If you
can do this in about 3 hours without looking, you own the project.
