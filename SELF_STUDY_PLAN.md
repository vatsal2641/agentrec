# SELF_STUDY_PLAN.md: how to continue without me

The code is finished and tested, and the experiments run end to end. What's left is for **you to
own it**.

**Time budget, assuming you start from zero in recommender systems and agents:**
- **Recommended:** about 6 weeks at 2 hours a day, 6 days a week (roughly 70 hours).
- **Compressed:** about 4 weeks at 3 hours a day, if interviews are close.
- Less than about 1.5 hours a day doesn't work well. The maths and the code need unbroken focus.

**Every 2-hour day has the same shape:**
- 45 minutes reading the LEARNING_NOTES section, with pen and paper for the toy examples.
- 45 minutes of code: the exercise first, then read the real module.
- 30 minutes on the quiz or interview questions *out loud*. Log misses in `INTERVIEW_WEAKNESSES.md`
  and tick `LEARNING_PROGRESS.md`.

**Rule:** if you fail a Gate, repeat that week's weak days before moving on. Don't put this repo on
your resume, or make it public, until you've passed Gate 5.

## Prerequisites check (day 0, 1 hour)
The notes assume you're comfortable with the items below. Test yourself. If any one takes more than
a few minutes, spend 1–2 extra days on it before Week 1.
- Python + numpy: indexing, broadcasting, `argsort`, sparse matrices (scipy `csr_matrix`).
- Linear algebra: dot product, cosine similarity, matrix multiply, matrix inverse.
- Probability: Bernoulli, expectation, Bayes' rule, the Beta distribution (what Beta(3, 9) looks
  like).
- ML: logistic regression, cross-entropy loss, gradient descent, overfitting, train/validation/test
  sets.

Good refreshers:
- 3Blue1Brown, *Essence of Linear Algebra* (YouTube).
- Google's free *Recommendation Systems* course, at developers.google.com/machine-learning/recommendation.
  It's short and a good first overview before §1.



## Week 0: setup (one evening, about 2–3 hours)
1. Create an **empty private** GitHub repo called `agentrec`. Don't add a README or a licence.
2. Unzip the delivered archive (it contains the git history) and run:
   ```bash
   cd agentrec
   git config user.name "<your name>"
   git config user.email "<id>+<username>@users.noreply.github.com"
   git rebase -r --root --exec "git commit --amend --no-edit --reset-author"   # makes you the author
   git remote add origin https://github.com/<username>/agentrec.git
   git push -u origin main
   ```
   The `Co-Authored-By: Claude` lines in the commit messages record honestly how the code was
   written. Keep them.
3. Set up the environment:
   `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
4. Run `pytest -q`. All 59 tests should pass.
5. Download MovieLens-1M (see `data/README.md`) and start `bash scripts/run_all.sh configs/ml1m.yaml`.
   Let it run in the background; expect roughly 30–60 minutes on a laptop CPU. Commit
   `results/ml1m/*.md`.
6. Run `python scripts/demo.py configs/synthetic.yaml` and play with it for 15 minutes. Don't try to
   understand it yet.

## Week 1: what a recommender is (LEARNING_NOTES §1–2)
- Day 1: §1.1–1.4. Do exercise 01.
- Day 2: §1.5–1.10. Draw the two-stage architecture from memory.
- Day 3: §1.11–1.15, then Quiz 1.
- Day 4: §2 (splits, leakage, negatives). Do exercise 03.
- Day 5: read `data/dataset.py` line by line. Explain each line out loud.
- Day 6: Quiz 2. Redo anything you missed. **Gate 1:** INTERVIEW_PREP L1 Q1–Q5 out loud, recorded.

## Week 2: baselines and metrics (§3, §5.3)
- Day 1: §5.3 metrics. Do exercise 02.
- Day 2: recompute one row of the E1 table by hand for 2 users (write a tiny script).
- Day 3: §3. Read `models/baselines.py`.
- Day 4: exercise 04 (the BPR gradient). Read `models/mf_bpr.py`.
- Day 5: change BPR's `reg` and `dim`, rerun E1, and explain the change *before* looking at it.
- Day 6: Quiz 3. **Gate 2:** L2 Q9–Q10, L3 Q14 and Q19.

## Week 3: retrieval and ranking (§4–5)
- Day 1: §4.1. Derive the two-tower backprop on paper.
- Day 2: break one line of `loss_and_grads` and watch the gradient test fail. Explain why it fails.
- Day 3: §4.2. Do exercise 05. Rerun E2 with different `n_lists` and `n_probe` values.
- Day 4: §5.1–5.2. Read `ranking/ranker.py`.
- Day 5: add one ranking feature (for example user–item decade distance) and rerun E3. Did it help,
  and why?
- Day 6: Quizzes 4–5. **Gate 3:** L1 Q1, L2 Q8, L3 Q20, L6 Q33.

## Week 4: bandits (§6). The hardest week, so go slowly.
- Day 1: §6.1–6.4. Do the UCB and Beta toy numbers by hand.
- Day 2: exercise 06. Read `bandits/mab.py`. Run the toy bandit with 3 different gap sizes.
- Day 3: §6.5–6.6. Read `bandits/linear.py`. Verify Sherman–Morrison on paper for d = 2.
- Day 4: §6.7–6.10. Read `bandits/env.py` and `run_policy` line by line.
- Day 5: change α and λ, *predict* the result, then run. Explain why greedy ≈ LinUCB here
  (EXPERIMENTS E4).
- Day 6: Quiz 6. **Gate 4:** L3 Q15–Q18, L5 Q25 and Q27, L8 B3.

## Week 5: feedback, LLM, agent, memory (§7–10)
- Day 1: §7. Do exercise 07. Trace one click through `FeedbackLoop.process` in a debugger.
- Day 2: §8–9. Read `agents/tools.py` and `agents/agent.py`. Do exercise 08.
- Day 3: install Ollama and a small tool-calling model (e.g. `qwen2.5:7b-instruct`). Run
  `python scripts/demo.py configs/ml1m.yaml --llm http://localhost:11434/v1 qwen2.5:7b-instruct`.
  If the client breaks, fixing it is part of the learning (it was never tested live).
- Day 4: run `python scripts/run_agent_eval.py configs/ml1m.yaml --n 50 --llm http://localhost:11434/v1 qwen2.5:7b-instruct`.
  Write down what you actually see.
- Day 5: write 20 *messy* requests the regex parser can't handle. Compare both planners by hand.
  This is the real test of the "agentic" claim.
- Day 6: §10 and Quizzes 7–10. L2 Q11, L1 Q6, L8 B5.

## Week 6: own it
- Day 1: update `EXPERIMENTS.md` and `RESUME.md` with your ML-1M numbers and your LLM findings.
- Days 2–3: exercise 09, the capstone. Rebuild the pipeline from memory in a single file of fewer
  than 300 lines.
- Day 4: the code-defence drill on 5 random files.
- Day 5: a full mock interview, all levels. Ask a friend, or use the tutor prompt below.
- Day 6: **Gate 5:** whiteboard "design an agentic recommender" in 20 minutes without notes (§11).
  Only after passing: put it on your resume and make the repo public.

## Continuing with another assistant (optional)
If you use any AI tutor later, paste this prompt:
> "I'm studying my repo AgentRec (two-tower retrieval + ranker + LinUCB bandit + tool-calling agent).
> Quiz me using INTERVIEW_PREP.md one question at a time, don't reveal answers, push back on vague
> answers, and log my misses in INTERVIEW_WEAKNESSES.md format."

## Honest boundaries (don't cross these on the resume)
- Don't say "real users", "production" or "deployed". Say "simulated users" for all bandit and
  feedback numbers.
- Don't say "LLM-based reasoning improved accuracy" unless your own `--llm` run shows it.
- The synthetic numbers in `results/synthetic/` are for development only. Quote ML-1M numbers.
