#!/usr/bin/env bash
# Reproduce every experiment for one config. usage: bash scripts/run_all.sh configs/ml1m.yaml
set -euo pipefail
CFG=${1:-configs/synthetic.yaml}
if [[ "$CFG" == *synthetic* ]]; then python scripts/make_synthetic_data.py; fi
python scripts/prepare_data.py "$CFG"
python scripts/run_baselines.py "$CFG"      # E1
python scripts/run_ranking.py "$CFG"        # E3 (also trains + caches stage-1 models and the ranker)
python scripts/run_retrieval.py "$CFG"      # E2
python scripts/run_bandits.py "$CFG"        # E4-E6 (simulated)
python scripts/run_agent_eval.py "$CFG"     # E8
python scripts/run_latency.py "$CFG"
python scripts/make_plots.py "$CFG" || echo "plots skipped (matplotlib missing?)"
python scripts/collect_results.py "$CFG"
if [[ "$CFG" == *synthetic* ]]; then python scripts/build_experiments_md.py; fi
