#!/usr/bin/env bash
# Local test split through the official run.py entry point.
set -euo pipefail
source "$HOME/PRML_pj3/scripts/env.sh"

srun -D "$REPO" -p debug --gres=gpu:1 -t 20 "$PY" run.py \
  --model-dir model/grpo/merged \
  --instances data/instances/test.jsonl \
  --output results/final_test.jsonl \
  --device cuda
