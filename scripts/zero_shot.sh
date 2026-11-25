#!/usr/bin/env bash
# Zero-shot baseline on the dev split (single GPU, short debug allocation).
set -euo pipefail
source "$HOME/PRML_pj3/scripts/env.sh"

srun -D "$REPO" -p debug --gres=gpu:1 -t 20 "$PY" -m code.eval.evaluate \
  --model-dir model/Qwen3-4B \
  --instances data/instances/dev.jsonl \
  --output results/zero_shot_dev.jsonl \
  --device cuda
