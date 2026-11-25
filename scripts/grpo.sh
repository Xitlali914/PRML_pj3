#!/usr/bin/env bash
#SBATCH --job-name=prml-grpo
#SBATCH --chdir=/home/g19/PRML_pj3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/grpo-%j.log

set -euo pipefail
source "$HOME/PRML_pj3/scripts/env.sh"

ARGS=(
  --base-model model/Qwen3-4B
  --instances-dir data/instances
  --output-dir model/grpo
)

if [ -d model/grpo/latest ]; then
  echo "resuming from model/grpo/latest"
  ARGS+=(--resume-dir model/grpo/latest)
else
  ARGS+=(--sft-adapter model/sft/adapter)
fi

"$PY" -m code.train.grpo "${ARGS[@]}"
