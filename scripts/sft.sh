#!/usr/bin/env bash
#SBATCH --job-name=prml-sft
#SBATCH --chdir=/home/g19/PRML_pj3
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/home/g19/PRML_pj3/logs/sft-%j.log

set -euo pipefail
source "$HOME/PRML_pj3/scripts/env.sh"

"$PY" -m code.train.sft \
  --base-model model/Qwen3-4B \
  --output-dir model/sft
