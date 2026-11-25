#!/usr/bin/env bash
# Generate and validate the deterministic dataset on a CPU allocation.
set -euo pipefail
source "$HOME/PRML_pj3/scripts/env.sh"

srun -p debug -c 4 -t 10 bash -lc "cd '$REPO' && '$PY' -m code.data.generate_all"
