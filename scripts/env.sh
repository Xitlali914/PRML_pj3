#!/usr/bin/env bash
# Shared environment for the course server (run with `source`).
set -euo pipefail

export REPO="${REPO:-$HOME/PRML_pj3}"
export PY="$HOME/.venv/bin/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

JAVA_HOME="$($PY -c 'import jdk4py; print(jdk4py.JAVA_HOME)' 2>/dev/null || true)"
if [ -n "${JAVA_HOME}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
  export JAVA_HOME
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

cd "${REPO}"
mkdir -p logs results model
