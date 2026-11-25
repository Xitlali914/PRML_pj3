#!/usr/bin/env bash
# One-time server setup: user-level JRE + model weights inside the repo.
set -euo pipefail

REPO="${REPO:-$HOME/PRML_pj3}"
PY="$HOME/.venv/bin/python"

uv pip install --python "$PY" jdk4py
JAVA="$($PY -c 'import pathlib, jdk4py; print(pathlib.Path(jdk4py.JAVA_HOME) / "bin" / "java")')"
test -x "$JAVA"
"$JAVA" -version

if [ ! -d "$REPO/model/Qwen3-4B" ]; then
  cp -a /shared/models/Qwen3-4B "$REPO/model/Qwen3-4B"
fi
du -sh "$REPO/model/Qwen3-4B"
