"""Central path configuration for PRML Project 3.

Every entry point resolves paths through this module so that moving the
repository (or the weights) only requires changing one place or passing the
documented CLI overrides.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
DATA = ROOT / "data"
MODEL = ROOT / "model"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

INSTANCES = DATA / "instances"
SFT = DATA / "sft"

BASE_MODEL_DIR = MODEL / "Qwen3-4B"
SFT_DIR = MODEL / "sft"
GRPO_DIR = MODEL / "grpo"


def ensure_output_dirs() -> None:
    for path in (DATA, INSTANCES, SFT, MODEL, RUNS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
