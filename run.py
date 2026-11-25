#!/usr/bin/env python
"""Official evaluation entry point.

Example:
    python run.py --model-dir model/grpo/merged \
        --instances data/instances/test.jsonl --output results/final_test.jsonl
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.eval.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
