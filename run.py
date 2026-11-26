#!/usr/bin/env python
"""Official evaluation entry point (placeholder).

The course has not released the final test-file format yet, so this entry point
is intentionally not implemented. The local test split can be evaluated with
``python -m code.eval.evaluate``; see README.md.

When the official format is available, implement:
  * instance loading and field mapping,
  * the same greedy 10-action episode loop as ``code.eval.evaluate``,
  * metrics and per-episode trajectory output.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Official evaluation entry point (not implemented yet)."
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--instances", default=None)
    parser.add_argument("--output", default=None)
    parser.parse_args(argv)
    print(
        "run.py is reserved for the official test configuration, which has not "
        "been released yet.",
        file=sys.stderr,
    )
    print(
        "Use the local test split instead:\n"
        "  python -m code.eval.evaluate --model-dir model/grpo/merged "
        "--instances data/instances/test.jsonl "
        "--output results/final_test.jsonl --device cuda",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
