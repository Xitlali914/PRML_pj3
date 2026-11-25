"""One-command data pipeline: generate, checksum, validate.

The pipeline is deterministic: running it on another machine with the same
vendored runtime must reproduce byte-identical files. The committed
``data/expected_checksums.json`` is compared on every run; a mismatch stops the
pipeline before training.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data import (  # noqa: E402
    build_expert,
    build_sft_turns,
    checksums,
    generate_instances,
    validate_data,
)
from code.paths import DATA, INSTANCES, SFT  # noqa: E402


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    expected_path = DATA / "expected_checksums.json"

    print("== 1/4 instances ==")
    generate_instances.build(INSTANCES)
    print("== 2/4 expert trajectories ==")
    build_expert.build(INSTANCES, SFT)
    print("== 3/4 SFT turns ==")
    build_sft_turns.build(INSTANCES, SFT)

    actual = checksums.compute(DATA)
    if expected_path.exists():
        problems = checksums.compare(expected_path, actual)
        if problems:
            print("[generate_all] checksum mismatch against the committed expected file:")
            for problem in problems:
                print(f"  - {problem}")
            raise SystemExit(1)
        print("[generate_all] checksums match the committed expected file")
    else:
        checksums.write_checksums(DATA, expected_path)
        print(f"[generate_all] wrote {expected_path}")

    print("== 4/4 validation ==")
    report = validate_data.validate(DATA, INSTANCES, SFT, expected_path)
    if not report["passed"]:
        for error in report["errors"]:
            print(f"[generate_all] ERROR: {error}")
        raise SystemExit(1)
    print("[generate_all] data pipeline finished successfully")
    print(f"[generate_all] counts: {report['counts']}")


if __name__ == "__main__":
    main()
