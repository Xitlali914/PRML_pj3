"""Validate generated data and replay every expert trajectory.

This script is the gate between data generation and training. It checks
structure, split hygiene, the SFT turn expansion and, most importantly, that
every expert trajectory can be replayed from a fresh environment and still
finishes successfully.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data import checksums  # noqa: E402
from code.data.io_utils import read_jsonl, write_json  # noqa: E402
from code.env.twx import TWXEnv  # noqa: E402
from code.paths import DATA, INSTANCES, SFT  # noqa: E402

EXPECTED_INSTANCE_COUNTS = {
    "sorting_train": 1000,
    "mapreader_train": 3000,
    "dev": 200,
    "test": 200,
}
EXPECTED_TRAJECTORIES = {"sorting": 600, "mapreader": 900}
EXPECTED_TOTAL_TURNS = 11100


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _check_instances(
    path: Path, split: str, errors: list[str]
) -> list[dict]:
    rows = read_jsonl(path)
    _require(len(rows) == EXPECTED_INSTANCE_COUNTS[split], f"{path.name}: wrong count", errors)
    seen: set[str] = set()
    for row in rows:
        for key in ("id", "game", "params", "seed", "fold", "difficulty", "task",
                    "observation", "inventory", "valid_actions"):
            _require(key in row, f"{row.get('id', '?')}: missing {key}", errors)
        _require(row["id"] not in seen, f"duplicate id {row['id']}", errors)
        seen.add(row["id"])
        _require(row["fold"] == ("train" if split.endswith("_train") else split),
                 f"{row['id']}: wrong fold", errors)
        _require(isinstance(row["valid_actions"], list) and row["valid_actions"],
                 f"{row['id']}: bad valid_actions", errors)
        _require(isinstance(row["inventory"], list), f"{row['id']}: inventory not list", errors)
        if row["game"] == "sorting":
            _require(row["difficulty"]["size"] in (3, 4, 5),
                     f"{row['id']}: bad sorting size", errors)
        elif row["game"] == "mapreader":
            _require(row["difficulty"]["distance"] in (1, 2, 3),
                     f"{row['id']}: bad map distance", errors)
        else:
            errors.append(f"{row['id']}: unknown game")
        if split in ("sorting_train", "mapreader_train"):
            _require(0 <= row["seed"] < 1000,
                     f"{row['id']}: train seed out of range", errors)
        if split == "dev":
            _require(10000 <= row["seed"] < 11000, f"{row['id']}: dev seed out of range", errors)
        if split == "test":
            _require(20000 <= row["seed"] < 21000, f"{row['id']}: test seed out of range", errors)
    return rows


def _expected_steps(instance: dict) -> int:
    if instance["game"] == "sorting":
        return 2 * int(instance["difficulty"]["size"])
    return 2 * int(instance["difficulty"]["distance"]) + 3


def _replay_trajectories(
    instances: dict[str, dict], trajectories: list[dict], errors: list[str]
) -> None:
    env = TWXEnv()
    try:
        for trajectory in trajectories:
            instance = instances.get(trajectory["instance_id"])
            if instance is None:
                errors.append(f"{trajectory['id']}: unknown instance")
                continue
            step = env.reset(
                instance["game"],
                instance["seed"],
                fold=instance["fold"],
                params=instance["params"],
            )
            _require(step.observation == instance["observation"],
                     f"{trajectory['id']}: initial observation mismatch", errors)
            for turn_index, turn in enumerate(trajectory["turns"]):
                _require(turn["valid_actions"] == env.valid_actions,
                         f"{trajectory['id']} t{turn_index}: valid_actions mismatch", errors)
                _require(turn["observation"] == env.observation,
                         f"{trajectory['id']} t{turn_index}: observation mismatch", errors)
                _require(turn["inventory"] == env.inventory,
                         f"{trajectory['id']} t{turn_index}: inventory mismatch", errors)
                _require(turn["action"] in env.valid_actions,
                         f"{trajectory['id']} t{turn_index}: action not valid", errors)
                step = env.step(turn["action"])
                _require(not step.invalid,
                         f"{trajectory['id']} t{turn_index}: invalid action", errors)
            _require(bool(step.success), f"{trajectory['id']}: did not succeed", errors)
            _require(abs(step.score - 1.0) < 1e-9, f"{trajectory['id']}: score != 1", errors)
            _require(len(trajectory["turns"]) == _expected_steps(instance),
                     f"{trajectory['id']}: unexpected step count", errors)
            if instance["game"] == "mapreader":
                _require(trajectory["turns"][0]["action"] == "read map",
                         f"{trajectory['id']}: first action is not read map", errors)
    finally:
        env.close()


def _check_turns(
    instances: dict[str, dict],
    trajectories: dict[str, dict],
    train_rows: list[dict],
    val_rows: list[dict],
    errors: list[str],
) -> None:
    _require(len(train_rows) + len(val_rows) == EXPECTED_TOTAL_TURNS,
             "turn count != 11100", errors)
    seen: set[str] = set()
    for row in train_rows + val_rows:
        _require(row["id"] not in seen, f"duplicate turn id {row['id']}", errors)
        seen.add(row["id"])
        _require(row["target_action"] in row["valid_actions"],
                 f"{row['id']}: target not in valid_actions", errors)
        _require(len(row["history"]) == row["turn_index"],
                 f"{row['id']}: history length mismatch", errors)
        trajectory = trajectories.get(row["trajectory_id"])
        if trajectory is None:
            errors.append(f"{row['id']}: unknown trajectory")
            continue
        turn = trajectory["turns"][row["turn_index"]]
        _require(row["target_action"] == turn["action"],
                 f"{row['id']}: target != trajectory action", errors)
        _require(row["observation"] == turn["observation"],
                 f"{row['id']}: observation != trajectory", errors)
        for history_index, item in enumerate(row["history"]):
            _require(item["action"] == trajectory["turns"][history_index]["action"],
                     f"{row['id']}: history action mismatch", errors)
            _require(
                item["observation"] == trajectory["turns"][history_index + 1]["observation"],
                f"{row['id']}: history observation mismatch",
                errors,
            )
        instance = instances[row["instance_id"]]
        _require(row["difficulty"] == instance["difficulty"],
                 f"{row['id']}: difficulty mismatch", errors)


def validate(
    data_dir: Path = DATA,
    instances_dir: Path = INSTANCES,
    sft_dir: Path = SFT,
    expected_path: Path | None = None,
) -> dict:
    errors: list[str] = []
    instance_sets = {
        "sorting_train": _check_instances(instances_dir / "sorting_train.jsonl",
                                          "sorting_train", errors),
        "mapreader_train": _check_instances(instances_dir / "mapreader_train.jsonl",
                                            "mapreader_train", errors),
        "dev": _check_instances(instances_dir / "dev.jsonl", "dev", errors),
        "test": _check_instances(instances_dir / "test.jsonl", "test", errors),
    }
    for name, rows in instance_sets.items():
        if len(rows) != EXPECTED_INSTANCE_COUNTS[name]:
            errors.append(f"{name}: expected {EXPECTED_INSTANCE_COUNTS[name]}, got {len(rows)}")

    instances = {
        row["id"]: row
        for rows in instance_sets.values()
        for row in rows
    }

    sorting_trajectories = read_jsonl(sft_dir / "sorting_expert.jsonl")
    map_trajectories = read_jsonl(sft_dir / "mapreader_expert.jsonl")
    for game, trajectories in (("sorting", sorting_trajectories),
                               ("mapreader", map_trajectories)):
        _require(len(trajectories) == EXPECTED_TRAJECTORIES[game],
                 f"{game}: expected {EXPECTED_TRAJECTORIES[game]} trajectories, "
                 f"got {len(trajectories)}", errors)
        for trajectory in trajectories:
            _require(trajectory["game"] == game, f"{trajectory['id']}: wrong game", errors)
            _require(trajectory["success"] is True, f"{trajectory['id']}: not successful", errors)
            _require(trajectory["score"] == 1.0, f"{trajectory['id']}: score not 1.0", errors)

    trajectories = {row["id"]: row for row in sorting_trajectories + map_trajectories}
    _replay_trajectories(instances, sorting_trajectories + map_trajectories, errors)

    train_rows = read_jsonl(sft_dir / "turns_train.jsonl")
    val_rows = read_jsonl(sft_dir / "turns_val.jsonl")
    _check_turns(instances, trajectories, train_rows, val_rows, errors)

    ordered_trajectories = sorted(trajectories)
    for index, trajectory_id in enumerate(ordered_trajectories):
        expected_split = "val" if index % 20 == 0 else "train"
        rows = [
            row for row in train_rows + val_rows if row["trajectory_id"] == trajectory_id
        ]
        for row in rows:
            _require(row["split"] == expected_split,
                     f"{row['id']}: split {row['split']} != {expected_split}", errors)
        _require(len(rows) == len(trajectories[trajectory_id]["turns"]),
                 f"{trajectory_id}: not all turns expanded", errors)

    hash_problems: list[str] = []
    if expected_path is not None and expected_path.exists():
        hash_problems = checksums.compare(expected_path, checksums.compute(data_dir))
        errors.extend(hash_problems)

    report = {
        "passed": not errors,
        "errors": errors,
        "counts": {
            "instances": {name: len(rows) for name, rows in instance_sets.items()},
            "trajectories": {
                "sorting": len(sorting_trajectories),
                "mapreader": len(map_trajectories),
            },
            "turns": {"train": len(train_rows), "val": len(val_rows)},
        },
        "hash_check": {
            "expected_file": str(expected_path) if expected_path else None,
            "checked": bool(expected_path and expected_path.exists()),
            "problems": hash_problems,
        },
        "difficulty": {
            "sorting": dict(sorted(Counter(
                row["difficulty"]["size"]
                for row in instance_sets["sorting_train"]
            ).items())),
            "mapreader": dict(sorted(Counter(
                row["difficulty"]["distance"]
                for row in instance_sets["mapreader_train"]
            ).items())),
        },
    }
    write_json(data_dir / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES)
    parser.add_argument("--sft-dir", type=Path, default=SFT)
    parser.add_argument(
        "--expected",
        type=Path,
        default=DATA / "expected_checksums.json",
        help="checksum file to verify (skipped when missing)",
    )
    args = parser.parse_args()
    report = validate(args.data_dir, args.instances_dir, args.sft_dir, args.expected)
    if not report["passed"]:
        for error in report["errors"]:
            print(f"[validate] ERROR: {error}")
        raise SystemExit(1)
    print(f"[validate] OK: {report['counts']}")


if __name__ == "__main__":
    main()
