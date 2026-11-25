"""Expand expert trajectories into turn-level SFT samples.

A turn sample is the context seen before an action plus the expert action.
Trajectories are split 95/5 into train/val by trajectory id (every 20th
trajectory goes to val) so that all turns of one episode stay on one side.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from code.paths import INSTANCES, SFT  # noqa: E402


def build(instances_dir: Path, sft_dir: Path, val_every: int = 20) -> dict:
    instances = {
        row["id"]: row
        for row in read_jsonl(instances_dir / "sorting_train.jsonl")
        + read_jsonl(instances_dir / "mapreader_train.jsonl")
    }
    trajectories = read_jsonl(sft_dir / "sorting_expert.jsonl") + read_jsonl(
        sft_dir / "mapreader_expert.jsonl"
    )
    trajectories.sort(key=lambda row: row["id"])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    val_trajectories = 0

    for trajectory_index, trajectory in enumerate(trajectories):
        instance = instances[trajectory["instance_id"]]
        split = "val" if trajectory_index % val_every == 0 else "train"
        if split == "val":
            val_trajectories += 1

        history: list[dict] = []
        turns = trajectory["turns"]
        for turn_index, turn in enumerate(turns):
            if turn["action"] not in turn["valid_actions"]:
                raise RuntimeError(
                    f"{trajectory['id']} turn {turn_index}: target action not valid: "
                    f"{turn['action']!r}"
                )
            row = {
                "id": f"{trajectory['id']}-t{turn_index:02d}",
                "instance_id": instance["id"],
                "trajectory_id": trajectory["id"],
                "turn_index": turn_index,
                "game": instance["game"],
                "difficulty": instance["difficulty"],
                "task": instance["task"],
                "observation": turn["observation"],
                "inventory": turn["inventory"],
                "valid_actions": turn["valid_actions"],
                "history": [dict(item) for item in history],
                "target_action": turn["action"],
                "split": split,
            }
            (val_rows if split == "val" else train_rows).append(row)
            if turn_index + 1 < len(turns):
                history.append(
                    {
                        "action": turn["action"],
                        "observation": turns[turn_index + 1]["observation"],
                    }
                )

    write_jsonl(sft_dir / "turns_train.jsonl", train_rows)
    write_jsonl(sft_dir / "turns_val.jsonl", val_rows)

    meta = {
        "trajectories": len(trajectories),
        "val_trajectories": val_trajectories,
        "turns": {
            "total": len(train_rows) + len(val_rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "by_game": {
            game: {
                "train": sum(1 for row in train_rows if row["game"] == game),
                "val": sum(1 for row in val_rows if row["game"] == game),
            }
            for game in ("sorting", "mapreader")
        },
        "val_rule": f"trajectory index % {val_every} == 0 (trajectories sorted by id)",
        "difficulty": {
            game: {
                str(bucket): count
                for bucket, count in sorted(
                    Counter(
                        row["difficulty"]["bucket"]
                        for row in train_rows + val_rows
                        if row["game"] == game
                    ).items()
                )
            }
            for game in ("sorting", "mapreader")
        },
    }
    write_json(sft_dir / "turns_meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES)
    parser.add_argument("--sft-dir", type=Path, default=SFT)
    parser.add_argument("--val-every", type=int, default=20)
    args = parser.parse_args()
    meta = build(args.instances_dir, args.sft_dir, args.val_every)
    print(f"[turns] {meta['turns']} by_game={meta['by_game']}")


if __name__ == "__main__":
    main()
