"""Generate the deterministic instance pools used by SFT, GRPO and evaluation.

Splits follow the official TextWorldExpress folds:
  train   seeds 0-999
  dev     seeds 10000-10999
  test    seeds 20000-20999

Sorting keeps every train instance (size 3/4/5). Map Reader scans the fixed
27-parameter grid and selects a balanced set of 1000 instances per distance
bucket with even spacing over the candidate list, which keeps all parameter
combinations represented.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data.io_utils import write_json, write_jsonl  # noqa: E402
from code.env.twx import TWX_VERSION, TWXEnv, verify_jar  # noqa: E402
from code.paths import INSTANCES  # noqa: E402

MAP_PARAMS: list[tuple[int, int, int]] = [
    (num_locations, max_distance, distractors)
    for num_locations in (6, 8, 10)
    for max_distance in (1, 2, 3)
    for distractors in (0, 1, 2)
]

TRAIN_MAP_QUOTAS = {1: 1000, 2: 1000, 3: 1000}
DEV_MAP_QUOTAS = {1: 34, 2: 33, 3: 33}
TEST_MAP_QUOTAS = {1: 34, 2: 33, 3: 33}
DEV_SORT_QUOTAS = {3: 34, 4: 33, 5: 33}
TEST_SORT_QUOTAS = {3: 34, 4: 33, 5: 33}


def params_string(num_locations: int, max_distance: int, distractors: int) -> str:
    return (
        f"numLocations={num_locations},"
        f"maxDistanceApart={max_distance},"
        f"maxDistractorItemsPerLocation={distractors}"
    )


def _sorting_record(env: TWXEnv, split: str, seed: int) -> dict:
    step = env.reset("sorting", seed, fold=split)
    size = int(env.properties["hidden_problemsize"])
    return {
        "id": f"sorting-{split}-{seed:06d}",
        "game": "sorting",
        "params": "",
        "seed": seed,
        "fold": split,
        "difficulty": {"bucket": f"size{size}", "size": size},
        "task": env.task,
        "observation": step.observation,
        "inventory": step.inventory,
        "valid_actions": step.valid_actions,
    }


def _mapreader_record(
    env: TWXEnv, split: str, index: int, seed: int, params: str
) -> dict:
    step = env.reset("mapreader", seed, fold=split, params=params)
    distance = int(env.properties["actualDistanceApart"])
    num_locations, max_distance, distractors = MAP_PARAMS[index]
    return {
        "id": f"mapreader-{split}-{index:02d}-{seed:06d}",
        "game": "mapreader",
        "params": params,
        "seed": seed,
        "fold": split,
        "difficulty": {
            "bucket": f"d{distance}",
            "distance": distance,
            "num_locations": num_locations,
            "max_distance": max_distance,
            "distractors": distractors,
        },
        "task": env.task,
        "observation": step.observation,
        "inventory": step.inventory,
        "valid_actions": step.valid_actions,
    }


def collect_sorting(env: TWXEnv, split: str, seeds: range) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = {3: [], 4: [], 5: []}
    for seed in seeds:
        record = _sorting_record(env, split, seed)
        buckets[record["difficulty"]["size"]].append(record)
    return buckets


def collect_mapreader(env: TWXEnv, split: str, seeds: range) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for index, (num_locations, max_distance, distractors) in enumerate(MAP_PARAMS):
        params = params_string(num_locations, max_distance, distractors)
        for seed in seeds:
            record = _mapreader_record(env, split, index, seed, params)
            distance = record["difficulty"]["distance"]
            if distance in buckets:
                buckets[distance].append(record)
    return buckets


def select_even(items: list[dict], quota: int, label: str) -> list[dict]:
    if len(items) < quota:
        raise RuntimeError(
            f"{label}: only {len(items)} candidates, need {quota}. "
            "Extend the scan range instead of silently shrinking the dataset."
        )
    if len(items) == quota:
        return list(items)
    indices = [i * len(items) // quota for i in range(quota)]
    if len(set(indices)) != quota:
        raise RuntimeError(f"{label}: non-unique selection indices")
    return [items[i] for i in indices]


def _sorted(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda record: record["id"])


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = TWXEnv()
    try:
        print("[instances] scanning sorting train (1000 seeds) ...")
        sorting_train = collect_sorting(env, "train", range(1000))
        sorting_train_rows = _sorted(
            sorting_train[3] + sorting_train[4] + sorting_train[5]
        )
        write_jsonl(out_dir / "sorting_train.jsonl", sorting_train_rows)
        print(f"[instances] sorting_train: {len(sorting_train_rows)} rows")

        print("[instances] scanning mapreader train (27000 resets) ...")
        map_train = collect_mapreader(env, "train", range(1000))
        map_train_rows: list[dict] = []
        for distance, quota in TRAIN_MAP_QUOTAS.items():
            selected = select_even(
                map_train[distance], quota, f"mapreader train d={distance}"
            )
            map_train_rows.extend(selected)
        map_train_rows = _sorted(map_train_rows)
        write_jsonl(out_dir / "mapreader_train.jsonl", map_train_rows)
        print(f"[instances] mapreader_train: {len(map_train_rows)} rows")

        print("[instances] scanning dev (10000-10999 / 10000-10199) ...")
        dev_sort = collect_sorting(env, "dev", range(10000, 11000))
        dev_rows: list[dict] = []
        for size, quota in DEV_SORT_QUOTAS.items():
            dev_rows.extend(
                select_even(dev_sort[size], quota, f"dev sorting size={size}")
            )
        dev_map = collect_mapreader(env, "dev", range(10000, 10200))
        for distance, quota in DEV_MAP_QUOTAS.items():
            dev_rows.extend(
                select_even(dev_map[distance], quota, f"dev mapreader d={distance}")
            )
        dev_rows = _sorted(dev_rows)
        write_jsonl(out_dir / "dev.jsonl", dev_rows)
        print(f"[instances] dev: {len(dev_rows)} rows")

        print("[instances] scanning test (20000-20999 / 20000-20199) ...")
        test_sort = collect_sorting(env, "test", range(20000, 21000))
        test_rows: list[dict] = []
        for size, quota in TEST_SORT_QUOTAS.items():
            test_rows.extend(
                select_even(test_sort[size], quota, f"test sorting size={size}")
            )
        test_map = collect_mapreader(env, "test", range(20000, 20200))
        for distance, quota in TEST_MAP_QUOTAS.items():
            test_rows.extend(
                select_even(test_map[distance], quota, f"test mapreader d={distance}")
            )
        test_rows = _sorted(test_rows)
        write_jsonl(out_dir / "test.jsonl", test_rows)
        print(f"[instances] test: {len(test_rows)} rows")
    finally:
        env.close()

    def sort_counter(rows: list[dict], key: str) -> dict[str, int]:
        counter = Counter(record["difficulty"][key] for record in rows)
        return {str(k): counter[k] for k in sorted(counter)}

    meta = {
        "env": {
            "package": "textworld_express",
            "version": TWX_VERSION,
            "jar_sha256": verify_jar(),
        },
        "mapreader_params": [
            {
                "index": index,
                "num_locations": values[0],
                "max_distance": values[1],
                "distractors": values[2],
            }
            for index, values in enumerate(MAP_PARAMS)
        ],
        "splits": {
            "train": {
                "sorting": {
                    "total": len(sorting_train_rows),
                    "sizes": sort_counter(sorting_train_rows, "size"),
                },
                "mapreader": {
                    "total": len(map_train_rows),
                    "distances": sort_counter(map_train_rows, "distance"),
                },
            },
            "dev": {
                "total": len(dev_rows),
                "sorting": sort_counter(
                    [r for r in dev_rows if r["game"] == "sorting"], "size"
                ),
                "mapreader": sort_counter(
                    [r for r in dev_rows if r["game"] == "mapreader"], "distance"
                ),
            },
            "test": {
                "total": len(test_rows),
                "sorting": sort_counter(
                    [r for r in test_rows if r["game"] == "sorting"], "size"
                ),
                "mapreader": sort_counter(
                    [r for r in test_rows if r["game"] == "mapreader"], "distance"
                ),
            },
        },
        "selection": {
            "sorting_train": "all seeds 0-999",
            "mapreader_train": "27 params x 1000 seeds, even-spaced 1000 per distance",
            "sorting_dev": "seeds 10000-10999, 34/33/33 by size",
            "mapreader_dev": "seeds 10000-10199 x 27 params, 34/33/33 by distance",
            "sorting_test": "seeds 20000-20999, 34/33/33 by size",
            "mapreader_test": "seeds 20000-20199 x 27 params, 34/33/33 by distance",
        },
    }
    write_json(out_dir / "meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=INSTANCES)
    args = parser.parse_args()
    meta = build(args.out_dir)
    print(f"[instances] wrote {args.out_dir}")
    print(f"[instances] meta: {meta['splits']}")


if __name__ == "__main__":
    main()
