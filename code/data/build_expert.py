"""Build verified expert trajectories for SFT cold start.

Sorting expert: parse the quantities from the valid `take` actions, convert
units to a common base within the instance's dimension, sort ascending, then
`take` / `put ... in box` one object at a time.

Map Reader expert: `read map`, parse the room graph, BFS to the coin room,
navigate using the directions visible in each room, take the coin, BFS back to
the start room, and put the coin in the box.

Every trajectory is replayed while it is built and is discarded unless it ends
with task success and the expected number of actions.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from code.env.twx import TWXEnv  # noqa: E402
from code.paths import INSTANCES, SFT  # noqa: E402

UNIT_TABLE = {
    "mg": ("mass", 1.0),
    "g": ("mass", 1_000.0),
    "kg": ("mass", 1_000_000.0),
    "ml": ("volume", 1.0),
    "l": ("volume", 1_000.0),
    "mm": ("length", 1.0),
    "cm": ("length", 10.0),
    "m": ("length", 1_000.0),
}
UNIT_RE = re.compile(r"^take\s+(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l|mm|cm|m)\s+of\s+(.+)$")
COUNT_RE = re.compile(r"^take\s+(\d+(?:\.\d+)?)\s+([a-z].+)$")

MAP_LINE_RE = re.compile(r"The (.+?) connects to (.+?)\.\s*$")
START_RE = re.compile(r"You are in the (.+?)\.")
TARGET_RE = re.compile(r"coin that is located in the (.+?),")
DIRECTION_RE = re.compile(
    r"To the (North|South|East|West) you see (?:the )?(.+?)\.", re.IGNORECASE
)


def parse_take(action: str) -> dict | None:
    unit_match = UNIT_RE.match(action)
    if unit_match:
        value = float(unit_match.group(1))
        unit = unit_match.group(2)
        dimension, multiplier = UNIT_TABLE[unit]
        return {
            "action": action,
            "object": action[len("take ") :].strip(),
            "value": value,
            "unit": unit,
            "dimension": dimension,
            "key": value * multiplier,
        }
    count_match = COUNT_RE.match(action)
    if count_match:
        value = float(count_match.group(1))
        return {
            "action": action,
            "object": action[len("take ") :].strip(),
            "value": value,
            "unit": None,
            "dimension": "count",
            "key": value,
        }
    return None


def snapshot(env: TWXEnv) -> dict:
    return {
        "observation": env.observation,
        "inventory": list(env.inventory),
        "valid_actions": list(env.valid_actions),
    }


def turn_record(before: dict, action: str, score_after: float) -> dict:
    return {
        "observation": before["observation"],
        "inventory": before["inventory"],
        "valid_actions": before["valid_actions"],
        "action": action,
        "score_after": score_after,
    }


def select_even(items: list[dict], quota: int) -> list[dict]:
    if len(items) < quota:
        raise RuntimeError(f"need {quota} items, only {len(items)} available")
    if len(items) == quota:
        return list(items)
    indices = [i * len(items) // quota for i in range(quota)]
    return [items[i] for i in indices]


def sorting_orders(takes: list[dict]) -> list[list[dict]]:
    ordered = sorted(takes, key=lambda item: item["key"])
    groups = [
        list(group) for _, group in itertools.groupby(ordered, key=lambda item: item["key"])
    ]
    permutations = [list(itertools.permutations(group)) for group in groups]
    orders: list[list[dict]] = []
    for combination in itertools.product(*permutations):
        orders.append([item for group in combination for item in group])
        if len(orders) >= 24:
            break
    return orders


def run_sorting(env: TWXEnv, instance: dict, order: list[dict]) -> list[dict] | None:
    env.reset("sorting", instance["seed"], fold=instance["fold"])
    turns: list[dict] = []
    step = None
    for item in order:
        take_action = item["action"]
        put_action = f"put {item['object']} in box"

        before = snapshot(env)
        if take_action not in before["valid_actions"]:
            return None
        step = env.step(take_action)
        turns.append(turn_record(before, take_action, step.score))
        if step.invalid or step.failure:
            return None

        before = snapshot(env)
        if put_action not in before["valid_actions"]:
            return None
        step = env.step(put_action)
        turns.append(turn_record(before, put_action, step.score))
        if step.invalid or step.failure:
            return None

    if step is None or not step.success or abs(step.score - 1.0) > 1e-9:
        return None
    return turns


def normalize_room(name: str) -> str:
    """Normalize room names.

    The map text prefixes the first room of a connection list with "the"
    (e.g. "connects to the launderette and pantry"), while observations
    sometimes omit it. Stripping the article makes the graph consistent.
    """
    name = name.strip().lower()
    while name.startswith("the "):
        name = name[4:]
    return name


def split_rooms(text: str) -> list[str]:
    rooms: list[str] = []
    for segment in text.split(", "):
        segment = segment.strip()
        if " and " in segment:
            rooms.extend(part.strip() for part in segment.split(" and "))
        else:
            rooms.append(segment)
    return [normalize_room(room) for room in rooms if room]


def parse_map(observation: str) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for line in observation.splitlines():
        match = MAP_LINE_RE.search(line.strip())
        if not match:
            continue
        room = normalize_room(match.group(1))
        for neighbour in split_rooms(match.group(2)):
            adjacency[room].add(neighbour)
            adjacency[neighbour].add(room)
    return dict(adjacency)


def bfs(adjacency: dict[str, set[str]], start: str, goal: str) -> list[str] | None:
    if start == goal:
        return [start]
    queue = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        for neighbour in sorted(adjacency.get(path[-1], ())):
            if neighbour in seen:
                continue
            next_path = path + [neighbour]
            if neighbour == goal:
                return next_path
            seen.add(neighbour)
            queue.append(next_path)
    return None


def find_direction(observation: str, room: str) -> str | None:
    target = normalize_room(room)
    for match in DIRECTION_RE.finditer(observation):
        if normalize_room(match.group(2)) == target:
            return match.group(1).lower()
    return None


def navigate(
    env: TWXEnv,
    adjacency: dict[str, set[str]],
    room_observation: str,
    current: str,
    goal: str,
    turns: list[dict],
) -> tuple[str, str] | None:
    path = bfs(adjacency, current, goal)
    if path is None:
        return None
    for next_room in path[1:]:
        direction = find_direction(room_observation, next_room)
        if direction is None:
            return None
        action = f"move {direction}"
        before = snapshot(env)
        if action not in before["valid_actions"]:
            return None
        step = env.step(action)
        turns.append(turn_record(before, action, step.score))
        if step.invalid:
            return None
        current = next_room
        room_observation = step.observation
    return current, room_observation


def run_mapreader(env: TWXEnv, instance: dict) -> list[dict] | None:
    env.reset("mapreader", instance["seed"], fold=instance["fold"], params=instance["params"])
    turns: list[dict] = []

    room_observation = env.observation
    before = snapshot(env)
    if "read map" not in before["valid_actions"]:
        return None
    step = env.step("read map")
    turns.append(turn_record(before, "read map", step.score))
    if step.invalid:
        return None
    adjacency = parse_map(step.observation)
    if not adjacency:
        return None

    start_match = START_RE.search(instance["observation"])
    target_match = TARGET_RE.search(instance["task"])
    if not start_match or not target_match:
        return None
    start_room = normalize_room(start_match.group(1))
    target_room = normalize_room(target_match.group(1))

    navigated = navigate(env, adjacency, room_observation, start_room, target_room, turns)
    if navigated is None:
        return None
    current, room_observation = navigated

    before = snapshot(env)
    if "take coin" not in before["valid_actions"]:
        return None
    step = env.step("take coin")
    turns.append(turn_record(before, "take coin", step.score))
    if step.invalid or step.failure:
        return None

    navigated = navigate(env, adjacency, room_observation, current, start_room, turns)
    if navigated is None:
        return None
    current, room_observation = navigated

    before = snapshot(env)
    if "put coin in box" not in before["valid_actions"]:
        return None
    step = env.step("put coin in box")
    turns.append(turn_record(before, "put coin in box", step.score))
    if step.invalid or step.failure:
        return None

    distance = int(instance["difficulty"]["distance"])
    if not step.success or len(turns) != 2 * distance + 3:
        return None
    return turns


def build(instances_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_instances = read_jsonl(instances_dir / "sorting_train.jsonl") + read_jsonl(
        instances_dir / "mapreader_train.jsonl"
    )
    sorting_instances = [row for row in train_instances if row["game"] == "sorting"]
    map_instances = [row for row in train_instances if row["game"] == "mapreader"]

    by_size: dict[int, list[dict]] = {3: [], 4: [], 5: []}
    for row in sorting_instances:
        by_size[row["difficulty"]["size"]].append(row)
    sorting_selected: list[dict] = []
    for size in (3, 4, 5):
        items = sorted(by_size[size], key=lambda row: row["seed"])
        sorting_selected.extend(items[:200])

    by_distance: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for row in map_instances:
        by_distance[row["difficulty"]["distance"]].append(row)
    map_selected: list[dict] = []
    for distance in (1, 2, 3):
        items = sorted(by_distance[distance], key=lambda row: row["id"])
        map_selected.extend(select_even(items, 300))

    env = TWXEnv()
    drops: Counter = Counter()
    sorting_trajectories: list[dict] = []
    try:
        print(f"[expert] sorting: {len(sorting_selected)} instances")
        for index, instance in enumerate(sorting_selected, 1):
            take_actions = [
                action for action in instance["valid_actions"] if action.startswith("take ")
            ]
            takes = [parse_take(action) for action in take_actions]
            if not takes or any(item is None for item in takes):
                drops["sorting_unparsed"] += 1
                continue
            dimensions = {item["dimension"] for item in takes}
            if len(dimensions) != 1:
                drops["sorting_mixed_dimension"] += 1
                continue
            turns = None
            for order in sorting_orders(takes):
                turns = run_sorting(env, instance, order)
                if turns is not None:
                    break
            if turns is None:
                drops["sorting_replay_failed"] += 1
                continue
            sorting_trajectories.append(
                {
                    "id": f"{instance['id']}-expert",
                    "instance_id": instance["id"],
                    "game": "sorting",
                    "planner": "sorting-expert-v1",
                    "success": True,
                    "score": 1.0,
                    "steps": len(turns),
                    "turns": turns,
                }
            )
            if index % 100 == 0:
                print(f"[expert] sorting {index}/{len(sorting_selected)}")

        print(f"[expert] mapreader: {len(map_selected)} instances")
        map_trajectories: list[dict] = []
        for index, instance in enumerate(map_selected, 1):
            turns = run_mapreader(env, instance)
            if turns is None:
                drops["mapreader_replay_failed"] += 1
                continue
            map_trajectories.append(
                {
                    "id": f"{instance['id']}-expert",
                    "instance_id": instance["id"],
                    "game": "mapreader",
                    "planner": "mapreader-expert-v1",
                    "success": True,
                    "score": 1.0,
                    "steps": len(turns),
                    "turns": turns,
                }
            )
            if index % 100 == 0:
                print(f"[expert] mapreader {index}/{len(map_selected)}")
    finally:
        env.close()

    write_jsonl(out_dir / "sorting_expert.jsonl", sorting_trajectories)
    write_jsonl(out_dir / "mapreader_expert.jsonl", map_trajectories)
    meta = {
        "counts": {
            "sorting": len(sorting_trajectories),
            "mapreader": len(map_trajectories),
            "sorting_turns": sum(len(item["turns"]) for item in sorting_trajectories),
            "mapreader_turns": sum(len(item["turns"]) for item in map_trajectories),
        },
        "selected": {"sorting": len(sorting_selected), "mapreader": len(map_selected)},
        "drops": {key: drops[key] for key in sorted(drops)},
        "planner": {"sorting": "sorting-expert-v1", "mapreader": "mapreader-expert-v1"},
    }
    write_json(out_dir / "expert_meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES)
    parser.add_argument("--out-dir", type=Path, default=SFT)
    args = parser.parse_args()
    meta = build(args.instances_dir, args.out_dir)
    print(f"[expert] wrote {args.out_dir}: {meta['counts']} drops={meta['drops']}")


if __name__ == "__main__":
    main()
