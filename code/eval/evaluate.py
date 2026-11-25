"""Greedy evaluation of a model on TextWorldExpress instances."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from code.data.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from code.env.prompt import (  # noqa: E402
    MAX_PROMPT_TOKENS,
    parse_action,
    render_prompt,
    repair_prompt,
)
from code.env.twx import TWXEnv  # noqa: E402
from code.paths import RESULTS  # noqa: E402
from code.train.model import load_eval_model, set_seed  # noqa: E402


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    output = model.generate(**inputs, **kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def act(
    model,
    tokenizer,
    prompt: str,
    valid_actions: list[str],
    device: str,
    max_new_tokens: int = 48,
) -> tuple[str, bool, str, int]:
    text = generate_text(model, tokenizer, prompt, device, max_new_tokens, do_sample=False)
    action, parsed = parse_action(text, valid_actions)
    retries = 0
    if not parsed:
        retries = 1
        text = generate_text(
            model, tokenizer, repair_prompt(prompt), device, max_new_tokens, do_sample=False
        )
        action, parsed = parse_action(text, valid_actions)
    return action, parsed, text, retries


def evaluate(
    model,
    tokenizer,
    instances: list[dict],
    device: str = "cuda",
    max_actions: int = 10,
) -> tuple[list[dict], dict]:
    env = TWXEnv()
    results: list[dict] = []
    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "success": 0, "steps_success": [], "actions": 0, "invalid": 0,
                 "unparsed": 0, "retries": 0}
    )
    try:
        for index, instance in enumerate(instances, 1):
            env.reset(
                instance["game"],
                instance["seed"],
                fold=instance["fold"],
                params=instance.get("params", ""),
            )
            history: list[tuple[str, str]] = []
            actions: list[str] = []
            observations: list[str] = []
            invalid = 0
            unparsed = 0
            retries = 0
            step = None

            for step_index in range(max_actions):
                prompt = render_prompt(
                    instance["task"],
                    env.observation,
                    env.inventory,
                    env.valid_actions,
                    history,
                    tokenizer=tokenizer,
                    max_prompt_tokens=MAX_PROMPT_TOKENS,
                )
                action, parsed, _text, retry_count = act(
                    model, tokenizer, prompt, env.valid_actions, device
                )
                retries += retry_count
                step = env.step(action)
                actions.append(action)
                observations.append(step.observation)
                history.append((action, step.observation))
                if step.invalid:
                    invalid += 1
                if not parsed:
                    unparsed += 1
                if step.success or step.failure or step_index + 1 == max_actions:
                    break

            success = bool(step is not None and step.success)
            steps = len(actions)
            row = {
                "id": instance["id"],
                "game": instance["game"],
                "seed": instance["seed"],
                "params": instance.get("params", ""),
                "success": success,
                "failure": bool(step is not None and step.failure),
                "score": float(step.score) if step is not None else 0.0,
                "steps": steps,
                "invalid_actions": invalid,
                "unparsed": unparsed,
                "retries": retries,
                "actions": actions,
                "observations": observations,
            }
            results.append(row)

            bucket = stats[instance["game"]]
            bucket["n"] += 1
            bucket["success"] += int(success)
            bucket["actions"] += steps
            bucket["invalid"] += invalid
            bucket["unparsed"] += unparsed
            bucket["retries"] += retries
            if success:
                bucket["steps_success"].append(steps)
            if index % 20 == 0:
                print(f"[eval] {index}/{len(instances)}")
    finally:
        env.close()

    metrics: dict[str, dict] = {}
    for game, values in stats.items():
        successes = values["steps_success"]
        metrics[game] = {
            "n": values["n"],
            "success_rate": values["success"] / values["n"] if values["n"] else 0.0,
            "avg_steps_on_success": (
                sum(successes) / len(successes) if successes else 0.0
            ),
            "invalid_action_ratio": (
                values["invalid"] / values["actions"] if values["actions"] else 0.0
            ),
            "unparsed": values["unparsed"],
            "retries": values["retries"],
        }

    total_n = sum(item["n"] for item in metrics.values())
    total_actions = sum(item["actions"] for item in stats.values())
    total_invalid = sum(item["invalid"] for item in stats.values())
    all_success_steps = [
        value for bucket in stats.values() for value in bucket["steps_success"]
    ]
    overall = {
        "n": total_n,
        "success_rate": (
            sum(item["success_rate"] * item["n"] for item in metrics.values()) / total_n
            if total_n
            else 0.0
        ),
        "avg_steps_on_success": (
            sum(all_success_steps) / len(all_success_steps) if all_success_steps else 0.0
        ),
        "invalid_action_ratio": total_invalid / total_actions if total_actions else 0.0,
        "score": 0.5 * metrics.get("sorting", {}).get("success_rate", 0.0)
        + 0.5 * metrics.get("mapreader", {}).get("success_rate", 0.0),
    }
    metrics["overall"] = overall
    return results, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model", default=None,
                        help="base model directory when --model-dir is a LoRA adapter")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    instances = read_jsonl(Path(args.instances))
    if args.limit:
        instances = instances[: args.limit]

    model, tokenizer = load_eval_model(args.model_dir, args.base_model, args.device)
    results, metrics = evaluate(
        model, tokenizer, instances, device=args.device, max_actions=args.max_actions
    )

    output = Path(args.output)
    write_jsonl(output, results)
    metrics_path = Path(args.metrics) if args.metrics else output.with_suffix(".metrics.json")
    write_json(metrics_path, metrics)
    print(f"[eval] results: {output}")
    print(f"[eval] metrics: {metrics_path}")
    print(f"[eval] summary: {metrics}")


if __name__ == "__main__":
    main()
