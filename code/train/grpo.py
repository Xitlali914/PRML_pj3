"""GRPO training on TextWorldExpress with turn-level samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402

from code.data.io_utils import read_json, read_jsonl, write_json  # noqa: E402
from code.paths import BASE_MODEL_DIR, GRPO_DIR, INSTANCES, RESULTS  # noqa: E402
from code.train.model import (  # noqa: E402
    load_tokenizer,
    load_trainable_adapter,
    merge_adapter,
    save_adapter,
    set_seed,
)
from code.train.rollout import RolloutEngine, TurnSample  # noqa: E402

BUCKETS = [
    ("sorting", 3),
    ("sorting", 4),
    ("sorting", 5),
    ("mapreader", 1),
    ("mapreader", 2),
    ("mapreader", 3),
]
EASY_BUCKETS = [("sorting", 3), ("sorting", 4), ("mapreader", 1), ("mapreader", 2)]
HARD_BUCKETS = [("sorting", 5), ("mapreader", 3)]


def load_buckets(instances_dir: Path) -> dict[tuple[str, int], list[dict]]:
    buckets: dict[tuple[str, int], list[dict]] = {key: [] for key in BUCKETS}
    for row in read_jsonl(instances_dir / "sorting_train.jsonl"):
        buckets[("sorting", int(row["difficulty"]["size"]))].append(row)
    for row in read_jsonl(instances_dir / "mapreader_train.jsonl"):
        buckets[("mapreader", int(row["difficulty"]["distance"]))].append(row)
    for key, rows in buckets.items():
        if not rows:
            raise RuntimeError(f"empty bucket {key}")
    return buckets


def choose_instances(
    buckets: dict[tuple[str, int], list[dict]],
    update: int,
    rng: random.Random,
    count: int = 8,
) -> list[dict]:
    if update <= 80:
        pool = [row for key in EASY_BUCKETS for row in buckets[key]]
        return rng.sample(pool, count)
    if update <= 200:
        pool = [row for key in BUCKETS for row in buckets[key]]
        return rng.sample(pool, count)
    hard_pool = [row for key in HARD_BUCKETS for row in buckets[key]]
    other_pool = [row for key in BUCKETS if key not in HARD_BUCKETS for row in buckets[key]]
    return rng.sample(hard_pool, 4) + rng.sample(other_pool, 4)


def collate_samples(
    samples: list[tuple[TurnSample, float]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_length = max(len(turn.input_ids) for turn, _ in samples)
    max_action = max(turn.action_len for turn, _ in samples)
    target_length = max_action + 1  # one extra position for the causal shift
    input_ids, attention, target_ids, action_mask, old_logp, advantages = [], [], [], [], [], []
    for turn, advantage in samples:
        left = max_length - len(turn.input_ids)
        input_ids.append([pad_token_id] * left + turn.input_ids)
        attention.append([0] * left + [1] * len(turn.input_ids))
        target_left = max_action - turn.action_len
        target_right = target_length - target_left - turn.action_len
        target_ids.append(
            [0] * target_left + turn.input_ids[-turn.action_len :] + [0] * target_right
        )
        action_mask.append(
            [0] * target_left + [1] * turn.action_len + [0] * target_right
        )
        old_logp.append(
            [0.0] * target_left + turn.old_logp + [0.0] * target_right
        )
        advantages.append(advantage)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "action_mask": torch.tensor(action_mask, dtype=torch.float32),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
    }


def token_log_probs(model, batch: dict[str, torch.Tensor], max_action: int) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        logits_to_keep=max_action,
    )
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)


def run_dev_eval(
    base_model: Path,
    adapter: Path,
    dev_instances: Path,
    output: Path,
    metrics: Path,
    device: str,
) -> dict:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    command = [
        sys.executable,
        "-m",
        "code.eval.evaluate",
        "--model-dir",
        str(adapter),
        "--base-model",
        str(base_model),
        "--instances",
        str(dev_instances),
        "--output",
        str(output),
        "--metrics",
        str(metrics),
        "--device",
        device,
    ]
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    return read_json(metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--instances-dir", type=Path, default=INSTANCES)
    parser.add_argument("--dev-instances", type=Path, default=Path("data/instances/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=GRPO_DIR)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-update", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl", type=float, default=0.001)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    buckets = load_buckets(args.instances_dir)
    tokenizer = load_tokenizer(args.base_model)
    model = load_trainable_adapter(args.base_model, args.sft_adapter, args.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    engine = RolloutEngine(
        model,
        tokenizer,
        device=args.device,
        group_size=args.group_size,
        max_actions=args.max_actions,
        max_new_tokens=args.max_new_tokens,
    )

    updates = args.updates
    estimated_steps = max(1, updates * 28)
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * estimated_steps)),
        num_training_steps=estimated_steps,
    )

    config = {
        "base_model": str(args.base_model),
        "sft_adapter": str(args.sft_adapter),
        "updates": updates,
        "group_size": args.group_size,
        "prompts_per_update": args.prompts_per_update,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "clip": args.clip,
        "kl": args.kl,
        "max_actions": args.max_actions,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "buckets": [list(key) for key in BUCKETS],
    }
    write_json(args.output_dir / "config.json", config)
    log_path = args.output_dir / "grpo_log.jsonl"
    log_handle = log_path.open("w", encoding="utf-8", newline="\n")

    rng = random.Random(args.seed)
    best_score = -1.0
    best_update = -1
    stats: Counter = Counter()

    try:
        for update in range(1, updates + 1):
            model.eval()
            model.config.use_cache = True
            instances = choose_instances(
                buckets, update, rng, count=args.prompts_per_update
            )
            samples: list[tuple[TurnSample, float]] = []
            update_rewards: list[float] = []
            skipped_groups = 0
            for instance in instances:
                trajectories = engine.run_group(instance, group_size=args.group_size)
                rewards = [trajectory.reward.total for trajectory in trajectories]
                mean = sum(rewards) / len(rewards)
                variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
                std = math.sqrt(variance)
                if std < 1e-6:
                    skipped_groups += 1
                    continue
                advantages = [(value - mean) / (std + 1e-4) for value in rewards]
                for trajectory, advantage in zip(trajectories, advantages):
                    update_rewards.append(trajectory.reward.total)
                    stats["success"] += int(trajectory.success)
                    stats["failure"] += int(trajectory.failure)
                    stats["invalid"] += trajectory.invalid
                    stats["unparsed"] += trajectory.unparsed
                    stats["trajectories"] += 1
                    for turn in trajectory.turns:
                        samples.append((turn, advantage))

            if not samples:
                stats["skipped_updates"] += 1
                record = {"update": update, "skipped": True}
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                log_handle.flush()
                continue

            rng.shuffle(samples)
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            losses = []
            micro_index = 0
            for start in range(0, len(samples), args.micro_batch):
                batch_samples = samples[start : start + args.micro_batch]
                batch = collate_samples(batch_samples, tokenizer.pad_token_id)
                batch = {key: value.to(args.device) for key, value in batch.items()}
                max_action = batch["target_ids"].shape[1]
                mask = batch["action_mask"]
                token_count = mask.sum().clamp(min=1.0)

                with torch.no_grad():
                    with model.disable_adapter():
                        ref_logp = token_log_probs(model, batch, max_action)
                policy_logp = token_log_probs(model, batch, max_action)
                ratio = torch.exp(policy_logp - batch["old_logp"])
                advantage = batch["advantages"].unsqueeze(-1)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantage
                policy_loss = -(torch.min(unclipped, clipped) * mask).sum()
                delta = ref_logp - policy_logp
                kl = (torch.exp(delta) - delta - 1.0) * mask
                loss = (policy_loss + args.kl * kl.sum()) / token_count
                losses.append(float(loss.detach()))
                (loss / args.grad_accum).backward()
                micro_index += 1

                if micro_index % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if micro_index % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            record = {
                "update": update,
                "loss": sum(losses) / len(losses),
                "reward": sum(update_rewards) / len(update_rewards)
                if update_rewards
                else 0.0,
                "samples": len(samples),
                "skipped_groups": skipped_groups,
                "lr": scheduler.get_last_lr()[0],
            }
            print(f"[grpo] {record}")
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()

            if update % args.eval_every == 0 or update == updates:
                adapter_dir = args.output_dir / f"update_{update}"
                save_adapter(model, adapter_dir, tokenizer)
                metrics = run_dev_eval(
                    args.base_model,
                    adapter_dir,
                    args.dev_instances,
                    RESULTS / f"grpo_update{update}.jsonl",
                    RESULTS / f"grpo_update{update}.metrics.json",
                    args.device,
                )
                score = 0.5 * metrics["sorting"]["success_rate"] + 0.5 * metrics[
                    "mapreader"
                ]["success_rate"]
                eval_record = {
                    "update": update,
                    "dev_score": score,
                    "sorting": metrics["sorting"],
                    "mapreader": metrics["mapreader"],
                }
                print(f"[grpo] dev {eval_record}")
                log_handle.write(json.dumps(eval_record, sort_keys=True) + "\n")
                log_handle.flush()
                if score > best_score:
                    best_score = score
                    best_update = update
    finally:
        log_handle.close()
        engine.close()

    if best_update < 0:
        raise RuntimeError("GRPO did not complete an evaluation")

    engine.model = None
    del engine
    best_adapter = args.output_dir / f"update_{best_update}"
    final_adapter = args.output_dir / "adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(best_adapter, final_adapter)

    del model
    torch.cuda.empty_cache()
    merge_adapter(args.base_model, final_adapter, args.output_dir / "merged", tokenizer)
    write_json(
        args.output_dir / "result.json",
        {"best_update": best_update, "best_dev_score": best_score, "config": config},
    )
    print(f"[grpo] best update {best_update} score {best_score:.4f}")


if __name__ == "__main__":
    main()
