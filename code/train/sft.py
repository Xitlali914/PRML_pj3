"""SFT cold start on expert turn samples.

Each sample is ``prompt tokens + action tokens``; the forward pass only keeps
the logits of the last ``max_action_len`` positions, so a 4B model fits on a
24GB GPU with micro-batch 4 and gradient accumulation 4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402

from code.data.io_utils import read_json, read_jsonl, write_json  # noqa: E402
from code.env.prompt import MAX_ACTION_TOKENS, MAX_PROMPT_TOKENS, render_prompt, target_text  # noqa: E402
from code.paths import BASE_MODEL_DIR, RESULTS, SFT, SFT_DIR  # noqa: E402
from code.train.model import (  # noqa: E402
    attach_lora,
    load_base_model,
    load_tokenizer,
    merge_adapter,
    save_adapter,
    set_seed,
)


class TurnDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_prompt_tokens: int = MAX_PROMPT_TOKENS):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        history = [(item["action"], item["observation"]) for item in row["history"]]
        prompt = render_prompt(
            row["task"],
            row["observation"],
            row["inventory"],
            row["valid_actions"],
            history,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
        )
        target = target_text(row["target_action"], self.tokenizer)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-self.max_prompt_tokens :]
        target_ids = target_ids[:MAX_ACTION_TOKENS]
        return {"prompt_ids": prompt_ids, "target_ids": target_ids}


class TurnCollator:
    def __init__(self, pad_token_id: int, max_target_tokens: int = MAX_ACTION_TOKENS):
        self.pad_token_id = pad_token_id
        self.max_target_tokens = max_target_tokens

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_prompt = max(len(item["prompt_ids"]) for item in batch)
        max_target = max(len(item["target_ids"]) for item in batch)
        max_target = min(max_target, self.max_target_tokens)
        block = max_target + 1  # target tokens + one extra slot
        keep = block + 1  # last prompt token + block, used by logits_to_keep

        input_ids = []
        attention = []
        target_ids = []
        target_mask = []
        for item in batch:
            prompt = item["prompt_ids"]
            target = item["target_ids"][:max_target]
            left_pad = max_prompt - len(prompt)
            right_pad = block - len(target)
            ids = [self.pad_token_id] * left_pad + prompt + target + [self.pad_token_id] * right_pad
            mask = [0] * left_pad + [1] * len(prompt) + [1] * len(target) + [0] * right_pad
            labels = target + [self.pad_token_id] * (keep - len(target))
            labels_mask = [1] * len(target) + [0] * (keep - len(target))
            input_ids.append(ids)
            attention.append(mask)
            target_ids.append(labels)
            target_mask.append(labels_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.float32),
        }


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
    parser.add_argument("--train-data", type=Path, default=SFT / "turns_train.jsonl")
    parser.add_argument("--val-data", type=Path, default=SFT / "turns_val.jsonl")
    parser.add_argument("--dev-instances", type=Path, default=Path("data/instances/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model, padding_side="right")
    model = attach_lora(load_base_model(args.base_model, args.device, for_training=True))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_dataset = TurnDataset(args.train_data, tokenizer)
    val_dataset = TurnDataset(args.val_data, tokenizer)
    collator = TurnCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_updates)),
        num_training_steps=total_updates,
    )

    config = {
        "base_model": str(args.base_model),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "epochs": args.epochs,
        "lr": args.lr,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_action_tokens": MAX_ACTION_TOKENS,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "seed": args.seed,
    }
    write_json(args.output_dir / "config.json", config)

    best_score = -1.0
    best_epoch = -1
    log_path = args.output_dir / "sft_log.jsonl"
    log_handle = log_path.open("w", encoding="utf-8", newline="\n")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_tokens = 0
            update_index = 0
            for step, batch in enumerate(train_loader, 1):
                batch = {key: value.to(args.device) for key, value in batch.items()}
                max_target = batch["target_ids"].shape[1]
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    logits_to_keep=max_target,
                )
                logits = outputs.logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
                gathered = log_probs.gather(
                    -1, batch["target_ids"].unsqueeze(-1)
                ).squeeze(-1)
                token_count = batch["target_mask"].sum().clamp(min=1.0)
                loss = -(gathered * batch["target_mask"]).sum() / token_count
                (loss / args.grad_accum).backward()
                running_loss += float(loss.detach())
                running_tokens += int(token_count)

                if step % args.grad_accum == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_index += 1
                    if update_index % args.log_every == 0:
                        record = {
                            "epoch": epoch,
                            "update": update_index,
                            "loss": running_loss / max(1, update_index * args.grad_accum),
                            "lr": scheduler.get_last_lr()[0],
                        }
                        print(f"[sft] {record}")
                        log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                        log_handle.flush()

            epoch_dir = args.output_dir / f"epoch_{epoch}"
            save_adapter(model, epoch_dir, tokenizer)
            metrics_path = RESULTS / f"sft_epoch{epoch}.metrics.json"
            output_path = RESULTS / f"sft_epoch{epoch}.jsonl"
            metrics = run_dev_eval(
                args.base_model,
                epoch_dir,
                args.dev_instances,
                output_path,
                metrics_path,
                args.device,
            )
            score = 0.5 * metrics["sorting"]["success_rate"] + 0.5 * metrics[
                "mapreader"
            ]["success_rate"]
            record = {
                "epoch": epoch,
                "dev_score": score,
                "sorting": metrics["sorting"],
                "mapreader": metrics["mapreader"],
            }
            print(f"[sft] dev epoch {epoch}: {record}")
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()
            if score > best_score:
                best_score = score
                best_epoch = epoch
    finally:
        log_handle.close()

    if best_epoch < 0:
        raise RuntimeError("SFT did not complete an epoch")

    best_adapter = args.output_dir / f"epoch_{best_epoch}"
    final_adapter = args.output_dir / "adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(best_adapter, final_adapter)

    del model
    torch.cuda.empty_cache()
    merge_adapter(args.base_model, final_adapter, args.output_dir / "merged", tokenizer)
    write_json(
        args.output_dir / "result.json",
        {"best_epoch": best_epoch, "best_dev_score": best_score},
    )
    print(f"[sft] best epoch {best_epoch} score {best_score:.4f}")
    print(f"[sft] adapter: {final_adapter}")
    print(f"[sft] merged: {args.output_dir / 'merged'}")


if __name__ == "__main__":
    main()
