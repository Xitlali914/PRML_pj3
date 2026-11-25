"""Model loading, LoRA attachment, adapter merging and seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_dir: str | Path, padding_side: str = "right") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_base_model(
    model_dir: str | Path,
    device: str = "cuda",
    for_training: bool = False,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if for_training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    return model


def attach_lora(model: AutoModelForCausalLM) -> PeftModel:
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").is_file()


def load_trainable_adapter(
    base_dir: str | Path, adapter_dir: str | Path, device: str = "cuda"
) -> PeftModel:
    base = load_base_model(base_dir, device=device, for_training=True)
    model = PeftModel.from_pretrained(
        base, str(adapter_dir), is_trainable=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def load_eval_model(
    model_dir: str | Path,
    base_dir: str | Path | None = None,
    device: str = "cuda",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = load_tokenizer(model_dir)
    if is_adapter_dir(model_dir):
        if base_dir is None:
            import json

            config = json.loads(
                (Path(model_dir) / "adapter_config.json").read_text(encoding="utf-8")
            )
            base_dir = config.get("base_model_name_or_path")
            if not base_dir:
                raise ValueError(
                    "adapter model needs --base-model or a base_model_name_or_path"
                )
        base = load_base_model(base_dir, device=device, for_training=False)
        model = PeftModel.from_pretrained(base, str(model_dir), is_trainable=False)
        model.to(device)
    else:
        model = load_base_model(model_dir, device=device, for_training=False)
    model.eval()
    return model, tokenizer


def save_adapter(model: PeftModel, output_dir: str | Path, tokenizer: AutoTokenizer) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def merge_adapter(
    base_dir: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    tokenizer: AutoTokenizer,
) -> None:
    """Merge on CPU so the GPU stays free for the next stage."""
    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir), torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = model.merge_and_unload()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
