# Model weights

This directory is git-ignored. Put the official Qwen3-4B snapshot here as:

```
model/Qwen3-4B/
```

The server copy is `cp -a /shared/models/Qwen3-4B model/Qwen3-4B`.
The code renders every prompt through this tokenizer's chat template with
`enable_thinking=False`.

Training writes:

```
model/sft/adapter/     # LoRA adapter of the selected SFT epoch
model/sft/merged/      # merged full weights, used for evaluation
model/grpo/adapter/    # LoRA adapter of the selected GRPO update
model/grpo/merged/     # merged full weights, used by run.py
```

All scripts accept `--base-model` / `--output-dir` overrides, so the weights can
live on another disk as long as the paths are passed explicitly.
