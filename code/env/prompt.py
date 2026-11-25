"""Chat-template prompt rendering for Qwen3-4B (thinking disabled).

The model provided by the course is the post-trained Qwen3-4B, so every prompt
is rendered through the tokenizer's chat template with ``enable_thinking=False``.
The structured game state is unchanged; only the renderer differs from a base
model setup.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

MAX_PROMPT_TOKENS = 1536
MAX_HISTORY = 6
MAX_HISTORY_OBS_TOKENS = 120
MAX_ACTION_TOKENS = 48

SYSTEM_PROMPT = (
    "You are an agent playing a text game. At each turn, choose exactly one action. "
    "Reply with the action inside <action></action> and nothing else."
)

_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)

REPAIR_INSTRUCTION = (
    "Your previous reply was not a valid action. Reply with exactly one action "
    "from the valid actions, inside <action></action>."
)


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    if tokenizer is None:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _count_tokens(text: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _user_content(
    task: str,
    observation: str,
    inventory: Sequence[str],
    valid_actions: Sequence[str],
    history: Sequence[tuple[str, str]],
) -> str:
    lines = [
        f"Task: {task.strip()}",
        "",
        "Current observation:",
        observation.strip(),
        "",
        f"Inventory: {', '.join(inventory) if inventory else 'nothing'}",
        "",
        "Valid actions:",
    ]
    lines.extend(f"- {action}" for action in valid_actions)
    lines.extend(["", "History:"])
    if history:
        for index, (action, obs) in enumerate(history, 1):
            lines.append(f"[{index}] {action} -> {obs}")
    else:
        lines.append("(none)")
    return "\n".join(lines)


def _render_chat(
    task: str,
    observation: str,
    inventory: Sequence[str],
    valid_actions: Sequence[str],
    history: Sequence[tuple[str, str]],
    tokenizer: Any,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _user_content(
                task, observation, inventory, valid_actions, history
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_prompt(
    task: str,
    observation: str,
    inventory: Sequence[str],
    valid_actions: Sequence[str],
    history: Sequence[tuple[str, str]],
    tokenizer: Any,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_history: int = MAX_HISTORY,
    max_history_obs_tokens: int = MAX_HISTORY_OBS_TOKENS,
) -> str:
    """Render one chat prompt.

    History is capped at ``max_history`` steps, each history observation is
    collapsed and truncated, and the final prompt is kept under
    ``max_prompt_tokens`` by dropping old history and then shortening the
    current observation.
    """
    if tokenizer is None:
        raise ValueError("the Qwen3-4B renderer requires a tokenizer")

    hist = [
        (action, _one_line(_truncate_to_tokens(obs, tokenizer, max_history_obs_tokens)))
        for action, obs in history
    ][-max_history:]

    text = _render_chat(task, observation, inventory, valid_actions, hist, tokenizer)
    if _count_tokens(text, tokenizer) <= max_prompt_tokens:
        return text

    while hist and _count_tokens(text, tokenizer) > max_prompt_tokens:
        hist.pop(0)
        text = _render_chat(task, observation, inventory, valid_actions, hist, tokenizer)
    if _count_tokens(text, tokenizer) <= max_prompt_tokens:
        return text

    ids = tokenizer.encode(observation, add_special_tokens=False)
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = tokenizer.decode(ids[:mid], skip_special_tokens=True)
        candidate_text = _render_chat(
            task, candidate, inventory, valid_actions, hist, tokenizer
        )
        if _count_tokens(candidate_text, tokenizer) <= max_prompt_tokens:
            lo = mid
        else:
            hi = mid - 1
    observation = tokenizer.decode(ids[:lo], skip_special_tokens=True)
    return _render_chat(task, observation, inventory, valid_actions, hist, tokenizer)


def target_text(action: str, tokenizer: Any) -> str:
    return f"<action>{action}</action>{tokenizer.eos_token}"


def repair_prompt(prompt: str, tokenizer: Any) -> str:
    """Append a corrective user turn after a malformed assistant reply."""
    eos = tokenizer.eos_token
    return (
        prompt
        + eos
        + "\n<|im_start|>user\n"
        + REPAIR_INSTRUCTION
        + eos
        + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _normalise(text: str) -> str:
    text = text.strip().strip("\"'`")
    text = text.rstrip(".")
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_action(text: str, valid_actions: Iterable[str]) -> tuple[str, bool]:
    """Extract one action from a model reply."""
    matches = _ACTION_RE.findall(text or "")
    if matches:
        candidate = matches[-1]
    else:
        candidate = ""
        for line in (text or "").splitlines():
            if line.strip():
                candidate = line
                break
    candidate = candidate.strip()
    normalised = _normalise(candidate)
    for action in valid_actions:
        if _normalise(action) == normalised:
            return action, True
    return candidate, False
