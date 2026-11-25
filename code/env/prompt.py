"""Plain-text prompt rendering and action parsing (Qwen3-4B-Base, no chat template)."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

MAX_PROMPT_TOKENS = 1536
MAX_HISTORY = 6
MAX_HISTORY_OBS_TOKENS = 120
MAX_ACTION_TOKENS = 48

_HEADER = (
    "You are an agent playing a text game. At each turn, choose exactly one action.\n"
    "Reply with the action inside <action></action> and nothing else."
)

_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)

REPAIR_SUFFIX = (
    "\n\nYour previous reply was not a valid action. Reply with exactly one action "
    "from the valid actions, inside <action></action>.\n\nAction:"
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


def _render(
    task: str,
    observation: str,
    inventory: Sequence[str],
    valid_actions: Sequence[str],
    history: Sequence[tuple[str, str]],
) -> str:
    lines = [
        _HEADER,
        "",
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
    lines.extend(["", "Action:"])
    return "\n".join(lines)


def render_prompt(
    task: str,
    observation: str,
    inventory: Sequence[str],
    valid_actions: Sequence[str],
    history: Sequence[tuple[str, str]],
    tokenizer: Any = None,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_history: int = MAX_HISTORY,
    max_history_obs_tokens: int = MAX_HISTORY_OBS_TOKENS,
) -> str:
    """Render one prompt.

    History is kept to the most recent ``max_history`` steps, each observation is
    collapsed to one line and truncated to ``max_history_obs_tokens`` tokens.
    If the prompt is still too long, the oldest history entries are dropped and
    finally the current observation is truncated from its end.
    """
    hist = [
        (action, _one_line(_truncate_to_tokens(obs, tokenizer, max_history_obs_tokens)))
        for action, obs in history
    ][-max_history:]

    text = _render(task, observation, inventory, valid_actions, hist)
    if tokenizer is None or _count_tokens(text, tokenizer) <= max_prompt_tokens:
        return text

    while hist and _count_tokens(text, tokenizer) > max_prompt_tokens:
        hist.pop(0)
        text = _render(task, observation, inventory, valid_actions, hist)
    if _count_tokens(text, tokenizer) <= max_prompt_tokens:
        return text

    ids = tokenizer.encode(observation, add_special_tokens=False)
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = tokenizer.decode(ids[:mid], skip_special_tokens=True)
        candidate_text = _render(task, candidate, inventory, valid_actions, hist)
        if _count_tokens(candidate_text, tokenizer) <= max_prompt_tokens:
            lo = mid
        else:
            hi = mid - 1
    observation = tokenizer.decode(ids[:lo], skip_special_tokens=True)
    return _render(task, observation, inventory, valid_actions, hist)


def target_text(action: str) -> str:
    return f"<action>{action}</action>"


def repair_prompt(prompt: str) -> str:
    return prompt + REPAIR_SUFFIX


def _normalise(text: str) -> str:
    text = text.strip().strip("\"'`")
    text = text.rstrip(".")
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_action(text: str, valid_actions: Iterable[str]) -> tuple[str, bool]:
    """Extract one action from a model reply.

    Returns ``(action, parsed)``. When the reply matches a valid action, the
    canonical action string is returned; otherwise the raw candidate string is
    returned (the environment will treat it as invalid) and ``parsed`` is False.
    """
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
