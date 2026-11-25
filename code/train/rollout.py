"""On-policy rollout for GRPO: G trajectories per instance, batched generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from code.env.prompt import (
    MAX_PROMPT_TOKENS,
    parse_action,
    render_prompt,
    repair_prompt,
)
from code.env.twx import TWXEnv
from code.train.reward import RewardParts, episode_reward


@dataclass
class TurnSample:
    input_ids: list[int]
    action_len: int
    old_logp: list[float]
    action: str
    parsed: bool


@dataclass
class Trajectory:
    turns: list[TurnSample] = field(default_factory=list)
    success: bool = False
    failure: bool = False
    score: float = 0.0
    invalid: int = 0
    unparsed: int = 0
    reward: RewardParts | None = None


class RolloutEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        group_size: int = 8,
        max_actions: int = 10,
        max_new_tokens: int = 48,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.max_actions = max_actions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer.padding_side = "left"
        self.envs = [TWXEnv() for _ in range(group_size)]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @torch.inference_mode()
    def _generate(self, prompts: list[str]) -> list[dict]:
        encodings = self.tokenizer(
            prompts, padding=True, return_tensors="pt", add_special_tokens=False
        )
        encodings = {key: value.to(self.device) for key, value in encodings.items()}
        output = self.model.generate(
            **encodings,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=1.0,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        prompt_length = encodings["input_ids"].shape[1]
        new_ids = output.sequences[:, prompt_length:]
        results: list[dict] = []
        for row, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            action_ids: list[int] = []
            old_logp: list[float] = []
            text = ""
            for step in range(new_ids.shape[1]):
                token = int(new_ids[row, step])
                if token == self.tokenizer.eos_token_id:
                    break
                log_probs = torch.log_softmax(
                    output.scores[step][row].float(), dim=-1
                )
                action_ids.append(token)
                old_logp.append(float(log_probs[token]))
                text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
                if "</action>" in text:
                    break
            results.append(
                {
                    "prompt_ids": prompt_ids,
                    "action_ids": action_ids,
                    "old_logp": old_logp,
                    "text": text,
                }
            )
        return results

    def run_group(self, instance: dict, group_size: int | None = None) -> list[Trajectory]:
        group_size = group_size or self.group_size
        envs = self.envs[:group_size]
        for env in envs:
            env.reset(
                instance["game"],
                instance["seed"],
                fold=instance["fold"],
                params=instance.get("params", ""),
            )
        active = [
            {
                "env": env,
                "history": [],
                "turns": [],
                "done": False,
                "success": False,
                "failure": False,
                "score": 0.0,
                "invalid": 0,
                "unparsed": 0,
            }
            for env in envs
        ]

        for _ in range(self.max_actions):
            todo = [index for index, item in enumerate(active) if not item["done"]]
            if not todo:
                break
            prompts = [
                render_prompt(
                    instance["task"],
                    active[index]["env"].observation,
                    active[index]["env"].inventory,
                    active[index]["env"].valid_actions,
                    active[index]["history"],
                    tokenizer=self.tokenizer,
                    max_prompt_tokens=MAX_PROMPT_TOKENS,
                )
                for index in todo
            ]
            generations = self._generate(prompts)

            failed = []
            for position, generation in enumerate(generations):
                action, parsed = parse_action(
                    generation["text"], active[todo[position]]["env"].valid_actions
                )
                if not parsed:
                    failed.append(position)
            if failed:
                repairs = self._generate(
                    [repair_prompt(prompts[position], self.tokenizer) for position in failed]
                )
                for position, generation in zip(failed, repairs):
                    generations[position] = generation

            for position, index in enumerate(todo):
                item = active[index]
                generation = generations[position]
                action, parsed = parse_action(
                    generation["text"], item["env"].valid_actions
                )
                step = item["env"].step(action)
                if step.invalid:
                    item["invalid"] += 1
                if not parsed:
                    item["unparsed"] += 1
                if generation["action_ids"]:
                    item["turns"].append(
                        TurnSample(
                            input_ids=generation["prompt_ids"] + generation["action_ids"],
                            action_len=len(generation["action_ids"]),
                            old_logp=generation["old_logp"],
                            action=action,
                            parsed=parsed,
                        )
                    )
                item["history"].append((action, step.observation))
                item["score"] = step.score
                if step.success or step.failure:
                    item["done"] = True
                    item["success"] = step.success
                    item["failure"] = step.failure

        trajectories: list[Trajectory] = []
        for item in active:
            reward = episode_reward(
                success=item["success"],
                failure=item["failure"],
                score=item["score"],
                steps=len(item["turns"]),
                invalid_count=item["invalid"],
                unparsed_count=item["unparsed"],
            )
            trajectories.append(
                Trajectory(
                    turns=item["turns"],
                    success=item["success"],
                    failure=item["failure"],
                    score=item["score"],
                    invalid=item["invalid"],
                    unparsed=item["unparsed"],
                    reward=reward,
                )
            )
        return trajectories
