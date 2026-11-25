"""Reward definition shared by GRPO (and logged during evaluation)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardParts:
    success: float
    progress: float
    failure: float
    invalid: float
    steps: float
    unparsed: float

    @property
    def total(self) -> float:
        return (
            self.success
            + self.progress
            + self.failure
            + self.invalid
            + self.steps
            + self.unparsed
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "success": self.success,
            "progress": self.progress,
            "failure": self.failure,
            "invalid": self.invalid,
            "steps": self.steps,
            "unparsed": self.unparsed,
            "total": self.total,
        }


def episode_reward(
    success: bool,
    failure: bool,
    score: float,
    steps: int,
    invalid_count: int,
    unparsed_count: int,
) -> RewardParts:
    return RewardParts(
        success=1.0 if success else 0.0,
        progress=0.5 * max(float(score), 0.0),
        failure=-0.5 if failure else 0.0,
        invalid=-0.05 * invalid_count,
        steps=-0.01 * steps,
        unparsed=-0.05 * unparsed_count,
    )
