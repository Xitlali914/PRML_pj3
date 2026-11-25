"""Thin wrapper around the vendored TextWorldExpress runtime.

The wrapper has three jobs:
  * import the vendored package (never a system installation),
  * verify the packaged JAR is the expected build,
  * expose a small, deterministic interface for data generation, rollout and
    evaluation, with the 10-action budget enforced by the caller.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from code.paths import THIRD_PARTY

EXPECTED_JAR_SHA256 = (
    "23da9ab053dde2de003b4556a2734af3225dc3a0ef8af7354df53aafb5c77d99"
)


def _ensure_java() -> None:
    """Use the user-level jdk4py JRE when the system has no java on PATH."""
    try:
        import jdk4py
    except ImportError:
        return
    try:
        java_home = Path(jdk4py.JAVA_HOME)
    except AttributeError:
        return
    java = java_home / "bin" / "java"
    if not java.is_file():
        return
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")


_ensure_java()

if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

import textworld_express  # noqa: E402  (must come after sys.path setup)
from textworld_express import TextWorldExpressEnv  # noqa: E402
from textworld_express.constants import JAR_PATH  # noqa: E402

if not Path(textworld_express.__file__).resolve().is_relative_to(THIRD_PARTY.resolve()):
    raise RuntimeError(
        "textworld_express was imported from outside third_party: "
        f"{textworld_express.__file__}"
    )

TWX_VERSION = textworld_express.__version__


def verify_jar() -> str:
    digest = hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest()
    if digest != EXPECTED_JAR_SHA256:
        raise RuntimeError(
            "TextWorldExpress JAR hash mismatch: "
            f"expected {EXPECTED_JAR_SHA256}, got {digest}"
        )
    return digest


def _normalise_inventory(value: object) -> list[str]:
    """The environment returns the inventory as a text block, not a list."""
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.lower().startswith("inventory:"):
        text = text[len("inventory:") :]
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line:
            continue
        if line.lower().startswith("your inventory is currently empty"):
            continue
        items.append(line)
    return items


@dataclass
class Step:
    observation: str
    score: float
    score_raw: float
    success: bool
    failure: bool
    valid_actions: list[str] = field(default_factory=list)
    inventory: list[str] = field(default_factory=list)
    invalid: bool = False


class TWXEnv:
    """One JVM-backed environment, reset between instances."""

    def __init__(self, env_step_limit: int = 100):
        verify_jar()
        self._env = TextWorldExpressEnv(envStepLimit=env_step_limit)
        self._loaded: tuple[str, str] | None = None

        self.observation = ""
        self.inventory: list[str] = []
        self.valid_actions: list[str] = []
        self.task = ""
        self.score = 0.0
        self.steps = 0
        self.properties: dict = {}

    def reset(
        self,
        game: str,
        seed: int,
        fold: str = "train",
        params: str = "",
    ) -> Step:
        key = (game, params)
        if self._loaded != key:
            self._env.load(game, params)
            self._loaded = key

        obs, infos = self._env.reset(
            seed=seed, gameFold=fold, generateGoldPath=False
        )
        self.steps = 0
        self.observation = obs
        self.inventory = _normalise_inventory(infos.get("inventory", ""))
        self.valid_actions = list(infos.get("validActions", []))
        self.task = infos.get("taskDescription", "")
        self.score = float(infos.get("score", 0.0))
        self.properties = dict(self._env.getGenerationProperties())
        return Step(
            observation=self.observation,
            score=self.score,
            score_raw=float(infos.get("scoreRaw", 0.0)),
            success=bool(infos.get("tasksuccess", False)),
            failure=bool(infos.get("taskfailure", False)),
            valid_actions=list(self.valid_actions),
            inventory=list(self.inventory),
        )

    def step(self, action: str) -> Step:
        obs, _reward, _done, infos = self._env.step(action)
        self.steps += 1
        self.observation = obs
        self.inventory = _normalise_inventory(infos.get("inventory", ""))
        self.valid_actions = list(infos.get("validActions", []))
        self.score = float(infos.get("score", 0.0))
        invalid = obs.strip().lower().startswith("unknown action")
        return Step(
            observation=obs,
            score=self.score,
            score_raw=float(infos.get("scoreRaw", 0.0)),
            success=bool(infos.get("tasksuccess", False)),
            failure=bool(infos.get("taskfailure", False)),
            valid_actions=list(self.valid_actions),
            inventory=list(self.inventory),
            invalid=invalid,
        )

    def close(self) -> None:
        env = self._env
        if env is None:
            return
        try:
            env.close()
        except Exception:
            pass
        # TextWorldExpressEnv.__del__ calls close() again at interpreter exit;
        # the JVM side is already gone by then, so make the second call a no-op.
        try:
            env.close = lambda: None  # type: ignore[method-assign]
        except Exception:
            pass
        self._env = None
