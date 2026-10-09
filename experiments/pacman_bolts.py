"""Environment and reward adapters for the smallClassic paper experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit, TransformReward

from npc_gym.bolts import BoltSpec, make_builtin_bolts
from npc_gym.envs import PacmanEnv, PacmanLayout
from npc_gym.wrappers import RestrainingBoltWrapper
from npc_gym.wrappers.pacman_wrappers import TrappedObservation

LAYOUT = "experiments/layouts/smallClassic.lay"
LAYOUT_SHA256 = "5a7cce3f11ba3ae21f33b736a021d25f39455b2c8e841e9536b3334603cae35c"
BUDGETS = {
    "vegan": {"dqn": 2_500_000, "ppo": 5_000_000},
    "vegetarian": {"dqn": 10_000_000, "ppo": 20_000_000},
    "hungry-vegan": {"dqn": 20_000_000, "ppo": 20_000_000},
}
RECIPES = {
    "vegan": "pacman/vegan-v0",
    "vegetarian": "pacman/vegetarian-orange-v0",
    "hungry-vegan": "pacman/hungry-vegan-v0",
    "trapped": "pacman/trapped-v1",
    "vegan-conflict": "pacman/vegan-conflict-v1",
    "hungry-vegan-penalty": "pacman/hungry-vegan-penalty-v1",
}
PILOT_NORMS = ("trapped", "vegan-conflict")
PILOT_BUDGETS = (5_000_000, 10_000_000, 20_000_000)
TRAPPED_PPO_EXTENSION_BUDGET = 30_000_000
PILOT_SEEDS = (100, 101, 102)
FINAL_SEEDS = tuple(range(8))
# Penalties follow Neufeld, Engesser and Tappler (KR 2026), except Trapped uses 1000.
# Values are in raw game-reward units, before scaling by 1/100.
PUNISHMENTS = {
    "vegan": {"VegetarianBlue": 1636.632265720805, "VegetarianOrange": 1636.632265720805},
    "vegetarian": {"VegetarianOrange": 3710.6783688206015},
    "hungry-vegan": {
        "Hungr": 4719.7584507181455,
        "VegetarianBlue": 714.172186332458,
        "VegetarianOrange": 714.172186332458,
    },
    "trapped": {"Trapped": 1000.0},
    "vegan-conflict": {"VegetarianBlue": 1000.0, "VegetarianOrange": 1000.0, "OblBlue": 3000.0},
    "hungry-vegan-penalty": {
        "Hungr": 3067.2038106118816,
        "VegetarianBlue": 426.1333673762767,
        "VegetarianOrange": 426.1333673762767,
        "CTD": 806.9335974146014,
    },
}


def experiment_id(norm: str, algorithm: str, steps: int | None = None) -> str:
    """Identify a fixed configuration, including the candidate budget for pilots."""
    if norm not in RECIPES or algorithm not in ("dqn", "ppo"):
        raise ValueError(f"Unsupported Pacman bolt configuration: {norm!r}, {algorithm!r}")
    if norm in PILOT_NORMS:
        budgets = PILOT_BUDGETS + ((TRAPPED_PPO_EXTENSION_BUDGET,) if norm == "trapped" and algorithm == "ppo" else ())
        if steps is None or steps not in budgets:
            raise ValueError(f"{norm} requires a candidate budget from {budgets}")
        return f"pacman-smallclassic-{algorithm}-bolts-{norm}-{steps // 1_000_000}m-v0"
    if steps is not None:
        raise ValueError(f"{norm} has a fixed budget; do not supply steps")
    return f"pacman-smallclassic-{algorithm}-bolts-{norm}-v0"


def layout_path(layout: str = LAYOUT) -> Path:
    """Resolve and validate the research-only asset, independently of cwd."""
    path = Path(__file__).resolve().parents[1] / layout
    if not path.is_file():
        raise FileNotFoundError(f"smallClassic experiments require the repository checkout asset: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != LAYOUT_SHA256:
        raise ValueError(f"smallClassic layout checksum differs from the paper configuration: {path}")
    return path


def make_environment(
    render_mode: Literal["human", "rgb_array"] | None,
    *,
    layout: str,
    features: str,
    ghost_behavior: Literal["random", "deterministic", "partly-deterministic"],
) -> gym.Env[Any, Any]:
    """Load the exact Berkeley layout while retaining the current random engine."""
    return PacmanEnv(
        layout=PacmanLayout.from_file(layout_path(layout)),
        features=features,
        ghost_behavior=ghost_behavior,
        render_mode=render_mode,
    )


def make_bolts(norm: str) -> Mapping[str, BoltSpec]:
    """Use minimized catalogue DFAs with the configured research punishment weights."""
    if norm not in RECIPES:
        raise ValueError(f"Unknown Pacman bolt norm base: {norm!r}")
    recipe = RECIPES[norm]
    return make_builtin_bolts(
        recipe,
        minimize=True,
        reward=0,
        rewards={f"{recipe}/{key}": -value for key, value in PUNISHMENTS[norm].items()},
    )


class TaskReward(gym.Wrapper[Any, Any, Any, Any]):
    """Keep bolt observations/diagnostics but evaluate the unscaled game reward."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, info = self.env.step(action)
        return observation, float(info["restraining_bolts"]["wrapped_reward"]), terminated, truncated, info


def wrap_bolts(env: gym.Env[Any, Any], training: bool, *, norm: str) -> gym.Env[Any, Any]:
    """Flatten bolt states before game features; scale after penalties.

    The inner 300-turn limit supplies truncation to the bolts on the same input
    as evaluation monitors. The runner retains its declared outer limit too.
    Evaluation advances identical automata but returns the raw task reward.
    Only Trapped appends its existing active-history flag to the 63 game features.
    """
    if norm == "trapped":
        env = TrappedObservation(env)
    env = RestrainingBoltWrapper(TimeLimit(env, max_episode_steps=300), bolts=make_bolts(norm))
    env = FlattenObservation(env)
    return TransformReward(env, lambda reward: float(reward) / 100) if training else TaskReward(env)
