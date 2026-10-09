"""Emergency penalties for the full-observation Storm Taxi paper experiment."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from npc_gym.bolts import BoltSpec, make_builtin_bolts
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID
from npc_gym.wrappers import RestrainingBoltWrapper

PUNISHMENTS = {
    "Warn Violations": 5.0,
    "Seven-Step Safety Violations": 5.0,
    "Three-Step Safety Violations": 5.0,
    "Stay Violations": 5.0,
}
WRAPPER_ID = "taxi/emergency-bolts-v4"
WARN_WRAPPER_ID = "taxi/warn-bolts-v0"


def make_bolts(norm: str = "emergency") -> Mapping[str, BoltSpec]:
    """Create minimized bolts for ``emergency`` or just its ``warn`` component.

    Charge each selected component event once; simultaneous penalties add.
    Preserve the existing rain-dependent deadlines and reset semantics.
    Derived Safety and Emergency totals do not incur additional penalties.
    """
    if norm not in {"emergency", "warn"}:
        raise ValueError(f"Unknown Taxi bolt norm base: {norm!r}; choose 'emergency' or 'warn'")
    bolts = make_builtin_bolts(
        EMERGENCY_NORM_ID,
        reward=0,
        minimize=True,
        rewards={f"{EMERGENCY_NORM_ID}/{name}": -value for name, value in PUNISHMENTS.items()},
    )
    if norm == "warn":
        key = f"{EMERGENCY_NORM_ID}/Warn Violations"
        return {key: bolts[key]}
    return bolts


class TaskReward(gym.Wrapper[Any, Any, Any, Any]):
    """Keep bolt observations during evaluation while reporting raw task rewards."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, info = self.env.step(action)
        return observation, float(info["restraining_bolts"]["wrapped_reward"]), terminated, truncated, info


def wrap_bolts(env: gym.Env[Any, Any], training: bool, *, norm: str = "emergency") -> gym.Env[Any, Any]:
    """Append discrete bolt states, subtracting unscaled penalties in training.

    Keep all nine environment fields and expose the 50-step truncation to the
    bolts. Evaluation advances identical bolt states but returns task rewards.
    """
    env = RestrainingBoltWrapper(TimeLimit(env, max_episode_steps=50), bolts=make_bolts(norm), state_encoding="index")
    return env if training else TaskReward(env)
