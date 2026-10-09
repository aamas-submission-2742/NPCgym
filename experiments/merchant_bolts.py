"""Agreed Merchant norm bases and penalties in raw task-reward units."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from npc_gym.bolts import BoltSpec, make_builtin_bolts
from npc_gym.monitors.merchant_monitors import DELIVERY_PACIFIST_NORM_ID, ENV_FRIENDLY_NORM_ID
from npc_gym.wrappers import RestrainingBoltWrapper

RECIPES = {
    "env-friendly": ENV_FRIENDLY_NORM_ID,
    "delivery-pacifist": DELIVERY_PACIFIST_NORM_ID,
}
WRAPPER_IDS = {
    "env-friendly": "merchant/env-friendly-bolts-v0",
    "delivery-pacifist": "merchant/delivery-pacifist-bolts-v1",
}
MINIMIZED_WRAPPER_IDS = {norm: f"merchant/{norm}-minimized-bolts-v0" for norm in RECIPES}
PUNISHMENTS = {
    "env-friendly": {"EnvFriendly": 300.0},
    "delivery-pacifist": {"Delivery": 10000.0, "Danger": 100.0, "CTD": 1000.0},
}


def make_bolts(norm: str, *, minimize: bool = False) -> Mapping[str, BoltSpec]:
    """Penalize each component once with the configured research weights.

    Simultaneous violations add their penalties. The monitor's derived total
    is diagnostic and never contributes an additional penalty. The default
    retains the original experiment encoding; new minimized recipes opt in.
    """
    if norm not in RECIPES:
        raise ValueError(f"Unknown Merchant bolt norm base: {norm!r}")
    recipe = RECIPES[norm]
    return make_builtin_bolts(
        recipe,
        minimize=minimize,
        reward=0,
        rewards={f"{recipe}/{key}": -value for key, value in PUNISHMENTS[norm].items()},
    )


class TaskReward(gym.Wrapper[Any, Any, Any, Any]):
    """Advance the same bolt states but report the unadjusted task reward."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, info = self.env.step(action)
        return observation, float(info["restraining_bolts"]["wrapped_reward"]), terminated, truncated, info


def wrap_bolts(env: gym.Env[Any, Any], training: bool, *, norm: str, minimize: bool = False) -> gym.Env[Any, Any]:
    """Append discrete bolt states to the supplied Merchant observation.

    Training subtracts penalties without reward scaling, matching the tabular
    baseline's units. Evaluation returns raw task rewards. Both retain the
    action mask and expose truncation to the bolts at the 150-step limit.
    """
    env = RestrainingBoltWrapper(
        TimeLimit(env, max_episode_steps=150), bolts=make_bolts(norm, minimize=minimize), state_encoding="index"
    )
    return env if training else TaskReward(env)
