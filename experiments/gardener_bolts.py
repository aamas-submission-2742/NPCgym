"""Gardener's collection/rescue norm base in raw task-reward units."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit

from npc_gym.bolts import BoltSpec, make_builtin_bolts
from npc_gym.envs.gardener.labels import GardenerLabel, PuddleDrained
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.builtins import RegexRecipe
from npc_gym.monitors.gardener_monitors import COLLECT_ONE_NORM_ID, rescue_recipe
from npc_gym.wrappers import RestrainingBoltWrapper

WRAPPER_ID = "gardener/collection-rescue-bolts-v0"
PUNISHMENTS = {"CollectOne": 10000.0, "Rescue": 1000.0, "Unpermitted": 100.0, "Drain": 10.0}


def make_bolts(*, num_frogs: int, num_puddles: int) -> Mapping[str, BoltSpec]:
    """Minimized DFAs charge per frog/puddle, with one terminal Collect One fee.

    Each frog has its own Rescue automaton. Multiple failures for that frog on
    one step cost 1,000; two frogs failing together cost 2,000. Derived monitor
    totals never add penalties. Collection without permission remains penalized
    even when it fulfills Rescue or Collect One. All bolts ignore reset labels.
    """
    for name, size in (("num_frogs", num_frogs), ("num_puddles", num_puddles)):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    bolts = dict(make_builtin_bolts(COLLECT_ONE_NORM_ID, reward=-PUNISHMENTS["CollectOne"]))
    for frog in range(num_frogs):
        recipe, bindings = rescue_recipe(frog)
        bolts[f"Rescue/{frog}"] = BoltSpec(recipe.compile(), bindings, -PUNISHMENTS["Rescue"], consume_initial=False)
        bolts[f"Unpermitted/{frog}"] = BoltSpec(
            RegexRecipe(".*[collected & !permitted]").compile(),
            {
                "collected": bindings["collected"],
                "permitted": lambda input: GardenerLabel.PERMITTED_COLLECT in input.labels,
            },
            -PUNISHMENTS["Unpermitted"],
            consume_initial=False,
        )
    for puddle in range(num_puddles):

        def drained(input: MonitorInput, puddle_id: int = puddle) -> bool:
            return any(isinstance(label, PuddleDrained) and label.puddle_id == puddle_id for label in input.labels)

        bolts[f"Drain/{puddle}"] = BoltSpec(
            RegexRecipe(".*[drained]").compile(),
            {"drained": drained},
            -PUNISHMENTS["Drain"],
            consume_initial=False,
        )
    return bolts


class TaskReward(gym.Wrapper[Any, Any, Any, Any]):
    """Keep identical bolt observations while reporting unadjusted task reward."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, info = self.env.step(action)
        return observation, float(info["restraining_bolts"]["wrapped_reward"]), terminated, truncated, info


def wrap_bolts(env: gym.Env[Any, Any], training: bool) -> gym.Env[Any, Any]:
    """Append one-hot DFA states; penalize training and evaluate raw returns.

    The inner 1,000-step limit exposes truncation to Collect One and Rescue on
    the same step as evaluation monitors. No reward scaling or shaping is used.
    """
    bolts = make_bolts(num_frogs=env.get_wrapper_attr("num_frogs"), num_puddles=env.get_wrapper_attr("num_puddles"))
    env = FlattenObservation(RestrainingBoltWrapper(TimeLimit(env, max_episode_steps=1000), bolts=bolts))
    return env if training else TaskReward(env)
