"""Real-environment regressions for signals caused by episode-ending transitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from npc_gym.envs import StormTaxiEnv
from npc_gym.envs.gardener.gardener import GardenerEnv
from npc_gym.envs.gardener.gardener import action_dict as gardener_actions
from npc_gym.envs.merchant.merchant import MerchantEnv
from npc_gym.envs.merchant.merchant import action_dict as merchant_actions
from npc_gym.envs.pacman.pacman_env import PacmanEnv
from npc_gym.evaluation import EvaluationSummary, TerminationClass, evaluate
from npc_gym.monitors.builtins import make_builtin_monitor
from npc_gym.monitors.gardener_monitors import COLLECT_ONE_NORM_ID
from npc_gym.monitors.merchant_monitors import DELIVERY_NORM_ID, PACIFIST_NORM_ID
from npc_gym.monitors.pacman_monitors import HUNGRY_NORM_ID
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID


class ArrangedReset(gym.Wrapper[Any, Any, Any, Any]):
    """Arrange private domain state after reset so the scripted steps end the episode."""

    def __init__(self, env: gym.Env[Any, Any], arrange: Callable[[Any], None]) -> None:
        super().__init__(env)
        self._arrange = arrange

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        result = self.env.reset(**kwargs)
        self._arrange(self.env.unwrapped)
        return result


def _evaluate_script(
    env: gym.Env[Any, Any], actions: tuple[int, ...], make_monitor: Callable[[str], Any], *norm_ids: str
) -> EvaluationSummary:
    remaining = list(actions)
    try:
        return evaluate(
            env,
            lambda observation, info: remaining.pop(0),
            monitors={monitor_id: lambda monitor_id=monitor_id: make_monitor(monitor_id) for monitor_id in norm_ids},
            seed=0,
        )
    finally:
        env.close()


def test_gardener_counts_a_frog_collected_by_the_truncating_transition():
    def arrange(env: GardenerEnv) -> None:
        env.agent = env.frogs[1].copy()
        env._move_frogs = lambda: None

    env = TimeLimit(ArrangedReset(GardenerEnv(size=15), arrange), max_episode_steps=1)
    summary = _evaluate_script(env, (gardener_actions["stay"],), make_builtin_monitor, COLLECT_ONE_NORM_ID)

    (episode,) = summary.episodes
    assert episode.termination is TerminationClass.TRUNCATED
    assert episode.monitor_counts == {COLLECT_ONE_NORM_ID: {"count": 0}}


def test_merchant_counts_the_fatal_fight_that_terminates_the_episode():
    def arrange(env: MerchantEnv) -> None:
        env.label = "D"  # the agent is attacked

    env = ArrangedReset(MerchantEnv(layout="basic", risk_death=1.0), arrange)
    summary = _evaluate_script(
        env, (merchant_actions["north"], merchant_actions["fight"]), make_builtin_monitor, PACIFIST_NORM_ID
    )

    (episode,) = summary.episodes
    assert episode.termination is TerminationClass.TERMINATED
    assert episode.monitor_counts == {PACIFIST_NORM_ID: {"Danger": 2, "CTD": 1, "Pacifist(total)": 3}}


def test_taxi_counts_the_missing_warning_on_the_truncating_transition():
    def arrange(env: StormTaxiEnv) -> None:
        # The first step observes rain; the second must respond, including when truncated.
        env.s = env.encode(0, 0, 4, 1, True, 0, 0, False, 2)

    env = TimeLimit(ArrangedReset(StormTaxiEnv(), arrange), max_episode_steps=2)
    summary = _evaluate_script(env, (0, 0), make_builtin_monitor, EMERGENCY_NORM_ID)

    (episode,) = summary.episodes
    assert episode.termination is TerminationClass.TRUNCATED
    assert episode.monitor_counts[EMERGENCY_NORM_ID]["Warn Violations"] == 1


def test_pacman_counts_the_score_reached_by_the_truncating_transition():
    def arrange(env: PacmanEnv) -> None:
        env._sim.score = 102

    env = TimeLimit(ArrangedReset(PacmanEnv(layout="small", features="essential"), arrange), max_episode_steps=1)
    summary = _evaluate_script(env, (0,), make_builtin_monitor, HUNGRY_NORM_ID)

    (episode,) = summary.episodes
    assert episode.termination is TerminationClass.TRUNCATED
    assert episode.monitor_counts == {HUNGRY_NORM_ID: {"count": 1}}


def test_initial_home_activates_delivery_in_repeated_one_step_episodes():
    env = TimeLimit(MerchantEnv(layout="basic", risk_death=0.0, sunset=1), max_episode_steps=1)
    try:
        result = evaluate(
            env,
            lambda observation, info: merchant_actions["east"],
            episodes=3,
            monitors={DELIVERY_NORM_ID: lambda: make_builtin_monitor(DELIVERY_NORM_ID)},
            seed=4,
        )
    finally:
        env.close()
    for episode in result.episodes:
        assert episode.length == 1
        assert episode.termination == "truncated"
        assert episode.monitor_counts == {DELIVERY_NORM_ID: {"count": 1}}
