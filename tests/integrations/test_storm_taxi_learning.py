"""Storm Taxi learning, persistence, monitored evaluation and policy fixing."""

from contextlib import ExitStack

import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit

from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import StormTaxiEnv
from npc_gym.evaluation import evaluate
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID, make_taxi_monitor
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant


def test_short_learning_save_load_and_monitored_evaluation_are_reproducible(tmp_path):
    with ExitStack() as stack:
        envs = [
            stack.enter_context(TimeLimit(IgnoreWeatherRelevant(env), max_episode_steps=20))
            for env in (StormTaxiEnv(), StormTaxiEnv())
        ]
        models = [TabularQLearning(env, seed=7, log_interval=None) for env in envs]
        for model in models:
            model.learn(400)
        assert models[0].q_table.keys() == models[1].q_table.keys()
        for key in models[0].q_table:
            np.testing.assert_array_equal(models[0].q_table[key], models[1].q_table[key])
        checkpoint = tmp_path / "storm-policy.zip"
        models[0].save(checkpoint)
        loaded = TabularQLearning.load(checkpoint, env=envs[1])
        assert models[0].q_table.keys() == loaded.q_table.keys()
        for key in models[0].q_table:
            np.testing.assert_array_equal(models[0].q_table[key], loaded.q_table[key])
        summaries = [
            evaluate(
                env,
                model,
                episodes=4,
                seed=19,
                monitors={EMERGENCY_NORM_ID: lambda: make_taxi_monitor(EMERGENCY_NORM_ID)},
            )
            for env, model in zip(envs, (models[0], loaded), strict=True)
        ]
        assert summaries[0].episodes == summaries[1].episodes


def test_composed_taxi_supports_real_warning_policy_fixes():
    pytest.importorskip("clingo")
    from npc_gym.policy_fixes import ASPPlanner, TaxiModel

    model, planner = TaxiModel(), ASPPlanner()
    monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
    interventions = 0
    with StormTaxiEnv() as env, StormTaxiEnv() as reference:
        for seed in range(4):
            _, info = env.reset(seed=seed)
            reference.reset(seed=seed)
            model.reset()
            monitor.reset(MonitorInput(info["labels"]))
            for _ in range(50):
                decision = planner.solve(model.problem(), dict.fromkeys(range(7), 0.0))
                actual, expected = env.step(decision.action), reference.step(decision.action)
                assert actual[:4] == expected[:4]
                assert actual[4]["labels"] == expected[4]["labels"]
                event = MonitorInput(actual[4]["labels"], actual[2], actual[3])
                model.advance(event)
                monitor.update(event)
                assert monitor.counts["Warn Violations"] == 0
                interventions += decision.changed
                if actual[2] or actual[3]:
                    break
    assert interventions > 0
