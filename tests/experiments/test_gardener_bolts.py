"""Gardener bolt accounting, observation parity and reproducible training."""

import inspect
import json
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from experiments.execution import (
    execute_training_batch,
    plan_training_batch,
    resolved_configuration,
    resolved_from_configuration,
)
from experiments.gardener_bolts import make_bolts, wrap_bolts
from experiments.specifications import GARDENER_BOLT_EXPERIMENT_IDS, REGISTRY, ExperimentRegistry, KeywordArguments
from npc_gym.envs.gardener.labels import FrogCollected, GardenerLabel, PuddleDrained
from npc_gym.monitors import MonitorInput


class TraceEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (27,), dtype=np.float32)
    action_space = gym.spaces.Discrete(5)
    num_frogs = 2
    num_puddles = 4

    def __init__(self, labels, *, truncate=False):
        self.labels = labels
        self.truncate = truncate

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        # Reset must not satisfy Collect One or trigger Rescue.
        return np.zeros(27, np.float32), {"labels": frozenset({FrogCollected(0), PuddleDrained(0, (1,))})}

    def step(self, action):
        labels = self.labels[self.index]
        self.index += 1
        end = self.index == len(self.labels)
        return (
            np.zeros(27, np.float32),
            9.9,
            end and not self.truncate,
            end and self.truncate,
            {"labels": frozenset(labels)},
        )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize(
    "labels,penalties",
    [
        ([set()], [10000]),
        ([{FrogCollected(0), FrogCollected(1)}], [200]),
        ([{FrogCollected(0), FrogCollected(1), GardenerLabel.PERMITTED_COLLECT}], [0]),
        ([{PuddleDrained(0, (0, 1)), PuddleDrained(1, (0,))}], [12020]),
        ([{PuddleDrained(0, (0, 1))}, {FrogCollected(0)}], [10, 1100]),
        (
            [{PuddleDrained(0, (0, 1))}] + [set()] * 5 + [{FrogCollected(0), FrogCollected(1)}],
            [10, 0, 0, 0, 0, 0, 2200],
        ),
    ],
)
def test_per_frog_penalties_match_monitors(labels, penalties, training, truncate):
    monitors = REGISTRY.resolve(GARDENER_BOLT_EXPERIMENT_IDS[0]).make_monitors()
    with wrap_bolts(TraceEnv(labels, truncate=truncate), training) as env:
        for _ in range(2):
            observation, info = env.reset(seed=7)
            assert env.observation_space.contains(observation)
            assert all(value == 0 for value in info["restraining_bolts"]["reward_adjustments"].values())
            for monitor in monitors.values():
                monitor.reset(MonitorInput(info["labels"]))
            total = 0
            for penalty in penalties:
                observation, reward, terminated, truncated, info = env.step(0)
                assert env.observation_space.contains(observation)
                assert reward == pytest.approx(9.9 - penalty if training else 9.9)
                assert sum(info["restraining_bolts"]["reward_adjustments"].values()) == -penalty
                for monitor in monitors.values():
                    monitor.update(MonitorInput(info["labels"], terminated, truncated))
                total += penalty
            assert total == (
                10000 * monitors["gardener/collect-one-v0"].count
                + 1000 * monitors["gardener/rescue-v2"].counts["Rescue"]
                + 100 * monitors["gardener/permission-aware-v0"].counts["Unpermitted"]
                + 10 * monitors["gardener/drain-v0"].counts["Drain"]
            )


def test_inner_limit_penalizes_failure_to_collect():
    with wrap_bolts(TraceEnv([set()] * 1001), True) as env:
        env.reset()
        for _ in range(999):
            _, reward, terminated, truncated, _ = env.step(4)
            assert reward == 9.9 and not terminated and not truncated
        _, reward, terminated, truncated, _ = env.step(4)
        assert not terminated and truncated
        assert reward == pytest.approx(9.9 - 10000)


@pytest.mark.parametrize("identifier", GARDENER_BOLT_EXPERIMENT_IDS)
def test_configuration_defaults_and_saved_round_trip(identifier):
    sb3 = pytest.importorskip("stable_baselines3")
    resolved = REGISTRY.resolve(identifier)
    backend = sb3.DQN if "-dqn-" in identifier else sb3.PPO
    defaults = inspect.signature(backend.__init__).parameters
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    assert all(value == defaults[key].default for key, value in kwargs.items() if key != "device")
    assert kwargs["gamma"] == 0.99
    settings = resolved.specification.run
    assert settings.training_steps == 1000000
    assert settings.max_episode_steps == settings.final_evaluation_episodes == 1000
    assert settings.intermediate_evaluation_frequency == 50000
    assert settings.intermediate_evaluation_episodes == 1000
    saved = resolved_configuration(resolved)
    assert resolved_configuration(resolved_from_configuration(saved)) == saved


def test_seeds_and_environment_parity(tmp_path):
    plans = plan_training_batch(tmp_path, experiment_ids=GARDENER_BOLT_EXPERIMENT_IDS, seeds=range(8))
    assert len(plans) == len({p.run_directory for p in plans}) == 16
    assert all(p.intermediate_evaluation_seed == 10000 + p.seed for p in plans)
    assert all(p.final_evaluation_seed == 10001 + p.seed for p in plans)
    resolved = REGISTRY.resolve(GARDENER_BOLT_EXPERIMENT_IDS[0])
    with resolved.make_env(training=True) as training, resolved.make_env(training=False) as evaluation:
        a, _ = training.reset(seed=37)
        b, _ = evaluation.reset(seed=37)
        np.testing.assert_array_equal(a, b)
        rng = np.random.default_rng(42)
        for _ in range(1000):
            action = int(rng.integers(5))
            a, _, ended, truncated, info = training.step(action)
            b, reward, other_end, other_truncated, other_info = evaluation.step(action)
            np.testing.assert_array_equal(a, b)
            assert training.observation_space.contains(a)
            assert (ended, truncated) == (other_end, other_truncated)
            assert info["labels"] == other_info["labels"]
            assert reward == other_info["restraining_bolts"]["wrapped_reward"]
            assert training.unwrapped.labeling_state() == evaluation.unwrapped.labeling_state()
            if ended or truncated:
                break


@pytest.mark.parametrize("identifier", GARDENER_BOLT_EXPERIMENT_IDS)
def test_training_reload_and_final_evaluation(tmp_path, identifier):
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    resolved = REGISTRY.resolve(identifier)
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    kwargs.update(
        {"buffer_size": 128, "learning_starts": 4}
        if "-dqn-" in identifier
        else {"n_steps": 32, "n_epochs": 1, "batch_size": 16}
    )
    settings = replace(
        resolved.specification.run,
        training_steps=64,
        intermediate_evaluation_frequency=64,
        intermediate_evaluation_episodes=1,
        final_evaluation_episodes=1,
    )
    registry = ExperimentRegistry(
        environments=(resolved.environment,),
        wrappers=resolved.wrappers,
        scenarios=(resolved.scenario,),
        algorithms=(replace(resolved.algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs)),),
        techniques=(resolved.technique,),
        experiments=(replace(resolved.specification, run=settings),),
    )
    (plan,) = plan_training_batch(tmp_path, experiment_ids=[identifier], seeds=[0], registry=registry)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        (result,) = execute_training_batch((plan,))
    finally:
        torch.set_num_threads(previous)
    assert result.actual_timesteps == 64
    assert json.loads(plan.run_path.read_text())["status"] == "complete"
    summary = json.loads((plan.run_directory / "evaluations/final/summary.json").read_text())
    assert summary["metadata"]["model_source"] == "saved-checkpoint"
    assert set(summary["aggregate"]["monitor_counts"]) == set(resolved.make_monitors())


@pytest.mark.parametrize("size", [-1, True, 1.5])
def test_invalid_object_counts(size):
    with pytest.raises(ValueError, match="num_frogs"):
        make_bolts(num_frogs=size, num_puddles=4)
    with pytest.raises(ValueError, match="num_puddles"):
        make_bolts(num_frogs=2, num_puddles=size)
