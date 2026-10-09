"""Merchant bolt penalties match monitor events without duplicate charges."""

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
from experiments.merchant_bolts import RECIPES, make_bolts, wrap_bolts
from experiments.specifications import (
    MERCHANT_ENVIRONMENT_ID,
    MERCHANT_EXPERIMENT_ID,
    MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS,
    MERCHANT_POLICY_FIX_EXPERIMENT_ID,
    REGISTRY,
    ExperimentRegistry,
    UnknownSpecificationIDError,
)
from npc_gym.envs.merchant.labels import MerchantLabel as M
from npc_gym.monitors import MonitorInput, MultiMonitor, make_builtin_monitor


class TraceEnv(gym.Env):
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, labels, rewards):
        self.labels = labels
        self.rewards = rewards

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return 0, {"labels": frozenset(self.labels[0])}

    def step(self, action):
        self.index += 1
        return (
            0,
            self.rewards[self.index - 1],
            self.index == len(self.labels) - 1,
            False,
            {"labels": frozenset(self.labels[self.index])},
        )


@pytest.mark.parametrize(
    "norm,labels,rewards,adjusted,counts",
    [
        (
            "delivery-pacifist",
            [(M.AT_HOME,), (M.AT_DANGER,), (M.AT_DANGER, M.FIGHT, M.SUNDOWN), (M.UNLOAD,)],
            [0, 0, -50],
            [-100, -11100, -50],
            {"Delivery": 1, "Danger": 2, "CTD": 1, "DeliveryPacifist": 4},
        ),
        (
            "delivery-pacifist",
            [(M.AT_HOME,), (M.AT_DANGER,), (M.FIGHT,), (M.SUNDOWN,)],
            [0, 0, 0],
            [-100, -1000, -10000],
            {"Delivery": 1, "Danger": 1, "CTD": 1, "DeliveryPacifist": 3},
        ),
        (
            "delivery-pacifist",
            [(M.AT_HOME,), (M.AT_DANGER,), (M.UNLOAD,), (M.AT_MARKET,), (M.SUNDOWN,)],
            [0, -50, 0, 0],
            [-100, -50, 0, 0],
            {"Delivery": 0, "Danger": 1, "CTD": 0, "DeliveryPacifist": 1},
        ),
        (
            "env-friendly",
            [(M.AT_TREE, M.HAS_WOOD), (M.EXTRACT,), (M.AT_MARKET, M.UNLOAD)],
            [50, 100],
            [-250, 100],
            {"count": 1},
        ),
    ],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("minimize", [False, True])
def test_rewards_and_counts_across_resets(norm, labels, rewards, adjusted, counts, training, minimize):
    monitor = make_builtin_monitor(RECIPES[norm])
    with wrap_bolts(TraceEnv(labels, rewards), training, norm=norm, minimize=minimize) as env:
        for _ in range(2):
            observation, info = env.reset(seed=7)
            monitor.reset(MonitorInput(info["labels"]))
            assert env.observation_space.contains(observation)
            for task_reward, expected in zip(rewards, adjusted, strict=True):
                observation, reward, terminated, truncated, info = env.step(0)
                monitor.update(MonitorInput(info["labels"], terminated, truncated))
                assert env.observation_space.contains(observation)
                assert reward == (expected if training else task_reward)
                assert info["restraining_bolts"]["wrapped_reward"] == task_reward
            assert (monitor.counts if isinstance(monitor, MultiMonitor) else {"count": monitor.count}) == counts


def test_unknown_norm_base():
    with pytest.raises(ValueError, match="Unknown Merchant bolt norm base"):
        make_bolts("missing")


@pytest.mark.parametrize("identifier", MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS)
def test_configuration_and_saved_round_trip(identifier):
    resolved = REGISTRY.resolve(identifier)
    baseline = REGISTRY.resolve(MERCHANT_EXPERIMENT_ID)
    assert resolved.algorithm == baseline.algorithm
    assert resolved.environment == baseline.environment
    if identifier == MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS[0]:
        assert resolved.wrappers[0] == baseline.wrappers[0]
    else:
        assert len(resolved.wrappers) == 1
        assert resolved.wrappers[0].kind == "reward"
    settings = resolved.specification.run
    assert settings.training_steps == 5_000_000
    assert settings.max_episode_steps == 150
    assert settings.final_evaluation_episodes == settings.intermediate_evaluation_episodes == 1000
    assert settings.intermediate_evaluation_frequency == 250_000
    assert "merchant/evolving-v0" not in dict(resolved.scenario.monitor_factories)
    configuration = resolved_configuration(resolved)
    assert resolved_configuration(resolved_from_configuration(configuration)) == configuration


def test_all_planned_seeds(tmp_path):
    identifiers = MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS
    plans = plan_training_batch(tmp_path, seeds=range(8), experiment_ids=identifiers)
    assert len(plans) == 16
    assert len({plan.run_directory for plan in plans}) == 16
    for identifier in identifiers:
        assert {plan.seed for plan in plans if plan.resolved.specification.id == identifier} == set(range(8))
    assert all(plan.intermediate_evaluation_seed == 10_000 + plan.seed for plan in plans)
    assert all(plan.final_evaluation_seed == 10_001 + plan.seed for plan in plans)


@pytest.mark.parametrize("training", [True, False])
def test_delivery_pacifist_observes_clock_and_deadline_in_training_and_evaluation(training):
    resolved = REGISTRY.resolve(MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS[1])
    assert resolved.specification.id == "merchant-tabular-bolts-delivery-pacifist-minimized-v1"
    with resolved.make_env(training=training) as env:
        observation, _ = env.reset(seed=7)
        assert len(observation["observation"]) == len(env.unwrapped.observation_space.spaces)
        assert observation["observation"][5] == 0
        assert env.observation_space.contains(observation)
        for step in range(1, 31):
            # Leave home, then advance time without changing position or inventory.
            observation, reward, terminated, truncated, info = env.step(0 if step == 1 else 6)
            assert not (terminated or truncated)
            assert observation["observation"][5] == min(step, 28) == env.unwrapped.labeling_state().time
            assert (M.SUNDOWN in info["labels"]) == (step >= 28)
            assert env.observation_space.contains(observation)
            assert reward == (-10000 if training and step == 28 else 0)


@pytest.mark.parametrize("identifier", MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS)
def test_train_checkpoint_reload_and_final_evaluation(tmp_path, identifier):
    resolved = REGISTRY.resolve(identifier)
    settings = replace(
        resolved.specification.run,
        training_steps=300,
        intermediate_evaluation_frequency=150,
        intermediate_evaluation_episodes=2,
        final_evaluation_episodes=2,
    )
    registry = ExperimentRegistry(
        environments=(resolved.environment,),
        wrappers=resolved.wrappers,
        scenarios=(resolved.scenario,),
        algorithms=(resolved.algorithm,),
        techniques=(resolved.technique,),
        experiments=(replace(resolved.specification, run=settings),),
    )
    (plan,) = plan_training_batch(tmp_path, seeds=[0], experiment_ids=[identifier], registry=registry)
    (result,) = execute_training_batch((plan,))
    assert result.actual_timesteps == 300
    metadata = json.loads(plan.run_path.read_text())
    assert metadata["status"] == "complete"
    summary = json.loads((plan.run_directory / "evaluations/final/summary.json").read_text())
    assert summary["metadata"]["model_source"] == "saved-checkpoint"
    assert set(summary["aggregate"]["monitor_counts"]) == set(dict(resolved.scenario.monitor_factories))


@pytest.mark.parametrize("identifier", MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS)
def test_training_and_evaluation_keep_observations_masks_and_task_rewards_identical(identifier):
    resolved = REGISTRY.resolve(identifier)
    with resolved.make_env(training=True) as training, resolved.make_env(training=False) as evaluation:
        observation, info = training.reset(seed=7)
        other, other_info = evaluation.reset(seed=7)
        assert observation["observation"] == other["observation"]
        np.testing.assert_array_equal(observation["automata"], other["automata"])
        assert (info["action_mask"] == other_info["action_mask"]).all()
        # Fight at home does nothing and reaches the exact episode limit.
        for step in range(150):
            observation, reward, terminated, truncated, info = training.step(6)
            other, task_reward, other_terminated, other_truncated, other_info = evaluation.step(6)
            assert observation["observation"] == other["observation"]
            np.testing.assert_array_equal(observation["automata"], other["automata"])
            assert training.observation_space.contains(observation)
            assert (terminated, truncated) == (other_terminated, other_truncated) == (False, step == 149)
            assert (info["action_mask"] == other_info["action_mask"]).all()
            assert task_reward == info["restraining_bolts"]["wrapped_reward"] == 0
            assert reward == sum(info["restraining_bolts"]["reward_adjustments"].values())


@pytest.mark.parametrize("index,dimensions", [(0, [4]), (1, [4, 2, 3])])
def test_selected_bolt_recipes_use_minimal_automata(index, dimensions):
    minimized = REGISTRY.resolve(MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS[index])
    norm = tuple(RECIPES)[index]
    for spec in make_bolts(norm, minimize=True).values():
        assert len(spec.definition.states) == len(spec.definition.minimized().states)
    with minimized.make_env(training=True) as env:
        assert env.observation_space["automata"].nvec.tolist() == dimensions
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)


def test_environment_selection_has_exactly_the_four_paper_recipes(tmp_path):
    expected = {MERCHANT_EXPERIMENT_ID, MERCHANT_POLICY_FIX_EXPERIMENT_ID, *MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS}
    plans = plan_training_batch(tmp_path, seeds=range(8), environment_ids=[MERCHANT_ENVIRONMENT_ID])
    assert len(plans) == 32
    assert {plan.resolved.specification.id for plan in plans} == expected
    baseline = REGISTRY.resolve(MERCHANT_EXPERIMENT_ID)
    fixed = REGISTRY.resolve(MERCHANT_POLICY_FIX_EXPERIMENT_ID)
    assert baseline.algorithm == fixed.algorithm
    assert baseline.environment == fixed.environment
    assert baseline.wrappers == fixed.wrappers
    for plan in plans:
        settings = plan.resolved.specification.run
        assert settings.training_steps == 5_000_000
        assert settings.final_evaluation_episodes == settings.intermediate_evaluation_episodes == 1000
        assert settings.max_episode_steps == 150
        assert set(plan.resolved.make_monitors()) == set(baseline.make_monitors())
    for retired in ("merchant-tabular-bolts-env-friendly-v0", "merchant-tabular-bolts-delivery-pacifist-v1"):
        with pytest.raises(UnknownSpecificationIDError):
            REGISTRY.resolve(retired)
