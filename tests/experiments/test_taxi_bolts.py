"""Taxi experiments retain full state and charge the agreed Emergency events."""

import json
from dataclasses import replace
from inspect import signature

import gymnasium as gym
import numpy as np
import pytest

from experiments.execution import (
    execute_training_batch,
    plan_training_batch,
    resolved_configuration,
    resolved_from_configuration,
)
from experiments.paper import EXPERIMENTS_BY_ENVIRONMENT
from experiments.specifications import (
    REGISTRY,
    TAXI_BOLT_EXPERIMENT_ID,
    TAXI_EXPERIMENT_ID,
    TAXI_POLICY_FIX_EXPERIMENT_ID,
    TAXI_WARN_BOLT_EXPERIMENT_ID,
    ExperimentRegistry,
)
from experiments.taxi_bolts import PUNISHMENTS, make_bolts, wrap_bolts
from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import StormTaxiEnv
from npc_gym.monitors import MonitorInput, make_builtin_monitor
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID

EXPERIMENT_IDS = (TAXI_BOLT_EXPERIMENT_ID, TAXI_WARN_BOLT_EXPERIMENT_ID)
RAIN = frozenset({"rain", "hurricane"})
ONSET = RAIN | {"newHurricane"}
SAFE = RAIN | {"atShelter"}


def test_taxi_paper_has_only_four_training_recipes_for_five_compared_policies():
    all_taxi_ids = {
        identifier
        for identifier in REGISTRY.experiment_ids
        if REGISTRY.resolve(identifier).environment.family == "taxi"
    }
    taxi_ids = set(EXPERIMENTS_BY_ENVIRONMENT["taxi"])
    assert taxi_ids == {TAXI_EXPERIMENT_ID, TAXI_POLICY_FIX_EXPERIMENT_ID, *EXPERIMENT_IDS}
    assert all_taxi_ids == taxi_ids
    baseline = REGISTRY.resolve(TAXI_EXPERIMENT_ID)
    assert [wrapper.id for wrapper in baseline.wrappers] == ["taxi/ignore-weather-relevant-v0"]
    for identifier in taxi_ids:
        resolved = REGISTRY.resolve(identifier)
        assert resolved.scenario == baseline.scenario
        assert tuple(name for name, _factory in resolved.scenario.monitor_factories) == (EMERGENCY_NORM_ID,)
        expected_steps = {
            TAXI_BOLT_EXPERIMENT_ID: 50_000_000,
            TAXI_WARN_BOLT_EXPERIMENT_ID: 10_000_000,
        }.get(identifier, 5_000_000)
        assert resolved.specification.run.training_steps == expected_steps
        assert resolved.environment.id == "taxi/storm-v1"
        assert resolved.specification.run.max_episode_steps == 50
        kwargs = resolved.algorithm.constructor_kwargs.to_dict()
        baseline_kwargs = baseline.algorithm.constructor_kwargs.to_dict()
        assert kwargs == baseline_kwargs
        assert kwargs["gamma"] == 0.99
        defaults = signature(TabularQLearning).parameters
        assert all(kwargs[name] == defaults[name].default for name in kwargs)
        assert kwargs["exploration_fraction"] * resolved.specification.run.training_steps == expected_steps / 10
        monitor = resolved.scenario.make_monitors()[EMERGENCY_NORM_ID]
        assert {name: len(member._runner.definition.states) for name, member in monitor.members.items()} == {
            "Warn Violations": 5,
            "Stay Violations": 3,
            "Seven-Step Safety Violations": 18,
            "Three-Step Safety Violations": 6,
        }


def test_corrected_labels_and_five_point_penalties_have_distinct_experiment_identity():
    assert PUNISHMENTS == {
        "Warn Violations": 5.0,
        "Seven-Step Safety Violations": 5.0,
        "Three-Step Safety Violations": 5.0,
        "Stay Violations": 5.0,
    }
    resolved = REGISTRY.resolve(TAXI_BOLT_EXPERIMENT_ID)
    assert resolved.specification.id == "taxi-tabular-bolts-emergency-50m-v1"
    assert [wrapper.id for wrapper in resolved.wrappers] == ["taxi/emergency-bolts-v4"]


def test_taxi_bolt_observations_use_minimal_automata():
    expected = {
        "Warn Violations": 5,
        "Stay Violations": 3,
        "Seven-Step Safety Violations": 18,
        "Three-Step Safety Violations": 6,
    }
    bolts = make_bolts()
    for key, spec in bolts.items():
        assert len(spec.definition.states) == expected[key.removeprefix(f"{EMERGENCY_NORM_ID}/")]
        assert len(spec.definition.minimized().states) == len(spec.definition.states)
        assert spec.reward == -PUNISHMENTS[key.removeprefix(f"{EMERGENCY_NORM_ID}/")]
    resolved = REGISTRY.resolve(TAXI_BOLT_EXPERIMENT_ID)
    with resolved.make_env(training=True) as env:
        assert sorted(env.observation_space["automata"].nvec) == sorted(expected.values())
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)


def test_warn_only_retains_10m_budget_and_observes_only_warn_state():
    resolved = REGISTRY.resolve(TAXI_WARN_BOLT_EXPERIMENT_ID)
    baseline = REGISTRY.resolve(TAXI_BOLT_EXPERIMENT_ID)
    assert resolved.specification.id == "taxi-tabular-bolts-warn-v5"
    assert resolved.algorithm == baseline.algorithm
    assert resolved.environment == baseline.environment
    assert resolved.scenario == baseline.scenario
    assert resolved.specification.run == replace(baseline.specification.run, training_steps=10_000_000)
    bolts = make_bolts("warn")
    assert list(bolts) == [f"{EMERGENCY_NORM_ID}/Warn Violations"]
    assert next(iter(bolts.values())).reward == -5.0
    with resolved.make_env(training=True) as env:
        np.testing.assert_array_equal(env.observation_space["automata"].nvec, [5])


def test_unknown_taxi_bolt_norm_base_is_rejected():
    with pytest.raises(ValueError, match="choose 'emergency' or 'warn'"):
        make_bolts("unknown")


class TraceEnv(gym.Env):
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, labels):
        self.labels = labels

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        # Reset labels must not activate Warn or a safety deadline.
        return 0, {"labels": ONSET}

    def step(self, action):
        labels = self.labels[self.index]
        self.index += 1
        return 0, -1, self.index == len(self.labels), False, {"labels": frozenset(labels)}


@pytest.mark.parametrize(
    "labels,penalties",
    [
        ([ONSET, *([RAIN] * 8)], [0, 5, *([0] * 7)]),
        ([{"rain"}, ONSET, *([RAIN] * 4)], [0, 5, 0, 0, 0, 0]),
        ([SAFE, RAIN, RAIN], [0, 5, 0]),
    ],
)
@pytest.mark.parametrize("training", [True, False])
def test_warn_only_ignores_safety_and_stay_events(labels, penalties, training):
    key = f"{EMERGENCY_NORM_ID}/Warn Violations"
    with wrap_bolts(TraceEnv(labels), training, norm="warn") as env:
        env.reset(seed=7)
        for penalty in penalties:
            observation, reward, _, _, info = env.step(0)
            assert env.observation_space.contains(observation)
            assert reward == (-1 - penalty if training else -1)
            assert info["restraining_bolts"]["reward_adjustments"] == {key: -penalty}


@pytest.mark.parametrize(
    "labels,penalties,expected_counts",
    [
        # No earlier rain: seven-step deadline, not three; expiry is counted once.
        ([ONSET, RAIN | {"warn"}, *([RAIN] * 7)], [0] * 7 + [5, 0], {"Seven-Step Safety Violations": 1}),
        # Earlier rain selects three steps, with a timely warning.
        (
            [{"rain"}, ONSET | {"warn"}, RAIN, RAIN, RAIN, RAIN],
            [0, 0, 0, 0, 5, 0],
            {"Three-Step Safety Violations": 1},
        ),
        # Simultaneous warning and Stay violations are charged once each.
        ([SAFE, RAIN, RAIN], [0, 10, 5], {"Warn Violations": 1, "Stay Violations": 2}),
        # Safety on the seven-step deadline clears the obligation; later exit violates Stay.
        ([ONSET, RAIN | {"warn"}, *([RAIN] * 5), SAFE, RAIN], [0] * 8 + [5], {"Stay Violations": 1}),
        # Safety on the three-step deadline likewise avoids its penalty.
        ([{"rain"}, ONSET | {"warn"}, RAIN, RAIN, SAFE], [0] * 5, {}),
        # A rain onset at termination has no future warning penalty.
        ([set(), {"rain"}], [0, 0], {}),
        # The terminal input can violate an existing deadline; ending early adds no penalty.
        ([ONSET, RAIN | {"warn"}, *([RAIN] * 6)], [0] * 7 + [5], {"Seven-Step Safety Violations": 1}),
        ([ONSET, RAIN | {"warn"}], [0, 0], {}),
    ],
)
@pytest.mark.parametrize("training", [True, False])
def test_component_events_penalties_and_reset(labels, penalties, expected_counts, training):
    monitor = make_builtin_monitor(EMERGENCY_NORM_ID)
    with wrap_bolts(TraceEnv(labels), training) as env:
        for _ in range(2):
            observation, info = env.reset(seed=7)
            monitor.reset(MonitorInput(info["labels"]))
            assert env.observation_space.contains(observation)
            for penalty in penalties:
                observation, reward, terminated, truncated, info = env.step(0)
                monitor.update(MonitorInput(info["labels"], terminated, truncated))
                assert env.observation_space.contains(observation)
                assert reward == (-1 - penalty if training else -1)
                assert sum(info["restraining_bolts"]["reward_adjustments"].values()) == -penalty
            assert {name: monitor.counts[name] for name in monitor.members if monitor.counts[name]} == expected_counts


@pytest.mark.parametrize("identifier", EXPERIMENT_IDS)
def test_configuration_and_round_trip(identifier):
    resolved = REGISTRY.resolve(identifier)
    old = REGISTRY.resolve(TAXI_EXPERIMENT_ID)
    assert resolved.algorithm == old.algorithm
    assert resolved.environment == old.environment
    assert resolved.scenario == old.scenario
    assert not any(wrapper.id == "taxi/ignore-weather-relevant-v0" for wrapper in resolved.wrappers)
    settings = resolved.specification.run
    assert settings.training_steps == (50_000_000 if identifier == TAXI_BOLT_EXPERIMENT_ID else 10_000_000)
    assert settings.max_episode_steps == 50
    assert settings.final_evaluation_episodes == settings.intermediate_evaluation_episodes == 1000
    assert settings.intermediate_evaluation_frequency == 250_000
    configuration = resolved_configuration(resolved)
    assert resolved_configuration(resolved_from_configuration(configuration)) == configuration


def test_all_planned_seeds():
    identifiers = EXPERIMENT_IDS
    plans = plan_training_batch("unused-taxi-output", seeds=range(8), experiment_ids=identifiers)
    assert len(plans) == len({plan.run_directory for plan in plans}) == 16
    for identifier in identifiers:
        assert {plan.seed for plan in plans if plan.resolved.specification.id == identifier} == set(range(8))
    assert all(plan.intermediate_evaluation_seed == 10_000 + plan.seed for plan in plans)
    assert all(plan.final_evaluation_seed == 10_001 + plan.seed for plan in plans)


@pytest.mark.parametrize("identifier", [TAXI_BOLT_EXPERIMENT_ID, TAXI_WARN_BOLT_EXPERIMENT_ID])
def test_full_state_and_training_evaluation_agree_through_truncation(identifier):
    bolts = REGISTRY.resolve(identifier)
    with (
        bolts.make_env(training=True) as train,
        bolts.make_env(training=False) as evaluation,
        gym.wrappers.TimeLimit(StormTaxiEnv(), max_episode_steps=50) as base,
    ):
        obs, _ = train.reset(seed=7)
        other, _ = evaluation.reset(seed=7)
        plain, _ = base.reset(seed=7)
        assert base.observation_space == gym.spaces.Discrete(352000)
        assert obs["observation"] == other["observation"] == plain
        assert len(tuple(base.unwrapped.decode(plain))) == 9
        for step in range(50):
            obs, reward, terminated, truncated, info = train.step(6)
            other, task, other_term, other_trunc, other_info = evaluation.step(6)
            plain, plain_reward, base_term, base_trunc, base_info = base.step(6)
            assert obs["observation"] == other["observation"] == plain == base.unwrapped.get_state()
            np.testing.assert_array_equal(obs["automata"], other["automata"])
            assert train.observation_space.contains(obs)
            assert (
                (terminated, truncated) == (other_term, other_trunc) == (base_term, base_trunc) == (False, step == 49)
            )
            assert task == plain_reward == info["restraining_bolts"]["wrapped_reward"] == -1
            assert reward == task + sum(info["restraining_bolts"]["reward_adjustments"].values())
            assert info["labels"] == other_info["labels"] == base_info["labels"]


@pytest.mark.parametrize("identifier", EXPERIMENT_IDS)
def test_train_reload_and_evaluate(tmp_path, identifier):
    resolved = REGISTRY.resolve(identifier)
    settings = replace(
        resolved.specification.run,
        training_steps=100,
        intermediate_evaluation_frequency=50,
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
    assert result.actual_timesteps == 100
    assert json.loads(plan.run_path.read_text())["status"] == "complete"
    summary = json.loads((plan.run_directory / "evaluations/final/summary.json").read_text())
    assert summary["metadata"]["model_source"] == "saved-checkpoint"
    assert set(summary["aggregate"]["monitor_counts"]) == {EMERGENCY_NORM_ID}
    for episode in summary["episodes"]:
        assert set(episode["monitor_counts"][EMERGENCY_NORM_ID]) == {
            "Warn Violations",
            "Stay Violations",
            "Seven-Step Safety Violations",
            "Three-Step Safety Violations",
            "Safety Violations",
            "Emergency Violations",
        }
