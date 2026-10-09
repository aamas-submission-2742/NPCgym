"""Fast, synthetic checks of this batch's configuration and reward accounting."""

import inspect
import json
from collections import deque
from dataclasses import replace
from itertools import product

import gymnasium as gym
import numpy as np
import pytest

from experiments import execution
from experiments.execution import plan_training_batch, resolved_configuration, resolved_from_configuration
from experiments.pacman_bolts import (
    BUDGETS,
    LAYOUT,
    PUNISHMENTS,
    RECIPES,
    experiment_id,
    make_bolts,
    make_environment,
    wrap_bolts,
)
from experiments.specifications import (
    PACMAN_BOLT_EXTENDED_PILOT_CONFIGURATIONS,
    PACMAN_BOLT_STUDY_CONFIGURATIONS,
    REGISTRY,
    ExperimentRegistry,
)
from npc_gym.bolts import make_builtin_bolts
from npc_gym.envs.pacman.labels import PacmanLabel as L
from npc_gym.evaluation import evaluate


@pytest.mark.parametrize(
    "norm,states",
    [
        ("vegan", (2, 2)),
        ("vegetarian", (2,)),
        ("hungry-vegan", (2, 2, 3)),
        ("trapped", (3,)),
        ("vegan-conflict", (2, 2, 5)),
        ("hungry-vegan-penalty", (2, 2, 3, 4)),
    ],
)
def test_minimized_states_preserve_every_acceptance_and_penalty(norm, states):
    reduced = make_bolts(norm)
    recipe = RECIPES[norm]
    original = make_builtin_bolts(
        recipe,
        minimize=False,
        reward=0,
        rewards={f"{recipe}/{key}": -value for key, value in PUNISHMENTS[norm].items()},
    )
    assert tuple(len(bolt.definition.states) for bolt in reduced.values()) == states
    for key, bolt in reduced.items():
        old = original[key]
        assert (bolt.reward, bolt.reporting, bolt.consume_initial) == (old.reward, old.reporting, old.consume_initial)
        left, right = old.definition, bolt.definition
        assert left.atoms == right.atoms
        valuations = [
            frozenset(atom for atom, present in zip(left.atoms, bits, strict=True) if present)
            for bits in product((False, True), repeat=len(left.atoms))
        ]
        pending = deque([(left.initial_state, right.initial_state)])
        reached = set(pending)
        while pending:
            a, b = pending.popleft()
            assert (a in left.final_states) == (b in right.final_states)
            for labels in valuations:
                pair = left.transition(labels, a), right.transition(labels, b)
                if pair not in reached:
                    reached.add(pair)
                    pending.append(pair)
    env = wrap_bolts(TraceEnv([set()]), False, norm=norm)
    try:
        assert env.observation_space.shape == (63 + (norm == "trapped") + sum(states),)
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
    finally:
        env.close()


@pytest.mark.parametrize("norm", BUDGETS)
@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
def test_configuration_and_round_trip(norm, algorithm):
    backend = pytest.importorskip("stable_baselines3")

    registry = REGISTRY
    resolved = registry.resolve(experiment_id(norm, algorithm))
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    defaults = inspect.signature((backend.DQN if algorithm == "dqn" else backend.PPO).__init__).parameters
    assert kwargs["gamma"] == 0.99
    assert all(value == defaults[key].default for key, value in kwargs.items() if key != "device")
    assert kwargs["device"] == "cpu"
    assert resolved.specification.run.training_steps == BUDGETS[norm][algorithm]
    assert resolved.specification.run.final_evaluation_episodes == 1000
    assert resolved.specification.run.max_episode_steps == 300
    assert resolved.wrappers[0].id == f"pacman/kr2026-{norm}-bolts-and-reward-v1"
    saved = resolved_configuration(resolved)
    assert resolved_configuration(resolved_from_configuration(saved, registry=registry)) == saved


def test_all_training_and_evaluation_seeds(tmp_path):
    plans = [
        plan
        for norm in BUDGETS
        for algorithm in BUDGETS[norm]
        for plan in plan_training_batch(tmp_path, seeds=range(8), experiment_ids=[experiment_id(norm, algorithm)])
    ]
    assert len(plans) == 48
    assert len({plan.run_directory for plan in plans}) == 48
    assert sum(plan.resolved.specification.run.training_steps for plan in plans) == 620_000_000
    assert all(plan.intermediate_evaluation_seed == 10000 + plan.seed for plan in plans)
    assert all(plan.final_evaluation_seed == 10001 + plan.seed for plan in plans)


class TraceEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (63,), dtype=np.float32)
    action_space = gym.spaces.Discrete(5)

    def __init__(self, labels):
        self.trace = labels

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(63, np.float32), {"labels": frozenset({L.SCORE_0})}

    def step(self, action):
        labels = self.trace[self.index]
        self.index += 1
        return np.zeros(63, np.float32), 9.0, self.index == len(self.trace), False, {"labels": frozenset(labels)}


@pytest.mark.parametrize("norm", BUDGETS)
@pytest.mark.parametrize("training", (False, True))
def test_exact_penalties_permission_and_reward_units(norm, training):
    # Miss Hungry once, then eat each ghost. Persistent deadline labels must
    # not create additional Hungry punishments or derived-total punishments.
    env = wrap_bolts(
        TraceEnv([{L.SCORE_GREATER_100}, {L.SCORE_GREATER_100}, {L.EAT_BLUE_GHOST}, {L.EAT_ORANGE_GHOST}]),
        training,
        norm=norm,
    )
    observation, _info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    weights = PUNISHMENTS[norm]
    penalties = [weights.get("Hungr", 0), 0, weights.get("VegetarianBlue", 0), weights["VegetarianOrange"]]
    for penalty in penalties:
        _observation, reward, _terminated, _truncated, info = env.step(0)
        assert sum(info["restraining_bolts"]["reward_adjustments"].values()) == pytest.approx(-penalty)
        assert reward == pytest.approx((9 - penalty) / 100 if training else 9)
    env.close()


def test_hungry_satisfied_on_deadline_and_evaluation_counts():
    registry = REGISTRY
    resolved = registry.resolve(experiment_id("hungry-vegan", "dqn"))
    env = wrap_bolts(
        TraceEnv([{L.SCORE_GREATER_100, L.EAT_BLUE_GHOST}, {L.SCORE_GREATER_100}]), False, norm="hungry-vegan"
    )
    summary = evaluate(
        env, lambda observation, info: 0, monitors=dict(resolved.scenario.monitor_factories), episodes=1, seed=0
    )
    assert summary.mean_return == 18
    assert summary.mean_monitor_counts["pacman/hungry-vegan-v0"]["Hungr"] == 0
    assert summary.mean_monitor_counts["pacman/vegan-v0"]["Vegan"] == 1
    assert summary.mean_monitor_counts["pacman/vegetarian-orange-v0"]["count"] == 0
    env.close()


@pytest.mark.parametrize("norm", RECIPES)
def test_wrapping_preserves_random_gameplay_and_base_features(norm):
    env = REGISTRY.resolve(
        experiment_id(norm, "dqn", 5_000_000) if norm in ("trapped", "vegan-conflict") else experiment_id(norm, "dqn")
    ).make_env(training=False)
    plain = make_environment(None, layout=LAYOUT, features="complete", ghost_behavior="random")
    rng = np.random.default_rng(91)
    try:
        for seed in range(3):
            obs, _info = env.reset(seed=seed)
            raw, _info = plain.reset(seed=seed)
            np.testing.assert_array_equal(obs[-64:-1] if norm == "trapped" else obs[-63:], raw)
            for _ in range(300):
                action = int(rng.integers(5))
                obs, reward, ended, truncated, info = env.step(action)
                raw, raw_reward, raw_ended, _, raw_info = plain.step(action)
                np.testing.assert_array_equal(obs[-64:-1] if norm == "trapped" else obs[-63:], raw)
                assert reward == raw_reward
                assert ended == raw_ended
                assert info["labels"] == raw_info["labels"]
                assert env.unwrapped.state() == plain.state()
                assert env.observation_space.contains(obs)
                if ended or truncated:
                    break
    finally:
        env.close()
        plain.close()


def test_inner_time_limit_reaches_bolts():
    env = wrap_bolts(TraceEnv([set()] * 301), True, norm="vegan")
    env.reset()
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.close()


@pytest.mark.parametrize("norm", RECIPES)
@pytest.mark.parametrize("algorithm", ("dqn", "ppo"))
def test_training_and_saved_checkpoint_evaluation(tmp_path, norm, algorithm):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    base = REGISTRY.resolve(
        experiment_id(norm, algorithm, 5_000_000)
        if norm in ("trapped", "vegan-conflict")
        else experiment_id(norm, algorithm)
    )
    specification = replace(
        base.specification,
        run=replace(
            base.specification.run,
            training_steps=2048,
            intermediate_evaluation_frequency=2048,
            intermediate_evaluation_episodes=2,
            final_evaluation_episodes=2,
        ),
    )
    registry = ExperimentRegistry(
        environments=(base.environment,),
        wrappers=base.wrappers,
        scenarios=(base.scenario,),
        algorithms=(base.algorithm,),
        techniques=(base.technique,),
        experiments=(specification,),
    )
    try:
        plans = execution.plan_training_batch(tmp_path, seeds=[0], registry=registry)
        (result,) = execution.execute_training_batch(plans)
        assert result.actual_timesteps == 2048
        document = json.loads(plans[0].run_path.read_text())
        assert document["status"] == "complete"
        assert result.final_evaluation.mean_return == result.final_evaluation.mean_metrics["score"]
        (repeated,) = execution.execute_evaluation_batch(
            execution.plan_evaluation_batch(tmp_path, registry=registry, name="repeat", seeds=[0])
        )
        assert repeated.summary == result.final_evaluation
        before = plans[0].run_path.read_bytes()
        with pytest.raises(execution.PreflightError, match="already exists"):
            execution.execute_training_batch(plans)
        assert plans[0].run_path.read_bytes() == before
    finally:
        torch.set_num_threads(previous)


def test_layout_validation_before_outputs(tmp_path, monkeypatch):
    from experiments import pacman_bolts

    bad = tmp_path / "altered.lay"
    bad.write_text("not the paper layout")
    with pytest.raises(ValueError, match="checksum"):
        pacman_bolts.layout_path(str(bad))
    with pytest.raises(FileNotFoundError, match="checkout asset"):
        pacman_bolts.layout_path(str(tmp_path / "missing.lay"))

    def missing(layout):
        raise FileNotFoundError("checkout asset missing")

    monkeypatch.setattr(pacman_bolts, "layout_path", missing)
    plans = execution.plan_training_batch(tmp_path / "runs", seeds=[0], experiment_ids=[experiment_id("vegan", "dqn")])
    with pytest.raises(FileNotFoundError, match="checkout asset"):
        execution.preflight_training(plans)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "norm,algorithm,steps,identifier",
    (*PACMAN_BOLT_STUDY_CONFIGURATIONS, *PACMAN_BOLT_EXTENDED_PILOT_CONFIGURATIONS),
)
def test_study_configurations(norm, algorithm, steps, identifier):
    resolved = REGISTRY.resolve(identifier)
    assert resolved.specification.run.training_steps == steps
    assert resolved.specification.run.final_evaluation_episodes == 1000
    assert resolved.specification.run.max_episode_steps == 300
    assert (
        resolved.algorithm.constructor_kwargs
        == REGISTRY.resolve(experiment_id("vegan", algorithm)).algorithm.constructor_kwargs
    )
    saved = resolved_configuration(resolved)
    assert resolved_configuration(resolved_from_configuration(saved)) == saved
    assert set(dict(resolved.scenario.monitor_factories)) == set(RECIPES.values())


@pytest.mark.parametrize(
    "norm,labels,penalties,flags",
    [
        (
            "trapped",
            [set(), {L.WEST_SIDE}, {L.EAT_BLUE_GHOST}, {L.SCORE_GREATER_400, L.EAT_BLUE_GHOST}, set()],
            [1000, 0, 1000, 0, 0],
            [1, 1, 1, 0, 0],
        ),
        (
            "vegan-conflict",
            [
                {L.ADJACENT_BLUE_GHOST},
                set(),
                set(),
                set(),
                {L.ADJACENT_BLUE_GHOST},
                {L.EAT_BLUE_GHOST},
                {L.EAT_ORANGE_GHOST},
            ],
            [0, 0, 0, 3000, 3000, 1000, 1000],
            None,
        ),
        (
            "vegan-conflict",
            [{L.ADJACENT_BLUE_GHOST}, set(), set(), {L.EAT_BLUE_GHOST}, set()],
            [0, 0, 0, 1000, 0],
            None,
        ),
        (
            "hungry-vegan-penalty",
            [{L.SCORE_GREATER_100}, {L.EAT_BLUE_GHOST}, set(), {L.EAT_ORANGE_GHOST}, {L.STAYED_STILL}],
            [3067.2038106118816, 426.1333673762767, 806.9335974146014, 426.1333673762767, 0],
            None,
        ),
    ],
)
@pytest.mark.parametrize("training", (False, True))
def test_additional_norm_penalties_and_history(norm, labels, penalties, flags, training):
    env = wrap_bolts(TraceEnv(labels), training, norm=norm)
    try:
        obs, _ = env.reset()
        if flags:
            assert obs[-1] == 1
        for index, penalty in enumerate(penalties):
            obs, reward, _, _, info = env.step(0)
            assert env.observation_space.contains(obs)
            assert sum(info["restraining_bolts"]["reward_adjustments"].values()) == pytest.approx(-penalty)
            assert reward == pytest.approx((9 - penalty) / 100 if training else 9)
            if flags:
                assert obs[-1] == flags[index]
    finally:
        env.close()


def test_candidate_budget_required_and_fixed_budget_rejected():
    with pytest.raises(ValueError, match="candidate budget"):
        experiment_id("trapped", "dqn")
    with pytest.raises(ValueError, match="candidate budget"):
        experiment_id("vegan-conflict", "dqn", 1_000_000)
    with pytest.raises(ValueError, match="candidate budget"):
        experiment_id("trapped", "dqn", 30_000_000)
    assert experiment_id("trapped", "ppo", 30_000_000).endswith("trapped-30m-v0")
    with pytest.raises(ValueError, match="fixed budget"):
        experiment_id("vegan", "dqn", 5_000_000)
