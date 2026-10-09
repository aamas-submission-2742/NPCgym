from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest
from gymnasium.wrappers import TimeLimit, TransformReward

from experiments.specifications import (
    ALGORITHMS,
    ENVIRONMENTS,
    EXPERIMENTS,
    GARDENER_ACTION_ID,
    GARDENER_BOLT_EXPERIMENT_IDS,
    GARDENER_DRAIN_OFTEN_EXPERIMENT_ID,
    GARDENER_EXPERIMENT_ID,
    GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID,
    GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID,
    GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID,
    GARDENER_PPO_EXPERIMENT_ID,
    MERCHANT_ENVIRONMENT_ID,
    MERCHANT_EXPERIMENT_ID,
    MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS,
    MERCHANT_OBSERVATION_ID,
    MERCHANT_POLICY_FIX_EXPERIMENT_ID,
    PACMAN_BOLT_EXPERIMENT_IDS,
    PACMAN_DQN_EXPERIMENT_ID,
    PACMAN_EXPERIMENT_ID,
    PACMAN_IMAGES_EXPERIMENT_ID,
    PACMAN_OFTEN_EXPERIMENT_ID,
    PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID,
    PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID,
    REGISTRY,
    SCENARIOS,
    TAXI_BOLT_EXPERIMENT_ID,
    TAXI_EXPERIMENT_ID,
    TAXI_POLICY_FIX_EXPERIMENT_ID,
    TAXI_WARN_BOLT_EXPERIMENT_ID,
    TECHNIQUES,
    WRAPPERS,
    AlgorithmSpecification,
    DuplicateSpecificationIDError,
    EnvironmentSpecification,
    ExperimentRegistry,
    ExperimentSpecification,
    KeywordArguments,
    SpecificationError,
    TechniqueSpecification,
    UnknownSpecificationIDError,
    UnsupportedCombinationError,
)
from npc_gym.wrappers.gardener_wrappers import IllegalActionPenaltyWrapper, StateFeatureObsWrapper
from npc_gym.wrappers.merchant_wrappers import IgnoreTimeObservation
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_BASELINES = {
    TAXI_EXPERIMENT_ID: {
        "environment": {
            "fickle_passenger": False,
        },
        "wrappers": ["taxi/ignore-weather-relevant-v0"],
        "monitors": ["taxi/emergency-v0"],
        "algorithm": {
            "learning_rate": 0.2,
            "gamma": 0.99,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.1,
            "use_action_mask": False,
            "log_interval": 10_000,
        },
        "run": (5_000_000, 50, 1_000, 1_000, 250_000, {}, "identity"),
    },
    MERCHANT_EXPERIMENT_ID: {
        "environment": {
            "layout": "basic",
            "risk_fight": 0.75,
            "risk_death": 0.25,
            "capacity": 5,
            "sunset": 28,
        },
        "wrappers": ["merchant/ignore-time-v0"],
        "monitors": [
            "merchant/env-friendly-v0",
            "merchant/delivery-v0",
            "merchant/pacifist-v0",
            "merchant/delivery-pacifist-v0",
        ],
        "algorithm": {
            "learning_rate": 0.5,
            "gamma": 0.99,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.2,
            "use_action_mask": True,
            "log_interval": 10_000,
        },
        "run": (5_000_000, 150, 1_000, 1_000, 250_000, {}, "identity"),
    },
    GARDENER_EXPERIMENT_ID: {
        "environment": {
            "size": 15,
            "grass_respawn": 50,
            "puddle_respawn": 20,
            "score_limit": 300,
            "frog_freeze": 5,
        },
        "wrappers": ["gardener/state-features-v0", GARDENER_ACTION_ID],
        "monitors": [
            "gardener/permission-aware-v0",
            "gardener/drain-v0",
            "gardener/rescue-v1",
        ],
        "algorithm": {
            "learning_rate": 5e-4,
            "buffer_size": 50_000,
            "learning_starts": 5_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "replay_buffer_class": None,
            "replay_buffer_kwargs": None,
            "optimize_memory_usage": False,
            "target_update_interval": 2_000,
            "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "max_grad_norm": 10.0,
            "stats_window_size": 100,
            "policy_kwargs": {"net_arch": [32, 32]},
            "verbose": 1,
            "device": "cpu",
        },
        "run": (
            100_000,
            1_000,
            1_000,
            1_000,
            5_000,
            {"log_interval": 100},
            "illegal actions become stay with penalty -1",
        ),
    },
    PACMAN_EXPERIMENT_ID: {
        "environment": {"layout": "small", "features": "complete", "dfas": None},
        "wrappers": ["pacman/training-reward-divide-100-v0"],
        "monitors": [
            "pacman/vegan-v0",
            "pacman/vegetarian-blue-v0",
            "pacman/vegetarian-orange-v0",
            "pacman/conditional-vegan-v0",
            "pacman/penalty-v0",
            "pacman/hungry-v0",
            "pacman/hungry-vegan-v0",
            "pacman/hungry-vegan-penalty-v1",
            "pacman/vegan-preference-v0",
            "pacman/hungry-vegetarian-v0",
            "pacman/trapped-v1",
        ],
        "algorithm": {
            "learning_rate": 3e-4,
            "n_steps": 2_048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "rollout_buffer_class": None,
            "rollout_buffer_kwargs": None,
            "target_kl": None,
            "stats_window_size": 100,
            "policy_kwargs": None,
            "verbose": 1,
            "device": "cpu",
        },
        "run": (5_000_000, 300, 1_000, 1_000, 250_000, {"log_interval": 1}, "reward divided by 100"),
    },
}


@pytest.mark.parametrize("experiment_id", sorted(EXPECTED_BASELINES))
def test_baseline_resolves_legacy_settings(experiment_id: str) -> None:
    expected = EXPECTED_BASELINES[experiment_id]
    resolved = REGISTRY.resolve(experiment_id)
    run = resolved.specification.run

    assert resolved.environment.constructor_kwargs.to_dict() == expected["environment"]
    assert [wrapper.id for wrapper in resolved.wrappers] == expected["wrappers"]
    assert list(resolved.make_monitors()) == expected["monitors"]
    assert resolved.algorithm.constructor_kwargs.to_dict() == expected["algorithm"]
    assert (
        run.training_steps,
        run.max_episode_steps,
        run.final_evaluation_episodes,
        run.intermediate_evaluation_episodes,
        run.intermediate_evaluation_frequency,
        run.learning_kwargs.to_dict(),
        run.training_reward_transform,
    ) == expected["run"]
    assert run.evaluation_seed_offset == 10_000
    assert run.deterministic_evaluation is True
    assert run.task_return_units == "unscaled task reward"
    assert "seed" not in resolved.algorithm.constructor_kwargs.to_dict()


@pytest.mark.parametrize("experiment_id", sorted(EXPECTED_BASELINES))
def test_factories_create_fresh_environments_and_monitors(experiment_id: str) -> None:
    resolved = REGISTRY.resolve(experiment_id)
    first_env = resolved.make_env(training=False)
    second_env = resolved.make_env(training=False)
    try:
        assert first_env is not second_env
        first_observation, _ = first_env.reset(seed=7)
        second_observation, _ = second_env.reset(seed=7)
        assert first_env.observation_space.contains(first_observation)
        assert second_env.observation_space.contains(second_observation)
    finally:
        first_env.close()
        second_env.close()

    first_monitors = resolved.make_monitors()
    second_monitors = resolved.make_monitors()
    assert first_monitors.keys() == second_monitors.keys()
    assert all(first_monitors[monitor_id] is not second_monitors[monitor_id] for monitor_id in first_monitors)


def test_training_and_evaluation_wrapper_resolution() -> None:
    taxi = REGISTRY.resolve(TAXI_EXPERIMENT_ID).make_env(training=True)
    merchant = REGISTRY.resolve(MERCHANT_EXPERIMENT_ID).make_env(training=True)
    gardener_train = REGISTRY.resolve(GARDENER_EXPERIMENT_ID).make_env(training=True)
    gardener_eval = REGISTRY.resolve(GARDENER_EXPERIMENT_ID).make_env(training=False)
    pacman_train = REGISTRY.resolve(PACMAN_EXPERIMENT_ID).make_env(training=True)
    pacman_eval = REGISTRY.resolve(PACMAN_EXPERIMENT_ID).make_env(training=False)
    try:
        assert isinstance(taxi, TimeLimit) and isinstance(taxi.env, IgnoreWeatherRelevant)
        assert isinstance(merchant, TimeLimit) and isinstance(merchant.env, IgnoreTimeObservation)
        assert isinstance(gardener_train.env, IllegalActionPenaltyWrapper)
        assert isinstance(gardener_train.env.env, StateFeatureObsWrapper)
        assert isinstance(gardener_eval.env, IllegalActionPenaltyWrapper)
        assert gardener_train.env._penalty == -1.0
        assert gardener_eval.env._penalty == 0.0
        assert isinstance(pacman_train.env, TransformReward)
        assert not isinstance(pacman_eval.env, TransformReward)
    finally:
        for env in (taxi, merchant, gardener_train, gardener_eval, pacman_train, pacman_eval):
            env.close()


def test_pacman_rewards_are_scaled_only_during_training() -> None:
    resolved = REGISTRY.resolve(PACMAN_EXPERIMENT_ID)
    train_env = resolved.make_env(training=True)
    eval_env = resolved.make_env(training=False)
    try:
        train_env.reset(seed=0)
        eval_env.reset(seed=0)
        rewards = [(train_env.step(3)[1], eval_env.step(3)[1]) for _ in range(10)]
    finally:
        train_env.close()
        eval_env.close()

    assert {task_reward for _training_reward, task_reward in rewards} == {9.0, -1.0}
    assert all(training_reward == pytest.approx(task_reward / 100) for training_reward, task_reward in rewards)


def test_render_arguments_apply_only_to_rendered_environments() -> None:
    calls: list[tuple[str | None, dict[str, object]]] = []

    def factory(render_mode: str | None, **kwargs: object) -> gym.Env[Any, Any]:
        calls.append((render_mode, kwargs))
        return gym.Env()

    environment = EnvironmentSpecification(
        "test/environment-v0",
        "test",
        factory,
        KeywordArguments.from_mapping({"size": 3}),
        KeywordArguments.from_mapping({"step_delay_ms": 500}),
    )
    environment.make()
    environment.make(render=True)

    assert calls == [(None, {"size": 3}), ("human", {"size": 3, "step_delay_ms": 500})]


def test_environment_is_closed_when_a_wrapper_fails() -> None:
    class ClosingEnv(gym.Env[int, int]):
        closed = False

        def close(self) -> None:
            self.closed = True

    env = ClosingEnv()

    def fail(_env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
        raise RuntimeError("wrapper failed")

    environment = replace(ENVIRONMENTS[0], factory=lambda render_mode: env, constructor_kwargs=KeywordArguments(()))
    wrapper = replace(WRAPPERS[0], factory=fail)
    registry = ExperimentRegistry(
        environments=(environment,),
        wrappers=(wrapper,),
        scenarios=SCENARIOS[:1],
        algorithms=ALGORITHMS[:1],
        techniques=TECHNIQUES,
        experiments=EXPERIMENTS[:1],
    )

    with pytest.raises(RuntimeError, match="wrapper failed"):
        registry.resolve(TAXI_EXPERIMENT_ID).make_env(training=True)
    assert env.closed


def test_resolution_and_keyword_arguments_are_deterministic_and_detached() -> None:
    assert REGISTRY.experiment_ids == tuple(
        sorted(
            (
                *EXPECTED_BASELINES,
                MERCHANT_POLICY_FIX_EXPERIMENT_ID,
                TAXI_POLICY_FIX_EXPERIMENT_ID,
                TAXI_BOLT_EXPERIMENT_ID,
                TAXI_WARN_BOLT_EXPERIMENT_ID,
                PACMAN_OFTEN_EXPERIMENT_ID,
                PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID,
                PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID,
                GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID,
                GARDENER_DRAIN_OFTEN_EXPERIMENT_ID,
                GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID,
                GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID,
                GARDENER_PPO_EXPERIMENT_ID,
                PACMAN_DQN_EXPERIMENT_ID,
                PACMAN_IMAGES_EXPERIMENT_ID,
                *PACMAN_BOLT_EXPERIMENT_IDS,
                *MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS,
                *GARDENER_BOLT_EXPERIMENT_IDS,
            )
        )
    )
    first = REGISTRY.resolve(GARDENER_EXPERIMENT_ID)
    second = REGISTRY.resolve(GARDENER_EXPERIMENT_ID)
    assert first == second

    arguments = first.algorithm.constructor_kwargs.to_dict()
    arguments["policy_kwargs"]["net_arch"].append(64)  # type: ignore[index,union-attr]
    assert first.algorithm.constructor_kwargs.to_dict()["policy_kwargs"] == {"net_arch": [32, 32]}


def test_keyword_arguments_reject_non_serializable_values() -> None:
    with pytest.raises(TypeError, match="unsupported value"):
        KeywordArguments.from_mapping({"callback": object()})

    with pytest.raises(TypeError, match="names must be strings"):
        KeywordArguments.from_mapping({1: "value"})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="must be finite"):
        KeywordArguments.from_mapping({"learning_rate": float("nan")})


def test_duplicate_component_and_experiment_ids_are_rejected() -> None:
    with pytest.raises(DuplicateSpecificationIDError, match="Duplicate environment ID"):
        ExperimentRegistry(environments=(ENVIRONMENTS[0], ENVIRONMENTS[0]))
    with pytest.raises(DuplicateSpecificationIDError, match="Duplicate experiment ID"):
        ExperimentRegistry(experiments=(EXPERIMENTS[0], EXPERIMENTS[0]))

    duplicate_wrapper = replace(EXPERIMENTS[0], wrapper_ids=EXPERIMENTS[0].wrapper_ids * 2)
    with pytest.raises(DuplicateSpecificationIDError, match="duplicate wrapper IDs"):
        _registry_with(duplicate_wrapper)

    duplicate_monitor = replace(SCENARIOS[0], monitor_factories=SCENARIOS[0].monitor_factories * 2)
    with pytest.raises(DuplicateSpecificationIDError, match="duplicate monitor IDs"):
        ExperimentRegistry(environments=ENVIRONMENTS, scenarios=(duplicate_monitor,))


@pytest.mark.parametrize(
    ("field", "unknown_id", "component"),
    [
        ("environment_id", "environment/missing-v0", "environment"),
        ("wrapper_ids", ("wrapper/missing-v0",), "wrapper"),
        ("scenario_id", "scenario/missing-v0", "scenario"),
        ("algorithm_id", "algorithm/missing-v0", "algorithm"),
        ("technique_id", "technique/missing-v0", "technique"),
    ],
)
def test_unknown_component_references_are_rejected(field: str, unknown_id: object, component: str) -> None:
    experiment = replace(EXPERIMENTS[0], **{field: unknown_id})
    with pytest.raises(UnknownSpecificationIDError, match=rf"Unknown {component} ID"):
        _registry_with(experiment)


@pytest.mark.parametrize(
    ("registry_arguments", "error", "match"),
    [
        ({"environments": (replace(ENVIRONMENTS[0], id="Taxi"),)}, SpecificationError, "lowercase versioned ID"),
        ({"experiments": (replace(EXPERIMENTS[0], id="taxi-baseline"),)}, SpecificationError, "lowercase versioned ID"),
        (
            {"experiments": (replace(EXPERIMENTS[0], id="a/../../escape-v0"),)},
            SpecificationError,
            "filesystem-safe lowercase versioned ID",
        ),
        ({"environments": (replace(ENVIRONMENTS[0], family=""),)}, SpecificationError, "must declare a family"),
        ({"environments": (replace(ENVIRONMENTS[0], factory=None),)}, TypeError, "factory must be callable"),
        ({"wrappers": (replace(WRAPPERS[0], factory=None),)}, TypeError, "factory must be callable"),
        ({"wrappers": (replace(WRAPPERS[0], environment_ids=frozenset()),)}, SpecificationError, "at least one"),
        ({"wrappers": (replace(WRAPPERS[0], execution_paths=frozenset({"jax"})),)}, SpecificationError, "paths"),
        (
            {"wrappers": (replace(WRAPPERS[0], environment_ids=frozenset({"environment/missing-v0"})),)},
            UnknownSpecificationIDError,
            "Unknown environment ID",
        ),
        ({"scenarios": (replace(SCENARIOS[0], environment_ids=frozenset()),)}, SpecificationError, "at least one"),
        (
            {"scenarios": (replace(SCENARIOS[0], monitor_factories=(("Emergency", object),)),)},
            SpecificationError,
            "lowercase versioned ID",
        ),
        (
            {"scenarios": (replace(SCENARIOS[0], monitor_factories=(("taxi/emergency-v0", None),)),)},
            TypeError,
            "factory must be callable",
        ),
        ({"algorithms": (replace(ALGORITHMS[0], execution_path="jax"),)}, SpecificationError, "execution path"),
        ({"algorithms": (replace(ALGORITHMS[0], environment_ids=frozenset()),)}, SpecificationError, "at least one"),
        ({"techniques": (replace(TECHNIQUES[0], execution_paths=frozenset()),)}, SpecificationError, "paths"),
    ],
)
def test_invalid_components_are_rejected(
    registry_arguments: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        ExperimentRegistry(**{"environments": ENVIRONMENTS, **registry_arguments})


def test_unknown_experiment_id_is_rejected() -> None:
    with pytest.raises(UnknownSpecificationIDError, match="Unknown experiment ID"):
        REGISTRY.resolve("missing-experiment-v0")


@pytest.mark.parametrize("retired", [0, 1])
def test_merchant_versions_identify_capacity_enforcement_and_reject_retired_ids(retired: int) -> None:
    assert MERCHANT_ENVIRONMENT_ID == "merchant/basic-v2"
    assert MERCHANT_EXPERIMENT_ID == "merchant-tabular-unconstrained-v2"
    resolved = REGISTRY.resolve(MERCHANT_EXPERIMENT_ID)
    assert resolved.environment.id == MERCHANT_ENVIRONMENT_ID
    with pytest.raises(UnknownSpecificationIDError, match=MERCHANT_EXPERIMENT_ID):
        REGISTRY.resolve(f"merchant-tabular-unconstrained-v{retired}")
    with pytest.raises(UnknownSpecificationIDError, match=MERCHANT_ENVIRONMENT_ID):
        _registry_with(replace(resolved.specification, environment_id=f"merchant/basic-v{retired}"))


def test_known_but_unsupported_component_combination_is_rejected() -> None:
    experiment = replace(EXPERIMENTS[0], scenario_id=EXPERIMENTS[1].scenario_id)
    with pytest.raises(UnsupportedCombinationError, match="does not support environment"):
        _registry_with(experiment)

    experiment = replace(EXPERIMENTS[0], algorithm_id=EXPERIMENTS[1].algorithm_id)
    with pytest.raises(UnsupportedCombinationError, match="does not support environment"):
        _registry_with(experiment)

    experiment = replace(EXPERIMENTS[0], wrapper_ids=(MERCHANT_OBSERVATION_ID,))
    with pytest.raises(UnsupportedCombinationError, match="does not support environment"):
        _registry_with(experiment)

    taxi_dqn = replace(ALGORITHMS[2], id="sb3/dqn-taxi-v0", environment_ids=frozenset({ENVIRONMENTS[0].id}))
    experiment = replace(EXPERIMENTS[0], algorithm_id=taxi_dqn.id)
    with pytest.raises(UnsupportedCombinationError, match="does not support execution path 'sb3'"):
        _registry_with(experiment, algorithms=(*ALGORITHMS, taxi_dqn))

    sb3_only = replace(TECHNIQUES[0], execution_paths=frozenset({"sb3"}))
    with pytest.raises(UnsupportedCombinationError, match="does not support execution path 'tabular'"):
        _registry_with(EXPERIMENTS[0], techniques=(sb3_only,))


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"training_steps": 0}, "training_steps must be a positive integer"),
        ({"max_episode_steps": True}, "max_episode_steps must be a positive integer"),
        ({"evaluation_seed_offset": -1}, "evaluation_seed_offset must be a non-negative integer"),
        ({"deterministic_evaluation": 1}, "deterministic_evaluation must be a bool"),
        ({"task_return_units": ""}, "task_return_units must not be empty"),
        ({"training_reward_transform": ""}, "training_reward_transform must not be empty"),
    ],
)
def test_invalid_run_settings_are_rejected(changes: dict[str, object], match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        replace(EXPERIMENTS[0].run, **changes)


def test_specifications_import_without_optional_dependencies() -> None:
    code = """
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src")]
import experiments.specifications as specifications

optional = ["stable_baselines3", "torch", "optuna", "matplotlib", "pandas"]
assert specifications.REGISTRY.experiment_ids
print(json.dumps({name: name in sys.modules for name in optional}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "matplotlib": False,
        "optuna": False,
        "pandas": False,
        "stable_baselines3": False,
        "torch": False,
    }


def _registry_with(
    experiment: ExperimentSpecification,
    *,
    algorithms: tuple[AlgorithmSpecification, ...] = ALGORITHMS,
    techniques: tuple[TechniqueSpecification, ...] = TECHNIQUES,
) -> ExperimentRegistry:
    return ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=algorithms,
        techniques=techniques,
        experiments=(experiment,),
    )


def test_new_pacman_maps_version_the_experiment():
    assert PACMAN_EXPERIMENT_ID == "pacman-ppo-unconstrained-v2"
    with pytest.raises(UnknownSpecificationIDError):
        REGISTRY.resolve("pacman-ppo-unconstrained-v0")


def test_retired_pacman_map_experiments_are_not_silently_reinterpreted():
    with pytest.raises(UnknownSpecificationIDError):
        REGISTRY.resolve("pacman-ppo-unconstrained-v1")


@pytest.mark.parametrize("specification", EXPERIMENTS, ids=lambda spec: spec.id)
def test_all_experiments_evaluate_1000_episodes_per_seed(specification):
    assert specification.run.final_evaluation_episodes == 1000
    assert specification.run.intermediate_evaluation_episodes == 1000


def test_every_merchant_recipe_monitors_all_paper_norms_without_changing_budget():
    expected = {
        "merchant/env-friendly-v0",
        "merchant/delivery-v0",
        "merchant/pacifist-v0",
        "merchant/delivery-pacifist-v0",
    }
    for specification in EXPERIMENTS:
        resolved = REGISTRY.resolve(specification.id)
        if resolved.environment.family == "merchant":
            assert set(resolved.make_monitors()) == expected
            assert specification.run.training_steps == 5_000_000
