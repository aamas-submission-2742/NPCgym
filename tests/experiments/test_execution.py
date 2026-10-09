from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

from experiments import execution, run
from experiments.execution import (
    ConfigurationMismatchError,
    PreflightError,
    SavedRunError,
    SelectionError,
    execute_evaluation_batch,
    execute_training_batch,
    plan_evaluation_batch,
    plan_training_batch,
    preflight_training,
    resolved_configuration,
)
from experiments.specifications import (
    ALGORITHMS,
    ENVIRONMENTS,
    EXPERIMENTS,
    GARDENER_EXPERIMENT_ID,
    MERCHANT_EXPERIMENT_ID,
    PACMAN_EXPERIMENT_ID,
    SCENARIOS,
    TAXI_EXPERIMENT_ID,
    TECHNIQUES,
    WRAPPERS,
    AlgorithmSpecification,
    EnvironmentSpecification,
    ExperimentRegistry,
    ExperimentSpecification,
    KeywordArguments,
    RunSettings,
    ScenarioSpecification,
    TechniqueSpecification,
)
from npc_gym.monitors import SimpleMonitor

ROOT = Path(__file__).resolve().parents[2]


def _tiny_registry(index: int) -> ExperimentRegistry:
    algorithm = ALGORITHMS[index]
    if algorithm.execution_path == "sb3":
        if algorithm.implementation.endswith(".DQN"):
            kwargs = {
                "learning_rate": 5e-4,
                "buffer_size": 50,
                "learning_starts": 0,
                "batch_size": 4,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "replay_buffer_class": None,
                "replay_buffer_kwargs": None,
                "optimize_memory_usage": False,
                "target_update_interval": 5,
                "exploration_fraction": 0.3,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "max_grad_norm": 10.0,
                "stats_window_size": 10,
                "policy_kwargs": {"net_arch": [8]},
                "verbose": 0,
                "device": "cpu",
            }
        else:
            kwargs = {
                "learning_rate": 3e-4,
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
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
                "stats_window_size": 10,
                "policy_kwargs": {"net_arch": [8]},
                "verbose": 0,
                "device": "cpu",
            }
        algorithm = replace(algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs))
    algorithms = (*ALGORITHMS[:index], algorithm, *ALGORITHMS[index + 1 :])
    settings = replace(
        EXPERIMENTS[index].run,
        training_steps=8 if algorithm.execution_path == "sb3" else 4,
        final_evaluation_episodes=1,
        intermediate_evaluation_episodes=1,
        intermediate_evaluation_frequency=4 if algorithm.execution_path == "sb3" else 2,
        learning_kwargs=KeywordArguments.from_mapping({"log_interval": 1} if algorithm.execution_path == "sb3" else {}),
    )
    experiment = replace(EXPERIMENTS[index], run=settings)
    return ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=algorithms,
        techniques=TECHNIQUES,
        experiments=(experiment,),
    )


@pytest.mark.parametrize(
    ("index", "experiment_id"),
    [
        (0, TAXI_EXPERIMENT_ID),
        (1, MERCHANT_EXPERIMENT_ID),
        (2, GARDENER_EXPERIMENT_ID),
        (3, PACMAN_EXPERIMENT_ID),
    ],
)
def test_tiny_train_save_load_and_evaluate_flows(tmp_path: Path, index: int, experiment_id: str) -> None:
    registry = _tiny_registry(index)
    (plan,) = plan_training_batch(tmp_path, seeds=[7], registry=registry, experiment_ids=[experiment_id])

    (result,) = execute_training_batch((plan,))

    document = json.loads(plan.run_path.read_text(encoding="utf-8"))
    assert document["status"] == "complete"
    assert document["experiment_id"] == experiment_id
    assert document["seed"] == 7
    assert document["seeds"] == {
        "training": 7,
        "intermediate_evaluation": 10_007,
        "final_evaluation": 10_008,
    }
    assert document["timesteps"] == {
        "requested": plan.resolved.specification.run.training_steps,
        "actual": result.actual_timesteps,
    }
    assert result.actual_timesteps >= plan.resolved.specification.run.training_steps
    assert document["artifacts"]["model"]["sha256"] == execution.sha256_file(plan.model_path)
    assert document["provenance"]["device"] == "cpu"
    assert plan.curve_path.is_file()
    assert (plan.run_directory / "evaluations" / "final" / "episodes.csv").is_file()
    summary = json.loads((plan.run_directory / "evaluations" / "final" / "summary.json").read_text())
    assert summary["schema_version"] == 5
    assert set(summary["aggregate"]["monitor_counts"]) == set(dict(plan.resolved.scenario.monitor_factories))
    assert summary["metadata"]["model_source"] == "saved-checkpoint"
    assert summary["metadata"]["model_sha256"] == result.model_checksum
    intermediate = min((plan.run_directory / "evaluations").glob("intermediate-*"))
    intermediate_summary = json.loads((intermediate / "summary.json").read_text())
    assert intermediate_summary["metadata"]["model_source"] == "in-memory-policy"
    assert intermediate_summary["metadata"]["model_sha256"] is None
    assert intermediate_summary["metadata"]["scheduled_thresholds"] == [0]
    assert intermediate_summary["metadata"]["timesteps"] == 0
    frequency = plan.resolved.specification.run.intermediate_evaluation_frequency
    expected = list(range(0, result.actual_timesteps + 1, frequency))
    with plan.curve_path.open() as stream:
        assert [int(row["timesteps"]) for row in csv.DictReader(stream)] == expected
    assert {path.name for path in (plan.run_directory / "evaluations").iterdir()} == {
        "final",
        *(f"intermediate-{step}" for step in expected),
    }
    if plan.resolved.algorithm.execution_path == "sb3":
        _assert_sb3_parameters_updated(plan)


def _assert_sb3_parameters_updated(plan: execution.TrainingPlan) -> None:
    import torch
    from stable_baselines3 import DQN, PPO

    algorithm_class = DQN if plan.resolved.algorithm.implementation.endswith(".DQN") else PPO
    saved_env = plan.resolved.make_env(training=False)
    initial_env = plan.resolved.make_env(training=True)
    try:
        saved = algorithm_class.load(plan.model_path, env=saved_env, device="cpu")
        initial = algorithm_class(
            plan.resolved.algorithm.policy,
            initial_env,
            seed=plan.seed,
            **plan.resolved.algorithm.constructor_kwargs.to_dict(),
        )
        saved_parameters = saved.policy.state_dict()
        initial_parameters = initial.policy.state_dict()
        assert any(not torch.equal(saved_parameters[name], initial_parameters[name]) for name in saved_parameters), (
            "tiny SB3 training must perform at least one parameter update"
        )
    finally:
        saved_env.close()
        initial_env.close()


def test_named_evaluation_uses_saved_configuration_and_never_changes_model(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    (training_plan,) = plan_training_batch(tmp_path, seeds=[3], registry=registry)
    execute_training_batch((training_plan,))
    checksum = execution.sha256_file(training_plan.model_path)

    current = registry.resolve(TAXI_EXPERIMENT_ID)
    changed_algorithm = replace(
        current.algorithm,
        constructor_kwargs=KeywordArguments.from_mapping(
            {**current.algorithm.constructor_kwargs.to_dict(), "learning_rate": 0.9}
        ),
    )
    changed_experiment = replace(current.specification, run=replace(current.specification.run, training_steps=99))
    changed_registry = ExperimentRegistry(
        environments=(current.environment,),
        wrappers=current.wrappers,
        scenarios=(current.scenario,),
        algorithms=(changed_algorithm,),
        techniques=(current.technique,),
        experiments=(changed_experiment,),
    )
    (evaluation_plan,) = plan_evaluation_batch(
        tmp_path,
        registry=changed_registry,
        experiment_ids=[TAXI_EXPERIMENT_ID],
        seeds=[3],
        name="replication",
        evaluation_seed=42,
        episodes=2,
    )
    assert evaluation_plan.resolved.algorithm.constructor_kwargs.to_dict()["learning_rate"] == 0.2
    assert evaluation_plan.resolved.specification.run.training_steps == 4

    (result,) = execute_evaluation_batch((evaluation_plan,))

    assert len(result.summary.episodes) == 2
    assert execution.sha256_file(training_plan.model_path) == checksum
    metadata = json.loads((result.output_directory / "summary.json").read_text())["metadata"]
    assert metadata["seed"] == 42
    assert metadata["model_sha256"] == checksum
    with pytest.raises(PreflightError, match="already exists"):
        execute_evaluation_batch((evaluation_plan,))
    execute_evaluation_batch((evaluation_plan,), overwrite=True)
    assert execution.sha256_file(training_plan.model_path) == checksum

    assert (
        run.main(
            [
                "evaluate",
                "--output",
                str(tmp_path),
                "--experiment",
                TAXI_EXPERIMENT_ID,
                "--seed",
                "3",
                "--name",
                "cli-check",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (training_plan.run_directory / "evaluations" / "cli-check").exists()


def test_cli_is_import_safe_and_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = """
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / 'src')]
import experiments.run
from experiments.execution import plan_training_batch, preflight_training
from experiments.specifications import TAXI_EXPERIMENT_ID
plans = plan_training_batch(Path.cwd() / 'output', seeds=[0], experiment_ids=[TAXI_EXPERIMENT_ID])
preflight_training(plans)
print(json.dumps([name for name in ('stable_baselines3', 'torch', 'optuna', 'matplotlib', 'pandas') if name in sys.modules]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)], check=True, capture_output=True, text=True, cwd=tmp_path
    )
    assert json.loads(completed.stdout) == []
    for command in (
        [sys.executable, str(ROOT / "experiments" / "run.py"), "--help"],
        [sys.executable, "-m", "experiments.run", "--help"],
    ):
        help_result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
        assert "{pacman-bolt-pilots,pacman-baselines,train,evaluate,report,retain}" in help_result.stdout

    output = tmp_path / "not-created"
    assert (
        run.main(
            [
                "train",
                "--output",
                str(output),
                "--experiment",
                TAXI_EXPERIMENT_ID,
                "--seed",
                "0",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)[0]["operation"] == "train"


def test_tabular_runs_do_not_import_optional_dependencies(tmp_path: Path) -> None:
    code = """
import json
import sys
from dataclasses import replace
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / 'src')]
from experiments import execution
from experiments.specifications import ALGORITHMS, ENVIRONMENTS, EXPERIMENTS, SCENARIOS, TECHNIQUES, WRAPPERS
from experiments.specifications import ExperimentRegistry
settings = dict(
    training_steps=4, final_evaluation_episodes=1, intermediate_evaluation_episodes=1, intermediate_evaluation_frequency=2
)
experiment = replace(EXPERIMENTS[0], run=replace(EXPERIMENTS[0].run, **settings))
registry = ExperimentRegistry(
    environments=ENVIRONMENTS,
    wrappers=WRAPPERS,
    scenarios=SCENARIOS,
    algorithms=ALGORITHMS,
    techniques=TECHNIQUES,
    experiments=(experiment,),
)
execution.execute_training_batch(execution.plan_training_batch(Path.cwd(), seeds=[0], registry=registry))
execution.execute_evaluation_batch(execution.plan_evaluation_batch(Path.cwd(), registry=registry, name='again'))
print(json.dumps([name for name in ('stable_baselines3', 'torch', 'optuna', 'matplotlib', 'pandas') if name in sys.modules]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)], check=True, capture_output=True, text=True, cwd=tmp_path
    )

    assert json.loads(completed.stdout) == []
    assert (tmp_path / TAXI_EXPERIMENT_ID / "seed-0" / "evaluations" / "again" / "summary.json").is_file()


def test_batch_preflight_happens_before_any_output_is_written(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    plans = plan_training_batch(tmp_path, seeds=[1, 2], registry=registry)
    plans[1].run_directory.mkdir(parents=True)

    with pytest.raises(PreflightError, match="already exists"):
        execute_training_batch(plans)

    assert not plans[0].run_directory.exists()


def test_batch_preflight_rejects_mixed_configurations_for_one_experiment_identity(tmp_path: Path) -> None:
    first_registry = _tiny_registry(0)
    changed = first_registry.resolve(TAXI_EXPERIMENT_ID)
    changed_experiment = replace(
        changed.specification,
        run=replace(changed.specification.run, training_steps=changed.specification.run.training_steps + 1),
    )
    second_registry = ExperimentRegistry(
        environments=(changed.environment,),
        wrappers=changed.wrappers,
        scenarios=(changed.scenario,),
        algorithms=(changed.algorithm,),
        techniques=(changed.technique,),
        experiments=(changed_experiment,),
    )
    first = plan_training_batch(tmp_path, seeds=[1], registry=first_registry)[0]
    second = plan_training_batch(tmp_path, seeds=[2], registry=second_registry)[0]

    with pytest.raises(ConfigurationMismatchError, match="different resolved settings within the training batch"):
        preflight_training((first, second))

    assert not first.run_directory.exists()
    assert not second.run_directory.exists()


def test_training_preflight_revalidates_manually_constructed_seed_bounds(tmp_path: Path) -> None:
    plan = plan_training_batch(tmp_path, seeds=[0], registry=_tiny_registry(0))[0]
    seed = 2**32
    invalid = replace(
        plan,
        seed=seed,
        intermediate_evaluation_seed=seed + plan.resolved.specification.run.evaluation_seed_offset,
        final_evaluation_seed=seed + plan.resolved.specification.run.evaluation_seed_offset + 1,
        run_directory=plan.output_root / plan.resolved.specification.id / f"seed-{seed}",
    )

    with pytest.raises(PreflightError, match=r"between 0 and 2\*\*32 - 1"):
        preflight_training((invalid,))

    assert not invalid.run_directory.exists()


def test_evaluation_preflight_revalidates_manually_constructed_numeric_fields(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    training_plan = plan_training_batch(tmp_path, seeds=[1], registry=registry)[0]
    execute_training_batch((training_plan,))
    plan = plan_evaluation_batch(tmp_path, registry=registry, name="check")[0]

    for invalid, match in (
        (replace(plan, seed=2**32), r"between 0 and 2\*\*32 - 1"),
        (replace(plan, evaluation_seed=2**32), r"between 0 and 2\*\*32 - 1"),
        (replace(plan, episodes=0), "positive integer"),
    ):
        with pytest.raises(PreflightError, match=match):
            execution.preflight_evaluation((invalid,))

    assert not plan.output_directory.exists()


def test_preflight_rejects_parent_collisions_and_unsafe_owned_artifacts(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    collision_root = tmp_path / "collision"
    collision_root.mkdir()
    (collision_plan,) = plan_training_batch(collision_root, seeds=[1], registry=registry)
    (collision_root / TAXI_EXPERIMENT_ID).write_text("not a directory")
    with pytest.raises(PreflightError, match="not a directory"):
        preflight_training((collision_plan,))

    (plan,) = plan_training_batch(tmp_path / "unsafe", seeds=[1], registry=registry)
    execute_training_batch((plan,))
    metadata_before = plan.run_path.read_bytes()
    plan.curve_path.unlink()
    plan.curve_path.symlink_to(tmp_path / "outside.csv")
    with pytest.raises(PreflightError, match="unsafe run artifact"):
        preflight_training((plan,), overwrite=True)
    assert plan.run_path.read_bytes() == metadata_before


def test_duplicate_empty_and_unsafe_selections_are_rejected(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    with pytest.raises(SelectionError, match="explicit training seed"):
        plan_training_batch(tmp_path, seeds=[], registry=registry)
    with pytest.raises(SelectionError, match="Duplicate training seed"):
        plan_training_batch(tmp_path, seeds=[1, 1], registry=registry)
    with pytest.raises(SelectionError, match="Duplicate experiment filter"):
        plan_training_batch(
            tmp_path,
            seeds=[1],
            registry=registry,
            experiment_ids=[TAXI_EXPERIMENT_ID, TAXI_EXPERIMENT_ID],
        )
    with pytest.raises(SelectionError, match="Unknown environment"):
        plan_training_batch(tmp_path, seeds=[1], registry=registry, environment_ids=["merchant/missing-v0"])
    with pytest.raises(SelectionError, match="safe lowercase"):
        plan_evaluation_batch(tmp_path, registry=registry, name="../../escape")
    with pytest.raises(SelectionError, match="training-owned"):
        plan_evaluation_batch(tmp_path, registry=registry, name="final")
    with pytest.raises(SelectionError, match=r"between 0 and 2\*\*32 - 1"):
        plan_training_batch(tmp_path, seeds=[2**32], registry=registry)
    with pytest.raises(SelectionError, match="derives evaluation seeds"):
        plan_training_batch(tmp_path, seeds=[2**32 - 2], registry=registry)


def test_overwrite_requires_matching_configuration_and_preserves_other_evaluations(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    (plan,) = plan_training_batch(tmp_path, seeds=[1], registry=registry)
    execute_training_batch((plan,))
    preserved = plan.run_directory / "evaluations" / "named" / "note.txt"
    preserved.parent.mkdir()
    preserved.write_text("keep", encoding="utf-8")

    with pytest.raises(PreflightError, match="pass --overwrite"):
        preflight_training((plan,))
    execute_training_batch((plan,), overwrite=True)
    assert preserved.read_text(encoding="utf-8") == "keep"

    changed = _tiny_registry(0).resolve(TAXI_EXPERIMENT_ID)
    changed = replace(changed.specification, run=replace(changed.specification.run, training_steps=5))
    changed_registry = ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=ALGORITHMS,
        techniques=TECHNIQUES,
        experiments=(changed,),
    )
    (changed_plan,) = plan_training_batch(tmp_path, seeds=[1], registry=changed_registry)
    with pytest.raises(ConfigurationMismatchError, match="different resolved settings"):
        preflight_training((changed_plan,), overwrite=True)
    (new_seed_plan,) = plan_training_batch(tmp_path, seeds=[2], registry=changed_registry)
    with pytest.raises(ConfigurationMismatchError, match="already has runs with different resolved settings"):
        preflight_training((new_seed_plan,))
    assert not new_seed_plan.run_directory.exists()


def test_missing_or_modified_models_are_rejected_before_evaluation(tmp_path: Path) -> None:
    registry = _tiny_registry(0)
    (plan,) = plan_training_batch(tmp_path, seeds=[1], registry=registry)
    execute_training_batch((plan,))
    plan.model_path.write_bytes(plan.model_path.read_bytes() + b"changed")
    with pytest.raises(SavedRunError, match="checksum"):
        plan_evaluation_batch(tmp_path, registry=registry, seeds=[1], name="check")
    plan.model_path.unlink()
    with pytest.raises(SavedRunError, match="missing"):
        plan_evaluation_batch(tmp_path, registry=registry, seeds=[1], name="check")


def test_unavailable_saved_component_and_missing_sb3_extra_are_preflight_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _tiny_registry(0)
    (plan,) = plan_training_batch(tmp_path, seeds=[1], registry=registry)
    execute_training_batch((plan,))
    document = json.loads(plan.run_path.read_text())
    document["configuration"]["algorithm"]["id"] = "tabular/unavailable-v9"
    plan.run_path.write_text(json.dumps(document))
    with pytest.raises(SavedRunError, match="does not match"):
        plan_evaluation_batch(tmp_path, registry=registry, seeds=[1], name="check")

    gardener = _tiny_registry(2)
    gardener_plans = plan_training_batch(tmp_path / "sb3", seeds=[1], registry=gardener)
    monkeypatch.setattr(
        execution,
        "_load_sb3_class",
        lambda _algorithm: (_ for _ in ()).throw(PreflightError("missing npc-gym[sb3]")),
    )
    with pytest.raises(PreflightError, match=r"npc-gym\[sb3\]"):
        preflight_training(gardener_plans)
    assert not (tmp_path / "sb3").exists()


def test_model_staging_preserves_checkpoint_and_cleans_failure(tmp_path: Path) -> None:
    target = tmp_path / "model.zip"
    target.write_bytes(b"original")

    class FailingModel:
        num_timesteps = 0

        def save(self, path: str | Path) -> None:
            Path(path).write_bytes(b"partial")
            raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        execution._save_model_staged(FailingModel(), target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_training_failure_closes_environment_and_records_incomplete_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, environments = _custom_registry()
    (plan,) = plan_training_batch(tmp_path, seeds=[5], registry=registry)

    class FailingLearner:
        num_timesteps = 3

        def predict(self, *_args: object, **_kwargs: object) -> int:
            return 0

        def learn(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("training failed")

    monkeypatch.setattr(execution, "_make_tabular_model", lambda *_args: FailingLearner())
    with pytest.raises(RuntimeError, match="training failed"):
        execute_training_batch((plan,))

    assert environments and environments[0].closed
    document = json.loads(plan.run_path.read_text())
    assert document["status"] == "incomplete"
    assert document["timesteps"]["actual"] == 3
    assert document["failure"] == {"type": "RuntimeError", "message": "training failed"}
    assert not list(plan.run_directory.glob(".*.tmp*"))


def test_custom_environment_and_scenario_use_the_existing_tabular_path(tmp_path: Path) -> None:
    registry, environments = _custom_registry()
    (plan,) = plan_training_batch(tmp_path, seeds=[5], registry=registry)

    (result,) = execute_training_batch((plan,))

    assert result.actual_timesteps == 2
    assert json.loads(plan.run_path.read_text())["status"] == "complete"
    assert environments and all(environment.closed for environment in environments)


def test_configuration_comparison_is_type_sensitive() -> None:
    assert KeywordArguments.from_mapping({"value": True}) != KeywordArguments.from_mapping({"value": 1})
    assert KeywordArguments.from_mapping({"value": 1}) != KeywordArguments.from_mapping({"value": 1.0})
    configuration = resolved_configuration(_tiny_registry(0).resolve(TAXI_EXPERIMENT_ID))
    changed = json.loads(json.dumps(configuration))
    changed["environment"]["constructor_kwargs"]["fickle_passenger"] = 0
    assert not execution._json_equal(configuration, changed)


class _ShortEnv(gym.Env[int, int]):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, closed: list[_ShortEnv]) -> None:
        self.closed_instances = closed
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.steps = 0
        return 0, {"labels": frozenset({"reset"})}

    def step(self, action: int):
        self.steps += 1
        return action, 1.0, self.steps == 1, False, {"labels": frozenset({"step"})}

    def close(self) -> None:
        self.closed = True
        self.closed_instances.append(self)


class _CountingMonitor(SimpleMonitor):
    def __init__(self):
        super().__init__(consume_initial=False)

    def detect(self, input):
        return True


def _custom_registry() -> tuple[ExperimentRegistry, list[_ShortEnv]]:
    environments: list[_ShortEnv] = []

    def make_env(_render_mode: str | None) -> _ShortEnv:
        return _ShortEnv(environments)

    environment = EnvironmentSpecification(
        "custom/environment-v0", "custom", make_env, KeywordArguments.from_mapping({})
    )
    scenario = ScenarioSpecification(
        "custom/scenario-v0", frozenset({environment.id}), (("custom/count-v0", _CountingMonitor),)
    )
    algorithm = AlgorithmSpecification(
        "custom/tabular-v0",
        "tabular",
        "npc_gym.algorithms.TabularQLearning",
        None,
        KeywordArguments.from_mapping({"log_interval": None}),
        frozenset({environment.id}),
    )
    technique = TechniqueSpecification("custom/technique-v0", frozenset({"tabular"}))
    settings = RunSettings(
        training_steps=2,
        max_episode_steps=2,
        final_evaluation_episodes=1,
        intermediate_evaluation_episodes=1,
        intermediate_evaluation_frequency=1,
        learning_kwargs=KeywordArguments.from_mapping({}),
        evaluation_seed_offset=100,
        deterministic_evaluation=True,
        task_return_units="unscaled task reward",
        training_reward_transform="identity",
    )
    experiment = ExperimentSpecification(
        "custom-tabular-v0", environment.id, (), scenario.id, algorithm.id, technique.id, settings
    )
    registry = ExperimentRegistry(
        environments=(environment,),
        scenarios=(scenario,),
        algorithms=(algorithm,),
        techniques=(technique,),
        experiments=(experiment,),
    )
    return registry, environments


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("deterministic", [True, False])
def test_runner_evaluations_preserve_tabular_learning(tmp_path, index, deterministic):
    import numpy as np

    registry = _tiny_registry(index)
    (plan,) = plan_training_batch(tmp_path, seeds=[7], registry=registry)
    resolved = replace(
        plan.resolved,
        specification=replace(
            plan.resolved.specification,
            run=replace(plan.resolved.specification.run, deterministic_evaluation=deterministic),
        ),
    )
    plan = replace(plan, resolved=resolved)
    baseline_env = resolved.make_env(training=True)
    loaded_env = resolved.make_env(training=False)
    try:
        baseline = execution._make_tabular_model(resolved, baseline_env, plan.seed)
        baseline.learn(resolved.specification.run.training_steps)
        execute_training_batch((plan,))
        evaluated = execution._load_model(resolved, plan.model_path, loaded_env)
        assert baseline.q_table.keys() == evaluated.q_table.keys()
        assert all(np.array_equal(values, evaluated.q_table[key]) for key, values in baseline.q_table.items())
        assert baseline._training_rng.bit_generator.state == evaluated._training_rng.bit_generator.state
    finally:
        baseline_env.close()
        loaded_env.close()


def test_coarse_vector_training_writes_one_result_for_all_crossed_thresholds(tmp_path, monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv

    (plan,) = plan_training_batch(tmp_path, seeds=[7], registry=_tiny_registry(2))
    resolved = replace(
        plan.resolved,
        specification=replace(
            plan.resolved.specification,
            run=replace(plan.resolved.specification.run, intermediate_evaluation_frequency=1),
        ),
    )
    plan = replace(plan, resolved=resolved)
    original = execution._make_sb3_model
    vectors = []

    def make_vector_model(resolved, _env, seed):
        vector = DummyVecEnv([lambda: resolved.make_env(training=True) for _ in range(3)])
        vectors.append(vector)
        return original(resolved, vector, seed)

    monkeypatch.setattr(execution, "_make_sb3_model", make_vector_model)
    try:
        (result,) = execute_training_batch((plan,))
        assert result.actual_timesteps == 9
        with plan.curve_path.open() as stream:
            assert [int(row["timesteps"]) for row in csv.DictReader(stream)] == [0, 3, 6, 9]
        outputs = plan.run_directory / "evaluations"
        assert {path.name for path in outputs.iterdir()} == {
            "intermediate-0",
            "intermediate-3",
            "intermediate-6",
            "intermediate-9",
            "final",
        }
        for step, thresholds in [(0, [0]), (3, [1, 2, 3]), (6, [4, 5, 6]), (9, [7, 8, 9])]:
            metadata = json.loads((outputs / f"intermediate-{step}" / "summary.json").read_text())["metadata"]
            assert metadata["timesteps"] == step
            assert metadata["scheduled_thresholds"] == thresholds
    finally:
        for vector in vectors:
            vector.close()


def test_final_ppo_evaluation_loads_checkpoint_after_last_optimization(tmp_path, monkeypatch):
    import torch

    (plan,) = plan_training_batch(tmp_path, seeds=[7], registry=_tiny_registry(3))
    original_evaluate = execution._evaluate_model
    original_load = execution._load_model
    intermediate_parameters = []
    loaded_parameters = []

    def evaluate_model(resolved, model, **kwargs):
        intermediate_parameters.append({key: value.clone() for key, value in model.policy.state_dict().items()})
        return original_evaluate(resolved, model, **kwargs)

    def load_model(resolved, path, env):
        model = original_load(resolved, path, env)
        loaded_parameters.append({key: value.clone() for key, value in model.policy.state_dict().items()})
        return model

    monkeypatch.setattr(execution, "_evaluate_model", evaluate_model)
    monkeypatch.setattr(execution, "_load_model", load_model)
    execute_training_batch((plan,))
    assert len(intermediate_parameters) == 3
    assert len(loaded_parameters) == 1
    assert any(not torch.equal(value, loaded_parameters[0][key]) for key, value in intermediate_parameters[-1].items())


@pytest.mark.parametrize("failure", ["evaluation", "writing"])
def test_initial_evaluation_failure_closes_resources_and_leaves_run_incomplete(tmp_path, monkeypatch, failure):
    registry, closed_environments = _custom_registry()
    (plan,) = plan_training_batch(tmp_path, seeds=[5], registry=registry)

    def fail(*args, **kwargs):
        raise RuntimeError("initial evaluation failed")

    monkeypatch.setattr(execution, "_evaluate_in" if failure == "evaluation" else "_record_intermediate", fail)
    with pytest.raises(RuntimeError, match="initial evaluation failed"):
        execute_training_batch((plan,))
    assert len(closed_environments) == 2
    assert all(env.closed for env in closed_environments)
    document = json.loads(plan.run_path.read_text())
    assert document["status"] == "incomplete"
    assert document["timesteps"]["actual"] == 0
    assert not plan.model_path.exists()
    assert not list(plan.run_directory.rglob("*.tmp"))


def test_cli_enables_tabular_info_progress_without_import_side_effects(tmp_path):
    code = """
import logging
import sys
from dataclasses import replace
from pathlib import Path
from experiments import execution, run
from experiments.specifications import REGISTRY, TAXI_EXPERIMENT_ID, KeywordArguments
assert logging.getLogger().handlers == []
(plan,) = execution.plan_training_batch(Path(sys.argv[1]), seeds=[0], experiment_ids=[TAXI_EXPERIMENT_ID])
resolved = plan.resolved
algorithm = replace(resolved.algorithm, constructor_kwargs=KeywordArguments.from_mapping(
    {**resolved.algorithm.constructor_kwargs.to_dict(), "log_interval": 1}
))
settings = replace(resolved.specification.run, training_steps=2, final_evaluation_episodes=1,
                   intermediate_evaluation_episodes=1, intermediate_evaluation_frequency=1)
resolved = replace(resolved, algorithm=algorithm, specification=replace(resolved.specification, run=settings))
plan = replace(plan, resolved=resolved)
run.plan_training_batch = lambda *args, **kwargs: (plan,)
raise SystemExit(run.main(["train", "--output", sys.argv[1], "--seed", "0"]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "tabular q-learning: timesteps=1" in completed.stderr
    assert "tabular q-learning: timesteps=2" in completed.stderr
    assert "completed " in completed.stdout


def test_named_evaluation_can_extend_monitors_without_changing_checkpoint(tmp_path):
    registry = _tiny_registry(0)
    (training,) = plan_training_batch(tmp_path, seeds=[0], registry=registry)
    execute_training_batch((training,))
    before = training.run_path.read_bytes(), training.model_path.read_bytes()
    current = training.resolved
    expanded = replace(
        current.scenario,
        monitor_factories=(*current.scenario.monitor_factories, ("extra-v0", _CountingMonitor)),
    )
    changed = ExperimentRegistry(
        environments=(current.environment,),
        wrappers=current.wrappers,
        scenarios=(expanded,),
        algorithms=(current.algorithm,),
        techniques=(current.technique,),
        experiments=(
            replace(current.specification, run=replace(current.specification.run, final_evaluation_episodes=3)),
        ),
    )
    with pytest.raises(SavedRunError, match="monitor"):
        plan_evaluation_batch(tmp_path, registry=changed, name="expanded")
    (plan,) = plan_evaluation_batch(tmp_path, registry=changed, name="expanded", current_monitors=True)
    assert plan.episodes == 3
    (result,) = execute_evaluation_batch((plan,))
    assert "extra-v0" in result.summary.mean_monitor_counts
    metadata = json.loads((plan.output_directory / "summary.json").read_text())["metadata"]
    assert "extra-v0" in metadata["monitor_ids"]
    assert before == (training.run_path.read_bytes(), training.model_path.read_bytes())
    config = execution.resolved_configuration(current)
    config["environment"]["id"] = "unavailable-v0"
    with pytest.raises(SavedRunError, match="environment"):
        execution.resolved_from_configuration(config, registry=changed, current_monitors=True)
