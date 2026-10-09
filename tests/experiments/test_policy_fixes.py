"""Reproducible Taxi and Merchant base-versus-fix experiments."""

import json
from dataclasses import replace

import numpy as np
import pytest

from experiments import execution, run
from experiments.specifications import (
    ALGORITHMS,
    ENVIRONMENTS,
    MERCHANT_POLICY_FIX_EXPERIMENT_ID,
    REGISTRY,
    SCENARIOS,
    TAXI_POLICY_FIX_EXPERIMENT_ID,
    TECHNIQUES,
    WRAPPERS,
    ExperimentRegistry,
)
from npc_gym.algorithms import TabularQLearning


def tiny_registry(experiment_id=MERCHANT_POLICY_FIX_EXPERIMENT_ID):
    resolved = REGISTRY.resolve(experiment_id)
    settings = replace(
        resolved.specification.run,
        training_steps=128,
        max_episode_steps=40,
        intermediate_evaluation_frequency=64,
        intermediate_evaluation_episodes=2,
        final_evaluation_episodes=2,
    )
    return ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=ALGORITHMS,
        techniques=TECHNIQUES,
        experiments=(replace(resolved.specification, run=settings),),
    )


def test_configuration_matches_eight_seed_benchmark(tmp_path):
    plans = execution.plan_training_batch(tmp_path, experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID], seeds=range(8))
    for plan in plans:
        resolved = plan.resolved
        run = resolved.specification.run
        assert run.training_steps == 5_000_000
        assert run.max_episode_steps == 150
        assert run.final_evaluation_episodes == 1000
        assert run.intermediate_evaluation_frequency == 5_000_000
        assert plan.final_evaluation_seed == plan.intermediate_evaluation_seed == 10_000
        assert resolved.algorithm.constructor_kwargs.to_dict() == {
            "learning_rate": 0.5,
            "gamma": 0.99,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.2,
            "exploration_fraction": 0.1,
            "use_action_mask": True,
            "log_interval": 10_000,
        }
        assert set(resolved.make_monitors()) == {
            "merchant/env-friendly-v0",
            "merchant/delivery-v0",
            "merchant/pacifist-v0",
            "merchant/delivery-pacifist-v0",
        }
        configuration = execution.resolved_configuration(resolved)
        assert execution.resolved_configuration(execution.resolved_from_configuration(configuration)) == configuration


@pytest.mark.parametrize("experiment_id", [MERCHANT_POLICY_FIX_EXPERIMENT_ID, TAXI_POLICY_FIX_EXPERIMENT_ID])
def test_train_and_re_evaluate_save_paired_results_without_changing_base(tmp_path, monkeypatch, experiment_id):
    pytest.importorskip("clingo")
    from experiments import policy_fixes

    registry = tiny_registry(experiment_id)
    plans = execution.plan_training_batch(tmp_path, registry=registry, seeds=[3])
    plan = plans[0]
    observed_seeds = []
    evaluate = policy_fixes.evaluate

    def record(*args, **kwargs):
        observed_seeds.append(kwargs["seed"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(policy_fixes, "evaluate", record)
    (result,) = execution.execute_training_batch(plans)
    assert observed_seeds == [10_000, 10_001] * 5  # Three curve points, final base and final fixed.
    document = execution.read_run_document(plan.run_path)
    assert document["status"] == "complete" and document["training_seconds"] > 0
    assert document["provenance"]["packages"]["clingo"]
    final = plan.run_directory / "evaluations" / "final"
    base = json.loads((final / "base" / "summary.json").read_text())
    fixed = json.loads((final / "summary.json").read_text())
    assert base["metadata"]["variant"] == "base" and "policy_fix" not in base["metadata"]
    assert fixed["metadata"]["variant"] == "fixed" and fixed["metadata"]["fallbacks"] == 0
    assert fixed["metadata"]["decisions"] == sum(e.length for e in result.final_evaluation.episodes)
    assert fixed["metadata"]["planning_seconds"] > 0
    assert fixed["metadata"]["policy_fix"]["use_action_mask"] == (experiment_id == MERCHANT_POLICY_FIX_EXPERIMENT_ID)
    if experiment_id == TAXI_POLICY_FIX_EXPERIMENT_ID:
        assert fixed["metadata"]["policy_fix"]["norm_component"] == "Warn Violations"
        assert result.final_evaluation.mean_monitor_counts["taxi/emergency-v0"]["Warn Violations"] == 0
    assert base["metadata"]["model_sha256"] == fixed["metadata"]["model_sha256"] == result.model_checksum
    assert base["metadata"]["episode_seed_rule"] == fixed["metadata"]["episode_seed_rule"] == "seed + episode index"
    env = plan.resolved.make_env(training=True)
    try:
        direct = execution._make_tabular_model(plan.resolved, env, 3).learn(128)
        saved = TabularQLearning.load(plan.model_path, env=env)
        assert direct.q_table.keys() == saved.q_table.keys()
        for key, values in direct.q_table.items():
            np.testing.assert_array_equal(values, saved.q_table[key])
    finally:
        env.close()
    before = plan.model_path.read_bytes()
    eval_plans = execution.plan_evaluation_batch(tmp_path, registry=registry, name="confirmation")
    execution.execute_evaluation_batch(eval_plans)
    assert plan.model_path.read_bytes() == before
    repeated = json.loads((eval_plans[0].output_directory / "summary.json").read_text())
    assert repeated["aggregate"] == fixed["aggregate"]
    with pytest.raises(execution.PreflightError, match="already exists"):
        execution.execute_evaluation_batch(eval_plans)
    execution.execute_evaluation_batch(eval_plans, overwrite=True)
    execution.execute_training_batch(plans, overwrite=True)
    assert eval_plans[0].output_directory.exists()  # Training overwrite preserves named evaluations.
    (final / "base" / "summary.json").unlink()
    with pytest.raises(execution.SavedRunError, match="required artifact"):
        execution.plan_evaluation_batch(tmp_path, registry=registry, name="missing-base")


@pytest.mark.parametrize("experiment_id", [MERCHANT_POLICY_FIX_EXPERIMENT_ID, TAXI_POLICY_FIX_EXPERIMENT_ID])
def test_evaluation_failure_does_not_publish_partial_pair(tmp_path, monkeypatch, experiment_id):
    pytest.importorskip("clingo")
    from experiments import policy_fixes

    registry = tiny_registry(experiment_id)
    plans = execution.plan_training_batch(tmp_path, registry=registry, seeds=[0])
    execution.execute_training_batch(plans)
    original = policy_fixes.evaluate_policy_fix

    def fail_fixed(*args, **kwargs):
        if kwargs.get("fixed", True):
            raise RuntimeError("fixed evaluation failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(policy_fixes, "evaluate_policy_fix", fail_fixed)
    eval_plans = execution.plan_evaluation_batch(tmp_path, registry=registry, name="failed")
    with pytest.raises(RuntimeError, match="fixed evaluation failed"):
        execution.execute_evaluation_batch(eval_plans)
    assert not eval_plans[0].output_directory.exists()
    assert not list((plans[0].run_directory / "evaluations").glob(".*.tmp"))


@pytest.mark.parametrize("experiment_id", [MERCHANT_POLICY_FIX_EXPERIMENT_ID, TAXI_POLICY_FIX_EXPERIMENT_ID])
def test_cli_missing_asp_fails_before_output(tmp_path, monkeypatch, capsys, experiment_id):
    real = execution.importlib.util.find_spec
    monkeypatch.setattr(execution.importlib.util, "find_spec", lambda name: None if name == "clingo" else real(name))
    assert (
        run.main(
            [
                "train",
                "--experiment",
                experiment_id,
                "--output",
                str(tmp_path / "runs"),
                "--seed",
                "0",
                "--dry-run",
            ]
        )
        == 2
    )
    assert "npc-gym[asp]" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_paired_seed_overflow_is_rejected(tmp_path):
    registry = tiny_registry()
    resolved = registry.resolve(MERCHANT_POLICY_FIX_EXPERIMENT_ID)
    bad = replace(
        resolved,
        specification=replace(
            resolved.specification, run=replace(resolved.specification.run, evaluation_seed_offset=2**32 - 1)
        ),
    )
    with pytest.raises(execution.SelectionError, match="episode seeds"):
        execution._evaluation_seeds(bad, 0)


def test_taxi_configuration_matches_eight_seed_benchmark(tmp_path):
    plans = execution.plan_training_batch(tmp_path, experiment_ids=[TAXI_POLICY_FIX_EXPERIMENT_ID], seeds=range(8))
    assert len(plans) == 8
    for plan in plans:
        resolved = plan.resolved
        settings = resolved.specification.run
        assert settings.training_steps == settings.intermediate_evaluation_frequency == 5_000_000
        assert settings.max_episode_steps == 50
        assert settings.final_evaluation_episodes == 1000
        assert plan.final_evaluation_seed == plan.intermediate_evaluation_seed == 10_000
        assert resolved.algorithm.constructor_kwargs.to_dict() == {
            "learning_rate": 0.2,
            "gamma": 0.99,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.1,
            "exploration_fraction": 0.1,
            "use_action_mask": False,
            "log_interval": 10_000,
        }
        assert list(resolved.make_monitors()) == ["taxi/emergency-v0"]
        assert resolved.environment.constructor_kwargs.to_dict() == {"fickle_passenger": False}
        assert resolved.specification.wrapper_ids == ("taxi/ignore-weather-relevant-v0",)
        configuration = execution.resolved_configuration(resolved)
        assert execution.resolved_configuration(execution.resolved_from_configuration(configuration)) == configuration


@pytest.mark.parametrize("experiment_id", [MERCHANT_POLICY_FIX_EXPERIMENT_ID, TAXI_POLICY_FIX_EXPERIMENT_ID])
def test_incompatible_action_mask_setting_is_rejected(experiment_id):
    resolved = REGISTRY.resolve(experiment_id)
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    kwargs["use_action_mask"] = not kwargs["use_action_mask"]
    from experiments.specifications import KeywordArguments

    resolved = replace(
        resolved, algorithm=replace(resolved.algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs))
    )
    with pytest.raises(execution.PreflightError, match="action masks"):
        execution._validate_execution_settings(resolved)
