"""The consolidated paper table keeps paired bases and all individual counts."""

import csv
import json
from dataclasses import replace

import pytest

from experiments.execution import (
    execute_evaluation_batch,
    execute_training_batch,
    plan_evaluation_batch,
    plan_training_batch,
)
from experiments.paper import (
    EXPERIMENTS_BY_ENVIRONMENT,
    PACMAN_TRAPPED_OFTEN_ID,
    PACMAN_VEGAN_OFTEN_ID,
    PACMAN_VEGETARIAN_OFTEN_ID,
    make_registry,
)
from experiments.report import _task_identity, create_report
from experiments.specifications import ExperimentRegistry, KeywordArguments


def test_smallclassic_task_identity_preserves_dynamics_and_observation_differences():
    registry = make_registry()
    from experiments.execution import resolved_configuration

    baseline = resolved_configuration(registry.resolve("pacman-dqn-unconstrained-v0"))
    bolt = resolved_configuration(registry.resolve("pacman-smallclassic-dqn-bolts-vegan-v0"))
    assert _task_identity(baseline) == _task_identity(bolt)
    for key, value in (("ghost_behavior", "deterministic"), ("layout", "small"), ("features", "pixels")):
        changed = json.loads(json.dumps(bolt))
        changed["environment"]["constructor_kwargs"][key] = value
        assert _task_identity(baseline) != _task_identity(changed)


@pytest.mark.parametrize("family", ["pacman", "gardener"])
def test_named_paper_report_groups_methods_and_deduplicates_common_base(tmp_path, family):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("clingo")
    pytest.importorskip("matplotlib")
    torch = pytest.importorskip("torch")
    registry = make_registry()
    ids = [
        "pacman-dqn-unconstrained-v0",
        "pacman-ppo-unconstrained-v1",
        PACMAN_VEGAN_OFTEN_ID,
        PACMAN_VEGETARIAN_OFTEN_ID,
        PACMAN_TRAPPED_OFTEN_ID,
        "pacman-smallclassic-dqn-bolts-vegan-v0",
        "pacman-smallclassic-ppo-bolts-vegan-v0",
    ]
    if family == "gardener":
        ids = EXPERIMENTS_BY_ENVIRONMENT[family]
    resolved = [registry.resolve(key) for key in ids]
    algorithms = {}
    experiments = []
    for item in resolved:
        kwargs = item.algorithm.constructor_kwargs.to_dict() | {"device": "cpu", "verbose": 0}
        if item.algorithm.implementation.endswith(".DQN"):
            kwargs.update(buffer_size=64, learning_starts=0, batch_size=4)
        else:
            kwargs.update(n_steps=8, batch_size=4, n_epochs=1)
        algorithms[item.algorithm.id] = replace(
            item.algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs)
        )
        settings = replace(
            item.specification.run,
            training_steps=8,
            max_episode_steps=6,
            intermediate_evaluation_frequency=8,
            intermediate_evaluation_episodes=2,
            final_evaluation_episodes=2,
        )
        often = replace(item.specification.often, pretraining_steps=8) if item.specification.often else None
        if family == "pacman" and often:
            often = replace(often, pretraining_max_episode_steps=6, evaluation_max_episode_steps=6)
        experiments.append(replace(item.specification, run=settings, often=often))
    environment_ids = frozenset(r.environment.id for r in resolved)

    def scoped(component):
        return replace(component, environment_ids=component.environment_ids & environment_ids)

    registry = ExperimentRegistry(
        environments={r.environment.id: r.environment for r in resolved}.values(),
        wrappers={w.id: scoped(w) for r in resolved for w in r.wrappers}.values(),
        scenarios={r.scenario.id: scoped(r.scenario) for r in resolved}.values(),
        algorithms=map(scoped, algorithms.values()),
        techniques={r.technique.id: r.technique for r in resolved}.values(),
        experiments=experiments,
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        execute_training_batch(plan_training_batch(tmp_path / "runs", registry=registry, seeds=[0]))
        execute_evaluation_batch(plan_evaluation_batch(tmp_path / "runs", registry=registry, name="paper"))
    finally:
        torch.set_num_threads(previous)
    target = create_report(
        tmp_path / "runs", tmp_path / "report", benchmark_ids=(f"{family}-paper",), evaluation_name="paper"
    )
    with (target / f"{family}-paper/table.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if family == "gardener":
        assert len(rows) == 13  # Three bases, four fixes, four OFTEN policies, two bolts.
        assert sum(row["optimized_norms"] == "None" for row in rows) == 3
        assert all(row["Episodes mean"] == "2.0" for row in rows)
        for name in ("Collect One", "Rescue per frog", "Unpermitted collection", "Drain", "No Collect"):
            assert all(float(row[name + " mean"]) >= 0 for row in rows)
        return
    assert len(rows) == 11  # Three baselines, three fixes, three OFTEN policies, two bolts.
    assert {row["method"] for row in rows if row["optimized_norms"] == "None"} == {
        "DQN (γ=0.99)",
        "PPO (γ=0.99)",
        "DQN for fixes/OFTEN (γ=0.99, Trapped features)",
    }
    assert all(row["Episodes mean"] == "2.0" for row in rows)
    for name in ("Blue eaten", "Orange eaten", "Hungry", "Trapped", "Missed blue obligation", "Missed pause"):
        assert all(float(row[name + " mean"]) >= 0 for row in rows)
    markdown = (target / "pacman-paper/table.md").read_text()
    assert markdown.index("**Baselines**") < markdown.index("**Vegan**") < markdown.index("**Vegetarian Orange**")
    assert markdown.count("**Baselines**") == 1


def test_pacman_paper_report_uses_evaluation_limit_and_rejects_mismatches(tmp_path):
    from types import SimpleNamespace

    from experiments.benchmarks import PACMAN_PAPER
    from experiments.execution import _write_evaluation_directory, resolved_configuration
    from experiments.report import _collect
    from npc_gym.evaluation import EpisodeResult, EvaluationSummary, TerminationClass
    from npc_gym.monitors.collection import monitor_counts

    registry = make_registry()
    runs = []
    for identifier in ("pacman-dqn-unconstrained-v0", PACMAN_VEGAN_OFTEN_ID):
        resolved = registry.resolve(identifier)
        configuration = resolved_configuration(resolved)
        directory = tmp_path / identifier / "seed-0"
        directory.mkdir(parents=True)
        document = {
            "configuration": configuration,
            "seed": 0,
            "timesteps": {"actual": resolved.specification.run.training_steps},
            "pretraining": {"actual_steps": 5_000_000, "seconds": 1},
            "teaching": {"seconds": 1},
        }
        (directory / "run.json").write_text(json.dumps(document))
        counts = monitor_counts(resolved.make_monitors())
        episode = EpisodeResult(
            episode=0,
            episode_return=0,
            length=1,
            termination=TerminationClass.TERMINATED,
            metrics={"won": 1, "lost": 0, "food_remaining": 0},
            monitor_counts=counts,
        )
        summary = EvaluationSummary.from_episodes([episode])
        for variant in ("", "base", "base-fixed") if resolved.specification.often else ("",):
            _write_evaluation_directory(
                directory / "evaluations/paper" / variant,
                summary,
                {"episode_count": 1, "task_return_units": "unscaled task reward"},
                overwrite=False,
            )
        runs.append(SimpleNamespace(configuration=configuration, run_directory=directory))
    rows = _collect(PACMAN_PAPER, runs, tmp_path, evaluation_name="paper")
    assert len(rows) == 3  # One shared baseline, fixes, OFTEN; teaching still has limit 500.
    assert all("Episode limit" not in row.samples[0] for row in rows)
    changed = json.loads(json.dumps(runs[1].configuration))
    changed["often"]["evaluation_max_episode_steps"] = 500
    original = runs[1].configuration
    runs[1].configuration = changed
    with pytest.raises(ValueError, match="episode limits"):
        _collect(PACMAN_PAPER, runs, tmp_path, evaluation_name="paper")
    runs[1].configuration = original
    changed = json.loads(json.dumps(runs[1].configuration))
    changed["environment"]["constructor_kwargs"]["layout"] = "small"
    runs[1].configuration = changed
    with pytest.raises(ValueError, match="different environment"):
        _collect(PACMAN_PAPER, runs, tmp_path, evaluation_name="paper")
