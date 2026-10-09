"""Small real experiments and numerical/reporting invariants."""

import csv
import json
import math
import shutil
from dataclasses import replace

import pytest

from experiments import execution, run
from experiments.benchmarks import BENCHMARKS
from experiments.specifications import (
    ALGORITHMS,
    ENVIRONMENTS,
    EXPERIMENTS,
    SCENARIOS,
    TECHNIQUES,
    WRAPPERS,
    ExperimentRegistry,
    KeywordArguments,
    PacmanOFTENSettings,
)


def smoke_registry():
    algorithms = []
    for algorithm in ALGORITHMS:
        kwargs = algorithm.constructor_kwargs.to_dict()
        if algorithm.implementation.endswith(".DQN"):
            kwargs.update(buffer_size=128, learning_starts=0, batch_size=8, verbose=0)
        elif algorithm.implementation.endswith(".PPO"):
            kwargs.update(n_steps=8, batch_size=8, n_epochs=1, verbose=0)
        algorithms.append(replace(algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs)))
    specs = []
    for experiment in EXPERIMENTS:
        settings = replace(
            experiment.run,
            training_steps=32,
            max_episode_steps=6,
            intermediate_evaluation_frequency=16,
            intermediate_evaluation_episodes=2,
            final_evaluation_episodes=2,
        )
        often = replace(experiment.often, pretraining_steps=32) if experiment.often else None
        if isinstance(often, PacmanOFTENSettings):
            often = replace(often, pretraining_max_episode_steps=6, evaluation_max_episode_steps=6)
        specs.append(replace(experiment, run=settings, often=often))
    return ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=algorithms,
        techniques=TECHNIQUES,
        experiments=specs,
    )


@pytest.fixture(scope="module")
def complete_runs(tmp_path_factory):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("clingo")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    pytest.importorskip("pygame")
    torch = pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("comparisons")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        plans = execution.plan_training_batch(root / "runs", registry=smoke_registry(), seeds=[0])
        execution.execute_training_batch(plans)
        yield root
    finally:
        torch.set_num_threads(previous)


def test_all_current_experiments_report_without_checkpoints(complete_runs):
    root = complete_runs
    # Numerical reporting must work on the versionable archive alone.
    for path in (root / "runs").rglob("*.zip"):
        path.unlink()
    assert run.main(["report", "--input", str(root / "runs"), "--outdir", str(root / "report")]) == 0
    for benchmark in BENCHMARKS:
        target = root / "report" / benchmark.id
        if benchmark.id.startswith("pacman-smallclassic-") and "-bolts-" not in benchmark.id:
            assert not target.exists()
            continue
        assert {p.name for p in target.iterdir()} == {
            "table.md",
            "table.csv",
            "curves.csv",
            "return.png",
            "return.pdf",
            "violations.png",
            "violations.pdf",
            "sources.json",
        }
        with (target / "curves.csv").open() as stream:
            curves = list(csv.DictReader(stream))
        for point in curves:
            assert math.isfinite(float(point["mean"]))
            assert float(point["std"]) == 0
        methods = {item["method"] for item in curves}
        if benchmark.id == "taxi-emergency":
            assert methods == {
                "TabularQLearning",
                "TabularQLearning base for fixes",
                "TabularQLearning + fixes",
                "TabularQLearning + bolts (Emergency)",
                "TabularQLearning + bolts (Warn)",
            }
            assert len(benchmark.experiments) == 4
            assert {p["metric"] for p in curves} == {
                "Return",
                "Warn Violations",
                "Stay Violations",
                "Seven-Step Safety Violations",
                "Three-Step Safety Violations",
                "Safety Violations",
                "Emergency Violations",
            }
        if benchmark.id == "merchant":
            assert methods == {
                "TabularQLearning",
                "TabularQLearning + fixes",
                "TabularQLearning + bolts (Environment Friendly)",
                "TabularQLearning + bolts (DeliveryPacifist)",
            }
            sources = json.loads((target / "sources.json").read_text())
            bolt_sources = [source for source in sources if " + bolts (" in source["method"]]
            assert bolt_sources and all(
                "minimized" in source["configuration"]["experiment_id"] for source in bolt_sources
            )
        if "-bolts-" in benchmark.id:
            assert len(methods) == len(benchmark.experiments)
            assert all(" + bolts (" in method for method in methods)
        if benchmark.id == "pacman-images-vegan":
            assert methods == {"DQN"}
        if benchmark.id in {"pacman-trapped", "pacman-vegetarian"}:
            other = "Vegetarian" if benchmark.id.endswith("trapped") else "Trapped"
            assert all(other not in method for method in methods)
        for method in methods:
            points = [point for point in curves if point["method"] == method]
            if points[0]["phase"] == "teaching":
                assert min(int(p["steps"]) for p in points) == 0
                assert max(int(p["steps"]) for p in points) == 32
        if benchmark.id == "gardener":
            sources = json.loads((target / "sources.json").read_text())
            assert len([row for row in sources if row["phase"] == "base training"]) == 1
            assert {p["metric"] for p in curves} == {"Return", "Unpermitted collection", "Drain", "Rescue"}
        if benchmark.id == "gardener-collection-rescue":
            assert methods == {"DQN + bolts (Collection and rescue)", "PPO + bolts (Collection and rescue)"}
            assert {p["metric"] for p in curves} == {
                "Return",
                "Collect One",
                "Rescue per frog",
                "Unpermitted collection",
                "Drain",
            }
    with (root / "report/merchant/table.csv").open() as stream:
        fields = csv.DictReader(stream).fieldnames
    assert "Market unloading rate mean" in fields
    assert "Delivered resources mean" not in fields
    before = (root / "report/index.md").read_bytes()
    assert run.main(["report", "--input", str(root / "runs"), "--outdir", str(root / "report")]) == 2
    assert (root / "report/index.md").read_bytes() == before


def test_report_validation_and_staging(complete_runs, monkeypatch):
    from experiments import report

    root = complete_runs
    with pytest.raises(ValueError, match="separate"):
        report.create_report(root / "runs", root / "runs" / "report")
    with pytest.raises(ValueError, match="Unknown benchmark"):
        report.create_report(root / "runs", root / "unknown", benchmark_ids=("missing",))

    def fail(*args):
        raise RuntimeError("figure failed")

    monkeypatch.setattr(report, "_write_benchmark", fail)
    with pytest.raises(RuntimeError, match="figure failed"):
        report.create_report(root / "runs", root / "failed", benchmark_ids=("gardener",))
    assert not (root / "failed").exists()
    assert not list(root.glob(".failed.*"))


def test_read_only_benchmark_collection_matches_report(complete_runs, tmp_path):
    from experiments.report import collect_benchmark, create_report

    root = complete_runs
    benchmark = next(item for item in BENCHMARKS if item.id == "merchant")
    rows = collect_benchmark(root / "runs", benchmark)
    target = create_report(root / "runs", tmp_path / "report", benchmark_ids=(benchmark.id,))
    with (target / "merchant/table.csv").open() as stream:
        reported = {record["method"]: record for record in csv.DictReader(stream)}
    assert len(rows) == 4
    for row in rows:
        assert set(row.samples) == {0}
        assert row.samples[0]["Return"] == float(reported[row.label]["Return mean"])
    with pytest.raises(ValueError, match="Named evaluations"):
        collect_benchmark(root / "runs", benchmark, evaluation_name="base")


def test_missing_monitored_norm_is_not_reported_as_zero(tmp_path):
    from experiments.report import _measurement

    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "metadata": {"episode_count": 1, "task_return_units": "unscaled task reward"},
                "episodes": [
                    {
                        "return": 1,
                        "length": 1,
                        "metrics": {"score": 1},
                        "termination": "truncated",
                        "monitor_counts": {},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="full monitored norm base"):
        _measurement(path, next(b for b in BENCHMARKS if b.id == "gardener"))


def test_across_seed_statistics_and_no_double_counted_diagnostics(tmp_path):
    from experiments.report import BenchmarkRow, _write_benchmark

    pytest.importorskip("matplotlib")
    benchmark = next(b for b in BENCHMARKS if b.id == "pacman-vegan")
    samples = {0: {"Return": 2.0, "Vegan": 1.0}, 1: {"Return": 6.0, "Vegan": 3.0}}
    row = BenchmarkRow("DQN", "None", "training", {}, samples, {seed: [(0, data)] for seed, data in samples.items()})
    _write_benchmark(tmp_path / "table", benchmark, [row])
    with (tmp_path / "table/table.csv").open() as stream:
        (result,) = list(csv.DictReader(stream))
    assert float(result["Return mean"]) == 4
    assert float(result["Return std"]) == pytest.approx(math.sqrt(8))
    assert float(result["Vegan mean"]) == 2


def test_different_environment_settings_are_not_compared(complete_runs):
    from experiments.plot_learning_curves import discover_run_groups
    from experiments.report import _collect

    benchmark = next(b for b in BENCHMARKS if b.id == "gardener")
    groups = discover_run_groups(complete_runs / "runs")
    runs = [run for group in groups if group.experiment_id in benchmark.experiments for run in group.runs]
    modified = {**runs[0].configuration, "run": {**runs[0].configuration["run"], "max_episode_steps": 9}}
    with pytest.raises(ValueError, match="different environment settings"):
        _collect(benchmark, [replace(runs[0], configuration=modified), *runs[1:]], complete_runs / "runs")


@pytest.mark.parametrize("environment_id", ["taxi/storm-v0", "merchant/basic-v2"])
@pytest.mark.parametrize("rain,storm", [(False, True), (True, True), (False, False)])
def test_task_comparison_handles_only_equivalent_storm_settings(environment_id, rain, storm):
    from copy import deepcopy

    from experiments.report import _task_identity

    current = {
        "environment": {
            "id": environment_id,
            "family": environment_id.split("/")[0],
            "constructor_kwargs": {"fickle_passenger": False},
            "render_kwargs": {},
        },
        "run": {"max_episode_steps": 50},
    }
    explicit = deepcopy(current)
    explicit["environment"]["constructor_kwargs"].update(is_rainy=rain, storm_risk=storm)
    original = deepcopy(explicit)
    equivalent = environment_id == "taxi/storm-v0" and rain is False and storm is True
    assert (_task_identity(current) == _task_identity(explicit)) is equivalent
    assert explicit == original
    changed = deepcopy(explicit)
    changed["environment"]["constructor_kwargs"]["fickle_passenger"] = True
    assert _task_identity(current) != _task_identity(changed)
    changed = deepcopy(explicit)
    changed["run"]["max_episode_steps"] = 100
    assert _task_identity(current) != _task_identity(changed)


def test_historical_pacman_reports_are_separate_from_current_runs(complete_runs, tmp_path):
    from experiments.report import create_report

    # Use synthetic smoke outputs, never archived research data as fixtures.
    source = complete_runs / "runs"
    root = tmp_path / "runs"
    mapping = {
        "pacman-dqn-unconstrained-v1": "pacman-dqn-unconstrained-v0",
        "pacman-ppo-unconstrained-v2": "pacman-ppo-unconstrained-v1",
        "pacman-dqn-often-v1": "pacman-dqn-often-v2",
        "pacman-dqn-often-vegetarian-v1": "pacman-dqn-often-vegetarian-v2",
        "pacman-dqn-often-trapped-v2": "pacman-dqn-often-trapped-v3",
    }
    for current, historical in mapping.items():
        shutil.copytree(source / current, root / current)
        shutil.copytree(source / current, root / historical)
        path = root / historical / "seed-0" / "run.json"
        document = json.loads(path.read_text())
        document["experiment_id"] = historical
        document["configuration"]["experiment_id"] = historical
        document["configuration"]["environment"]["constructor_kwargs"]["layout"] = "smallClassic"
        path.write_text(json.dumps(document))
    create_report(root, tmp_path / "report")
    for name in ("vegan", "vegetarian", "trapped"):
        for prefix, layout in (("pacman-", "small"), ("pacman-smallclassic-", "smallClassic")):
            folder = tmp_path / "report" / (prefix + name)
            rows = json.loads((folder / "sources.json").read_text())
            assert rows
            assert all(row["configuration"]["environment"]["constructor_kwargs"]["layout"] == layout for row in rows)
            with (folder / "table.csv").open() as stream:
                assert all(int(row["seeds"]) == 1 for row in csv.DictReader(stream))
    # Explicit selection also works when there are no current experiment IDs.
    for current in mapping:
        shutil.rmtree(root / current)
    selected = tuple(f"pacman-smallclassic-{name}" for name in ("vegan", "vegetarian", "trapped"))
    create_report(root, tmp_path / "historical-only", benchmark_ids=selected)
    assert {p.name for p in (tmp_path / "historical-only").iterdir()} == {*selected, "index.md"}


def test_merchant_is_one_comparison_with_four_policies_and_all_norms():
    merchant = [benchmark for benchmark in BENCHMARKS if benchmark.id.startswith("merchant")]
    assert len(merchant) == 1
    assert len(merchant[0].experiments) == 4
    assert {label for label, *_ in merchant[0].counts} == {
        "Environment Friendly",
        "Delivery",
        "Danger",
        "CTD",
        "Pacifist total",
        "DeliveryPacifist total",
    }


def test_named_evaluations_report_the_selected_measurements_only(complete_runs, tmp_path):
    from experiments.report import create_report

    root = complete_runs / "runs"
    merchant = next(benchmark for benchmark in BENCHMARKS if benchmark.id == "merchant")
    for identifier in merchant.experiments:
        folder = root / identifier / "seed-0"
        shutil.copytree(folder / "evaluations/final", folder / "evaluations/extended")
        path = folder / "evaluations/extended/summary.json"
        data = json.loads(path.read_text())
        for episode in data["episodes"]:
            episode["return"] = 123.0
        path.write_text(json.dumps(data))
    create_report(root, tmp_path / "named", benchmark_ids=("merchant",), evaluation_name="extended")
    with (tmp_path / "named/merchant/table.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert sum(float(row["Return mean"]) == 123 for row in rows) == 4
    with (tmp_path / "named/merchant/curves.csv").open() as stream:
        assert {int(row["steps"]) for row in csv.DictReader(stream)} == {32}
    with pytest.raises(ValueError):
        create_report(root, tmp_path / "unsafe", evaluation_name="../outside")
