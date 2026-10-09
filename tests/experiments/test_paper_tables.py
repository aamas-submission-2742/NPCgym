"""Current paper policies, seed aggregation and isolated LaTeX output."""

import json
import math
from dataclasses import replace

import pytest

from experiments import generate_paper_tables as tables
from experiments.benchmarks import TARGETS
from experiments.execution import resolved_configuration
from experiments.paper import EXPERIMENTS_BY_ENVIRONMENT, PACMAN_VEGAN_OFTEN_ID, make_registry
from experiments.report import BenchmarkRow
from experiments.specifications import RESTRAINING_BOLT_TECHNIQUE_ID, UNCONSTRAINED_TECHNIQUE_ID


def policy(configuration, *, target="None", phase="training"):
    return BenchmarkRow(
        "test policy",
        target,
        phase,
        configuration,
        {
            seed: {
                "Episodes": 1000.0,
                "Return": 100.0 + seed,
                "Episode length": 42.0,
                "Win rate": 0.75,
                "Death rate": 0.1,
                "Blue eaten": float(seed),
                "Orange eaten": float(7 - seed),
                "Hungry": 1.0,
                "Trapped": 0.0,
                "Missed blue obligation": 0.5,
                "Missed pause": 0.25,
                "Collect One": 0.2,
                "No Collect": 0.3,
                **{
                    metric: 0.25
                    for columns in tables.GRID_COLUMNS.values()
                    for c in columns
                    for metric in c.metrics
                    if metric not in {"Return", "Death rate"}
                },
            }
            for seed in range(8)
        },
        sources=["test-policy/seed-0/evaluations/paper"],
    )


@pytest.fixture
def result_rows(monkeypatch):
    """Synthetic measurements; configurations come from the current paper selection."""
    registry = make_registry()
    by_environment = {}
    for environment, identifiers in EXPERIMENTS_BY_ENVIRONMENT.items():
        rows = []
        for identifier in identifiers:
            config = resolved_configuration(registry.resolve(identifier))
            if config.get("often"):
                target = TARGETS[config["often"]["norm_id"]]
                rows.extend(
                    (
                        policy(config, phase="base training"),
                        policy(config, target=target, phase="reference"),
                        policy(config, target=target, phase="teaching"),
                    )
                )
            elif config["technique"]["id"] == UNCONSTRAINED_TECHNIQUE_ID:
                rows.append(policy(config))
            elif config["technique"]["id"] == RESTRAINING_BOLT_TECHNIQUE_ID:
                if environment == "pacman":
                    target = next(
                        target
                        for slug, target in (
                            ("hungry-vegan-penalty", "Hungry Vegan Penalty"),
                            ("hungry-vegan", "Hungry Vegan"),
                            ("vegan-conflict", "Vegan Conflict"),
                            ("vegan", "Vegan"),
                            ("vegetarian", "Vegetarian Orange"),
                            ("trapped", "Trapped"),
                        )
                        if f"bolts-{slug}-" in identifier
                    )
                elif environment == "merchant":
                    target = "Environment Friendly" if "env-friendly" in identifier else "DeliveryPacifist"
                elif environment == "taxi":
                    target = "Warn" if "warn" in identifier else "Emergency"
                else:
                    target = "Collection and rescue"
                rows.append(policy(config, target=target))
            else:
                target = "Warn" if environment == "taxi" else "Environment Friendly"
                rows.extend((policy(config), policy(config, target=target)))
        by_environment[environment] = rows

    def collect(root, benchmark, *, evaluation_name):
        assert set(benchmark.experiments) == set(EXPERIMENTS_BY_ENVIRONMENT[root.name])
        assert evaluation_name == ("final" if root.name == "taxi" else "paper")
        return by_environment[root.name]

    monkeypatch.setattr(tables, "collect_benchmark", collect)
    return by_environment


def test_compact_paper_selection_omits_excluded_rows_and_auxiliary_bases(tmp_path, result_rows):
    output = tables.generate_tables(tmp_path / "results", tmp_path / "tables")
    assert {path.name for path in output.iterdir()} == {"pacman_baselines.tex", "grid_baselines.tex", "sources.json"}
    sources = json.loads((output / "sources.json").read_text())
    expected = {
        "taxi": ["Q-learning", r"\quad + fixes (\textit{warn})", "Q-learning + bolts (50M)"],
        "merchant": ["Q-learning", r"\quad + fixes (\textit{env.})", "Q-learning + bolts"],
        "merchant-pacifist": ["Q-learning", "Q-learning + bolts"],
        "gardener": [
            "PPO (100k)",
            "DQN + fixes (perm.)",
            "DQN + fixes (drain)",
            "DQN + fixes (perm.+drain)",
            "DQN + OFTEN (perm.)",
            "DQN + OFTEN (drain)",
            "DQN + OFTEN (perm.+drain)",
            "DQN + bolts (1M)",
            "PPO + bolts (1M)",
        ],
    }
    for group, labels in expected.items():
        assert [r["method"] for r in sources if r["group"] == group] == labels
    pacman_rows = {r["method"] for r in sources if r["table"] == "pacman"}
    assert pacman_rows == {"DQN", "PPO", "DQN + fixes", "DQN + OFTEN", "DQN + bolts", "PPO + bolts"}
    assert all(r["seeds"] == list(range(8)) for r in sources)
    assert all(r["statistics"]["Episodes"]["mean"] == 1000 for r in sources)
    assert not any("base" in r["method"] for r in sources)
    pacman = (output / "pacman_baselines.tex").read_text()
    assert pacman.count("DQN + fixes &") == 1
    assert pacman.count("DQN + OFTEN &") == 1
    assert "--" in pacman  # No fixes/OFTEN for three of the norm bases.
    assert "0.95" not in pacman
    assert "Trapped" not in pacman
    assert not any(r["optimized_norms"] == "Trapped" for r in sources if r["table"] == "pacman")
    assert r"\begin{tabular}{@{}l rrrrrrrrrrr@{}}" in pacman
    grid = (output / "grid_baselines.tex").read_text()
    assert grid.index(r"\textbf{Taxi}") < grid.index(r"\textbf{Merchant}") < grid.index(r"\textbf{Gardener}")
    assert grid.count(r"\textbf{Merchant}") == 2
    assert r"\begin{tabular}{@{}l rrrr@{}}" in grid
    assert r"\begin{table}[t]" in grid
    assert r"\resizebox{\linewidth}" in grid
    assert "Warn)" not in grid
    assert "Collect One" not in grid
    assert "No Collect" not in grid
    assert "Danger" not in grid
    assert "Safety (7)" not in grid
    assert all("Collect One" in r["statistics"] for r in sources if r["table"] == "gardener")
    assert not (tmp_path / "results").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        tables.generate_tables(tmp_path / "results", output)


def test_component_sums_use_seed_covariance_and_between_seed_spread():
    config = resolved_configuration(make_registry().resolve(PACMAN_VEGAN_OFTEN_ID))
    row = policy(config)
    average, spread = tables._statistics(row, tables.Column("Viol.", tables.GHOSTS))
    assert average == 7
    assert spread == 0  # Opposing seed effects cancel; do not add component deviations.
    average, spread = tables._statistics(row, tables.Column("Blue", ("Blue eaten",)))
    assert average == 3.5
    assert spread == pytest.approx(math.sqrt(6))
    assert tables._cell(row, tables.WIN) == r"75.0{\tiny$\,\pm\,$0.0}"
    assert tables._cell(row, tables.Column("Zero", ("Trapped",))) == r"$<$0.001"
    assert tables._cell(None, tables.WIN) == "--"
    assert tables._rounded(0.25, 1) == "0.3"
    with pytest.raises(ValueError, match="missing measurement"):
        tables._cell(row, tables.Column("Unknown", ("unknown",)))


@pytest.mark.parametrize("section,index", [("taxi", 3), ("merchant-pacifist", 2)])
def test_compact_aggregate_columns_sum_components_per_seed(section, index):
    config = resolved_configuration(make_registry().resolve(PACMAN_VEGAN_OFTEN_ID))
    row = policy(config)
    column = tables.GRID_COLUMNS[section][index]
    left, right = column.metrics
    for seed, sample in row.samples.items():
        sample[left] = float(seed)
        sample[right] = float(7 - seed)
    assert tables._statistics(row, column) == (7, 0)
    assert tables._cell(row, column) == r"7.00{\tiny$\,\pm\,$0.00}"


def test_missing_selected_grid_policy_is_rejected(tmp_path, result_rows):
    result_rows["gardener"][:] = [
        row for row in result_rows["gardener"] if not (tables._algorithm(row) == "PPO" and row.targets == "None")
    ]
    with pytest.raises(ValueError, match="expected one PPO/base/None policy"):
        tables.generate_tables(tmp_path / "results", tmp_path / "tables", table="grid")


@pytest.mark.parametrize("problem", ["seed", "episodes"])
def test_incomplete_paper_protocol_is_rejected(tmp_path, result_rows, problem):
    row = result_rows["pacman"][0]
    if problem == "seed":
        del row.samples[7]
    else:
        row.samples[0]["Episodes"] = 999
    with pytest.raises(ValueError, match="expected training seeds" if problem == "seed" else "1000 evaluation"):
        tables.generate_tables(tmp_path / "results", tmp_path / "tables", table="pacman")
    assert not (tmp_path / "tables").exists()


def test_conflicting_pacman_policies_are_rejected(tmp_path, result_rows):
    row = result_rows["pacman"][0]
    result_rows["pacman"].append(replace(row, samples={**row.samples}))
    with pytest.raises(ValueError, match="Multiple Pacman policies"):
        tables.generate_tables(tmp_path / "results", tmp_path / "tables", table="pacman")


def test_unrecognized_pacman_target_is_not_silently_omitted(tmp_path, result_rows):
    next(row for row in result_rows["pacman"] if row.phase == "reference").targets = "Unknown norm base"
    with pytest.raises(ValueError, match="does not cover every selected policy"):
        tables.generate_tables(tmp_path / "results", tmp_path / "tables", table="pacman")


def test_path_separation_and_failure_cleanup(tmp_path, result_rows, monkeypatch):
    root = tmp_path / "results"
    for input_path, output in ((root, root / "tables"), (root / "results", root)):
        with pytest.raises(ValueError, match="separate"):
            tables.generate_tables(input_path, output)
    with pytest.raises(ValueError, match="table must be"):
        tables.generate_tables(root, tmp_path / "tables", table="unknown")
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        tables.generate_tables(root, linked / "tables")

    original = tables.Path.write_text

    def fail(path, *args, **kwargs):
        if path.name == "sources.json":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(tables.Path, "write_text", fail)
    with pytest.raises(OSError, match="disk full"):
        tables.generate_tables(root, tmp_path / "tables", table="pacman")
    assert not (tmp_path / "tables").exists()
    assert not list(tmp_path.glob(".tables.*"))


def test_cli_writes_selected_table_and_sources(tmp_path, result_rows, capsys):
    assert (
        tables.main(
            [
                "--results",
                str(tmp_path / "results"),
                "--output",
                str(tmp_path / "tables"),
                "--table",
                "pacman",
                "--details",
                "--print-latex",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert r"\begin{table*}" in output
    assert "test-policy/seed-0/evaluations/paper/summary.json" in output
    assert not (tmp_path / "tables/grid_baselines.tex").exists()
