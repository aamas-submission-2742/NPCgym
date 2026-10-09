import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from experiments import plot_learning_curves
from experiments.execution import resolved_configuration
from experiments.specifications import REGISTRY, TAXI_EXPERIMENT_ID


def write_run(root: Path, seed: int, returns: tuple[float, float], *, status: str = "complete", change: int = 0):
    configuration = resolved_configuration(REGISTRY.resolve(TAXI_EXPERIMENT_ID))
    configuration["run"]["training_steps"] += change
    run_directory = root / TAXI_EXPERIMENT_ID / f"seed-{seed}"
    run_directory.mkdir(parents=True)
    (run_directory / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "experiment_id": TAXI_EXPERIMENT_ID,
                "seed": seed,
                "configuration": configuration,
            }
        ),
        encoding="utf-8",
    )
    with (run_directory / "learning_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "schema_version",
                "timesteps",
                "episodes",
                "return_mean",
                "return_std",
                "length_mean",
                "length_std",
                "count/taxi%2Femergency-v0/Emergency%20Violations_mean",
                "count/taxi%2Femergency-v0/Emergency%20Violations_std",
            ],
        )
        writer.writeheader()
        for timestep, return_value in zip((100, 200), returns, strict=True):
            writer.writerow(
                {
                    "schema_version": 4,
                    "timesteps": timestep,
                    "episodes": 5,
                    "return_mean": return_value,
                    "return_std": 0,
                    "length_mean": 2,
                    "length_std": 0,
                    "count/taxi%2Femergency-v0/Emergency%20Violations_mean": return_value / 2,
                    "count/taxi%2Femergency-v0/Emergency%20Violations_std": 0,
                }
            )
    return run_directory


def test_metadata_discovery_aggregates_only_exact_complete_configurations(tmp_path, capsys):
    write_run(tmp_path, 0, (2, 6))
    write_run(tmp_path, 1, (4, 10))
    write_run(tmp_path, 2, (100, 100), status="incomplete")
    write_run(tmp_path, 3, (20, 20), change=1)

    groups = plot_learning_curves.discover_run_groups(tmp_path)

    assert {(group.experiment_id, tuple(run.seed for run in group.runs)) for group in groups} == {
        (TAXI_EXPERIMENT_ID, (0, 1)),
        (TAXI_EXPERIMENT_ID, (3,)),
    }
    multi_seed_group = next(group for group in groups if len(group.runs) == 2)
    mean, standard_deviation, monitors, paths = plot_learning_curves.load_and_average_group(multi_seed_group)
    assert mean["return_mean"].tolist() == [3, 8]
    assert standard_deviation["return_mean"].tolist() == pytest.approx([2**0.5, 2 * 2**0.5])
    assert monitors == ["count/taxi%2Femergency-v0/Emergency%20Violations_mean"]
    assert [Path(path).parent.name for path in paths] == ["seed-0", "seed-1"]
    assert "Excluding incomplete run" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("schema_version", "1", "schema version"),
        ("schema_version", "2", "schema version"),
        ("schema_version", "5", "schema version"),
        ("timesteps", "100.5", "integers"),
        ("episodes", "0", "integers"),
        ("return_mean", "nan", "non-finite"),
        ("return_mean", "not-a-number", "non-numeric"),
    ],
)
def test_metadata_curves_reject_invalid_numeric_values(tmp_path, column, value, match):
    run_directory = write_run(tmp_path, 0, (2, 6))
    rows = list(csv.DictReader((run_directory / "learning_curve.csv").open()))
    rows[0][column] = value
    with (run_directory / "learning_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    (group,) = plot_learning_curves.discover_run_groups(tmp_path)
    with pytest.raises(ValueError, match=match):
        plot_learning_curves.load_and_average_group(group)


def test_metadata_curves_reject_grid_and_monitor_mismatches(tmp_path):
    write_run(tmp_path, 0, (2, 6))
    second = write_run(tmp_path, 1, (4, 10))
    path = second / "learning_curve.csv"
    text = path.read_text().replace("200,", "300,", 1)
    path.write_text(text)
    (group,) = plot_learning_curves.discover_run_groups(tmp_path)
    with pytest.raises(ValueError, match="timestep grid"):
        plot_learning_curves.load_and_average_group(group)

    path.write_text(
        text.replace("count/taxi%2Femergency-v0/Emergency%20Violations", "count/taxi%2Fother-v0/Emergency%20Violations")
    )
    with pytest.raises(ValueError, match="absent from the resolved scenario"):
        plot_learning_curves.load_and_average_group(group)


def test_plot_outputs_are_protected_and_figures_close_on_failure(tmp_path, monkeypatch):
    figure, _axis = plt.subplots()
    existing = tmp_path / "curve.png"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="--overwrite"):
        plot_learning_curves.save_figure(figure, tmp_path / "curve")
    assert existing.read_bytes() == b"keep"
    dotted = plot_learning_curves.save_figure(figure, tmp_path / "taxi-q0.5-v0-learning-curve")
    assert [path.name for path in dotted] == ["taxi-q0.5-v0-learning-curve.png", "taxi-q0.5-v0-learning-curve.pdf"]
    plt.close(figure)

    def fail(*_args, **_kwargs):
        raise OSError("plot failed")

    monkeypatch.setattr(plot_learning_curves, "save_figure", fail)
    with pytest.raises(OSError, match="plot failed"):
        plot_learning_curves.make_individual_plot(
            averaged_mean=plot_learning_curves.pd.DataFrame({"timesteps": [1], "return_mean": [1]}),
            averaged_std=None,
            monitor_columns=[],
            title="test",
            output_stem=tmp_path / "failed",
            show_errors=False,
        )
    assert plt.get_fignums() == []


def test_plot_cli_writes_metadata_discovered_group_and_requires_overwrite(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "runs"
    output_path = tmp_path / "plots"
    write_run(input_path, 0, (2, 6))
    write_run(input_path, 1, (4, 10))
    arguments = [
        "plot_learning_curves.py",
        "--input",
        str(input_path),
        "--outdir",
        str(output_path),
        "--experiment",
        TAXI_EXPERIMENT_ID,
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    assert plot_learning_curves.main() == 0
    assert {path.name for path in output_path.iterdir()} == {
        "combined_learning_curves.pdf",
        "combined_learning_curves.png",
        f"{TAXI_EXPERIMENT_ID}-learning-curve.pdf",
        f"{TAXI_EXPERIMENT_ID}-learning-curve.png",
    }
    assert plot_learning_curves.main() == 2
    assert "--overwrite" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*arguments, "--overwrite"])
    assert plot_learning_curves.main() == 0
    assert plt.get_fignums() == []


def write_permission_run(root, seed, count):
    from npc_gym.evaluation import EpisodeResult, EvaluationSummary, LearningCurveCSVWriter, TerminationClass

    directory = write_run(root, seed, (0, 0))
    metadata_path = directory / "run.json"
    metadata = json.loads(metadata_path.read_text())
    permission = "gardener/collect-permission-v0"
    forbidden = "gardener/no-collect-v0"
    metadata["configuration"]["scenario"]["monitor_ids"] = [permission, forbidden]
    metadata_path.write_text(json.dumps(metadata))
    episode = EpisodeResult(
        0,
        1.0,
        1,
        TerminationClass.TRUNCATED,
        {},
        monitor_counts={permission: {"CollectPerm": count}, forbidden: {"NoCollect": count}},
    )
    with LearningCurveCSVWriter(directory / "learning_curve.csv", overwrite=True) as writer:
        writer.write(100, EvaluationSummary.from_episodes([episode]))
    return directory


def test_plot_uses_permission_measurement_without_duplicate_signal_summaries(tmp_path):
    write_permission_run(tmp_path, 0, 1)
    write_permission_run(tmp_path, 1, 3)
    (group,) = plot_learning_curves.discover_run_groups(tmp_path)
    mean, std, columns, _paths = plot_learning_curves.load_and_average_group(group)
    permission = "count/gardener%2Fcollect-permission-v0/CollectPerm_mean"
    assert columns == [permission, "count/gardener%2Fno-collect-v0/NoCollect_mean"]
    assert mean[permission].tolist() == [2]
    assert std[permission].tolist() == pytest.approx([2**0.5])
    figure, axis = plt.subplots()
    try:
        _, count_axis = plot_learning_curves.plot_instance_on_axes(
            axis, mean, std, columns, "Gardener", True, False, True
        )
        assert count_axis.lines[0].get_ydata().tolist() == [2]
        assert "CollectPerm" in count_axis.lines[0].get_label()
        assert "NoCollect" in count_axis.lines[1].get_label()
        assert count_axis.get_ylabel() == "Mean event count"
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "monitor columns"),
        ("malformed", "malformed count"),
        ("unknown", "absent from the resolved scenario"),
        ("negative", "nonnegative"),
        ("old", "regenerate with version 4"),
    ],
)
def test_plot_rejects_invalid_permission_measurements(tmp_path, change, match):
    directory = write_permission_run(tmp_path, 0, 1)
    path = directory / "learning_curve.csv"
    frame = plot_learning_curves.pd.read_csv(path)
    prefix = "count/gardener%2Fcollect-permission-v0/CollectPerm"
    if change == "missing":
        frame = frame.drop(columns=[f"{prefix}_mean", f"{prefix}_std"])
    elif change == "malformed":
        frame = frame.rename(columns=lambda name: name.replace("gardener%2F", "gardener%2f"))
    elif change == "unknown":
        frame = frame.rename(columns=lambda name: name.replace("count/gardener%2F", "count/other%2F"))
    elif change == "negative":
        frame[f"{prefix}_mean"] = -1
    elif change == "old":
        frame = frame.drop(columns=[name for name in frame if name.startswith("count/")])
        frame["schema_version"] = 1
    frame.to_csv(path, index=False)
    (group,) = plot_learning_curves.discover_run_groups(tmp_path)
    with pytest.raises(ValueError, match=match):
        plot_learning_curves.load_and_average_group(group)


@pytest.mark.parametrize("failure", ["missing", "malformed", "name", "unsafe", "collision"])
def test_plot_cli_reports_expected_refusals_without_tracebacks(tmp_path, failure):
    import subprocess

    input_path = tmp_path / "runs"
    output_path = tmp_path / "plots"
    args = ["--input", str(input_path), "--outdir", str(output_path)]
    if failure != "missing":
        directory = write_run(input_path, 0, (2, 6))
        if failure == "malformed":
            (directory / "learning_curve.csv").write_text("invalid,csv\n1,2\n")
        elif failure == "name":
            args.extend(["--combined-name", "../escape"])
        elif failure == "unsafe":
            output_path.mkdir()
            (output_path / "combined_learning_curves.png").symlink_to(tmp_path / "elsewhere")
            args.append("--overwrite")
        elif failure == "collision":
            args.extend(["--combined-name", f"{TAXI_EXPERIMENT_ID}-learning-curve"])
    completed = subprocess.run(
        [sys.executable, str(Path(plot_learning_curves.__file__)), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 2
    assert completed.stderr.startswith("error: ")
    assert "Traceback" not in completed.stderr
    assert not output_path.exists() or failure == "unsafe"


@pytest.mark.parametrize("version", [1, 2, 3, 5])
def test_plot_rejects_archived_or_future_curves_without_rewriting_them(tmp_path, version):
    directory = write_run(tmp_path, 0, (2, 6))
    path = directory / "learning_curve.csv"
    frame = plot_learning_curves.pd.read_csv(path)
    frame["schema_version"] = version
    frame = frame.rename(columns=lambda name: name.replace("signal/", "norm/"))
    frame.to_csv(path, index=False)
    archived = path.read_bytes()
    (group,) = plot_learning_curves.discover_run_groups(tmp_path)
    with pytest.raises(ValueError, match="unsupported learning-curve schema version; regenerate with version 4"):
        plot_learning_curves.load_and_average_group(group)
    assert path.read_bytes() == archived
