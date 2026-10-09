"""Public CSV plotting, including statistics, optional imports and safe outputs."""

import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from matplotlib.figure import Figure

from npc_gym.evaluation import (
    EpisodeResult,
    EvaluationSummary,
    LearningCurveCSVWriter,
    TerminationClass,
    plot_learning_curve,
)


def write_curve(path, *, offset=0, timesteps=(0, 10), counts=True):
    with LearningCurveCSVWriter(path) as writer:
        for step in timesteps:
            summary = EvaluationSummary.from_episodes(
                [
                    EpisodeResult(
                        index,
                        offset + step + 2 * index,
                        3,
                        TerminationClass.TERMINATED,
                        {"score": 1},
                        {"test/norm": {"violations / step": index}} if counts else {},
                    )
                    for index in range(2)
                ]
            )
            writer.write(step, summary)
    return path


@pytest.fixture
def figures(monkeypatch):
    captured = []
    save = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.append(
            {
                "axes": [
                    {
                        "lines": [(line.get_label(), line.get_xydata().copy()) for line in axis.lines],
                        "bands": [band.get_paths()[0].vertices.copy() for band in axis.collections],
                    }
                    for axis in figure.axes
                ],
                "titles": [text.get_text() for text in figure.texts],
            }
        )
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return captured


@pytest.mark.parametrize(("extension", "magic"), [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")])
def test_one_run_plots_recorded_returns_counts_and_episode_spread(tmp_path, figures, extension, magic):
    source = write_curve(tmp_path / "curve.csv")
    output = tmp_path / f"plot.{extension}"

    assert plot_learning_curve(source, output) == output

    assert output.read_bytes().startswith(magic)
    returns, counts = figures[0]["axes"]
    np.testing.assert_array_equal(returns["lines"][0][1], [[0, 1], [10, 11]])
    assert counts["lines"][0][0] == "test/norm / violations / step"
    np.testing.assert_array_equal(counts["lines"][0][1][:, 1], [0.5, 0.5])
    assert set(returns["bands"][0][:, 1]) == {0, 2, 10, 12}
    assert "evaluation episodes" in figures[0]["titles"][0]
    assert set(tmp_path.iterdir()) == {source, output}


def test_multiple_runs_weight_means_equally_and_show_sample_spread(tmp_path, figures):
    first = write_curve(tmp_path / "one.csv")
    second = write_curve(tmp_path / "two.csv", offset=4)
    # Episode sample size does not weight the run mean.
    rewrite(second, "episodes", "100")
    plot_learning_curve([first, second], tmp_path / "plot.png")

    returns = figures[0]["axes"][0]
    np.testing.assert_array_equal(returns["lines"][0][1][:, 1], [3, 13])
    assert min(returns["bands"][0][:, 1]) == pytest.approx(3 - 8**0.5)
    assert "2 runs" in figures[0]["titles"][0]


def test_select_counts_and_disable_bands(tmp_path, figures):
    source = write_curve(tmp_path / "curve.csv")
    plot_learning_curve(source, tmp_path / "selected.svg", counts=[("test/norm", "violations / step")])
    plot_learning_curve(source, tmp_path / "returns.svg", counts=[], show_std=False)
    assert len(figures[0]["axes"]) == 2
    assert len(figures[1]["axes"]) == 1
    assert not figures[1]["axes"][0]["bands"]
    assert not figures[1]["titles"]


def test_one_point_without_monitors_is_supported(tmp_path, figures):
    source = write_curve(tmp_path / "curve.csv", timesteps=(0,), counts=False)
    plot_learning_curve(source, tmp_path / "plot.png")
    assert len(figures[0]["axes"]) == 1


def rewrite(path, column, value):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[-1][column] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("schema_version", "3", "schema version"),
        ("timesteps", "0", "increasing"),
        ("timesteps", "1.5", "numeric"),
        ("episodes", "0", "positive episodes"),
        ("return_mean", "nan", "return_mean"),
        ("return_std", "-1", "return_std"),
        ("count/test%2Fnorm/violations%20%2F%20step_mean", "-1", "count/"),
    ],
)
def test_invalid_curve_values_leave_no_plot(tmp_path, column, value, message):
    source = write_curve(tmp_path / "curve.csv")
    rewrite(source, column, value)
    with pytest.raises(ValueError, match=message):
        plot_learning_curve(source, tmp_path / "plot.png")
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("contents", ["", "schema_version,schema_version\n4,4\n", "not,a,curve\n1,2,3\n"])
def test_invalid_headers_are_rejected(tmp_path, contents):
    source = tmp_path / "curve.csv"
    source.write_text(contents)
    with pytest.raises(ValueError, match="columns"):
        plot_learning_curve(source, tmp_path / "plot.png")


@pytest.mark.parametrize("change", ["empty", "unpaired", "extra", "missing"])
def test_empty_curves_and_malformed_rows_are_rejected(tmp_path, change):
    source = write_curve(tmp_path / "curve.csv")
    lines = source.read_text().splitlines()
    if change == "empty":
        lines = lines[:1]
    elif change == "unpaired":
        lines[0] = lines[0].replace("metric/score_std", "metric/other_std")
    elif change == "extra":
        lines[-1] += ",1"
    else:
        lines[-1] = lines[-1].rsplit(",", 1)[0]
    source.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        plot_learning_curve(source, tmp_path / "plot.png")
    assert list(tmp_path.iterdir()) == [source]


def test_mismatched_runs_duplicates_and_unknown_selections_are_rejected(tmp_path):
    first = write_curve(tmp_path / "one.csv")
    second = write_curve(tmp_path / "two.csv", timesteps=(0, 20))
    third = write_curve(tmp_path / "three.csv", counts=False)
    for sources, kwargs, message in [
        ([], {}, "at least one"),
        ([first, first], {}, "Duplicate"),
        ([first, second], {}, "matching"),
        ([first, third], {}, "matching"),
        (first, {"counts": [("missing", "count")]}, "Unknown"),
        (first, {"counts": ["test/norm"]}, "pair"),
        (first, {"counts": [("test/norm", "violations / step")] * 2}, "duplicate"),
    ]:
        with pytest.raises(ValueError, match=message):
            plot_learning_curve(sources, tmp_path / "plot.png", **kwargs)


def test_output_protection_and_failed_render_cleanup(tmp_path, monkeypatch):
    source = write_curve(tmp_path / "source.png")
    original = source.read_bytes()
    output = tmp_path / "plot.png"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        plot_learning_curve(source, output)
    with pytest.raises(ValueError, match="differ"):
        plot_learning_curve(source, source, overwrite=True)
    alias = tmp_path / "alias.png"
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="differ"):
        plot_learning_curve(source, alias, overwrite=True)
    alias.unlink()
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        plot_learning_curve(source, alias, overwrite=True)
    alias.unlink()
    with pytest.raises(ValueError, match=".png, .pdf or .svg"):
        plot_learning_curve(source, tmp_path / "plot.csv")

    def fail(*args, **kwargs):
        raise OSError("render failed")

    with monkeypatch.context() as patch:
        patch.setattr(Figure, "savefig", fail)
        for target in (output, tmp_path / "new.png"):
            with pytest.raises(OSError, match="render failed"):
                plot_learning_curve(source, target, overwrite=True)
    assert output.read_bytes() == b"keep"
    assert source.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source, output}
    plot_learning_curve(source, output, overwrite=True)
    assert output.read_bytes().startswith(b"\x89PNG")


def test_plotting_is_lazy_and_explains_missing_extra(tmp_path):
    source = write_curve(tmp_path / "curve.csv")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class WithoutMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'matplotlib':
            raise ModuleNotFoundError("No matplotlib", name="matplotlib")
sys.meta_path.insert(0, WithoutMatplotlib())
from npc_gym.evaluation import plot_learning_curve
assert not {'matplotlib', 'pandas', 'stable_baselines3'} & sys.modules.keys()
try:
    plot_learning_curve(sys.argv[1], sys.argv[2])
except ImportError as error:
    assert 'npc-gym[plots]' in str(error)
else:
    raise AssertionError('plotting worked without matplotlib')
""",
            str(source),
            str(tmp_path / "plot.png"),
        ],
        check=True,
    )


def test_plotting_preserves_existing_figures_and_backend(tmp_path):
    import matplotlib
    import matplotlib.pyplot as plt

    source = write_curve(tmp_path / "curve.csv")
    figure = plt.figure()
    backend = matplotlib.get_backend()
    numbers = plt.get_fignums()
    try:
        plot_learning_curve(source, tmp_path / "plot.png")
        assert plt.get_fignums() == numbers
        assert matplotlib.get_backend() == backend
        figure.add_subplot().plot([0, 1], [1, 0])
        figure.savefig(tmp_path / "existing.png")
    finally:
        plt.close(figure)


def test_documented_training_recording_and_plotting_workflow(tmp_path):
    guide = Path(__file__).resolve().parents[2] / "docs/evaluation.rst"
    block = guide.read_text().split("This complete example", 1)[1].split(".. code-block:: python\n", 1)[1]
    code = dedent(block.split("\nThe short training budget", 1)[0])
    subprocess.run(
        [sys.executable, "-c", code + "\nimport sys\nassert 'stable_baselines3' not in sys.modules\n"],
        cwd=tmp_path,
        check=True,
    )
    with (tmp_path / "learning_curve.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["timesteps"]) for row in rows] == [0, 250, 500, 750, 1000]
    assert all(int(row["episodes"]) == 5 for row in rows)
    assert "count/emergency/Emergency%20Violations_mean" in rows[0]
    assert len(json.loads((tmp_path / "final.json").read_text())["episodes"]) == 20
    with (tmp_path / "episodes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 20
    assert (tmp_path / "learning_curve.png").read_bytes().startswith(b"\x89PNG")


def test_minimal_plotting_example_uses_only_csvs(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "examples/plot_learning_curves.py"
    results = tmp_path / "experiments/results/merchant/merchant-tabular-unconstrained-v2"
    for seed in range(8):
        directory = results / f"seed-{seed}"
        directory.mkdir(parents=True)
        summary = EvaluationSummary.from_episodes(
            [
                EpisodeResult(
                    0,
                    float(seed),
                    1,
                    TerminationClass.TERMINATED,
                    {},
                    {"merchant/delivery-v0": {"count": 1}, "merchant/env-friendly-v0": {"count": 2}},
                )
            ]
        )
        with LearningCurveCSVWriter(directory / "learning_curve.csv") as writer:
            writer.write(0, summary)
            writer.write(10, summary)
    before = {path: path.read_bytes() for path in results.rglob("*.csv")}
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(script), run_name="__main__")

    assert (tmp_path / "experiments/output/merchant-learning-curve.png").read_bytes().startswith(b"\x89PNG")
    assert all(path.read_bytes() == data for path, data in before.items())
