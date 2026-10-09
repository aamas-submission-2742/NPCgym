"""Plot saved evaluation curves without a training framework or run metadata."""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile
from urllib.parse import quote, unquote

from npc_gym.evaluation.writers import SCHEMA_VERSION


@dataclass(frozen=True)
class _Curve:
    timesteps: tuple[int, ...]
    values: dict[str, list[float]]


def plot_learning_curve(
    source: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    counts: Sequence[tuple[str, str]] | None = None,
    show_std: bool = True,
    overwrite: bool = False,
) -> Path:
    """Save return and monitor-count curves from ``LearningCurveCSVWriter`` CSVs.

    ``source`` is one CSV or a nonempty sequence of independent runs with matching
    columns and strictly increasing timestep grids. Multiple runs receive equal
    weight; bands show sample standard deviation across their episode means.
    For one run, bands show the recorded population standard deviation across
    evaluation episodes. ``show_std=False`` omits bands.

    ``counts`` selects (monitor name, member name) pairs, in display order; None
    plots all counts and an empty sequence plots only returns. Counts are never
    summed. Values retain their recorded units; callers must ensure that supplied
    runs use comparable tasks, rewards, monitors and evaluation protocols.

    ``output`` must end in .png, .pdf or .svg, have an existing parent directory,
    and differ from every input. Existing files require ``overwrite=True``;
    output symlinks are refused. Failed rendering leaves existing files intact.
    Returns the output Path. Requires the optional ``npc-gym[plots]`` extra only
    when called; rendering is headless and does not change the Matplotlib backend.
    """
    paths = [Path(source)] if isinstance(source, (str, Path)) else [Path(path) for path in source]
    if not paths:
        raise ValueError("source must contain at least one learning-curve CSV")
    for index, path in enumerate(paths):
        if any(path.samefile(previous) for previous in paths[:index]):
            raise ValueError(f"Duplicate learning-curve input: {path}")
    destination = Path(output)
    image_format = destination.suffix.lower().lstrip(".")
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("output must end in .png, .pdf or .svg")
    if destination.is_symlink():
        raise ValueError("output must not be a symlink")
    if any(destination.resolve() == path.resolve() for path in paths) or (
        destination.exists() and any(destination.samefile(path) for path in paths)
    ):
        raise ValueError("output must differ from every input CSV")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    if destination.exists() and not destination.is_file():
        raise ValueError("output must be a regular file")

    curves = [_read_curve(path) for path in paths]
    first = curves[0]
    for path, curve in zip(paths[1:], curves[1:], strict=True):
        if curve.timesteps != first.timesteps or curve.values.keys() != first.values.keys():
            raise ValueError(f"{path}: runs must have matching columns and timestep grids")
    available = [key[:-5] for key in first.values if key.startswith("count/") and key.endswith("_mean")]
    selected = available if counts is None else [_count_key(pair) for pair in counts]
    if len(set(selected)) != len(selected):
        raise ValueError("counts must not contain duplicate selections")
    for key in selected:
        if key not in available:
            raise ValueError(f"Unknown monitor count: {key}")

    try:
        from matplotlib.figure import Figure
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise ImportError("Plotting requires the optional extra: pip install 'npc-gym[plots]'") from error

    figure = Figure(figsize=(7, 6 if selected else 3.5), layout="constrained")
    temporary: Path | None = None
    try:
        axes = [figure.add_subplot(2 if selected else 1, 1, 1)]
        if selected:
            axes.append(figure.add_subplot(2, 1, 2, sharex=axes[0]))
        for key, axis in [("return", axes[0]), *((key, axes[1]) for key in selected)]:
            values = [curve.values[f"{key}_mean"] for curve in curves]
            means = [fmean(point) for point in zip(*values, strict=True)]
            deviations = (
                first.values[f"{key}_std"]
                if len(curves) == 1
                else [stdev(point) for point in zip(*values, strict=True)]
            )
            label = "Return" if key == "return" else " / ".join(unquote(part) for part in key.split("/")[1:])
            (line,) = axis.plot(first.timesteps, means, marker=".", label=label)
            if show_std:
                axis.fill_between(
                    first.timesteps,
                    [mean - std for mean, std in zip(means, deviations, strict=True)],
                    [mean + std for mean, std in zip(means, deviations, strict=True)],
                    color=line.get_color(),
                    alpha=0.2,
                )
        axes[0].set_ylabel("Mean episode return")
        if selected:
            axes[1].set_ylabel("Mean count per episode")
            axes[1].legend(fontsize="small")
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Training transitions")
        if show_std:
            population = "evaluation episodes" if len(curves) == 1 else f"{len(curves)} runs"
            figure.suptitle(f"Shading: ±1 standard deviation across {population}", fontsize="small")
        # Publish only a fully rendered figure, without truncating an existing file.
        with NamedTemporaryFile(dir=destination.parent, prefix=".npc-gym-plot-", delete=False) as stream:
            temporary = Path(stream.name)
            figure.savefig(stream, format=image_format, dpi=150)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return destination
    finally:
        figure.clear()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _count_key(pair: tuple[str, str]) -> str:
    if (
        not isinstance(pair, (tuple, list))
        or len(pair) != 2
        or not all(isinstance(name, str) and name for name in pair)
    ):
        raise ValueError("Each count selection must be a (monitor name, member name) pair")
    return "count/" + "/".join(quote(name, safe="") for name in pair)


def _read_curve(path: Path) -> _Curve:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"schema_version", "timesteps", "episodes", "return_mean", "return_std", "length_mean", "length_std"}
        if len(set(fields)) != len(fields) or not required.issubset(fields):
            raise ValueError(f"{path}: missing or duplicate learning-curve columns")
        values: dict[str, list[float]] = {
            field: [] for field in fields if field not in {"schema_version", "timesteps", "episodes"}
        }
        for field in values:
            base, _, statistic = field.rpartition("_")
            parts = base.split("/")
            valid_base = base in {"return", "length"} or (
                (parts[0] == "count" and len(parts) == 3 or parts[0] == "metric" and len(parts) >= 2) and all(parts[1:])
            )
            if not valid_base or statistic not in {"mean", "std"} or {base + "_mean", base + "_std"} - values.keys():
                raise ValueError(f"{path}: invalid or unpaired learning-curve column: {field}")
        timesteps: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{row_number}: row does not match the CSV header")
            try:
                version, step, episodes = (int(row[key]) for key in ("schema_version", "timesteps", "episodes"))
                numbers = {field: float(row[field]) for field in values}
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: invalid numeric value") from error
            if version != SCHEMA_VERSION:
                raise ValueError(f"{path}:{row_number}: expected learning-curve schema version {SCHEMA_VERSION}")
            if step < 0 or episodes < 1 or (timesteps and step <= timesteps[-1]):
                raise ValueError(f"{path}:{row_number}: require increasing nonnegative timesteps and positive episodes")
            for field, value in numbers.items():
                if not math.isfinite(value) or (
                    (field.endswith("_std") or field.startswith(("count/", "length_"))) and value < 0
                ):
                    raise ValueError(f"{path}:{row_number}: invalid value for {field}")
                values[field].append(value)
            timesteps.append(step)
    if not timesteps:
        raise ValueError(f"{path}: learning curve contains no evaluation points")
    return _Curve(tuple(timesteps), values)
