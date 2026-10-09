"""
Create per-experiment and combined learning-curve plots from complete runs.

The CLI discovers ``run.json`` metadata below an experiment output root and
accepts only complete runs with version-3 learning curves. It groups exact
resolved configurations across seeds, validates their schemas and timestep
grids, and plots unscaled task returns with between-seed standard deviation.
For each configuration group, the CLI:

1. finds all complete seed runs;
2. averages ``return_mean`` and the selected monitor columns across seeds at each
   timestep;
3. writes one individual plot per instance; and
4. writes one combined paper-style figure with the same-height panels in a row.

Values are plotted without implicit rescaling.

Usage
-----
    python experiments/plot_learning_curves.py --outdir plots

By default, the script reads runs from ``experiments/output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

import matplotlib as mpl

# Use a non-interactive backend so the script also works on servers/CI.
mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RUN_SCHEMA_VERSION = 1
CONFIGURATION_SCHEMA_VERSION = 1
CURVE_SCHEMA_VERSION = 4
_SAFE_EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*-v[0-9]+$")

# Long CSV column names can be shortened here for the plot legends.
RENAME_MONITORS: dict[str, str] = {}

# Standard scalar column names.
TIMESTEP_COLUMN = "timesteps"
REWARD_COLUMN = "return_mean"

# Figure defaults.
INDIVIDUAL_FIGSIZE = (5.6, 3.7)
COMBINED_PANEL_SIZE = (3.8, 3.3)
DPI = 300


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCurve:
    """One complete metadata-owned learning curve."""

    experiment_id: str
    seed: int
    task_return_units: str
    configuration: Mapping[str, object]
    configuration_digest: str
    run_directory: Path
    curve_path: Path


@dataclass(frozen=True)
class RunGroup:
    """Complete runs with exactly matching resolved configurations."""

    experiment_id: str
    configuration_digest: str
    configuration: Mapping[str, object]
    runs: tuple[RunCurve, ...]


def discover_run_groups(input_path: Path, experiment_ids: Iterable[str] = ()) -> list[RunGroup]:
    """Discover complete runs through ``run.json`` and group exact configurations."""
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")
    requested = tuple(experiment_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate experiment selection")
    selected = set(requested)
    grouped: dict[tuple[str, str], list[RunCurve]] = {}
    seen_runs: set[tuple[str, int]] = set()
    for metadata_path in sorted(input_path.rglob("run.json")):
        run = _read_run_metadata(metadata_path)
        experiment_id = _required_string(run, "experiment_id", metadata_path)
        if not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError(f"{metadata_path}: experiment ID is not a safe path component")
        if selected and experiment_id not in selected:
            continue
        seed = _required_integer(run, "seed", metadata_path, minimum=0)
        expected_directory = input_path / experiment_id / f"seed-{seed}"
        if metadata_path.parent.resolve() != expected_directory.resolve():
            raise ValueError(f"{metadata_path}: metadata path does not match experiment ID and seed")
        identity = (experiment_id, seed)
        if identity in seen_runs:
            raise ValueError(f"Duplicate run metadata for experiment {experiment_id!r}, seed {seed}")
        seen_runs.add(identity)
        if run.get("status") != "complete":
            print(f"Excluding incomplete run: {metadata_path.parent}", file=sys.stderr)
            continue
        configuration = _required_mapping(run, "configuration", metadata_path)
        if configuration.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
            raise ValueError(f"{metadata_path}: unsupported configuration schema version")
        if configuration.get("experiment_id") != experiment_id:
            raise ValueError(f"{metadata_path}: configuration experiment ID does not match run metadata")
        run_settings = _required_mapping(configuration, "run", metadata_path)
        units = _required_string(run_settings, "task_return_units", metadata_path)
        canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        curve_path = metadata_path.parent / "learning_curve.csv"
        if not curve_path.is_file() or curve_path.is_symlink():
            raise ValueError(f"{metadata_path}: complete run is missing a safe learning_curve.csv")
        grouped.setdefault((experiment_id, digest), []).append(
            RunCurve(
                experiment_id,
                seed,
                units,
                configuration,
                digest,
                metadata_path.parent,
                curve_path,
            )
        )
    complete_experiments = {experiment_id for experiment_id, _digest in grouped}
    if selected - complete_experiments:
        missing = ", ".join(sorted(selected - complete_experiments))
        raise FileNotFoundError(f"No complete runs found for: {missing}")
    if not grouped:
        raise FileNotFoundError(f"No complete run metadata found below: {input_path}")
    return [
        RunGroup(experiment_id, digest, runs[0].configuration, tuple(sorted(runs, key=lambda run: run.seed)))
        for (experiment_id, digest), runs in sorted(grouped.items())
    ]


def load_and_average_group(
    group: RunGroup,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Strictly validate and aggregate one exact-configuration group across seeds."""
    if not group.runs:
        raise ValueError("A run group must contain at least one run")
    if len({run.seed for run in group.runs}) != len(group.runs):
        raise ValueError(f"{group.experiment_id}: duplicate training seeds")
    if len({run.task_return_units for run in group.runs}) != 1:
        raise ValueError(f"{group.experiment_id}: task-return units differ across seeds")
    if group.runs[0].task_return_units != "unscaled task reward":
        raise ValueError(f"{group.experiment_id}: learning curves must use unscaled task reward units")
    scenario = _required_mapping(group.configuration, "scenario", group.runs[0].curve_path)
    monitor_ids = scenario.get("monitor_ids")
    if not isinstance(monitor_ids, list) or not all(isinstance(item, str) and item for item in monitor_ids):
        raise ValueError(f"{group.experiment_id}: configuration has invalid monitor IDs")
    expected_monitors = set(monitor_ids)
    frames = []
    expected_columns: tuple[str, ...] | None = None
    expected_timesteps: tuple[int, ...] | None = None
    monitor_columns: list[str] = []
    for run in group.runs:
        canonical = json.dumps(run.configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != group.configuration_digest:
            raise ValueError(f"{run.curve_path}: resolved configuration differs within group")
        frame = pd.read_csv(run.curve_path)
        columns = tuple(str(column) for column in frame.columns)
        required = {
            "schema_version",
            "timesteps",
            "episodes",
            "return_mean",
            "return_std",
            "length_mean",
            "length_std",
        }
        if not required.issubset(columns):
            missing = ", ".join(sorted(required - set(columns)))
            raise ValueError(f"{run.curve_path}: missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{run.curve_path}: learning curve has no evaluation points")
        for column in columns:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{run.curve_path}: column {column!r} contains non-numeric values") from error
            if not frame[column].map(lambda value: math.isfinite(float(value))).all():
                raise ValueError(f"{run.curve_path}: column {column!r} contains non-finite values")
            if column.startswith("count/") and (frame[column] < 0).any():
                raise ValueError(f"{run.curve_path}: count column {column!r} must be nonnegative")
        if not (frame["schema_version"] == CURVE_SCHEMA_VERSION).all():
            raise ValueError(f"{run.curve_path}: unsupported learning-curve schema version; regenerate with version 4")
        _validate_dynamic_columns(columns, run.curve_path, expected_monitors)
        if expected_columns is None:
            expected_columns = columns
            monitor_columns = sorted(
                column for column in columns if column.startswith("count/") and column.endswith("_mean")
            )
            actual_monitors = {unquote(column.split("/")[1]) for column in monitor_columns}
            if actual_monitors != expected_monitors:
                raise ValueError(f"{run.curve_path}: monitor columns do not match the resolved scenario")
        elif columns != expected_columns:
            raise ValueError(f"{run.curve_path}: monitor or metric columns differ across seeds")
        _require_integral_column(frame, "schema_version", run.curve_path, minimum=CURVE_SCHEMA_VERSION)
        _require_integral_column(frame, "timesteps", run.curve_path, minimum=0)
        _require_integral_column(frame, "episodes", run.curve_path, minimum=1)
        timesteps = tuple(int(value) for value in frame["timesteps"])
        if tuple(sorted(set(timesteps))) != timesteps:
            raise ValueError(f"{run.curve_path}: timesteps must be unique and strictly increasing")
        if expected_timesteps is None:
            expected_timesteps = timesteps
        elif timesteps != expected_timesteps:
            raise ValueError(f"{run.curve_path}: timestep grid differs across seeds")
        frame["Seed"] = run.seed
        frames.append(frame)

    assert expected_columns is not None
    value_columns = ["return_mean", *monitor_columns]
    combined = pd.concat([frame[["timesteps", "Seed", *value_columns]] for frame in frames], ignore_index=True)
    grouped = combined.groupby("timesteps", as_index=False)[value_columns]
    averaged_mean = pd.DataFrame(grouped.mean()).sort_values("timesteps")
    averaged_std = pd.DataFrame(grouped.std()).sort_values("timesteps").fillna(0.0)
    return averaged_mean, averaged_std, monitor_columns, [str(run.curve_path) for run in group.runs]


def _read_run_metadata(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read run metadata {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path}: run metadata must be a JSON object")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported run schema version")
    if value.get("status") not in {"complete", "incomplete"}:
        raise ValueError(f"{path}: invalid completion status")
    return value


def _required_mapping(value: Mapping[str, object], key: str, path: Path) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, dict) or not all(isinstance(child, str) for child in item):
        raise ValueError(f"{path}: {key} must be a JSON object")
    return item


def _required_string(value: Mapping[str, object], key: str, path: Path) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path}: {key} must be a non-empty string")
    return item


def _required_integer(value: Mapping[str, object], key: str, path: Path, *, minimum: int) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
        raise ValueError(f"{path}: {key} must be an integer >= {minimum}")
    return item


def _validate_dynamic_columns(columns: tuple[str, ...], path: Path, monitor_ids: set[str]) -> None:
    dynamic = {column for column in columns if column.startswith(("count/", "metric/"))}
    for column in dynamic:
        if column.endswith("_mean"):
            partner = f"{column[:-5]}_std"
        elif column.endswith("_std"):
            partner = f"{column[:-4]}_mean"
        else:
            raise ValueError(f"{path}: malformed aggregate column {column!r}")
        if partner not in dynamic:
            raise ValueError(f"{path}: aggregate column {column!r} has no matching mean/std column")
        if column.startswith("count/"):
            base = column.rsplit("_", 1)[0]
            parts = base.split("/")
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"{path}: malformed count column {column!r}")
            try:
                monitor_id, key = (unquote(part, errors="strict") for part in parts[1:])
            except UnicodeError as error:
                raise ValueError(f"{path}: malformed count encoding in {column!r}") from error
            if [quote(monitor_id, safe=""), quote(key, safe="")] != parts[1:]:
                raise ValueError(f"{path}: malformed count encoding in {column!r}")
            if monitor_id not in monitor_ids:
                raise ValueError(f"{path}: count monitor {monitor_id!r} is absent from the resolved scenario")


def _require_integral_column(frame: pd.DataFrame, column: str, path: Path, *, minimum: int) -> None:
    values = frame[column]
    if not ((values >= minimum) & (values == values.astype("int64"))).all():
        raise ValueError(f"{path}: column {column!r} must contain integers >= {minimum}")


def timestep_formatter(value: float, _: int) -> str:
    """Format timesteps compactly for x-axis tick labels."""

    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def label_for_monitor(column: str) -> str:
    """Return a user-friendly label for a monitor column."""

    label = RENAME_MONITORS.get(column, column)
    if column.startswith("count/") and label == column:
        _, monitor, member = column.removesuffix("_mean").split("/")
        return f"{unquote(monitor)}: {unquote(member)}"
    return label


def plot_instance_on_axes(
    ax_reward: Axes,
    averaged_mean: pd.DataFrame,
    averaged_std: pd.DataFrame | None,
    monitor_columns: list[str],
    title: str,
    show_right_ylabel: bool,
    compact_legend: bool,
    show_errors: bool,
) -> tuple[Axes, Axes]:
    """Draw one learning curve with reward on the left axis and monitors on the right."""

    ax_monitor = ax_reward.twinx()
    x = averaged_mean[TIMESTEP_COLUMN]

    reward_line = ax_reward.plot(
        x,
        averaged_mean[REWARD_COLUMN],
        linewidth=2.4,
        color="black",
        label="Mean task return ± between-seed std." if show_errors else "Mean task return",
        zorder=4,
    )

    if show_errors and averaged_std is not None:
        reward_mean = averaged_mean[REWARD_COLUMN]
        reward_std = averaged_std[REWARD_COLUMN]
        ax_reward.fill_between(
            x,
            reward_mean - reward_std,
            reward_mean + reward_std,
            color="black",
            alpha=0.12,
        )

    monitor_lines = []
    for column in monitor_columns:
        line = ax_monitor.plot(
            x,
            averaged_mean[column],
            linewidth=1.55,
            alpha=0.92,
            label=label_for_monitor(column),
            zorder=3,
        )
        if show_errors and averaged_std is not None:
            color = line[0].get_color()
            monitor_mean = averaged_mean[column]
            monitor_std = averaged_std[column]
            ax_monitor.fill_between(x, monitor_mean - monitor_std, monitor_mean + monitor_std, color=color, alpha=0.12)
        monitor_lines.extend(line)

    ax_reward.set_title(title, fontsize=11.5, pad=8)
    ax_reward.set_xlabel("Timesteps")
    ax_reward.set_ylabel("Average return")
    if show_right_ylabel:
        ax_monitor.set_ylabel("Mean event count")
    else:
        ax_monitor.set_ylabel("")

    ax_reward.xaxis.set_major_formatter(FuncFormatter(timestep_formatter))
    ax_reward.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax_reward.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_monitor.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_reward.grid(True, axis="both", linewidth=0.55, alpha=0.35)
    ax_reward.set_axisbelow(True)

    # Anchor monitor axis at zero for consistent count interpretation.
    ax_monitor.set_ylim(bottom=0)

    # Give both axes a small amount of breathing room.
    ax_reward.margins(x=0.03, y=0.08)
    ax_monitor.margins(x=0.03, y=0.08)

    lines = reward_line + monitor_lines
    labels = [str(line.get_label()) for line in lines]

    if compact_legend:
        ax_reward.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            fontsize=7.0,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.7,
        )
    else:
        ax_reward.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=8.0,
            frameon=False,
            columnspacing=1.0,
            handlelength=2.0,
        )

    return ax_reward, ax_monitor


def save_figure(fig: Figure, output_stem: Path, *, overwrite: bool = False) -> list[Path]:
    """Save a figure as PNG and PDF and return the output paths."""

    output_paths = _plot_paths(output_stem)
    existing = [path for path in output_paths if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(f"Plot output already exists; pass --overwrite: {existing[0]}")
    for path in output_paths:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return output_paths


def make_individual_plot(
    averaged_mean: pd.DataFrame,
    averaged_std: pd.DataFrame | None,
    monitor_columns: list[str],
    title: str,
    output_stem: Path,
    show_errors: bool,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Create and save one individual instance plot."""

    fig, ax = plt.subplots(figsize=INDIVIDUAL_FIGSIZE)
    try:
        plot_instance_on_axes(
            ax_reward=ax,
            averaged_mean=averaged_mean,
            averaged_std=averaged_std,
            monitor_columns=monitor_columns,
            title=title,
            show_right_ylabel=True,
            compact_legend=False,
            show_errors=show_errors,
        )
        fig.tight_layout()
        return save_figure(fig, output_stem, overwrite=overwrite)
    finally:
        plt.close(fig)


def make_combined_plot(
    plotted_instances: list[tuple[str, pd.DataFrame, pd.DataFrame | None, list[str]]],
    output_stem: Path,
    show_errors: bool,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Create and save a single-row combined plot for all instances."""

    n_panels = len(plotted_instances)
    width = COMBINED_PANEL_SIZE[0] * n_panels
    height = COMBINED_PANEL_SIZE[1]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=(width, height),
        squeeze=False,
    )
    axes_row = list(axes[0])

    try:
        for index, (title, averaged_mean, averaged_std, monitor_columns) in enumerate(plotted_instances):
            ax = axes_row[index]
            plot_instance_on_axes(
                ax_reward=ax,
                averaged_mean=averaged_mean,
                averaged_std=averaged_std,
                monitor_columns=monitor_columns,
                title=title,
                show_right_ylabel=(index == n_panels - 1),
                compact_legend=True,
                show_errors=show_errors,
            )

            # Avoid repeating the left y-axis label on every panel.
            if index > 0:
                ax.set_ylabel("")

        # Leave space for the per-panel legends below the axes.
        fig.subplots_adjust(wspace=0.45, bottom=0.31, top=0.84)
        return save_figure(fig, output_stem, overwrite=overwrite)
    finally:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create averaged learning-curve plots from complete experiment runs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR,
        help="Experiment output root containing run.json files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument("--experiment", action="append", default=[], help="exact experiment ID; repeatable")
    parser.add_argument(
        "--combined-name",
        default="combined_learning_curves",
        help="Filename stem for the combined plot.",
    )
    parser.set_defaults(show_errors=True)
    parser.add_argument(
        "--no-errors",
        action="store_false",
        dest="show_errors",
        help="Disable standard-deviation error shading.",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing selected PNG/PDF outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return _plot(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _plot(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.combined_name):
        raise ValueError(f"Combined plot name must be a safe path component: {args.combined_name!r}")
    groups = discover_run_groups(args.input, args.experiment)
    loaded_groups = [(group, *load_and_average_group(group)) for group in groups]
    group_counts: dict[str, int] = {}
    for group in groups:
        group_counts[group.experiment_id] = group_counts.get(group.experiment_id, 0) + 1
    output_stems = []
    for group in groups:
        suffix = f"-{group.configuration_digest[:8]}" if group_counts[group.experiment_id] > 1 else ""
        output_stems.append(args.outdir / f"{group.experiment_id}{suffix}-learning-curve")
    combined_stem = args.outdir / args.combined_name
    _preflight_plot_outputs([*output_stems, combined_stem], overwrite=args.overwrite)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plotted_instances: list[tuple[str, pd.DataFrame, pd.DataFrame | None, list[str]]] = []
    for loaded, output_stem in zip(loaded_groups, output_stems, strict=True):
        group, averaged_mean, averaged_std, monitor_columns, matched_files = loaded
        title = group.experiment_id
        make_individual_plot(
            averaged_mean=averaged_mean,
            averaged_std=averaged_std if args.show_errors else None,
            monitor_columns=monitor_columns,
            title=title,
            output_stem=output_stem,
            show_errors=args.show_errors,
            overwrite=args.overwrite,
        )
        plotted_instances.append((title, averaged_mean, averaged_std if args.show_errors else None, monitor_columns))
        print(f"Plotted {title}: {len(matched_files)} complete seed runs, {len(monitor_columns)} monitors.")

    make_combined_plot(
        plotted_instances=plotted_instances,
        output_stem=combined_stem,
        show_errors=args.show_errors,
        overwrite=args.overwrite,
    )
    print(f"Finished. Wrote plots to: {args.outdir.resolve()}")

    return 0


def _plot_paths(stem: Path) -> list[Path]:
    # Append rather than replace suffixes: experiment IDs and plot names may contain dots.
    return [stem.with_name(f"{stem.name}{suffix}") for suffix in (".png", ".pdf")]


def _preflight_plot_outputs(stems: Iterable[Path], *, overwrite: bool) -> None:
    paths = [path for stem in stems for path in _plot_paths(stem)]
    if len(set(paths)) != len(paths):
        raise ValueError("Plot selections produce colliding output paths")
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(f"Plot output already exists; pass --overwrite: {existing[0]}")
    unsafe = [path for path in existing if path.is_symlink() or not path.is_file()]
    if unsafe:
        raise ValueError(f"Refusing to replace unsafe plot output: {unsafe[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
