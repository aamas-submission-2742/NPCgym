"""Tables and learning curves comparing methods within a declared norm base."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, cast

from experiments.benchmarks import BENCHMARKS, GARDENER_PAPER, PACMAN_PAPER, TARGETS, Benchmark
from experiments.pacman_bolts import LAYOUT, RECIPES
from experiments.specifications import (
    POLICY_FIX_TECHNIQUE_ID,
    RESTRAINING_BOLT_TECHNIQUE_ID,
    UNCONSTRAINED_TECHNIQUE_ID,
)

if TYPE_CHECKING:
    from experiments.plot_learning_curves import RunGroup

_TASK_LABELS = {
    "success": "Success rate",
    "won": "Win rate",
    "lost": "Loss rate",
    "food_remaining": "Food remaining",
    "score": "Score",
    "unload_market": "Market unloading rate",
    "death": "Death rate",
    "blue_eaten": "Blue ghosts eaten",
}


@dataclass
class BenchmarkRow:
    """One policy variant with equally weighted per-seed measurements and source paths."""

    label: str
    targets: str
    phase: str
    configuration: dict[str, Any]
    samples: dict[int, dict[str, float | None]] = field(default_factory=dict)
    curves: dict[int, list[tuple[int, dict[str, float | None]]]] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic-link input: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _number(value: Any, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numerical results in {path}")
    return float(value)


def _measurement(path: Path, benchmark: Benchmark) -> tuple[dict[str, float | None], dict[str, Any]]:
    document = _json(path)
    episodes = document.get("episodes")
    if document.get("schema_version") != 5 or not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Expected nonempty version-5 episode results: {path}")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or any(not isinstance(episode, dict) for episode in episodes):
        raise ValueError(f"Invalid evaluation metadata or episodes: {path}")
    if metadata.get("episode_count") != len(episodes) or metadata.get("task_return_units") != "unscaled task reward":
        raise ValueError(f"Inconsistent episode count or reward units: {path}")
    values: dict[str, float | None] = {"Episodes": float(len(episodes))}
    try:
        for name, key in (("Return", "return"), ("Episode length", "length")):
            values[name] = mean(_number(episode[key], path) for episode in episodes)
        for key in benchmark.task_metrics:
            values[_TASK_LABELS[key]] = mean(_number(episode["metrics"][key], path) for episode in episodes)
        if benchmark.id.startswith("gardener"):
            values["Success rate"] = mean(
                float(episode["termination"] in {"terminated", "terminated_and_truncated"}) for episode in episodes
            )
        for label, monitor, member in benchmark.counts:
            counts = [_number(episode["monitor_counts"][monitor][member], path) for episode in episodes]
            if min(counts) < 0:
                raise ValueError(f"Negative monitor count: {path}")
            values[label] = mean(counts)
    except KeyError as error:
        raise ValueError(
            f"Missing benchmark metric {error} in {path}; evaluate the full monitored norm base"
        ) from error
    return values, metadata


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]


def _evaluation_limit(configuration: dict[str, Any]) -> int:
    limit = (configuration.get("often") or {}).get(
        "evaluation_max_episode_steps", configuration["run"]["max_episode_steps"]
    )
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Evaluation episode limit must be a positive integer")
    return cast(int, limit)


def _task_identity(configuration: dict[str, Any]) -> str:
    """Compare task settings, including the explicit flags for the same Storm Taxi dynamics."""
    environment = dict(configuration["environment"])
    kwargs = dict(environment["constructor_kwargs"])
    if environment["id"] in {"taxi/storm-v0", "taxi/storm-v1"}:
        # The composed storm environment fixes these formerly explicit settings.
        # Other values must remain distinct; preserve the source metadata itself.
        for name, value in (("is_rainy", False), ("storm_risk", True)):
            if kwargs.get(name) is value:
                del kwargs[name]
    if (
        environment["id"] == "pacman/small-classic-v0"
        and kwargs.get("layout") == "smallClassic"
        and kwargs.get("dfas") is None
    ):
        environment["id"] = "pacman/smallclassic-random-bolts-v0"
        kwargs.pop("dfas", None)
        kwargs.update(layout=LAYOUT)
        kwargs.setdefault("ghost_behavior", "random")
    environment["constructor_kwargs"] = kwargs
    return _identity((environment, _evaluation_limit(configuration)))


def _base_identity(configuration: dict[str, Any], *, pretrained: bool) -> str:
    """Identify matching standalone/pretrained policies without their monitoring targets."""
    often = configuration.get("often") or {}
    return _identity(
        (
            {key: configuration[key] for key in ("environment", "wrappers", "algorithm")},
            often["pretraining_steps"] if pretrained else configuration["run"]["training_steps"],
            often.get("pretraining_max_episode_steps", configuration["run"]["max_episode_steps"])
            if pretrained
            else configuration["run"]["max_episode_steps"],
            _evaluation_limit(configuration),
        )
    )


def _collect(benchmark: Benchmark, runs: list[Any], root: Path, evaluation_name: str = "final") -> list[BenchmarkRow]:
    rows: dict[tuple[str, str], BenchmarkRow] = {}
    tasks = {_task_identity(run.configuration) for run in runs}
    if len(tasks) != 1:
        raise ValueError(f"{benchmark.id}: different environment settings or episode limits; report these separately")
    standalone = {
        _base_identity(run.configuration, pretrained=False)
        for run in runs
        if run.configuration["technique"]["id"] == UNCONSTRAINED_TECHNIQUE_ID
    }
    for run in runs:
        folder = run.run_directory
        document = _json(folder / "run.json")
        config = document["configuration"]
        often = config.get("often")
        fixed = config["technique"]["id"] == POLICY_FIX_TECHNIQUE_ID
        algorithm = config["algorithm"]["implementation"].rsplit(".", 1)[-1]
        variants: tuple[tuple[str, str, str, str], ...]
        if often:
            if often.get("norm_id") not in TARGETS:
                raise ValueError(f"OFTEN results must identify a supported policy-fix recipe: {folder}")
            target = TARGETS[often["norm_id"]]
            variants = (
                ("base", f"{algorithm} base", "None", "base training"),
                ("base-fixed", f"{algorithm} + fixes ({target})", target, "reference"),
                ("policy", f"{algorithm} + OFTEN ({target})", target, "teaching"),
            )
        elif fixed:
            target = "Warn" if benchmark.id.startswith("taxi") else "Environment Friendly"
            variants = (("policy", f"{algorithm} + fixes", target, "training"),)
            if config["environment"]["family"] != "merchant":
                variants = (("base", f"{algorithm} base for fixes", "None", "training"), *variants)
        elif config["technique"]["id"] == RESTRAINING_BOLT_TECHNIQUE_ID:
            targets_by_wrapper = {
                f"pacman/kr2026-{norm}-bolts-and-reward-v{version}": norm for norm in RECIPES for version in (0, 1)
            }
            targets_by_wrapper.update(
                {
                    "merchant/env-friendly-bolts-v0": "Environment Friendly",
                    "merchant/delivery-pacifist-bolts-v1": "DeliveryPacifist",
                    "merchant/env-friendly-minimized-bolts-v0": "Environment Friendly",
                    "merchant/delivery-pacifist-minimized-bolts-v0": "DeliveryPacifist",
                    "taxi/emergency-bolts-v2": "Emergency",
                    "taxi/emergency-bolts-v3": "Emergency",
                    "taxi/emergency-bolts-v4": "Emergency",
                    "taxi/warn-bolts-v0": "Warn",
                    "gardener/collection-rescue-bolts-v0": "Collection and rescue",
                }
            )
            targets_found = [targets_by_wrapper[w["id"]] for w in config["wrappers"] if w["id"] in targets_by_wrapper]
            if len(targets_found) != 1:
                raise ValueError(f"Restraining-bolt results must identify exactly one supported norm base: {folder}")
            name = targets_found[0]
            target = name.replace("-", " ").title() if name in RECIPES else name
            variants = (("policy", f"{algorithm} + bolts ({target})", target, "training"),)
        else:
            variants = (("policy", algorithm, "None", "training"),)
        for variant, label, targets, phase in variants:
            if (
                benchmark.id == PACMAN_PAPER.id
                and often
                and variant == "base"
                and _base_identity(config, pretrained=True) in standalone
            ):
                continue
            final = folder / "evaluations" / evaluation_name
            if often and variant == "base" and evaluation_name == "final":
                final = folder / "evaluations" / "intermediate-0"
            elif variant != "policy":
                final = final / variant
            values, metadata = _measurement(final / "summary.json", benchmark)
            if metadata.get("episode_limit", _evaluation_limit(config)) != _evaluation_limit(config):
                raise ValueError(f"Evaluation episode limit differs from its declared configuration: {final}")
            # A base's training does not depend on its subsequent teaching recipe.
            identity = config
            if often and variant == "base":
                identity = {key: config[key] for key in ("environment", "wrappers", "algorithm", "scenario")}
                if benchmark.id in {PACMAN_PAPER.id, GARDENER_PAPER.id}:
                    # Monitoring targets do not change the pretrained policy.
                    # Conflicting repeated measurements still fail below.
                    identity.pop("scenario")
                identity = {
                    **identity,
                    "steps": often["pretraining_steps"],
                    "episode_limit": _evaluation_limit(config),
                    "pretraining_episode_limit": often.get(
                        "pretraining_max_episode_steps", config["run"]["max_episode_steps"]
                    ),
                    "evaluation_frequency": often.get(
                        "pretraining_evaluation_frequency", config["run"]["intermediate_evaluation_frequency"]
                    ),
                }
            key = (variant, _identity(identity))
            row = rows.setdefault(key, BenchmarkRow(label, targets, phase, identity))
            measurements = dict(values)
            values.update(
                {
                    "Training steps": float(
                        document["pretraining"]["actual_steps"] if often else document["timesteps"]["actual"]
                    ),
                    "Teaching transitions": float(
                        document["timesteps"]["actual"] if often and variant == "policy" else 0
                    ),
                    "Training seconds": (
                        None
                        if document["pretraining"].get("seconds_include_evaluation", False)
                        else document["pretraining"]["seconds"]
                    )
                    if often
                    else document.get("training_seconds"),
                    "Teaching seconds": document["teaching"]["seconds"] if often and variant == "policy" else 0.0,
                    "Fix ms/decision": (
                        1000 * metadata["planning_seconds"] / metadata["decisions"]
                        if metadata.get("decisions")
                        else None
                    ),
                }
            )
            for value in values.values():
                if value is not None:
                    _number(value, final)
            seed = document["seed"]
            row.sources.append(str(final.relative_to(root)))
            if seed in row.samples:
                if any(row.samples[seed][name] != value for name, value in measurements.items()):
                    raise ValueError(f"Conflicting repeated base for seed {seed}: {folder}")
                continue
            row.samples[seed] = values
            points: list[tuple[int, dict[str, float | None]]] = []
            if phase == "reference" and evaluation_name == "final":
                points = [(0, measurements), (document["timesteps"]["actual"], measurements)]
            else:
                parent = folder / "evaluations" / "base" if phase == "base training" else folder / "evaluations"
                same_base_limit = not often or often.get(
                    "pretraining_max_episode_steps", config["run"]["max_episode_steps"]
                ) == _evaluation_limit(config)
                show_curve = evaluation_name == "final" and (phase != "base training" or same_base_limit)
                for point in parent.glob("intermediate-*") if show_curve else ():
                    steps = int(point.name.removeprefix("intermediate-"))
                    if fixed and variant == "base":
                        continue  # Only final raw-base data exist for these runs.
                    data, _ = _measurement(point / "summary.json", benchmark)
                    points.append((steps, data))
                final_step = int(
                    _number(values["Teaching transitions"] if phase == "teaching" else values["Training steps"], final)
                )
                points = [point for point in points if point[0] != final_step]
                points.append((final_step, measurements))
            row.curves[seed] = sorted(points)
    # Different hyperparameters remain distinct rows, even for the same learner.
    labels = [row.label for row in rows.values()]
    for key, row in rows.items():
        if labels.count(row.label) > 1:
            row.label += f" [{key[1][:6]}]"
    if benchmark.id == PACMAN_PAPER.id:
        for row in rows.values():
            if row.targets == "None":
                algorithm = row.configuration["algorithm"]["implementation"].rsplit(".", 1)[-1]
                gamma = row.configuration["algorithm"]["constructor_kwargs"]["gamma"]
                trapped = any(w["id"] == "pacman/trapped-observation-v1" for w in row.configuration["wrappers"])
                name = f"{algorithm} for fixes/OFTEN" if row.phase == "base training" else algorithm
                row.label = f"{name} (γ={gamma}" + (", Trapped features" if trapped else "") + ")"
        return sorted(rows.values(), key=_paper_order)
    return sorted(
        rows.values(), key=lambda row: (0 if row.targets == "None" else 2 if row.phase == "teaching" else 1, row.label)
    )


_PAPER_GROUPS = (
    "Baselines",
    "Vegan",
    "Vegetarian Orange",
    "Trapped",
    "HungryVegan",
    "VeganConflict",
    "HungryPenaltyVegan",
)


def _paper_group(row: BenchmarkRow) -> str:
    return {
        "None": "Baselines",
        "Vegan": "Vegan",
        "Vegetarian Orange": "Vegetarian Orange",
        "Vegetarian": "Vegetarian Orange",
        "Trapped": "Trapped",
        "Hungry Vegan": "HungryVegan",
        "Vegan Conflict": "VeganConflict",
        "Hungry Vegan Penalty": "HungryPenaltyVegan",
    }[row.targets]


def _paper_order(row: BenchmarkRow) -> tuple[int, int, str]:
    if row.targets == "None":
        gamma = row.configuration["algorithm"]["constructor_kwargs"]["gamma"]
        return 0, int(gamma != 0.99), row.label
    method = (
        0 if row.phase == "reference" else 1 if row.phase == "teaching" else 2 if row.label.startswith("DQN") else 3
    )
    return _PAPER_GROUPS.index(_paper_group(row)), method, row.label


def _stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark(path: Path, benchmark: Benchmark, rows: list[BenchmarkRow]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.mkdir()
    metrics = list(rows[0].samples[next(iter(rows[0].samples))])
    table = []
    markdown = [
        f"# {benchmark.title}",
        "",
        "Values are means ± sample standard deviations across training seeds.",
        "",
        "| Method | Optimized norms | Seeds | " + " | ".join(metrics) + " |",
        "|---|---|---:|" + "---:|" * len(metrics),
    ]
    curves: list[dict[str, Any]] = []
    previous_group = None
    for row in rows:
        if benchmark.id == PACMAN_PAPER.id:
            group = _paper_group(row)
            if group != previous_group:
                markdown.append("| **" + group + "** |" + " |" * (len(metrics) + 2))
                previous_group = group
        record: dict[str, Any] = {"method": row.label, "optimized_norms": row.targets, "seeds": len(row.samples)}
        cells = [row.label, row.targets, str(len(row.samples))]
        for metric in metrics:
            values = [sample[metric] for sample in row.samples.values()]
            if any(value is None for value in values):
                avg = spread = None
                cells.append("—")
            else:
                avg, spread = _stats([float(value) for value in values if value is not None])
                cells.append(
                    f"{avg:g}"
                    if spread == 0 and metric in {"Episodes", "Training steps", "Teaching transitions"}
                    else f"{avg:.3g} ± {spread:.2g}"
                )
            record[metric + " mean"] = avg
            record[metric + " std"] = spread
        table.append(record)
        markdown.append("| " + " | ".join(cells) + " |")
        grids = {tuple(step for step, _ in points) for points in row.curves.values()}
        if len(grids) != 1:
            raise ValueError(f"{row.label}: curve grids differ across seeds; use matching evaluation schedules")
        for index, steps in enumerate(next(iter(grids))):
            for metric in ("Return", *(item[0] for item in benchmark.counts)):
                avg, spread = _stats([_number(points[index][1][metric], path) for points in row.curves.values()])
                curves.append(
                    {
                        "method": row.label,
                        "phase": row.phase,
                        "steps": steps,
                        "metric": metric,
                        "mean": avg,
                        "std": spread,
                        "seeds": len(row.samples),
                    }
                )
    markdown += [
        "",
        "Training and teaching budgets and times are separate. Missing measurements are shown as —.",
        "Evaluation sample sizes are listed; only the base/fixes/OFTEN comparisons from the same run are paired.",
        "Task metrics use their recorded units; win/success values are proportions.",
        "",
    ]
    (path / "table.md").write_text("\n".join(markdown))
    _write_csv(path / "table.csv", table)
    _write_csv(path / "curves.csv", curves)
    for stem, plotted in (("return", ("Return",)), ("violations", tuple(item[0] for item in benchmark.counts))):
        figure, axes = plt.subplots(len(plotted), 1, figsize=(9, 3.5 * len(plotted)), squeeze=False)
        try:
            for axis, metric in zip(axes[:, 0], plotted, strict=True):
                for row in rows:
                    points = [point for point in curves if point["method"] == row.label and point["metric"] == metric]
                    x = [point["steps"] for point in points]
                    y = [point["mean"] for point in points]
                    band = [point["std"] for point in points]
                    axis.plot(x, y, label=row.label, linestyle="--" if row.phase == "reference" else "-", marker=".")
                    axis.fill_between(
                        x,
                        [a - b for a, b in zip(y, band, strict=True)],
                        [a + b for a, b in zip(y, band, strict=True)],
                        alpha=0.12,
                    )
                axis.set_ylabel(metric)
                axis.set_xlabel("Training steps / OFTEN teaching transitions")
                axis.grid(alpha=0.2)
            axes[0, 0].set_title(benchmark.title)
            axes[-1, 0].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
            figure.tight_layout()
            for suffix in ("png", "pdf"):
                figure.savefig(path / f"{stem}.{suffix}", bbox_inches="tight")
        finally:
            plt.close(figure)
    (path / "sources.json").write_text(
        json.dumps(
            [
                {
                    "method": row.label,
                    "phase": row.phase,
                    "optimized_norms": row.targets,
                    "configuration": row.configuration,
                    "seeds": sorted(row.samples),
                    "evaluations": row.sources,
                }
                for row in rows
            ],
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def collect_benchmark(input_path: Path, benchmark: Benchmark, *, evaluation_name: str = "final") -> list[BenchmarkRow]:
    """Read validated policy measurements without writing tables or figures.

    Requires completed runs for every benchmark experiment ID. Identical
    pretrained bases are deduplicated, with conflicting seed measurements
    rejected. Each sample is an episode mean; aggregate seeds with equal weight.
    Available seed counts are preserved rather than enforcing a paper protocol.
    Requires the ``plots`` extra for the shared run and curve validation.
    """
    from experiments.execution import SelectionError, _result_name
    from experiments.plot_learning_curves import discover_run_groups

    if evaluation_name != "final":
        try:
            _result_name(evaluation_name)
        except SelectionError as error:
            raise ValueError(str(error)) from error
    root = input_path.resolve()
    groups = discover_run_groups(root, benchmark.experiments)
    return _validated_rows(root, benchmark, groups, evaluation_name)


def _validated_rows(
    root: Path, benchmark: Benchmark, groups: Sequence[RunGroup], evaluation_name: str
) -> list[BenchmarkRow]:
    from experiments.plot_learning_curves import load_and_average_group

    for group in groups:
        load_and_average_group(group)
    return _collect(benchmark, [run for group in groups for run in group.runs], root, evaluation_name)


def create_report(
    input_path: Path, output_path: Path, *, benchmark_ids: tuple[str, ...] = (), evaluation_name: str = "final"
) -> Path:
    """Create one table and return/violation figures per available benchmark.

    Named evaluations report their final points without mixing in training curves.
    Reads numerical results only. Rejects missing monitored outcomes, mixed
    curve grids and duplicate seeds with conflicting bases. Seed means have
    equal weight. Builds in a temporary sibling directory, refuses overlapping
    input/output trees and existing output, and removes temporary files on error.
    """
    from experiments.execution import SelectionError, _result_name
    from experiments.plot_learning_curves import discover_run_groups

    if evaluation_name != "final":
        try:
            _result_name(evaluation_name)
        except SelectionError as error:
            raise ValueError(str(error)) from error

    root = input_path.resolve()
    target = output_path.absolute()
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError("Report output must not contain symbolic links")
    target = target.resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Report input and output directories must be separate")
    if target.exists():
        raise FileExistsError(f"Report output already exists; choose a new directory: {target}")
    benchmarks = (*BENCHMARKS, *(item for item in (PACMAN_PAPER, GARDENER_PAPER) if item.id in benchmark_ids))
    unknown = set(benchmark_ids) - {item.id for item in benchmarks}
    if unknown:
        raise ValueError(f"Unknown benchmark IDs: {sorted(unknown)}")
    groups = discover_run_groups(root)
    selected = []
    for benchmark in benchmarks:
        if benchmark_ids and benchmark.id not in benchmark_ids:
            continue
        matching = [group for group in groups if group.experiment_id in benchmark.experiments]
        if not matching:
            if benchmark_ids:
                raise ValueError(f"No completed runs for benchmark {benchmark.id}")
            continue
        selected.append((benchmark, _validated_rows(root, benchmark, matching, evaluation_name)))
    if not selected:
        raise ValueError("No completed runs match the declared benchmarks")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        index = ["# Experiment comparisons", ""]
        for benchmark, rows in selected:
            _write_benchmark(temporary / benchmark.id, benchmark, rows)
            index.append(f"- [{benchmark.title}]({benchmark.id}/table.md)")
        (temporary / "index.md").write_text("\n".join(index) + "\n")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target
