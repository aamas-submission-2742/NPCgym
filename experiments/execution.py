"""Safe, sequential execution for versioned NPC Gym experiment specifications."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast
from uuid import uuid4

import gymnasium as gym

from experiments.specifications import (
    GARDENER_ENVIRONMENT_ID,
    GARDENER_PERMISSION_DRAIN_SCENARIO_ID,
    GARDENER_SCENARIO_ID,
    MERCHANT_ENV_FRIENDLY_SCENARIO_ID,
    MERCHANT_ENVIRONMENT_ID,
    OFTEN_TECHNIQUE_ID,
    PACMAN_BOLTS_ENVIRONMENT_ID,
    PACMAN_ENVIRONMENT_ID,
    PACMAN_IMAGES_ENVIRONMENT_ID,
    PACMAN_VEGAN_SCENARIO_ID,
    PACMAN_VEGETARIAN_SCENARIO_ID,
    POLICY_FIX_TECHNIQUE_ID,
    REGISTRY,
    TAXI_ENVIRONMENT_ID,
    TAXI_SCENARIO_ID,
    AlgorithmSpecification,
    ExperimentRegistry,
    GardenerOFTENSettings,
    KeywordArguments,
    PacmanOFTENSettings,
    ResolvedExperiment,
    RunSettings,
)
from npc_gym.algorithms import TabularQLearning
from npc_gym.evaluation import (
    EpisodeCSVWriter,
    EvaluationJSONWriter,
    EvaluationSummary,
    LearningCurveCSVWriter,
    evaluate,
)
from npc_gym.monitors.gardener_monitors import DRAIN_NORM_ID, NO_COLLECT_NORM_ID, PERMISSION_AWARE_NORM_ID
from npc_gym.monitors.merchant_monitors import ENV_FRIENDLY_NORM_ID
from npc_gym.monitors.pacman_monitors import TRAPPED_NORM_ID
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID

RUN_SCHEMA_VERSION = 1
CONFIGURATION_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
MODEL_FILENAME = "model.zip"
RUN_FILENAME = "run.json"
CURVE_FILENAME = "learning_curve.csv"

_SAFE_RESULT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SEED_DIRECTORY = re.compile(r"^seed-(0|[1-9][0-9]*)$")
_TABULAR_IMPLEMENTATION = "npc_gym.algorithms.TabularQLearning"
_SB3_IMPLEMENTATIONS = frozenset({"stable_baselines3.DQN", "stable_baselines3.PPO"})
_OWNED_INTERMEDIATE = re.compile(r"^intermediate-(0|[1-9][0-9]*)$")
_MAX_SEED = 2**32 - 1


class ExecutionError(RuntimeError):
    """Base class for experiment planning and execution failures."""


class SelectionError(ExecutionError):
    """Raised when filters do not describe a unique, non-empty batch."""


class PreflightError(ExecutionError):
    """Raised before a batch starts when it cannot run safely."""


class SavedRunError(ExecutionError):
    """Raised when saved metadata is malformed, incomplete, or unavailable."""


class ConfigurationMismatchError(PreflightError):
    """Raised when an experiment identity is already bound to other settings."""


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """One resolved experiment/seed pair and its owned output paths."""

    resolved: ResolvedExperiment
    seed: int
    intermediate_evaluation_seed: int
    final_evaluation_seed: int
    output_root: Path
    run_directory: Path

    @property
    def model_path(self) -> Path:
        return self.run_directory / MODEL_FILENAME

    @property
    def run_path(self) -> Path:
        return self.run_directory / RUN_FILENAME

    @property
    def curve_path(self) -> Path:
        return self.run_directory / CURVE_FILENAME


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """One saved checkpoint and a distinct evaluation output target."""

    resolved: ResolvedExperiment
    seed: int
    evaluation_seed: int
    episodes: int
    name: str
    run_directory: Path
    model_path: Path
    output_directory: Path
    model_checksum: str


@dataclass(frozen=True, slots=True)
class RunResult:
    """A completed training run's key persisted results."""

    run_directory: Path
    actual_timesteps: int
    model_checksum: str
    final_evaluation: EvaluationSummary


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """A completed saved-checkpoint evaluation."""

    output_directory: Path
    summary: EvaluationSummary


class _Model(Protocol):
    num_timesteps: int

    def save(self, path: str | os.PathLike[str]) -> None: ...


def select_experiments(
    registry: ExperimentRegistry = REGISTRY,
    *,
    experiment_ids: Sequence[str] = (),
    environment_ids: Sequence[str] = (),
    wrapper_ids: Sequence[str] = (),
    scenario_ids: Sequence[str] = (),
    algorithm_ids: Sequence[str] = (),
    technique_ids: Sequence[str] = (),
) -> tuple[ResolvedExperiment, ...]:
    """Resolve a deterministic non-empty selection using repeatable exact-ID filters."""
    requested = {
        "experiment": _unique("experiment", experiment_ids),
        "environment": _unique("environment", environment_ids),
        "wrapper": _unique("wrapper", wrapper_ids),
        "scenario": _unique("scenario", scenario_ids),
        "algorithm": _unique("algorithm", algorithm_ids),
        "technique": _unique("technique", technique_ids),
    }
    candidates = tuple(registry.resolve(item) for item in (requested["experiment"] or registry.experiment_ids))
    known = {
        "environment": {registry.resolve(item).environment.id for item in registry.experiment_ids},
        "wrapper": {wrapper.id for item in registry.experiment_ids for wrapper in registry.resolve(item).wrappers},
        "scenario": {registry.resolve(item).scenario.id for item in registry.experiment_ids},
        "algorithm": {registry.resolve(item).algorithm.id for item in registry.experiment_ids},
        "technique": {registry.resolve(item).technique.id for item in registry.experiment_ids},
    }
    for kind in ("environment", "wrapper", "scenario", "algorithm", "technique"):
        unknown = set(requested[kind]) - known[kind]
        if unknown:
            raise SelectionError(f"Unknown {kind} filter(s): {', '.join(sorted(unknown))}")

    selected = []
    for resolved in candidates:
        if requested["environment"] and resolved.environment.id not in requested["environment"]:
            continue
        if requested["wrapper"] and not set(requested["wrapper"]).issubset(
            {wrapper.id for wrapper in resolved.wrappers}
        ):
            continue
        if requested["scenario"] and resolved.scenario.id not in requested["scenario"]:
            continue
        if requested["algorithm"] and resolved.algorithm.id not in requested["algorithm"]:
            continue
        if requested["technique"] and resolved.technique.id not in requested["technique"]:
            continue
        selected.append(resolved)
    if not selected:
        raise SelectionError("Experiment filters selected no supported configurations")
    return tuple(sorted(selected, key=lambda item: item.specification.id))


def plan_training_batch(
    output_root: str | os.PathLike[str],
    *,
    seeds: Sequence[int],
    registry: ExperimentRegistry = REGISTRY,
    experiment_ids: Sequence[str] = (),
    environment_ids: Sequence[str] = (),
    wrapper_ids: Sequence[str] = (),
    scenario_ids: Sequence[str] = (),
    algorithm_ids: Sequence[str] = (),
    technique_ids: Sequence[str] = (),
) -> tuple[TrainingPlan, ...]:
    """Resolve all selected training runs without touching the filesystem."""
    selected = select_experiments(
        registry,
        experiment_ids=experiment_ids,
        environment_ids=environment_ids,
        wrapper_ids=wrapper_ids,
        scenario_ids=scenario_ids,
        algorithm_ids=algorithm_ids,
        technique_ids=technique_ids,
    )
    checked_seeds = _seeds(seeds, required=True)
    root = Path(output_root).expanduser().resolve(strict=False)
    plans = []
    for resolved in selected:
        _validate_execution_settings(resolved)
        for seed in checked_seeds:
            intermediate_seed, final_seed = _evaluation_seeds(resolved, seed)
            plans.append(
                TrainingPlan(
                    resolved=resolved,
                    seed=seed,
                    intermediate_evaluation_seed=intermediate_seed,
                    final_evaluation_seed=final_seed,
                    output_root=root,
                    run_directory=root / resolved.specification.id / f"seed-{seed}",
                )
            )
    _distinct_paths([plan.run_directory for plan in plans])
    return tuple(plans)


def preflight_training(plans: Sequence[TrainingPlan], *, overwrite: bool = False) -> None:
    """Validate an entire training batch before any run creates output."""
    if not plans:
        raise SelectionError("Training batch is empty")
    roots = {plan.output_root for plan in plans}
    for root in roots:
        _check_output_parent(root)
    _distinct_paths([plan.run_directory for plan in plans])
    for plan in plans:
        _require_safe_run_path(plan)
        try:
            _seed(plan.seed, "training seed")
            _seed(plan.intermediate_evaluation_seed, "intermediate evaluation seed")
            _seed(plan.final_evaluation_seed, "final evaluation seed")
        except SelectionError as error:
            raise PreflightError(str(error)) from error
        expected = _evaluation_seeds(plan.resolved, plan.seed)
        if (plan.intermediate_evaluation_seed, plan.final_evaluation_seed) != expected:
            raise PreflightError("Training plan evaluation seeds do not match its resolved settings")
    _preflight_batch_identities(plans)
    for plan in plans:
        _check_output_parent(plan.run_directory)
        _check_dependency(plan.resolved)
        _preflight_experiment_identity(plan)
        _preflight_training_target(plan, overwrite=overwrite)


def execute_training_batch(plans: Sequence[TrainingPlan], *, overwrite: bool = False) -> tuple[RunResult, ...]:
    """Preflight and execute training plans sequentially."""
    preflight_training(plans, overwrite=overwrite)
    return tuple(_execute_training(plan, overwrite=overwrite) for plan in plans)


def plan_evaluation_batch(
    output_root: str | os.PathLike[str],
    *,
    registry: ExperimentRegistry = REGISTRY,
    experiment_ids: Sequence[str] = (),
    environment_ids: Sequence[str] = (),
    wrapper_ids: Sequence[str] = (),
    scenario_ids: Sequence[str] = (),
    algorithm_ids: Sequence[str] = (),
    technique_ids: Sequence[str] = (),
    seeds: Sequence[int] = (),
    name: str,
    evaluation_seed: int | None = None,
    episodes: int | None = None,
    current_monitors: bool = False,
) -> tuple[EvaluationPlan, ...]:
    """Resolve saved runs and named evaluation targets without writing.

    Episode counts default to the current experiment recipe. With
    ``current_monitors=True``, use its monitor factories too; saved environment,
    learner and wrapper settings remain authoritative. Source artifacts and
    checkpoints are unchanged.
    """
    _result_name(name)
    selected = select_experiments(
        registry,
        experiment_ids=experiment_ids,
        environment_ids=environment_ids,
        wrapper_ids=wrapper_ids,
        scenario_ids=scenario_ids,
        algorithm_ids=algorithm_ids,
        technique_ids=technique_ids,
    )
    checked_seeds = _seeds(seeds, required=False)
    if evaluation_seed is not None:
        evaluation_seed = _seed(evaluation_seed, "evaluation seed")
    if episodes is not None:
        episodes = _positive_integer(episodes, "episodes")
    root = Path(output_root).expanduser().resolve(strict=False)
    plans = []
    for selected_experiment in selected:
        experiment_directory = root / selected_experiment.specification.id
        run_directories = (
            [experiment_directory / f"seed-{seed}" for seed in checked_seeds]
            if checked_seeds
            else _discover_seed_directories(experiment_directory)
        )
        for run_directory in run_directories:
            document = read_run_document(run_directory / RUN_FILENAME)
            if document["status"] != "complete":
                raise SavedRunError(f"Run is not complete: {run_directory}")
            saved_seed = _nonnegative_integer(document.get("seed"), "run.seed")
            match = _SEED_DIRECTORY.fullmatch(run_directory.name)
            if match is None or saved_seed != int(match.group(1)):
                raise SavedRunError(f"Saved seed does not match run directory: {run_directory}")
            resolved = resolved_from_configuration(
                document.get("configuration"), registry=registry, current_monitors=current_monitors
            )
            if resolved.specification.id != selected_experiment.specification.id:
                raise SavedRunError(f"Saved experiment identity does not match run directory: {run_directory}")
            _validate_complete_run(document, resolved, saved_seed, run_directory)
            model_path = run_directory / MODEL_FILENAME
            if not model_path.is_file() or model_path.is_symlink():
                raise SavedRunError(f"Required model is missing or unsafe: {model_path}")
            checksum = sha256_file(model_path)
            recorded_checksum = _model_checksum_from_run(document)
            if recorded_checksum != checksum:
                raise SavedRunError(f"Model checksum does not match run metadata: {model_path}")
            saved_seeds = _mapping(document.get("seeds"), "run.seeds")
            default_seed = _nonnegative_integer(saved_seeds.get("final_evaluation"), "run.seeds.final_evaluation")
            plan_episodes = episodes or selected_experiment.specification.run.final_evaluation_episodes
            plans.append(
                EvaluationPlan(
                    resolved=resolved,
                    seed=saved_seed,
                    evaluation_seed=default_seed if evaluation_seed is None else evaluation_seed,
                    episodes=plan_episodes,
                    name=name,
                    run_directory=run_directory,
                    model_path=model_path,
                    output_directory=run_directory / "evaluations" / name,
                    model_checksum=checksum,
                )
            )
    if not plans:
        raise SelectionError("Evaluation filters selected no saved runs")
    _distinct_paths([plan.output_directory for plan in plans])
    return tuple(plans)


def preflight_evaluation(plans: Sequence[EvaluationPlan], *, overwrite: bool = False) -> None:
    """Validate every selected saved evaluation before writing any result."""
    if not plans:
        raise SelectionError("Evaluation batch is empty")
    _distinct_paths([plan.output_directory for plan in plans])
    for plan in plans:
        _result_name(plan.name)
        try:
            _seed(plan.seed, "saved training seed")
            _seed(plan.evaluation_seed, "evaluation seed")
            _positive_integer(plan.episodes, "episodes")
        except SelectionError as error:
            raise PreflightError(str(error)) from error
        if (
            plan.resolved.specification.often is not None or plan.resolved.technique.id == POLICY_FIX_TECHNIQUE_ID
        ) and plan.evaluation_seed + plan.episodes - 1 > _MAX_SEED:
            raise PreflightError("Paired evaluation episode seeds exceed 2**32 - 1")
        if plan.model_path != plan.run_directory / MODEL_FILENAME:
            raise PreflightError(f"Model path is not the run-owned model.zip: {plan.model_path}")
        if plan.output_directory != plan.run_directory / "evaluations" / plan.name:
            raise PreflightError(f"Evaluation path is not derived from its selected name: {plan.output_directory}")
        if not plan.model_path.is_file() or plan.model_path.is_symlink():
            raise PreflightError(f"Required model is missing or unsafe: {plan.model_path}")
        _check_dependency(plan.resolved)
        document = read_run_document(plan.run_directory / RUN_FILENAME)
        _validate_complete_run(document, plan.resolved, plan.seed, plan.run_directory)
        if sha256_file(plan.model_path) != plan.model_checksum:
            raise PreflightError(f"Model changed after evaluation planning: {plan.model_path}")
        _require_within(plan.output_directory, plan.run_directory, "evaluation output")
        _check_no_symlink_components(plan.output_directory, stop=plan.run_directory)
        _check_output_parent(plan.output_directory)
        output_exists = plan.output_directory.exists() or plan.output_directory.is_symlink()
        if output_exists and not overwrite:
            raise PreflightError(f"Evaluation output already exists; pass --overwrite: {plan.output_directory}")
        if output_exists and (plan.output_directory.is_symlink() or not plan.output_directory.is_dir()):
            raise PreflightError(f"Evaluation output is not a directory: {plan.output_directory}")


def execute_evaluation_batch(
    plans: Sequence[EvaluationPlan], *, overwrite: bool = False, render: bool = False
) -> tuple[EvaluationResult, ...]:
    """Preflight and evaluate saved checkpoints sequentially without modifying models."""
    preflight_evaluation(plans, overwrite=overwrite)
    results = []
    for plan in plans:
        summary = _evaluate_saved_to_directory(
            plan.resolved,
            plan.model_path,
            plan.output_directory,
            episodes=plan.episodes,
            seed=plan.evaluation_seed,
            render=render,
            checksum=plan.model_checksum,
            overwrite=overwrite,
        )
        results.append(EvaluationResult(plan.output_directory, summary))
    return tuple(results)


def resolved_configuration(resolved: ResolvedExperiment) -> dict[str, object]:
    """Serialize all seed-independent settings needed to reproduce a saved run."""
    run = resolved.specification.run
    return {
        **({"often": asdict(resolved.specification.often)} if resolved.specification.often is not None else {}),
        "schema_version": CONFIGURATION_SCHEMA_VERSION,
        "experiment_id": resolved.specification.id,
        "environment": {
            "id": resolved.environment.id,
            "family": resolved.environment.family,
            "constructor_kwargs": resolved.environment.constructor_kwargs.to_dict(),
            "render_kwargs": resolved.environment.render_kwargs.to_dict(),
        },
        "wrappers": [{"id": wrapper.id, "kind": wrapper.kind} for wrapper in resolved.wrappers],
        "scenario": {
            "id": resolved.scenario.id,
            "monitor_ids": [monitor_id for monitor_id, _factory in resolved.scenario.monitor_factories],
        },
        "algorithm": {
            "id": resolved.algorithm.id,
            "execution_path": resolved.algorithm.execution_path,
            "implementation": resolved.algorithm.implementation,
            "policy": resolved.algorithm.policy,
            "constructor_kwargs": resolved.algorithm.constructor_kwargs.to_dict(),
            "optional_extra": resolved.algorithm.optional_extra,
        },
        "technique": {"id": resolved.technique.id},
        "run": {
            "training_steps": run.training_steps,
            "max_episode_steps": run.max_episode_steps,
            "final_evaluation_episodes": run.final_evaluation_episodes,
            "intermediate_evaluation_episodes": run.intermediate_evaluation_episodes,
            "intermediate_evaluation_frequency": run.intermediate_evaluation_frequency,
            "learning_kwargs": run.learning_kwargs.to_dict(),
            "evaluation_seed_offset": run.evaluation_seed_offset,
            "deterministic_evaluation": run.deterministic_evaluation,
            "task_return_units": run.task_return_units,
            "training_reward_transform": run.training_reward_transform,
        },
    }


def resolved_from_configuration(
    value: object, *, registry: ExperimentRegistry = REGISTRY, current_monitors: bool = False
) -> ResolvedExperiment:
    """Rebuild saved settings, optionally adopting the current evaluation monitors.

    Only monitor selection can be overridden; environment, wrapper, algorithm
    and technique identities must still match the registered experiment.
    """
    document = _mapping(value, "configuration")
    _exact_keys(
        document,
        {"schema_version", "experiment_id", "environment", "wrappers", "scenario", "algorithm", "technique", "run"}
        | ({"often"} if "often" in document else set()),
        "configuration",
    )
    if _integer(document.get("schema_version"), "configuration.schema_version") != CONFIGURATION_SCHEMA_VERSION:
        raise SavedRunError(f"Unsupported configuration schema version {document.get('schema_version')!r}")
    experiment_id = _string(document.get("experiment_id"), "configuration.experiment_id")
    try:
        current = registry.resolve(experiment_id)
    except ValueError as error:
        raise SavedRunError(f"Saved experiment or component version is unavailable: {error}") from error

    environment = _mapping(document.get("environment"), "configuration.environment")
    _exact_keys(environment, {"id", "family", "constructor_kwargs", "render_kwargs"}, "configuration.environment")
    if _string(environment.get("id"), "configuration.environment.id") != current.environment.id:
        raise SavedRunError("Saved environment version does not match its registered experiment")
    if _string(environment.get("family"), "configuration.environment.family") != current.environment.family:
        raise SavedRunError("Saved environment family does not match its registered factory")
    saved_environment = replace(
        current.environment,
        constructor_kwargs=KeywordArguments.from_mapping(
            _mapping(environment.get("constructor_kwargs"), "configuration.environment.constructor_kwargs")
        ),
        render_kwargs=KeywordArguments.from_mapping(
            _mapping(environment.get("render_kwargs"), "configuration.environment.render_kwargs")
        ),
    )

    wrappers = _sequence(document.get("wrappers"), "configuration.wrappers")
    if len(wrappers) != len(current.wrappers):
        raise SavedRunError("Saved wrapper order or version is unavailable")
    wrapper_ids = []
    for index, item in enumerate(wrappers):
        wrapper = _mapping(item, f"configuration.wrappers[{index}]")
        _exact_keys(wrapper, {"id", "kind"}, f"configuration.wrappers[{index}]")
        wrapper_ids.append(_string(wrapper.get("id"), f"configuration.wrappers[{index}].id"))
        if _string(wrapper.get("kind"), f"configuration.wrappers[{index}].kind") != current.wrappers[index].kind:
            raise SavedRunError("Saved wrapper kind does not match its registered factory")
    if wrapper_ids != [wrapper.id for wrapper in current.wrappers]:
        raise SavedRunError("Saved wrapper order or version is unavailable")

    scenario = _mapping(document.get("scenario"), "configuration.scenario")
    _exact_keys(scenario, {"id", "monitor_ids"}, "configuration.scenario")
    if _string(scenario.get("id"), "configuration.scenario.id") != current.scenario.id and not current_monitors:
        raise SavedRunError("Saved scenario version does not match its registered experiment")
    monitor_ids = [
        _string(item, "configuration.scenario.monitor_ids[]")
        for item in _sequence(scenario.get("monitor_ids"), "configuration.scenario.monitor_ids")
    ]
    if (
        monitor_ids != [monitor_id for monitor_id, _factory in current.scenario.monitor_factories]
        and not current_monitors
    ):
        raise SavedRunError("Saved monitor order or version is unavailable")

    algorithm = _mapping(document.get("algorithm"), "configuration.algorithm")
    _exact_keys(
        algorithm,
        {"id", "execution_path", "implementation", "policy", "constructor_kwargs", "optional_extra"},
        "configuration.algorithm",
    )
    fixed_algorithm_fields = {
        "id": current.algorithm.id,
        "execution_path": current.algorithm.execution_path,
        "implementation": current.algorithm.implementation,
        "policy": current.algorithm.policy,
        "optional_extra": current.algorithm.optional_extra,
    }
    for field, expected in fixed_algorithm_fields.items():
        if not _json_equal(algorithm.get(field), expected):
            raise SavedRunError(f"Saved algorithm {field} does not match its registered version")
    saved_algorithm = replace(
        current.algorithm,
        constructor_kwargs=KeywordArguments.from_mapping(
            _mapping(algorithm.get("constructor_kwargs"), "configuration.algorithm.constructor_kwargs")
        ),
    )

    technique = _mapping(document.get("technique"), "configuration.technique")
    _exact_keys(technique, {"id"}, "configuration.technique")
    if _string(technique.get("id"), "configuration.technique.id") != current.technique.id:
        raise SavedRunError("Saved technique version does not match its registered experiment")

    run_value = _mapping(document.get("run"), "configuration.run")
    _exact_keys(
        run_value,
        {
            "training_steps",
            "max_episode_steps",
            "final_evaluation_episodes",
            "intermediate_evaluation_episodes",
            "intermediate_evaluation_frequency",
            "learning_kwargs",
            "evaluation_seed_offset",
            "deterministic_evaluation",
            "task_return_units",
            "training_reward_transform",
        },
        "configuration.run",
    )
    try:
        saved_run = RunSettings(
            training_steps=_integer(run_value.get("training_steps"), "configuration.run.training_steps"),
            max_episode_steps=_integer(run_value.get("max_episode_steps"), "configuration.run.max_episode_steps"),
            final_evaluation_episodes=_integer(
                run_value.get("final_evaluation_episodes"), "configuration.run.final_evaluation_episodes"
            ),
            intermediate_evaluation_episodes=_integer(
                run_value.get("intermediate_evaluation_episodes"), "configuration.run.intermediate_evaluation_episodes"
            ),
            intermediate_evaluation_frequency=_integer(
                run_value.get("intermediate_evaluation_frequency"),
                "configuration.run.intermediate_evaluation_frequency",
            ),
            learning_kwargs=KeywordArguments.from_mapping(
                _mapping(run_value.get("learning_kwargs"), "configuration.run.learning_kwargs")
            ),
            evaluation_seed_offset=_integer(
                run_value.get("evaluation_seed_offset"), "configuration.run.evaluation_seed_offset"
            ),
            deterministic_evaluation=_boolean(
                run_value.get("deterministic_evaluation"), "configuration.run.deterministic_evaluation"
            ),
            task_return_units=_string(run_value.get("task_return_units"), "configuration.run.task_return_units"),
            training_reward_transform=_string(
                run_value.get("training_reward_transform"), "configuration.run.training_reward_transform"
            ),
        )
    except (TypeError, ValueError) as error:
        raise SavedRunError(f"Invalid saved run settings: {error}") from error
    often = None
    if ("often" in document) != (current.specification.often is not None):
        raise SavedRunError("Saved OFTEN settings do not match the registered technique")
    if "often" in document:
        values = _mapping(document["often"], "configuration.often")
        assert current.specification.often is not None
        _exact_keys(values, set(asdict(current.specification.often)), "configuration.often")
        try:
            often = type(current.specification.often)(**values)
        except (TypeError, ValueError) as error:
            raise SavedRunError(f"Invalid saved OFTEN settings: {error}") from error
    saved_specification = replace(current.specification, run=saved_run, often=often)
    resolved = ResolvedExperiment(
        saved_specification,
        saved_environment,
        current.wrappers,
        current.scenario,
        saved_algorithm,
        current.technique,
    )
    _validate_execution_settings(resolved)
    return resolved


def read_run_document(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and minimally validate one version-1 run document."""
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise SavedRunError(f"Run metadata is missing or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SavedRunError(f"Cannot read run metadata {source}: {error}") from error
    document = dict(_mapping(value, "run"))
    if _integer(document.get("schema_version"), "run.schema_version") != RUN_SCHEMA_VERSION:
        raise SavedRunError(f"Unsupported run schema version {document.get('schema_version')!r}")
    status = _string(document.get("status"), "run.status")
    if status not in {"incomplete", "complete"}:
        raise SavedRunError(f"Invalid run status {status!r}")
    return document


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a lowercase SHA-256 checksum for one regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_as_dict(plan: TrainingPlan | EvaluationPlan) -> dict[str, object]:
    """Return a JSON-ready dry-run description."""
    common: dict[str, object] = {
        "experiment_id": plan.resolved.specification.id,
        "training_seed": plan.seed,
        "configuration": resolved_configuration(plan.resolved),
        "run_directory": str(plan.run_directory),
    }
    if isinstance(plan, TrainingPlan):
        common.update(
            {
                "operation": "train",
                "intermediate_evaluation_seed": plan.intermediate_evaluation_seed,
                "final_evaluation_seed": plan.final_evaluation_seed,
                "requested_timesteps": plan.resolved.specification.run.training_steps,
                "model_path": str(plan.model_path),
            }
        )
    else:
        common.update(
            {
                "operation": "evaluate",
                "evaluation_name": plan.name,
                "evaluation_seed": plan.evaluation_seed,
                "episodes": plan.episodes,
                "model_path": str(plan.model_path),
                "output_directory": str(plan.output_directory),
            }
        )
    return common


def _execute_training(plan: TrainingPlan, *, overwrite: bool) -> RunResult:
    _prepare_run_directory(plan, overwrite=overwrite)
    resolved = plan.resolved
    run_settings = resolved.specification.run
    document = _initial_run_document(plan)
    _write_json_atomic(plan.run_path, document)
    train_env: gym.Env[Any, Any] | None = None
    model: Any = None
    actual_timesteps = 0
    try:
        pretraining = resolved
        if isinstance(resolved.specification.often, PacmanOFTENSettings):
            pretraining = replace(
                resolved,
                specification=replace(
                    resolved.specification,
                    run=replace(
                        run_settings,
                        max_episode_steps=resolved.specification.often.pretraining_max_episode_steps,
                    ),
                ),
            )
        train_env = pretraining.make_env(training=True)
        curve_writer = LearningCurveCSVWriter(plan.curve_path)
        with curve_writer:
            if resolved.specification.often is not None:
                from experiments.often import train_often

                model = train_often(plan, document, curve_writer, train_env)
            elif resolved.algorithm.execution_path == "tabular":
                model = _make_tabular_model(resolved, train_env, plan.seed)

                evaluation_seconds = 0.0

                def evaluate_tabular(timesteps: int) -> None:
                    nonlocal evaluation_seconds
                    started = perf_counter()
                    summary = _evaluate_model(
                        resolved,
                        model,
                        episodes=run_settings.intermediate_evaluation_episodes,
                        seed=plan.intermediate_evaluation_seed,
                    )
                    _record_intermediate(plan, document, curve_writer, (timesteps,), timesteps, summary)
                    evaluation_seconds += perf_counter() - started

                def tabular_callback(current_model: Any, step: Any) -> None:
                    if step.timesteps % run_settings.intermediate_evaluation_frequency == 0:
                        evaluate_tabular(step.timesteps)

                started = perf_counter()
                evaluate_tabular(0)
                model.learn(
                    run_settings.training_steps,
                    callback=tabular_callback,
                    **run_settings.learning_kwargs.to_dict(),
                )
                if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
                    document["training_seconds"] = perf_counter() - started - evaluation_seconds
            else:
                model = _make_sb3_model(resolved, train_env, plan.seed)
                from npc_gym.integrations.sb3 import SB3EvaluationCallback

                evaluation_seconds = 0.0

                def evaluator(current_model: Any) -> EvaluationSummary:
                    nonlocal evaluation_seconds
                    before = perf_counter()
                    try:
                        return _evaluate_model(
                            resolved,
                            current_model,
                            episodes=run_settings.intermediate_evaluation_episodes,
                            seed=plan.intermediate_evaluation_seed,
                        )
                    finally:
                        evaluation_seconds += perf_counter() - before

                def on_evaluation(timesteps: int, summary: EvaluationSummary) -> None:
                    nonlocal evaluation_seconds
                    before = perf_counter()
                    thresholds = sb3_callback.evaluations[-1].thresholds
                    _record_intermediate(plan, document, curve_writer, thresholds, timesteps, summary)
                    evaluation_seconds += perf_counter() - before

                sb3_callback = SB3EvaluationCallback(
                    collect_training_episodes=False,
                    evaluator=evaluator,
                    evaluation_frequency=run_settings.intermediate_evaluation_frequency,
                    on_evaluation=on_evaluation,
                )
                started = perf_counter()
                try:
                    model.learn(
                        total_timesteps=run_settings.training_steps,
                        callback=sb3_callback,
                        **run_settings.learning_kwargs.to_dict(),
                    )
                finally:
                    document["training_seconds"] = perf_counter() - started - evaluation_seconds
        actual_timesteps = (
            document["timesteps"]["actual"] if resolved.specification.often is not None else int(model.num_timesteps)
        )
        document["timesteps"]["actual"] = actual_timesteps
        _save_model_staged(cast(_Model, model), plan.model_path)
        checksum = sha256_file(plan.model_path)
        document["artifacts"]["model"]["sha256"] = checksum
    except BaseException as error:
        if resolved.specification.often is not None:
            actual_timesteps = document["timesteps"]["actual"]
        elif model is not None:
            actual_timesteps = int(getattr(model, "num_timesteps", actual_timesteps))
        _mark_incomplete(plan.run_path, document, actual_timesteps, error)
        raise
    finally:
        if train_env is not None:
            train_env.close()

    try:
        final_summary = _evaluate_saved_to_directory(
            resolved,
            plan.model_path,
            plan.run_directory / "evaluations" / "final",
            episodes=run_settings.final_evaluation_episodes,
            seed=plan.final_evaluation_seed,
            render=False,
            checksum=checksum,
            timesteps=actual_timesteps,
            overwrite=False,
        )
        document["status"] = "complete"
        document["completed_at"] = _now()
        document["timesteps"]["actual"] = actual_timesteps
        document["failure"] = None
        _write_json_atomic(plan.run_path, document)
        return RunResult(plan.run_directory, actual_timesteps, checksum, final_summary)
    except BaseException as error:
        _mark_incomplete(plan.run_path, document, actual_timesteps, error)
        raise


def _make_tabular_model(resolved: ResolvedExperiment, env: gym.Env[Any, Any], seed: int) -> TabularQLearning[Any]:
    if resolved.algorithm.implementation != _TABULAR_IMPLEMENTATION:
        raise PreflightError(f"Unsupported tabular implementation {resolved.algorithm.implementation!r}")
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    constructor = cast(Callable[..., TabularQLearning[Any]], TabularQLearning)
    return constructor(env, seed=seed, **kwargs)


def _make_sb3_model(resolved: ResolvedExperiment, env: gym.Env[Any, Any], seed: int) -> Any:
    constructor = _load_sb3_class(resolved.algorithm)
    assert resolved.algorithm.policy is not None
    return constructor(resolved.algorithm.policy, env, seed=seed, **resolved.algorithm.constructor_kwargs.to_dict())


def _load_model(resolved: ResolvedExperiment, path: Path, env: gym.Env[Any, Any]) -> Any:
    if resolved.algorithm.execution_path == "tabular":
        if resolved.algorithm.implementation != _TABULAR_IMPLEMENTATION:
            raise SavedRunError(f"Unsupported tabular implementation {resolved.algorithm.implementation!r}")
        return TabularQLearning.load(path, env=env)
    constructor = _load_sb3_class(resolved.algorithm)
    kwargs: dict[str, object] = {"env": env}
    device = resolved.algorithm.constructor_kwargs.to_dict().get("device")
    if device is not None:
        kwargs["device"] = device
    return constructor.load(path, **kwargs)


def _load_sb3_class(algorithm: AlgorithmSpecification) -> Any:
    if algorithm.implementation not in _SB3_IMPLEMENTATIONS:
        raise PreflightError(f"Unsupported SB3 implementation {algorithm.implementation!r}")
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as error:
        raise PreflightError("SB3 execution requires the optional 'npc-gym[sb3]' dependencies") from error
    return {"stable_baselines3.DQN": DQN, "stable_baselines3.PPO": PPO}[algorithm.implementation]


def _evaluate_model(
    resolved: ResolvedExperiment,
    model: Any,
    *,
    episodes: int,
    seed: int,
    render: bool = False,
) -> EvaluationSummary:
    env = resolved.make_env(training=False, render=render)
    try:
        return _evaluate_in(env, resolved, model, episodes=episodes, seed=seed)
    finally:
        env.close()


def _evaluate_checkpoint(
    resolved: ResolvedExperiment,
    model_path: Path,
    *,
    episodes: int,
    seed: int,
    render: bool,
) -> EvaluationSummary:
    env = resolved.make_env(training=False, render=render)
    try:
        return _evaluate_in(env, resolved, _load_model(resolved, model_path, env), episodes=episodes, seed=seed)
    finally:
        env.close()


def _evaluate_saved_to_directory(
    resolved: ResolvedExperiment,
    model_path: Path,
    target: Path,
    *,
    episodes: int,
    seed: int,
    render: bool,
    checksum: str,
    overwrite: bool,
    timesteps: int | None = None,
) -> EvaluationSummary:
    metadata = _evaluation_metadata(
        resolved,
        episodes=episodes,
        seed=seed,
        model_source="saved-checkpoint",
        model_checksum=checksum,
        timesteps=timesteps,
    )
    base_evaluation = None
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        from experiments.policy_fixes import evaluate_policy_fix

        env = resolved.make_env(training=False, render=render)
        try:
            model = _load_model(resolved, model_path, env)
            base, base_stats = evaluate_policy_fix(env, resolved, model, episodes=episodes, seed=seed, fixed=False)
            summary, stats = evaluate_policy_fix(env, resolved, model, episodes=episodes, seed=seed)
            base_metadata = {**metadata, **base_stats}
            base_metadata.pop("policy_fix")
            base_evaluation = (base, base_metadata)
            metadata.update(stats)
        finally:
            env.close()
    else:
        summary = _evaluate_checkpoint(resolved, model_path, episodes=episodes, seed=seed, render=render)
    base_fixed_evaluation = None
    if resolved.specification.often is not None:
        from experiments.often import evaluate_base_fixed

        base_path = model_path.parent / "base.zip"
        base_fixed_evaluation = evaluate_base_fixed(resolved, base_path, episodes=episodes, seed=seed, render=render)
        if target.name != "final":
            # Named reevaluations need the same monitor set on the paired base.
            base_summary = _evaluate_checkpoint(resolved, base_path, episodes=episodes, seed=seed, render=render)
            base_metadata = _evaluation_metadata(
                resolved,
                episodes=episodes,
                seed=seed,
                model_source="saved-base-checkpoint",
                model_checksum=sha256_file(base_path),
                timesteps=resolved.specification.often.pretraining_steps,
            )
            base_metadata.update(variant="base", timesteps_unit="base DQN environment steps")
            base_evaluation = (base_summary, base_metadata)
    _write_evaluation_directory(
        target,
        summary,
        metadata,
        overwrite=overwrite,
        base_evaluation=base_evaluation,
        base_fixed_evaluation=base_fixed_evaluation,
    )
    return summary


def _evaluate_in(
    env: gym.Env[Any, Any], resolved: ResolvedExperiment, model: Any, *, episodes: int, seed: int
) -> EvaluationSummary:
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        from experiments.policy_fixes import evaluate_policy_fix

        summary, _ = evaluate_policy_fix(env, resolved, model, episodes=episodes, seed=seed)
        return summary
    deterministic = resolved.specification.run.deterministic_evaluation
    policy: Any
    if resolved.algorithm.execution_path == "tabular":

        def policy(observation: Any, info: Mapping[str, Any]) -> Any:
            return model.predict(observation, info, deterministic=deterministic)

    else:
        from npc_gym.integrations.sb3 import SB3Policy

        policy = SB3Policy(model, deterministic=deterministic)
    monitors = dict(resolved.scenario.monitor_factories)
    if resolved.specification.often is not None:
        if seed + episodes - 1 > _MAX_SEED:
            raise SelectionError("Paired evaluation episode seeds exceed 2**32 - 1")
        results = []
        for index in range(episodes):
            summary = evaluate(env, policy, episodes=1, monitors=monitors, seed=seed + index)
            results.append(replace(summary.episodes[0], episode=index))
        return EvaluationSummary.from_episodes(results)
    return evaluate(env, policy, episodes=episodes, monitors=monitors, seed=seed)


def _record_intermediate(
    plan: TrainingPlan,
    document: dict[str, Any],
    curve_writer: LearningCurveCSVWriter,
    thresholds: tuple[int, ...],
    timesteps: int,
    summary: EvaluationSummary,
    *,
    base_fixed_evaluation: tuple[EvaluationSummary, Mapping[str, object]] | None = None,
) -> None:
    output = plan.run_directory / "evaluations" / f"intermediate-{timesteps}"
    metadata = _evaluation_metadata(
        plan.resolved,
        episodes=len(summary.episodes),
        seed=plan.intermediate_evaluation_seed,
        model_source="in-memory-policy",
        model_checksum=None,
        timesteps=timesteps,
        scheduled_thresholds=thresholds,
    )
    _write_evaluation_directory(output, summary, metadata, overwrite=False, base_fixed_evaluation=base_fixed_evaluation)
    curve_writer.write(timesteps, summary)
    document["timesteps"]["actual"] = timesteps
    _write_json_atomic(plan.run_path, document)


def _write_evaluation_directory(
    target: Path,
    summary: EvaluationSummary,
    metadata: Mapping[str, object],
    *,
    overwrite: bool,
    base_evaluation: tuple[EvaluationSummary, Mapping[str, object]] | None = None,
    base_fixed_evaluation: tuple[EvaluationSummary, Mapping[str, object]] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        with EpisodeCSVWriter(temporary / "episodes.csv") as writer:
            writer.write_all(summary.episodes)
        with EvaluationJSONWriter(temporary / "summary.json") as writer:
            writer.write(summary, metadata=metadata)
        if base_evaluation is not None:
            base_summary, base_metadata = base_evaluation
            _write_evaluation_directory(temporary / "base", base_summary, base_metadata, overwrite=False)
        if base_fixed_evaluation is not None:
            fixed_summary, fixed_metadata = base_fixed_evaluation
            _write_evaluation_directory(temporary / "base-fixed", fixed_summary, fixed_metadata, overwrite=False)
        if target.exists() or target.is_symlink():
            if not overwrite:
                raise FileExistsError(target)
            if target.is_symlink() or not target.is_dir():
                raise PreflightError(f"Refusing to replace unsafe evaluation output: {target}")
            shutil.rmtree(target)
        temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _save_model_staged(model: _Model, target: Path) -> None:
    stage = target.parent / f".{target.stem}.{uuid4().hex}.tmp.zip"
    possible = (stage, Path(f"{stage}.zip"))
    try:
        model.save(stage)
        actual = next((path for path in possible if path.is_file()), None)
        if actual is None:
            raise ExecutionError("Model save completed without producing the staged checkpoint")
        os.replace(actual, target)
    finally:
        for path in possible:
            if path.exists() or path.is_symlink():
                path.unlink()


def _initial_run_document(plan: TrainingPlan) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "incomplete",
        "experiment_id": plan.resolved.specification.id,
        "seed": plan.seed,
        "configuration": resolved_configuration(plan.resolved),
        "seeds": {
            "training": plan.seed,
            "intermediate_evaluation": plan.intermediate_evaluation_seed,
            "final_evaluation": plan.final_evaluation_seed,
        },
        "timesteps": {"requested": plan.resolved.specification.run.training_steps, "actual": 0},
        "artifacts": {
            "model": {"path": MODEL_FILENAME, "sha256": None},
            "learning_curve": CURVE_FILENAME,
            "evaluations": "evaluations",
        },
        "provenance": _provenance(plan.resolved),
        "started_at": _now(),
        "completed_at": None,
        "failure": None,
    }


def _evaluation_metadata(
    resolved: ResolvedExperiment,
    *,
    episodes: int,
    seed: int,
    model_source: str,
    model_checksum: str | None,
    timesteps: int | None = None,
    scheduled_thresholds: tuple[int, ...] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "experiment_id": resolved.specification.id,
        "episode_count": episodes,
        "seed": seed,
        "task_return_units": resolved.specification.run.task_return_units,
        "model_source": model_source,
        "model_sha256": model_checksum,
        "scenario_id": resolved.scenario.id,
        "monitor_ids": [identifier for identifier, _factory in resolved.scenario.monitor_factories],
        "episode_limit": (
            resolved.specification.often.evaluation_max_episode_steps
            if isinstance(resolved.specification.often, PacmanOFTENSettings)
            else resolved.specification.run.max_episode_steps
        ),
    }
    if resolved.specification.often is not None:
        metadata["episode_seed_rule"] = "seed + episode index"
        metadata["timesteps_unit"] = "combined OFTEN transitions after pretraining"
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        metadata["episode_seed_rule"] = "seed + episode index"
        taxi = resolved.environment.id == TAXI_ENVIRONMENT_ID
        metadata["policy_fix"] = {
            "norm_id": EMERGENCY_NORM_ID if taxi else ENV_FRIENDLY_NORM_ID,
            **({"norm_component": "Warn Violations"} if taxi else {}),
            "horizon": 1,
            "reuse_solver": True,
            "use_action_mask": not taxi,
            "objectives": {"violations": {"priority": 2, "weight": 1}, "policy": {"priority": 1, "weight": 1}},
        }
    if timesteps is not None:
        metadata["timesteps"] = timesteps
    if scheduled_thresholds is not None:
        metadata["scheduled_thresholds"] = list(scheduled_thresholds)
    return metadata


def _provenance(resolved: ResolvedExperiment) -> dict[str, object]:
    packages: dict[str, str | None] = {}
    names = ["npc-gym", "gymnasium", "numpy"]
    if resolved.specification.often is not None or resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        names.append("clingo")
    if resolved.algorithm.execution_path == "sb3":
        names.extend(("stable-baselines3", "torch"))
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "git": _git_provenance(),
        "device": resolved.algorithm.constructor_kwargs.to_dict().get("device", "cpu"),
    }


def _git_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "revision": None, "dirty": None}
    return {"available": True, "revision": revision, "dirty": dirty}


def _preflight_batch_identities(plans: Sequence[TrainingPlan]) -> None:
    configurations: dict[tuple[Path, str], dict[str, object]] = {}
    for plan in plans:
        key = (plan.output_root, plan.resolved.specification.id)
        configuration = resolved_configuration(plan.resolved)
        previous = configurations.setdefault(key, configuration)
        if not _json_equal(previous, configuration):
            raise ConfigurationMismatchError(
                f"Experiment {plan.resolved.specification.id!r} has different resolved settings within the "
                "training batch"
            )


def _preflight_experiment_identity(plan: TrainingPlan) -> None:
    configuration = resolved_configuration(plan.resolved)
    for run_directory in _discover_seed_directories(plan.run_directory.parent):
        run_path = run_directory / RUN_FILENAME
        if run_directory == plan.run_directory or not (run_path.exists() or run_path.is_symlink()):
            continue
        if not _json_equal(read_run_document(run_path).get("configuration"), configuration):
            raise ConfigurationMismatchError(
                f"Experiment {plan.resolved.specification.id!r} already has runs with different resolved settings: "
                f"{run_directory}"
            )


def _preflight_training_target(plan: TrainingPlan, *, overwrite: bool) -> None:
    target = plan.run_directory
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or not target.is_dir():
        raise PreflightError(f"Run output path is not a safe directory: {target}")
    if not overwrite:
        raise PreflightError(f"Run output already exists; pass --overwrite: {target}")
    existing = read_run_document(plan.run_path)
    if existing.get("experiment_id") != plan.resolved.specification.id or existing.get("seed") != plan.seed:
        raise ConfigurationMismatchError(f"Existing run ownership does not match selected run: {target}")
    if not _json_equal(existing.get("configuration"), resolved_configuration(plan.resolved)):
        raise ConfigurationMismatchError(
            f"Existing experiment identity has different resolved settings and cannot be overwritten: {target}"
        )
    _preflight_owned_artifacts(plan)


def _preflight_owned_artifacts(plan: TrainingPlan) -> None:
    models = (
        (plan.run_directory / "base.zip", plan.run_directory / "base_learning_curve.csv")
        if plan.resolved.specification.often is not None
        else ()
    )
    for path in (plan.model_path, plan.curve_path, *models):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise PreflightError(f"Refusing to replace unsafe run artifact: {path}")
    evaluations = plan.run_directory / "evaluations"
    if not evaluations.exists() and not evaluations.is_symlink():
        return
    if evaluations.is_symlink() or not evaluations.is_dir():
        raise PreflightError(f"Refusing to use unsafe evaluations directory: {evaluations}")
    for path in evaluations.iterdir():
        if (path.name in {"final", "base"} or _OWNED_INTERMEDIATE.fullmatch(path.name)) and (
            path.is_symlink() or not path.is_dir()
        ):
            raise PreflightError(f"Refusing to replace unsafe evaluation artifact: {path}")


def _prepare_run_directory(plan: TrainingPlan, *, overwrite: bool) -> None:
    target = plan.run_directory
    target.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for path in (
        plan.run_path,
        plan.curve_path,
        *((target / "base_learning_curve.csv",) if plan.resolved.specification.often is not None else ()),
    ):
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise PreflightError(f"Refusing to replace unsafe run artifact: {path}")
            path.unlink()
    evaluations = target / "evaluations"
    if evaluations.exists() or evaluations.is_symlink():
        if evaluations.is_symlink() or not evaluations.is_dir():
            raise PreflightError(f"Refusing to use unsafe evaluations directory: {evaluations}")
        for path in evaluations.iterdir():
            if path.name in {"final", "base"} or _OWNED_INTERMEDIATE.fullmatch(path.name):
                if path.is_symlink() or not path.is_dir():
                    raise PreflightError(f"Refusing to replace unsafe evaluation artifact: {path}")
                shutil.rmtree(path)


def _mark_incomplete(path: Path, document: dict[str, Any], timesteps: int, error: BaseException) -> None:
    document["status"] = "incomplete"
    document["completed_at"] = None
    document["timesteps"]["actual"] = timesteps
    document["failure"] = {"type": type(error).__name__, "message": str(error)}
    try:
        _write_json_atomic(path, document)
    except OSError as write_error:
        error.add_note(f"Additionally failed to update incomplete run metadata: {write_error}")


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _check_dependency(resolved: ResolvedExperiment) -> None:
    if resolved.environment.id == PACMAN_BOLTS_ENVIRONMENT_ID:
        from experiments.pacman_bolts import layout_path

        layout_path(str(resolved.environment.constructor_kwargs.to_dict()["layout"]))
    if resolved.specification.often is not None and importlib.util.find_spec("clingo") is None:
        raise PreflightError("OFTEN experiments include policy-fix evaluation and require 'npc-gym[asp,sb3]'")
    if resolved.environment.id == PACMAN_IMAGES_ENVIRONMENT_ID and any(
        importlib.util.find_spec(module) is None for module in ("pygame", "PIL")
    ):
        raise PreflightError("Pacman image experiments require the optional 'npc-gym[sb3,render]' dependencies")
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID and importlib.util.find_spec("clingo") is None:
        raise PreflightError("Policy-fix experiments require the optional 'npc-gym[asp]' dependency")
    if resolved.algorithm.execution_path == "sb3":
        _load_sb3_class(resolved.algorithm)


def _evaluation_seeds(resolved: ResolvedExperiment, seed: int) -> tuple[int, int]:
    often = resolved.specification.often
    if often is not None:
        episodes = max(
            resolved.specification.run.final_evaluation_episodes,
            resolved.specification.run.intermediate_evaluation_episodes,
        )
        if seed + often.teaching_seed_offset > _MAX_SEED or often.evaluation_seed + episodes - 1 > _MAX_SEED:
            raise SelectionError("OFTEN training or evaluation seeds exceed 2**32 - 1")
        if isinstance(often, PacmanOFTENSettings) and seed + often.pretraining_evaluation_seed_offset > _MAX_SEED:
            raise SelectionError("OFTEN base evaluation seed exceeds 2**32 - 1")
        return often.evaluation_seed, often.evaluation_seed
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        run = resolved.specification.run
        first = run.evaluation_seed_offset
        if first + max(run.final_evaluation_episodes, run.intermediate_evaluation_episodes) - 1 > _MAX_SEED:
            raise SelectionError("Paired evaluation episode seeds exceed 2**32 - 1")
        return first, first
    first = seed + resolved.specification.run.evaluation_seed_offset
    if first + 1 > _MAX_SEED:
        raise SelectionError(f"Training seed {seed} derives evaluation seeds above 2**32 - 1")
    return first, first + 1


def _validate_execution_settings(resolved: ResolvedExperiment) -> None:
    often = resolved.specification.often
    if (often is not None) != (resolved.technique.id == OFTEN_TECHNIQUE_ID):
        raise PreflightError("OFTEN requires teaching settings and the OFTEN technique")
    if often is not None:
        recipes = {
            PACMAN_ENVIRONMENT_ID: {PACMAN_VEGAN_SCENARIO_ID, PACMAN_VEGETARIAN_SCENARIO_ID, TRAPPED_NORM_ID},
            "pacman/small-classic-v0": {PACMAN_VEGAN_SCENARIO_ID, PACMAN_VEGETARIAN_SCENARIO_ID, TRAPPED_NORM_ID},
            GARDENER_ENVIRONMENT_ID: {
                NO_COLLECT_NORM_ID,
                DRAIN_NORM_ID,
                PERMISSION_AWARE_NORM_ID,
                GARDENER_PERMISSION_DRAIN_SCENARIO_ID,
            },
        }
        if (
            often.norm_id not in recipes.get(resolved.environment.id, set())
            or (resolved.environment.id == GARDENER_ENVIRONMENT_ID and resolved.scenario.id != GARDENER_SCENARIO_ID)
            or (resolved.environment.family == "pacman" and resolved.scenario.id != often.norm_id)
            or resolved.algorithm.implementation != "stable_baselines3.DQN"
            or isinstance(often, GardenerOFTENSettings) != (resolved.environment.id == GARDENER_ENVIRONMENT_ID)
        ):
            raise PreflightError(
                "OFTEN requires a supported Pacman/Gardener scenario, matching teaching settings and DQN"
            )
        if resolved.specification.run.learning_kwargs.to_dict():
            raise PreflightError("OFTEN uses its own collection schedule, without learner keyword overrides")
        if often.norm_id == TRAPPED_NORM_ID and often.horizon != 1:
            raise PreflightError("Trapped policy fixes require horizon=1")
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        supported = {
            MERCHANT_ENVIRONMENT_ID: (MERCHANT_ENV_FRIENDLY_SCENARIO_ID, True),
            TAXI_ENVIRONMENT_ID: (TAXI_SCENARIO_ID, False),
        }
        if (
            (resolved.scenario.id, resolved.algorithm.constructor_kwargs.to_dict().get("use_action_mask"))
            != supported.get(resolved.environment.id)
            or resolved.algorithm.implementation != _TABULAR_IMPLEMENTATION
            or not resolved.specification.run.deterministic_evaluation
        ):
            raise PreflightError(
                "Policy fixes require deterministic tabular evaluation of Merchant Environment Friendly "
                "with action masks or Taxi Warn without action masks"
            )
    constructor = resolved.algorithm.constructor_kwargs.to_dict()
    reserved_constructor = {"env", "seed", "policy"} & constructor.keys()
    if reserved_constructor:
        raise PreflightError(f"Algorithm constructor settings reserve: {', '.join(sorted(reserved_constructor))}")
    learning = resolved.specification.run.learning_kwargs.to_dict()
    reserved_learning = {"total_timesteps", "callback"} & learning.keys()
    if reserved_learning:
        raise PreflightError(f"Learning settings reserve: {', '.join(sorted(reserved_learning))}")
    algorithm = resolved.algorithm
    if algorithm.execution_path == "tabular":
        if algorithm.implementation != _TABULAR_IMPLEMENTATION or algorithm.policy is not None:
            raise PreflightError(f"Unsupported tabular algorithm declaration {algorithm.id!r}")
    elif algorithm.execution_path == "sb3":
        if algorithm.implementation not in _SB3_IMPLEMENTATIONS or not algorithm.policy:
            raise PreflightError(f"Unsupported SB3 algorithm declaration {algorithm.id!r}")
    else:
        raise PreflightError(f"Unsupported execution path {algorithm.execution_path!r}")


def _require_safe_run_path(plan: TrainingPlan) -> None:
    expected = plan.output_root / plan.resolved.specification.id / f"seed-{plan.seed}"
    if plan.run_directory != expected:
        raise PreflightError(f"Run path is not derived from its safe experiment identity: {plan.run_directory}")
    _require_within(plan.run_directory, plan.output_root, "run output")
    _check_no_symlink_components(plan.run_directory, stop=plan.output_root)


def _require_within(path: Path, parent: Path, description: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise PreflightError(f"{description.capitalize()} escapes its selected root: {path}") from error


def _check_no_symlink_components(path: Path, *, stop: Path) -> None:
    current = path
    while current != stop:
        if current.is_symlink():
            raise PreflightError(f"Output path contains a symbolic link: {current}")
        if current.parent == current:
            raise PreflightError(f"Output path is not below selected root: {path}")
        current = current.parent


def _check_output_parent(root: Path) -> None:
    current = root
    while not current.exists():
        if current.parent == current:
            raise PreflightError(f"No existing parent for output path: {root}")
        current = current.parent
    if not current.is_dir():
        raise PreflightError(f"Output parent is not a directory: {current}")
    if not os.access(current, os.W_OK | os.X_OK):
        raise PreflightError(f"Output parent is not writable: {current}")


def _discover_seed_directories(experiment_directory: Path) -> list[Path]:
    if not experiment_directory.is_dir() or experiment_directory.is_symlink():
        return []
    result = []
    for path in experiment_directory.iterdir():
        if path.is_dir() and not path.is_symlink() and _SEED_DIRECTORY.fullmatch(path.name):
            result.append(path)
    return sorted(result, key=lambda path: int(path.name.removeprefix("seed-")))


def _model_checksum_from_run(document: Mapping[str, Any]) -> str:
    artifacts = _mapping(document.get("artifacts"), "run.artifacts")
    model = _mapping(artifacts.get("model"), "run.artifacts.model")
    if model.get("path") != MODEL_FILENAME:
        raise SavedRunError("Saved model path must be the run-owned model.zip")
    checksum = _string(model.get("sha256"), "run.artifacts.model.sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise SavedRunError("Saved model checksum must be a lowercase SHA-256 digest")
    return checksum


def _validate_complete_run(
    document: Mapping[str, Any], resolved: ResolvedExperiment, seed: int, run_directory: Path
) -> None:
    experiment_id = resolved.specification.id
    if document.get("experiment_id") != experiment_id:
        raise SavedRunError(f"Saved experiment ID does not match run directory: {run_directory}")
    seeds = _mapping(document.get("seeds"), "run.seeds")
    intermediate, final = _evaluation_seeds(resolved, seed)
    expected_seeds = {"training": seed, "intermediate_evaluation": intermediate, "final_evaluation": final}
    if not _json_equal(seeds, expected_seeds):
        raise SavedRunError(f"Saved seed streams do not match resolved settings: {run_directory}")
    timesteps = _mapping(document.get("timesteps"), "run.timesteps")
    requested = _positive_saved_integer(timesteps.get("requested"), "run.timesteps.requested")
    actual = _nonnegative_integer(timesteps.get("actual"), "run.timesteps.actual")
    if requested != resolved.specification.run.training_steps or actual < requested:
        raise SavedRunError(f"Saved completion timesteps do not match resolved settings: {run_directory}")
    artifacts = _mapping(document.get("artifacts"), "run.artifacts")
    if artifacts.get("learning_curve") != CURVE_FILENAME or artifacts.get("evaluations") != "evaluations":
        raise SavedRunError(f"Saved artifact paths are not run-owned: {run_directory}")
    if resolved.specification.often is not None:
        base = _mapping(artifacts.get("base"), "run.artifacts.base")
        path = run_directory / "base.zip"
        if base.get("path") != "base.zip" or path.is_symlink() or not path.is_file():
            raise SavedRunError("Complete OFTEN run is missing its safe base checkpoint")
        if sha256_file(path) != base.get("sha256"):
            raise SavedRunError("Base checkpoint checksum does not match run metadata")
    required_files: tuple[Path, ...] = (
        run_directory / CURVE_FILENAME,
        run_directory / "evaluations" / "final" / "episodes.csv",
        run_directory / "evaluations" / "final" / "summary.json",
    )
    if resolved.technique.id == POLICY_FIX_TECHNIQUE_ID:
        required_files += (
            run_directory / "evaluations" / "final" / "base" / "episodes.csv",
            run_directory / "evaluations" / "final" / "base" / "summary.json",
        )
    for path in required_files:
        if not path.is_file() or path.is_symlink():
            raise SavedRunError(f"Complete run is missing a safe required artifact: {path}")


def _result_name(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_RESULT_NAME.fullmatch(value) or value in {".", ".."}:
        raise SelectionError(f"Evaluation name must be a safe lowercase path component; got {value!r}")
    if value in {"final", "base"} or _OWNED_INTERMEDIATE.fullmatch(value):
        raise SelectionError(
            "Named evaluations cannot use the training-owned base, final or intermediate-<timesteps> names"
        )
    return value


def _unique(name: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise SelectionError(f"{name.capitalize()} filters must be non-empty strings")
    if len(set(result)) != len(result):
        raise SelectionError(f"Duplicate {name} filter")
    return result


def _seeds(values: Sequence[int], *, required: bool) -> tuple[int, ...]:
    result = tuple(_seed(value, "training seed") for value in values)
    if required and not result:
        raise SelectionError("At least one explicit training seed is required")
    if len(set(result)) != len(result):
        raise SelectionError("Duplicate training seed")
    return tuple(sorted(result))


def _seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SEED:
        raise SelectionError(f"{name} must be an integer between 0 and 2**32 - 1; got {value!r}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SelectionError(f"{name} must be a positive integer; got {value!r}")
    return value


def _positive_saved_integer(value: object, name: str) -> int:
    checked = _integer(value, name)
    if checked <= 0:
        raise SavedRunError(f"{name} must be positive; got {value!r}")
    return checked


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SavedRunError(f"{name} must be an integer; got {value!r}")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    checked = _integer(value, name)
    if checked < 0:
        raise SavedRunError(f"{name} must be non-negative; got {value!r}")
    return checked


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise SavedRunError(f"{name} must be a boolean; got {value!r}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SavedRunError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SavedRunError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SavedRunError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise SavedRunError(f"{name} must contain exactly {', '.join(sorted(expected))}")


def _json_equal(left: object, right: object) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SavedRunError(f"Value is not finite JSON data: {error}") from error


def _distinct_paths(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        raise SelectionError("Selected runs contain colliding output paths")


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CONFIGURATION_SCHEMA_VERSION",
    "CURVE_FILENAME",
    "DEFAULT_OUTPUT_DIRECTORY",
    "MODEL_FILENAME",
    "RUN_FILENAME",
    "RUN_SCHEMA_VERSION",
    "ConfigurationMismatchError",
    "EvaluationPlan",
    "EvaluationResult",
    "ExecutionError",
    "PreflightError",
    "RunResult",
    "SavedRunError",
    "SelectionError",
    "TrainingPlan",
    "execute_evaluation_batch",
    "execute_training_batch",
    "plan_as_dict",
    "plan_evaluation_batch",
    "plan_training_batch",
    "preflight_evaluation",
    "preflight_training",
    "read_run_document",
    "resolved_configuration",
    "resolved_from_configuration",
    "select_experiments",
    "sha256_file",
]
