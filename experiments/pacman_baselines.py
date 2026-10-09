"""Re-evaluate the shared smallClassic baseline checkpoints with six paper norm bases."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from experiments.pacman_bolts import LAYOUT, LAYOUT_SHA256, RECIPES, layout_path, make_environment
from experiments.parallel import validate_cpus
from npc_gym.evaluation import EpisodeCSVWriter, EvaluationJSONWriter, EvaluationSummary, evaluate
from npc_gym.monitors import make_builtin_monitor

MONITORS = tuple(RECIPES.values())
BASELINES = {"dqn": "pacman-dqn-unconstrained-v0", "ppo": "pacman-ppo-unconstrained-v1"}


@dataclass(frozen=True)
class BaselineEvaluation:
    """A hash-verified source checkpoint and a separate evaluation destination."""

    algorithm: str
    seed: int
    source: Path
    target: Path
    checksum: str
    evaluation_seed: int
    episodes: int = 1000


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plan_evaluations(source: Path, output: Path) -> tuple[BaselineEvaluation, ...]:
    """Validate all sixteen original baselines without modifying source or output.

    Requires complete 5M-step, gamma .99, 63-feature smallClassic checkpoints,
    their original 1000-episode final evaluations and matching model hashes.
    Output must be a fresh directory outside the source tree.
    """
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Baseline output must be separate from the source tree")
    if output.exists():
        raise FileExistsError(f"Baseline evaluation output already exists: {output}")
    layout_path()
    plans = []
    for algorithm, identifier in BASELINES.items():
        for seed in range(8):
            folder = source / identifier / f"seed-{seed}"
            doc = _read(folder / "run.json")
            config = doc["configuration"]
            run = config["run"]
            env = config["environment"]["constructor_kwargs"]
            expected_learner = f"stable_baselines3.{algorithm.upper()}"
            if (
                doc["status"] != "complete"
                or doc["seed"] != seed
                or doc["experiment_id"] != identifier
                or config["algorithm"]["implementation"] != expected_learner
                or config["algorithm"]["constructor_kwargs"]["gamma"] != 0.99
                or config["technique"]["id"] != "technique/unconstrained-v0"
                or env.get("features") != "complete"
                or env.get("layout") != "smallClassic"
                or run["training_steps"] != 5_000_000
                or doc["timesteps"]["actual"] < 5_000_000
                or run["max_episode_steps"] != 300
                or run["final_evaluation_episodes"] != 1000
                or not run["deterministic_evaluation"]
                or [w["id"] for w in config["wrappers"]] != ["pacman/training-reward-divide-100-v0"]
            ):
                raise ValueError(f"Not a complete matching paper baseline: {folder}")
            checksum = _hash(folder / "model.zip")
            if checksum != doc["artifacts"]["model"]["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {folder}")
            previous = _read(folder / "evaluations/final/summary.json")
            evaluation_seed = doc["seeds"]["final_evaluation"]
            if (
                previous["metadata"]["seed"] != evaluation_seed
                or len(previous["episodes"]) != 1000
                or previous["metadata"]["model_sha256"] != checksum
            ):
                raise ValueError(f"Original evaluation does not match checkpoint/seeds: {folder}")
            plans.append(
                BaselineEvaluation(
                    algorithm, seed, folder, output / identifier / f"seed-{seed}", checksum, evaluation_seed
                )
            )
    return tuple(plans)


def compare_episodes(summary: EvaluationSummary, previous: dict[str, Any]) -> dict[str, Any]:
    """Compare task measurements and previously monitored norms episode by episode."""
    if len(summary.episodes) != len(previous["episodes"]):
        raise ValueError("Original and repeated evaluation episode counts differ")
    task_differences = []
    norm_differences: dict[str, int] = {}
    for current, old in zip(summary.episodes, previous["episodes"], strict=True):
        if (
            current.episode_return != old["return"]
            or current.length != old["length"]
            or current.termination.value != old["termination"]
            or dict(current.metrics) != old["metrics"]
        ):
            task_differences.append(current.episode)
        for norm in MONITORS:
            if norm in old["monitor_counts"] and dict(current.monitor_counts[norm]) != old["monitor_counts"][norm]:
                norm_differences[norm] = norm_differences.get(norm, 0) + 1
    return {"task_mismatched_episodes": task_differences, "norm_mismatched_episode_counts": norm_differences}


def _load_policy(plan: BaselineEvaluation, env: Any) -> Any:
    import torch
    from stable_baselines3 import DQN, PPO

    from npc_gym.integrations.sb3 import SB3Policy

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    model = (DQN if plan.algorithm == "dqn" else PPO).load(plan.source / "model.zip", env=env, device="cpu")
    return SB3Policy(model, deterministic=True)


def evaluate_checkpoint(plan: BaselineEvaluation) -> dict[str, Any]:
    """Evaluate one existing checkpoint; retain an explicit incomplete status on failure."""
    if _hash(plan.source / "model.zip") != plan.checksum:
        raise ValueError(f"Checkpoint changed after planning: {plan.source}")
    plan.target.mkdir(parents=True, exist_ok=False)
    document = {
        **asdict(plan),
        "source": str(plan.source),
        "target": str(plan.target),
        "schema_version": 1,
        "status": "incomplete",
        "started_at": datetime.now(UTC).isoformat(),
        "source_run": _read(plan.source / "run.json"),
        "monitor_ids": MONITORS,
        "layout_sha256": LAYOUT_SHA256,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "features": "complete",
        "ghost_behavior": "random",
        "max_episode_steps": 300,
        "task_return_units": "unscaled task reward",
        "deterministic_evaluation": True,
    }
    _write(plan.target / "evaluation.json", document)
    env: gym.Env[Any, Any] = TimeLimit(
        make_environment(None, layout=LAYOUT, features="complete", ghost_behavior="random"), max_episode_steps=300
    )
    try:
        policy = _load_policy(plan, env)
        summary = evaluate(
            env,
            policy,
            monitors={key: partial(make_builtin_monitor, key) for key in MONITORS},
            episodes=plan.episodes,
            seed=plan.evaluation_seed,
        )
        comparison = compare_episodes(summary, _read(plan.source / "evaluations/final/summary.json"))
        if comparison["task_mismatched_episodes"]:
            raise ValueError(f"Task results differ from the original evaluation: {plan.source}")
        metadata = {
            "experiment_id": BASELINES[plan.algorithm],
            "episode_count": plan.episodes,
            "seed": plan.evaluation_seed,
            "task_return_units": "unscaled task reward",
            "model_sha256": plan.checksum,
            "model_source": "saved-checkpoint",
            "monitor_ids": MONITORS,
            "comparison_with_original": comparison,
        }
        with EpisodeCSVWriter(plan.target / "episodes.csv") as writer:
            writer.write_all(summary.episodes)
        with EvaluationJSONWriter(plan.target / "summary.json") as writer:
            writer.write(summary, metadata=metadata)
        document.update(
            status="complete", completed_at=datetime.now(UTC).isoformat(), comparison_with_original=comparison
        )
        _write(plan.target / "evaluation.json", document)
        return {"algorithm": plan.algorithm, "seed": plan.seed, "return": summary.mean_return, **comparison}
    finally:
        env.close()


def _worker(plans: tuple[BaselineEvaluation, ...], cpu: int) -> list[dict[str, Any]]:
    os.sched_setaffinity(0, [cpu])
    os.environ.update(
        dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return [evaluate_checkpoint(plan) for plan in plans]


def execute_evaluations(source: Path, output: Path, cpus: list[int]) -> None:
    """Run CPU-pinned evaluation workers, with no training or source modifications."""
    validate_cpus(cpus)
    plans = plan_evaluations(source, output)
    output.mkdir(parents=True, exist_ok=False)
    document: dict[str, Any] = {
        "schema_version": 1,
        "status": "incomplete",
        "source": str(source.resolve()),
        "cpus": cpus,
        "monitor_ids": MONITORS,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evaluator_sha256": _hash(Path(__file__)),
        "python": platform.python_version(),
        "packages": {
            name: metadata.version(name) for name in ("npc-gym", "gymnasium", "numpy", "stable-baselines3", "torch")
        },
    }
    _write(output / "batch.json", document)
    with ProcessPoolExecutor(
        max_workers=min(len(cpus), len(plans)), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_worker, plans[index :: len(cpus)], cpu) for index, cpu in enumerate(cpus) if index < len(plans)
        ]
        document["results"] = [row for future in futures for row in future.result()]
    document.update(status="complete", completed_at=datetime.now(UTC).isoformat())
    _write(output / "batch.json", document)
