"""Opt-in, CPU-pinned execution of the full standalone learning examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import runpy
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from npc_gym.evaluation import EvaluationSummary

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("pacman", "gardener", "taxi", "merchant")
SMOKE = {
    "NPC_GYM_TRAINING_STEPS": "16",
    "NPC_GYM_EVALUATION_EPISODES": "2",
    "NPC_GYM_MAX_EPISODE_STEPS": "40",
}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def physical_cpus(cpuinfo: str, allowed: set[int]) -> list[int]:
    """Select one allowed logical CPU per physical socket/core from Linux cpuinfo."""
    representatives: dict[tuple[int, int], int] = {}
    for block in cpuinfo.strip().split("\n\n"):
        fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" not in fields or int(fields["processor"]) not in allowed:
            continue
        if "physical id" not in fields or "core id" not in fields:
            raise ValueError("CPU topology is unavailable; specify distinct physical cores with --cpus")
        key = (int(fields["physical id"]), int(fields["core id"]))
        cpu = int(fields["processor"])
        representatives[key] = min(cpu, representatives.get(key, cpu))
    if not representatives:
        raise ValueError("CPU topology is unavailable; specify distinct physical cores with --cpus")
    return sorted(representatives.values())


def cpu_slots(cpus: Sequence[int], jobs: int, cores_per_job: int) -> list[list[int]]:
    """Allocate disjoint affinity sets, rejecting oversubscription."""
    if jobs < 1 or cores_per_job < 1:
        raise ValueError("jobs and cores-per-job must be positive")
    if len(set(cpus)) != len(cpus) or any(cpu < 0 for cpu in cpus):
        raise ValueError("cpus must contain distinct nonnegative CPU IDs")
    if jobs * cores_per_job > len(cpus):
        raise ValueError("jobs * cores-per-job exceeds available selected CPUs")
    return [list(cpus[index * cores_per_job : (index + 1) * cores_per_job]) for index in range(jobs)]


def worker_environment(seed: int, cpus: Sequence[int], *, smoke: bool) -> dict[str, str]:
    """Discard inherited example overrides; limit native threads before imports."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("NPC_GYM_")}
    env.update({key: str(len(cpus)) for key in THREAD_VARIABLES})
    env.update({"PYTHONUNBUFFERED": "1", "NPC_GYM_SEED": str(seed), "OMP_DYNAMIC": "FALSE"})
    if smoke:
        env.update(SMOKE)
    return env


def _write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def _worker(example: str, directory: Path, cpus: list[int], smoke: bool) -> int:
    # Set affinity before importing NumPy/Torch so helper threads inherit it.
    os.sched_setaffinity(0, cpus)
    signal.signal(signal.SIGTERM, _interrupt)
    import torch

    from npc_gym.evaluation import EvaluationJSONWriter

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    script = ROOT / "examples" / f"{example}.py"
    script_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    with TemporaryDirectory(prefix="npc-gym-validation-") as temporary_models:
        os.environ["NPC_GYM_MODEL_PATH"] = str(Path(temporary_models) / f"{example}.zip")
        namespace = runpy.run_path(str(script), run_name="learning_validation_example")
        main = cast("Callable[[], tuple[EvaluationSummary, int]]", namespace["main"])
        started = time.perf_counter()
        summary, actual_steps = main()
    metadata = {
        "example": example,
        "mode": "smoke" if smoke else "full",
        "assessment": "smoke_only" if smoke else "pending_author_review",
        "configuration": {
            key: value for key, value in namespace.items() if key.isupper() and isinstance(value, (int, str))
        },
        "evaluation_seed": namespace["SEED"] + 10_000,
        "actual_training_steps": actual_steps,
        "wall_seconds": time.perf_counter() - started,
        "cpus": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "example_sha256": script_digest,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("npc-gym", "numpy", "gymnasium", "torch", "stable-baselines3")},
    }
    temporary = directory / "evaluation.tmp"
    try:
        with EvaluationJSONWriter(temporary) as writer:
            writer.write(summary, metadata=metadata)
        temporary.replace(directory / "evaluation.json")
    finally:
        temporary.unlink(missing_ok=True)
    return 0


def run_batch(
    output: Path, examples: Sequence[str], seeds: Sequence[int], slots: list[list[int]], *, smoke: bool
) -> int:
    """Run isolated processes, retain logs/results, and stop children on interruption.

    The output directory must not exist. Exit zero means all executions completed;
    full-run learning quality remains subject to author review.
    """
    output.mkdir(parents=True, exist_ok=False)
    pending = deque((example, seed) for example in examples for seed in seeds)
    available = deque(slots)
    active: list[tuple[subprocess.Popen[bytes], list[int], dict[str, Any]]] = []
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    sources = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "examples").glob("*.py")) + [Path(__file__)]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    document: dict[str, Any] = {
        "schema_version": 1,
        "mode": "smoke" if smoke else "full",
        "status": "running",
        "assessment": "smoke_only" if smoke else "pending_author_review",
        "git_revision": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": digest.hexdigest(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "cpu_slots": slots,
        "examples": list(examples),
        "seeds": list(seeds),
        "runs": [],
    }
    manifest = output / "manifest.json"
    started = time.perf_counter()
    try:
        while pending or active:
            while pending and available:
                example, seed = pending.popleft()
                cpus = available.popleft()
                directory = output / f"{example}-seed-{seed}"
                directory.mkdir()
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_worker",
                    "--example",
                    example,
                    "--output",
                    str(directory),
                    "--cpus",
                    ",".join(map(str, cpus)),
                ]
                if smoke:
                    command.append("--smoke")
                with (directory / "run.log").open("xb") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=directory,
                        env=worker_environment(seed, cpus, smoke=smoke),
                    )
                record = {"example": example, "seed": seed, "cpus": cpus, "status": "running", "pid": process.pid}
                document["runs"].append(record)
                active.append((process, cpus, record))
                print(f"started {example} seed={seed} cpus={cpus}", flush=True)
                _write_json(manifest, document)
            for process, cpus, record in active[:]:
                code = process.poll()
                if code is None:
                    continue
                record.update(status="complete" if code == 0 else "failed", exit_code=code)
                active.remove((process, cpus, record))
                available.append(cpus)
                print(f"{record['status']} {record['example']} seed={record['seed']} exit={code}", flush=True)
                _write_json(manifest, document)
            if active:
                time.sleep(0.2)
        document["status"] = "complete" if all(run["exit_code"] == 0 for run in document["runs"]) else "failed"
        return 0 if document["status"] == "complete" else 1
    finally:
        for process, _, _ in active:
            if process.poll() is None:
                process.terminate()
        for process, _, record in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            record.update(status="interrupted", exit_code=process.returncode)
        if document["status"] == "running":
            document["status"] = "interrupted"
        document["wall_seconds"] = time.perf_counter() - started
        _write_json(manifest, document)


def main(argv: Sequence[str] | None = None) -> int:
    """Require an explicit full/smoke mode and a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "smoke", "_worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example", choices=EXAMPLES, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--jobs", type=int, help="concurrent processes; defaults to available physical cores or run count"
    )
    parser.add_argument("--cores-per-job", type=int, default=1)
    parser.add_argument("--cpus", help="comma-separated logical CPU IDs; select one hardware thread per physical core")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("CPU-pinned validation requires Linux sched_setaffinity")
        allowed = set(os.sched_getaffinity(0))
        cpus = (
            [int(cpu) for cpu in args.cpus.split(",")]
            if args.cpus
            else physical_cpus(Path("/proc/cpuinfo").read_text(), allowed)
        )
        if not cpus or not set(cpus) <= allowed:
            raise ValueError("--cpus must select CPUs within this process's allowed affinity")
        if args.mode == "_worker":
            if not args.example or len(args.example) != 1:
                raise ValueError("a worker requires exactly one --example")
            return _worker(args.example[0], args.output.resolve(), cpus, args.smoke)
        if args.smoke:
            raise ValueError("use the smoke mode for reduced budgets")
        examples = args.example or list(EXAMPLES)
        seeds = args.seed if args.seed is not None else ([0] if args.mode == "smoke" else [0, 1, 2])
        if len(set(examples)) != len(examples) or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("examples and nonnegative seeds must be unique")
        if args.cores_per_job < 1:
            raise ValueError("cores-per-job must be positive")
        jobs = args.jobs if args.jobs is not None else min(len(examples) * len(seeds), len(cpus) // args.cores_per_job)
        slots = cpu_slots(cpus, jobs, args.cores_per_job)
        signal.signal(signal.SIGTERM, _interrupt)
        return run_batch(args.output.resolve(), examples, seeds, slots, smoke=args.mode == "smoke")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; children stopped and partial outputs retained", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
