"""CPU-pinned subprocess scheduling for the standard paper experiment runner."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

from experiments.execution import ExecutionError, TrainingPlan, preflight_training, resolved_configuration
from experiments.specifications import REGISTRY


def validate_cpus(cpus: Sequence[int]) -> None:
    """Reject unavailable or duplicate CPU assignments before creating output."""
    if not hasattr(os, "sched_getaffinity") or shutil.which("taskset") is None:
        raise ValueError("--cpus requires Linux CPU affinity support and taskset")
    if not cpus or len(set(cpus)) != len(cpus) or any(cpu < 0 for cpu in cpus):
        raise ValueError("--cpus must list distinct nonnegative CPU IDs")
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError(f"Unavailable CPUs: {sorted(set(cpus) - os.sched_getaffinity(0))}")


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def execute_parallel_training(plans: Sequence[TrainingPlan], cpus: Sequence[int], *, paper: bool = False) -> None:
    """Run one single-threaded learner per CPU, retaining logs and queue status.

    All plans are preflighted before launch. The output root must have no prior
    batch.json or batch-logs directory. Children use the standard CLI/registry;
    longer runs start first. Failed jobs are retained while other jobs continue.
    Interruption terminates and reaps this dispatcher's remaining children.
    """
    validate_cpus(cpus)
    preflight_training(plans)
    registry = REGISTRY
    if paper:
        from experiments.paper import make_registry

        registry = make_registry()
    for plan in plans:
        if resolved_configuration(plan.resolved) != resolved_configuration(
            registry.resolve(plan.resolved.specification.id)
        ):
            raise ValueError("Parallel training requires the selected registry's exact configurations")
    roots = {plan.output_root for plan in plans}
    if len(roots) != 1:
        raise ValueError("Parallel plans must share an output root")
    root = roots.pop()
    manifest, logs = root / "batch.json", root / "batch-logs"
    if any(path.exists() or path.is_symlink() for path in (manifest, logs)):
        raise FileExistsError("Parallel batch output already exists; select a new --output directory")
    records: list[dict[str, Any]] = [
        {"experiment": p.resolved.specification.id, "seed": p.seed, "status": "queued"}
        for p in sorted(plans, key=lambda plan: -plan.resolved.specification.run.training_steps)
    ]
    document: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "cpus": list(cpus),
        "threads_per_worker": 1,
        "runs": records,
    }
    logs.mkdir(parents=True)

    def save() -> None:
        temporary = manifest.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            temporary.replace(manifest)
        finally:
            temporary.unlink(missing_ok=True)

    env = dict(os.environ)
    env.update(
        dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
    )
    env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    pending = deque(records)
    active: dict[int, tuple[subprocess.Popen[bytes], int, dict[str, Any]]] = {}

    def start(cpu: int) -> None:
        if not pending:
            return
        row = pending.popleft()
        command = [
            "taskset",
            "-c",
            str(cpu),
            sys.executable,
            str(Path(__file__).with_name("run.py")),
            "train",
            "--output",
            str(root),
            "--experiment",
            row["experiment"],
            "--seed",
            str(row["seed"]),
            "--threads",
            "1",
        ]
        if paper:
            command.append("--paper")
        with (logs / f"{row['experiment']}-seed-{row['seed']}.log").open("xb") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        row.update(status="running", cpu=cpu, pid=process.pid)
        active[process.pid] = (process, cpu, row)
        save()

    original_affinity = os.sched_getaffinity(0)
    previous_handler = signal.signal(signal.SIGTERM, _interrupt)
    try:
        os.sched_setaffinity(0, cpus)
        save()
        for cpu in cpus:
            start(cpu)
        while active:
            pid, status = os.wait()
            process, cpu, row = active.pop(pid)
            process.returncode = os.waitstatus_to_exitcode(status)
            row.update(status="complete" if process.returncode == 0 else "failed", exit_code=process.returncode)
            save()
            start(cpu)
        if any(row["status"] != "complete" for row in records):
            raise ExecutionError(f"Some training runs failed; see {manifest} and {logs}")
        document["status"] = "complete"
    except BaseException:
        document["status"] = "incomplete"
        raise
    finally:
        for process, _cpu, row in active.values():
            process.terminate()
            row["status"] = "interrupted"
        for process, _cpu, _row in active.values():
            process.wait()
        document["completed_at"] = datetime.now(UTC).isoformat()
        save()
        signal.signal(signal.SIGTERM, previous_handler)
        os.sched_setaffinity(0, original_affinity)
