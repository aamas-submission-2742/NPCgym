"""Subprocess dispatch, affinity, failures and output ownership."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from experiments import parallel, run
from experiments.execution import ExecutionError, plan_training_batch
from experiments.specifications import PACMAN_BOLT_EXPERIMENT_IDS, TAXI_EXPERIMENT_ID

pytestmark = pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or shutil.which("taskset") is None, reason="Linux taskset required"
)


@pytest.mark.parametrize("failure", (False, True))
def test_dispatches_every_seed_and_preserves_logs(tmp_path, monkeypatch, failure):
    cpus = sorted(os.sched_getaffinity(0))[:2]
    before = os.sched_getaffinity(0)
    plans = plan_training_batch(tmp_path, seeds=[0, 1, 2], experiment_ids=[TAXI_EXPERIMENT_ID])
    real_popen = subprocess.Popen

    def tiny_worker(command, **kwargs):
        assert command[-2:] == ["--threads", "1"]
        assert command[command.index("--experiment") + 1] == TAXI_EXPERIMENT_ID
        seed = int(command[command.index("--seed") + 1])
        code = (
            "import os,json; print(json.dumps({'cpu':sorted(os.sched_getaffinity(0)),"
            "'threads':os.environ['OMP_NUM_THREADS']})); "
            f"raise SystemExit({1 if failure and seed == 1 else 0})"
        )
        return real_popen([*command[:3], sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(parallel.subprocess, "Popen", tiny_worker)
    if failure:
        with pytest.raises(ExecutionError, match="Some training runs failed"):
            parallel.execute_parallel_training(plans, cpus)
    else:
        parallel.execute_parallel_training(plans, cpus)
    assert os.sched_getaffinity(0) == before
    metadata = json.loads((tmp_path / "batch.json").read_text())
    assert metadata["status"] == ("incomplete" if failure else "complete")
    assert len(metadata["runs"]) == 3
    assert {row["seed"] for row in metadata["runs"]} == {0, 1, 2}
    for row in metadata["runs"]:
        log = tmp_path / "batch-logs" / f"{TAXI_EXPERIMENT_ID}-seed-{row['seed']}.log"
        assert json.loads(log.read_text()) == {"cpu": [row["cpu"]], "threads": "1"}
        assert row["status"] == ("failed" if failure and row["seed"] == 1 else "complete")
    snapshot = (tmp_path / "batch.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        parallel.execute_parallel_training(plans, cpus)
    assert (tmp_path / "batch.json").read_bytes() == snapshot


def test_invalid_affinity_and_cli_dry_run_write_nothing(tmp_path, capsys):
    cpu = min(os.sched_getaffinity(0))
    for values in ([], [-1], [cpu, cpu], [max(os.sched_getaffinity(0)) + 1]):
        with pytest.raises(ValueError):
            parallel.validate_cpus(values)
    command = [
        "train",
        "--output",
        str(tmp_path / "runs"),
        "--technique",
        "technique/restraining-bolts-v0",
        "--seed",
        "0",
        "--cpus",
        str(cpu),
        "--dry-run",
    ]
    pytest.importorskip("stable_baselines3")
    assert run.main(command) == 0
    text = capsys.readouterr().out
    assert all(identifier in text for identifier in PACMAN_BOLT_EXPERIMENT_IDS)
    assert not (tmp_path / "runs").exists()
    assert run.main([*command, "--overwrite"]) == 2
    assert not (tmp_path / "runs").exists()


def test_pilot_queue_is_separate_from_final_seeds(tmp_path, capsys):
    pytest.importorskip("stable_baselines3")
    output = tmp_path / "pilots"
    assert (
        run.main(
            ["pacman-bolt-pilots", "--output", str(output), "--cpus", str(min(os.sched_getaffinity(0))), "--dry-run"]
        )
        == 0
    )
    plans = json.loads(capsys.readouterr().out)
    assert len(plans) == 52
    assert sum(p["requested_timesteps"] for p in plans) == 740_000_000
    assert len({p["run_directory"] for p in plans}) == 52
    for p in plans:
        final = "hungry-vegan-penalty" in p["experiment_id"]
        assert p["training_seed"] in (range(8) if final else (100, 101, 102))
    assert not output.exists()
