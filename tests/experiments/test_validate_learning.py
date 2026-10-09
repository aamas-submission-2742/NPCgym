"""Exercise CPU allocation, isolated execution, and failures using tiny budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from experiments import validate_learning as validation


def test_topology_selects_one_allowed_thread_per_physical_core():
    topology = "\n\n".join(
        f"processor: {cpu}\nphysical id: {socket}\ncore id: {core}"
        for cpu, socket, core in [(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 1, 0)]
    )
    assert validation.physical_cpus(topology, {0, 1, 2, 3, 4}) == [0, 1, 4]
    assert validation.physical_cpus(topology, {2, 3}) == [2, 3]
    with pytest.raises(ValueError, match="topology"):
        validation.physical_cpus("processor: 0", {0})


@pytest.mark.parametrize(
    ("cpus", "jobs", "cores", "message"),
    [([0, 1], 3, 1, "exceeds"), ([0, 0], 2, 1, "distinct"), ([0], 0, 1, "positive"), ([0], 1, 0, "positive")],
)
def test_allocation_rejects_oversubscription_and_invalid_inputs(cpus, jobs, cores, message):
    with pytest.raises(ValueError, match=message):
        validation.cpu_slots(cpus, jobs, cores)


def test_slots_and_full_budget_environment(monkeypatch):
    monkeypatch.setenv("NPC_GYM_TRAINING_STEPS", "1")
    monkeypatch.setenv("NPC_GYM_SEED", "999")
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    assert validation.cpu_slots([0, 2, 4, 6], 2, 2) == [[0, 2], [4, 6]]
    env = validation.worker_environment(2, [4, 6], smoke=False)
    assert "NPC_GYM_TRAINING_STEPS" not in env
    assert env["NPC_GYM_SEED"] == "2"
    assert all(env[key] == "2" for key in validation.THREAD_VARIABLES)


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="CPU-pinned gate requires Linux")
def test_smoke_runs_every_example_and_refuses_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "results"
    monkeypatch.setenv("NPC_GYM_TRAINING_STEPS", "999999999")
    cpus = sorted(os.sched_getaffinity(0))[:2]
    command = [
        sys.executable,
        str(validation.ROOT / "experiments/validate_learning.py"),
        "smoke",
        "--output",
        str(output),
        "--cpus",
        ",".join(map(str, cpus)),
        "--jobs",
        str(len(cpus)),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["assessment"] == "smoke_only"
    assert len(manifest["runs"]) == 4
    assert {tuple(run["cpus"]) for run in manifest["runs"]} == {(cpu,) for cpu in cpus}
    for example in validation.EXAMPLES:
        directory = output / f"{example}-seed-0"
        evaluation = json.loads((directory / "evaluation.json").read_text())
        metadata = evaluation["metadata"]
        assert metadata["actual_training_steps"] == 16
        assert metadata["configuration"]["TRAINING_STEPS"] == 16
        if example in {"pacman", "gardener"}:
            assert metadata["configuration"]["ALGORITHM"] == "dqn"
        assert metadata["evaluation_seed"] == 10000
        assert metadata["torch_threads"] == metadata["torch_interop_threads"] == 1
        assert metadata["cpus"] in [[cpu] for cpu in cpus]
        assert metadata["wall_seconds"] > 0
        assert len(evaluation["episodes"]) == 2
        assert evaluation["aggregate"]["metrics"]
        assert evaluation["aggregate"]["monitor_counts"]
        assert {path.name for path in directory.iterdir()} == {"evaluation.json", "run.log"}
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    refused = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False)
    assert refused.returncode == 2
    assert "File exists" in refused.stderr
    assert before == {path: path.read_bytes() for path in before}


@pytest.fixture
def fake_cpuinfo(monkeypatch):
    """Keep scheduler failure tests independent of the host's Linux topology file."""
    read_text = validation.Path.read_text

    def read(path, *args, **kwargs):
        if path == validation.Path("/proc/cpuinfo"):
            return "model name: test CPU\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(validation.Path, "read_text", read)


def test_failed_worker_retains_log_and_does_not_report_completion(tmp_path, monkeypatch, fake_cpuinfo):
    popen = subprocess.Popen

    def failing_worker(command, **kwargs):
        return popen([sys.executable, "-c", "raise RuntimeError('test worker failure')"], **kwargs)

    monkeypatch.setattr(validation.subprocess, "Popen", failing_worker)
    # Git provenance also uses Popen; bypass it here without replacing worker execution.
    monkeypatch.setattr(
        validation.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "revision")
    )
    output = tmp_path / "failed"
    assert validation.run_batch(output, ["taxi"], [0, 1], [[0]], smoke=True) == 1
    document = json.loads((output / "manifest.json").read_text())
    assert document["status"] == "failed"
    assert all(run["status"] == "failed" for run in document["runs"])
    assert "test worker failure" in (output / "taxi-seed-0/run.log").read_text()
    assert not list(output.rglob("evaluation.json"))
    assert not list(output.rglob("*.tmp"))


def test_interruption_stops_children_and_marks_partial_output(tmp_path, monkeypatch, fake_cpuinfo):
    popen = subprocess.Popen
    children = []

    def slow_worker(command, **kwargs):
        child = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    sleep = validation.time.sleep
    interrupted = False

    def interrupt(seconds):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        sleep(seconds)

    monkeypatch.setattr(validation.subprocess, "Popen", slow_worker)
    monkeypatch.setattr(
        validation.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "revision")
    )
    monkeypatch.setattr(validation.time, "sleep", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        validation.run_batch(output, ["taxi"], [0, 1], [[0]], smoke=True)
    assert children and all(child.poll() is not None for child in children)
    document = json.loads((output / "manifest.json").read_text())
    assert document["status"] == "interrupted"
    assert document["runs"][0]["status"] == "interrupted"
    assert not (output / "taxi-seed-1").exists()


def test_long_training_requires_explicit_invocation():
    import tomllib

    config = tomllib.loads((validation.ROOT / "pyproject.toml").read_text())
    import configparser

    tox = configparser.ConfigParser(interpolation=None)
    tox.read_string(config["tool"]["tox"]["legacy_tox_ini"])
    assert "learning" not in [name.strip() for name in tox["tox"]["env_list"].split(",")]
    assert "smoke" in tox["testenv:learning-smoke"]["commands"]
    assert tox["testenv:learning-smoke"]["platform"] == "linux"
    assert "{posargs}" in tox["testenv:learning"]["commands"]
    with pytest.raises(SystemExit):
        validation.main([])
