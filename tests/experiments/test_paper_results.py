"""Paper selection and publication use small, generated data only."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import parallel, results, run
from experiments.execution import plan_training_batch, preflight_training
from experiments.paper import EXPERIMENTS_BY_ENVIRONMENT, GARDENER_MONITORS, PACMAN_VEGAN_OFTEN_ID, make_registry
from experiments.specifications import MERCHANT_EXPERIMENT_ID, MERCHANT_POLICY_FIX_EXPERIMENT_ID


def test_paper_selection_matches_tables_and_uses_smallclassic(tmp_path):
    registry = make_registry()
    plans = plan_training_batch(tmp_path / "runs", registry=registry, seeds=range(8))
    assert len(plans) == 264
    assert {k: len(v) for k, v in EXPERIMENTS_BY_ENVIRONMENT.items()} == {
        "taxi": 4,
        "merchant": 4,
        "gardener": 8,
        "pacman": 17,
    }
    assert {p.seed for p in plans} == set(range(8))
    for plan in plans:
        assert plan.resolved.specification.run.final_evaluation_episodes == 1000
        assert plan.resolved.specification.run.intermediate_evaluation_episodes == 1000
        if plan.resolved.environment.family == "gardener":
            assert tuple(plan.resolved.make_monitors()) == GARDENER_MONITORS
        if plan.resolved.environment.family == "pacman":
            assert "smallClassic" in str(plan.resolved.environment.constructor_kwargs.to_dict()["layout"])
            assert set(plan.resolved.make_monitors()) == {
                "pacman/vegan-v0",
                "pacman/vegetarian-orange-v0",
                "pacman/hungry-vegan-v0",
                "pacman/trapped-v1",
                "pacman/vegan-conflict-v1",
                "pacman/hungry-vegan-penalty-v1",
            }
    baseline = registry.resolve("pacman-dqn-unconstrained-v0")
    assert "pacman/vegan-conflict-v1" in baseline.make_monitors()
    for identifier in EXPERIMENTS_BY_ENVIRONMENT["pacman"][:5]:
        with registry.resolve(identifier).make_env(training=False) as env:
            observation, _ = env.reset(seed=3)
            assert env.observation_space.contains(observation)
            env.step(0)
    assert not (tmp_path / "runs").exists()


def test_paper_cli_selects_four_merchant_recipes_without_writing(tmp_path, capsys):
    pytest.importorskip("clingo")
    assert (
        run.main(
            [
                "train",
                "--paper",
                "--environment",
                "merchant/basic-v2",
                "--output",
                str(tmp_path / "runs"),
                "--seed",
                "0",
                "--dry-run",
            ]
        )
        == 0
    )
    plans = json.loads(capsys.readouterr().out)
    assert {plan["experiment_id"] for plan in plans} == set(EXPERIMENTS_BY_ENVIRONMENT["merchant"])
    assert not (tmp_path / "runs").exists()


def test_parallel_paper_workers_keep_the_selected_registry(tmp_path, monkeypatch):
    pytest.importorskip("clingo")
    pytest.importorskip("stable_baselines3")
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux affinity required")
    identifier = PACMAN_VEGAN_OFTEN_ID
    plans = plan_training_batch(tmp_path, registry=make_registry(), experiment_ids=[identifier], seeds=[0])
    preflight_training(plans)
    real_popen = subprocess.Popen

    def worker(command, **kwargs):
        assert command[-1] == "--paper"
        assert command[command.index("--experiment") + 1] == identifier
        return real_popen([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr(parallel.subprocess, "Popen", worker)
    parallel.execute_parallel_training(plans, [min(os.sched_getaffinity(0))], paper=True)
    assert json.loads((tmp_path / "batch.json").read_text())["status"] == "complete"


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _runs(root, identifier=MERCHANT_POLICY_FIX_EXPERIMENT_ID):
    checksum = "a" * 64
    for seed in range(8):
        folder = root / identifier / f"seed-{seed}"
        _write(
            folder / "run.json",
            {
                "schema_version": 1,
                "status": "complete",
                "seed": seed,
                "experiment_id": identifier,
                "configuration": {"environment": {"family": "merchant"}},
                "artifacts": {"model": {"sha256": checksum}},
            },
        )
        (folder / "learning_curve.csv").write_text("steps,return\n1,2\n")
        _write(
            folder / "evaluations/paper/summary.json",
            {
                "episodes": [{"return": seed}],
                "metadata": {"model_sha256": checksum},
            },
        )
        (folder / "evaluations/paper/episodes.csv").write_text("episode,return\n1,2\n")
        _write(folder / "evaluations/paper/base/summary.json", {})
        _write(folder / "evaluations/other/summary.json", {})
        (folder / "model.zip").write_bytes(b"local model")
        (folder / "run.log").write_text("local log")


def test_export_exact_files_hashes_and_no_source_changes(tmp_path):
    source = tmp_path / "working"
    _runs(source)
    _runs(source, MERCHANT_EXPERIMENT_ID)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    target = results.retain_results(
        [source], tmp_path / "published", "merchant", experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID]
    )
    manifest = json.loads((target / "archive.json").read_text())
    assert manifest["completed_training_runs"] == 8
    assert len(manifest["files_sha256"]) == 32
    for relative, checksum in manifest["files_sha256"].items():
        assert (target / relative).read_bytes() == (source / relative).read_bytes()
        assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == checksum
    assert {p.suffix for p in target.rglob("*") if p.is_file()} == {".json", ".csv"}
    assert not list(target.rglob("base"))
    assert not list(target.rglob("other"))
    assert not (target / MERCHANT_EXPERIMENT_ID).exists()
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        results.retain_results([source], target.parent, "merchant", experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID])


@pytest.mark.parametrize(
    "problem", ["missing-seed", "incomplete", "missing-episodes", "wrong-checkpoint", "extra-seed"]
)
def test_export_rejects_invalid_batch_before_writing(tmp_path, problem):
    source = tmp_path / "working"
    _runs(source)
    folder = source / MERCHANT_POLICY_FIX_EXPERIMENT_ID / "seed-7"
    if problem == "missing-seed":
        folder.rename(folder.with_name("missing"))
    elif problem == "incomplete":
        metadata = json.loads((folder / "run.json").read_text())
        metadata["status"] = "incomplete"
        _write(folder / "run.json", metadata)
    elif problem == "missing-episodes":
        (folder / "evaluations/paper/episodes.csv").unlink()
    elif problem == "wrong-checkpoint":
        _write(folder / "evaluations/paper/summary.json", {"episodes": [1], "metadata": {"model_sha256": "b" * 64}})
    else:
        (folder.parent / "seed-8").mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        results.retain_results(
            [source], tmp_path / "published", "merchant", experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID]
        )
    assert not (tmp_path / "published").exists()


def test_export_overlap_ambiguity_and_cleanup(tmp_path, monkeypatch):
    source = tmp_path / "working"
    _runs(source)
    with pytest.raises(ValueError, match="separate"):
        results.retain_results([source], source, "merchant", experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID])
    second = tmp_path / "second"
    _runs(second)
    with pytest.raises(ValueError, match="exactly one source"):
        results.retain_results(
            [source, second], tmp_path / "published", "merchant", experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID]
        )

    def fail_copy(*args):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(results.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="simulated"):
        results.retain_results(
            [source], tmp_path / "published", "merchant", experiment_ids=[MERCHANT_POLICY_FIX_EXPERIMENT_ID]
        )
    assert list((tmp_path / "published").iterdir()) == []


def test_export_preserves_often_base_training_curves(tmp_path):
    identifier = "gardener-dqn-often-drain-v0"
    source = tmp_path / "working"
    _runs(source, identifier)
    for seed in range(8):
        folder = source / identifier / f"seed-{seed}"
        metadata = json.loads((folder / "run.json").read_text())
        metadata["configuration"].update(environment={"family": "gardener"}, often={})
        _write(folder / "run.json", metadata)
        (folder / "base_learning_curve.csv").write_text("steps,return\n0,0\n")
    target = results.retain_results([source], tmp_path / "published", "gardener", experiment_ids=[identifier])
    manifest = json.loads((target / "archive.json").read_text())
    for seed in range(8):
        relative = f"{identifier}/seed-{seed}/base_learning_curve.csv"
        assert relative in manifest["files_sha256"]
        assert (target / relative).read_bytes() == (source / relative).read_bytes()
    (source / identifier / "seed-0/base_learning_curve.csv").unlink()
    with pytest.raises(ValueError, match="base-training curve is missing"):
        results.retain_results([source], tmp_path / "missing", "gardener", experiment_ids=[identifier])
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("environment", ["pacman", "gardener"])
def test_export_selects_common_monitor_evaluation_and_preserves_curves(tmp_path, environment):
    identifier = f"{environment}-dqn-unconstrained-v0"
    source = tmp_path / "working"
    _runs(source, identifier)
    for seed in range(8):
        folder = source / identifier / f"seed-{seed}"
        metadata = json.loads((folder / "run.json").read_text())
        metadata["configuration"]["environment"]["family"] = environment
        _write(folder / "run.json", metadata)
        _write(folder / "evaluations/final/summary.json", {"obsolete": True})
        _write(folder / "evaluations/intermediate-0/summary.json", {"curve": True})
    target = results.retain_results([source], tmp_path / "published", environment, experiment_ids=[identifier])
    manifest = json.loads((target / "archive.json").read_text())
    assert manifest["evaluation_name"] == "paper"
    for seed in range(8):
        folder = target / identifier / f"seed-{seed}"
        assert (folder / "evaluations/paper/summary.json").is_file()
        assert (folder / "evaluations/intermediate-0/summary.json").is_file()
        assert not (folder / "evaluations/final").exists()
        assert not (folder / "evaluations/other").exists()


@pytest.fixture
def validated_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "EXPERIMENTS_BY_ENVIRONMENT", {"merchant": (MERCHANT_EXPERIMENT_ID,)})
    working = tmp_path / "working"
    _runs(working, MERCHANT_EXPERIMENT_ID)
    for seed in range(8):
        summary = working / MERCHANT_EXPERIMENT_ID / f"seed-{seed}" / "evaluations/paper/summary.json"
        _write(
            summary,
            {
                "episodes": [{"return": seed}] * 1000,
                "metadata": {
                    "episode_count": 1000,
                    "experiment_id": MERCHANT_EXPERIMENT_ID,
                    "seed": seed,
                    "model_sha256": "a" * 64,
                },
            },
        )
    published = tmp_path / "published"
    results.retain_results([working], published, "merchant")
    return published


def test_validator_is_read_only_and_cli_reports_coverage(validated_archive, capsys):
    before = {p: p.read_bytes() for p in validated_archive.rglob("*") if p.is_file()}
    assert results.validate_results(validated_archive) == {"merchant": 8}
    assert results.main(["--input", str(validated_archive)]) == 0
    assert json.loads(capsys.readouterr().out) == {"merchant": 8}
    assert before == {p: p.read_bytes() for p in validated_archive.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "coverage", "episodes", "identity", "symlink"])
def test_validator_rejects_damaged_archive(validated_archive, damage):
    folder = validated_archive / "merchant"
    manifest_path = folder / "archive.json"
    manifest = json.loads(manifest_path.read_text())
    run = folder / MERCHANT_EXPERIMENT_ID / "seed-0"
    summary = run / "evaluations/paper/summary.json"
    if damage == "changed":
        (run / "learning_curve.csv").write_text("changed")
    elif damage == "missing":
        summary.unlink()
    elif damage == "extra":
        (run / "extra.json").write_text("{}")
    elif damage == "coverage":
        manifest["seeds"] = [0]
    elif damage == "symlink":
        original = summary.read_bytes()
        summary.unlink()
        outside = validated_archive.parent / "external.json"
        outside.write_bytes(original)
        summary.symlink_to(outside)
    else:
        document = json.loads(summary.read_text())
        if damage == "episodes":
            document["episodes"].pop()
        else:
            document["metadata"]["model_sha256"] = "b" * 64
        _write(summary, document)
        manifest["files_sha256"][str(summary.relative_to(folder))] = hashlib.sha256(summary.read_bytes()).hexdigest()
    _write(manifest_path, manifest)
    with pytest.raises(ValueError):
        results.validate_results(validated_archive)
