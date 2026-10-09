"""Retain and validate completed numerical results in the paper's directory layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.execution import read_run_document
from experiments.paper import EXPERIMENTS_BY_ENVIRONMENT


def validate_results(root: Path) -> dict[str, int]:
    """Check the complete paper archive without writing or requiring models.

    Require exactly the selected recipes and seeds, manifest/file agreement,
    unchanged SHA-256 hashes, complete runs and 1,000 final episodes. Original
    training configurations remain valid when paper evaluations add monitors.
    Return completed training-run counts by environment; invalid data raise
    ``ValueError`` or the specific filesystem error for a missing input.
    """
    counts = {}
    for environment, identifiers in EXPERIMENTS_BY_ENVIRONMENT.items():
        folder = root / environment
        manifest = _json(folder / "archive.json")
        evaluation = "final" if environment == "taxi" else "paper"
        if (
            manifest.get("schema_version") != 1
            or manifest.get("environment") != environment
            or manifest.get("experiments") != sorted(identifiers)
            or manifest.get("seeds") != list(range(8))
            or manifest.get("completed_training_runs") != len(identifiers) * 8
            or manifest.get("evaluation_name") != evaluation
        ):
            raise ValueError(f"Paper archive has incomplete or inconsistent coverage: {folder}")
        hashes = manifest.get("files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"Paper archive has no file hashes: {folder}")
        actual = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}
        if actual != set(hashes) | {"archive.json"}:
            raise ValueError(f"Archive files differ from the manifest: {folder}")
        for relative, checksum in hashes.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ValueError(f"Unsafe archive path: {relative}")
            source = folder / path
            if any(part.is_symlink() for part in (source, *source.parents)):
                raise ValueError(f"Archive paths must not be symlinks: {source}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Archive checksum differs: {source}")
        if {path.name for path in folder.iterdir() if path.is_dir()} != set(identifiers):
            raise ValueError(f"Archive recipe directories differ: {folder}")
        for identifier in identifiers:
            experiment = folder / identifier
            if {path.name for path in experiment.iterdir() if path.is_dir()} != {f"seed-{seed}" for seed in range(8)}:
                raise ValueError(f"Expected exactly seeds 0–7: {experiment}")
            for seed in range(8):
                run = experiment / f"seed-{seed}"
                document = read_run_document(run / "run.json")
                if (
                    document["status"] != "complete"
                    or document["seed"] != seed
                    or document["experiment_id"] != identifier
                    or document["configuration"]["environment"]["family"] != environment
                ):
                    raise ValueError(f"Run is incomplete or has mismatched identity: {run}")
                summary = _json(run / "evaluations" / evaluation / "summary.json")
                episodes = summary.get("episodes")
                metadata = summary.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise TypeError(f"Expected final evaluation metadata: {run}")
                if not isinstance(episodes, list) or len(episodes) != 1000 or metadata.get("episode_count") != 1000:
                    raise ValueError(f"Expected 1,000 final evaluation episodes: {run}")
                if (
                    metadata.get("experiment_id") != identifier
                    or metadata.get("model_sha256") != document["artifacts"]["model"]["sha256"]
                ):
                    raise ValueError(f"Final evaluation identity differs from its training run: {run}")
                for required in (run / "learning_curve.csv", run / "evaluations" / evaluation / "episodes.csv"):
                    if not required.is_file():
                        raise ValueError(f"Required result file is missing: {required}")
        counts[environment] = len(identifiers) * 8
    return counts


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def retain_results(
    inputs: Sequence[Path], output: Path, environment: str, *, experiment_ids: Sequence[str] = ()
) -> Path:
    """Copy selected complete eight-seed experiments into ``output/environment``.

    Inputs contain experiment/seed directories. Multiple inputs permit batches
    trained separately. Every selected run must be complete, with its final
    summary, episodes and curve. Merchant, Gardener and Pacman use the named ``paper`` evaluation;
    Merchant omits redundant raw-base measurements. Other environments retain
    training, intermediate and final evaluations, including OFTEN comparisons.
    Metadata and numerical files are copied byte-for-byte; hashes record the
    result. Models, logs, unselected recipes and unrelated named evaluations
    are excluded. Existing destinations and overlapping trees are rejected.
    Validation finishes before publication; failure removes temporary output.
    """
    if environment not in EXPERIMENTS_BY_ENVIRONMENT:
        raise ValueError(f"Unknown paper environment: {environment}")
    selected = tuple(experiment_ids) or EXPERIMENTS_BY_ENVIRONMENT[environment]
    if len(set(selected)) != len(selected) or not set(selected) <= set(EXPERIMENTS_BY_ENVIRONMENT[environment]):
        raise ValueError("Select distinct paper experiment IDs belonging to the requested environment")
    selected = tuple(sorted(selected))
    roots = tuple(path.expanduser().resolve() for path in inputs)
    target = output.expanduser().resolve() / environment
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide distinct input roots")
    if any(root == target or root in target.parents or target in root.parents for root in roots):
        raise ValueError("Retained output must be separate from all input trees")
    if target.exists():
        raise FileExistsError(f"Results already exist: {target}")
    if any(not root.is_dir() for root in roots):
        raise ValueError("Every input root must be an existing directory")

    files: dict[Path, Path] = {}
    evaluation_name = "paper" if environment in {"merchant", "gardener", "pacman"} else "final"
    for identifier in selected:
        matches = [root / identifier for root in roots if (root / identifier).is_dir()]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source for {identifier}; found {len(matches)}")
        folder = matches[0]
        if {p.name for p in folder.glob("seed-*")} != {f"seed-{seed}" for seed in range(8)}:
            raise ValueError(f"Expected all eight seeds 0–7, without extra seeds: {folder}")
        for seed in range(8):
            run = folder / f"seed-{seed}"
            document = read_run_document(run / "run.json")
            if document["status"] != "complete" or document["seed"] != seed or document["experiment_id"] != identifier:
                raise ValueError(f"Run is incomplete or has mismatched identity: {run}")
            if document["configuration"]["environment"]["family"] != environment:
                raise ValueError(f"Run belongs to another environment: {run}")
            final = run / "evaluations" / evaluation_name
            summary = _json(final / "summary.json")
            if not summary.get("episodes"):
                raise ValueError(f"Final evaluation has no episodes: {final}")
            if summary.get("metadata", {}).get("model_sha256") != document["artifacts"]["model"]["sha256"]:
                raise ValueError(f"Final evaluation checkpoint hash differs from its training run: {final}")
            for required in (run / "learning_curve.csv", final / "episodes.csv"):
                if not required.is_file():
                    raise ValueError(f"Required result file is missing: {required}")
            paths = [run / "run.json", run / "learning_curve.csv"]
            if document["configuration"].get("often") is not None:
                base_curve = run / "base_learning_curve.csv"
                if not base_curve.is_file():
                    raise ValueError(f"Required base-training curve is missing: {base_curve}")
                paths.append(base_curve)
            evaluations = (
                [final]
                if environment == "merchant"
                else [
                    path
                    for path in (run / "evaluations").iterdir()
                    if path.is_dir()
                    and (path.name in {evaluation_name, "base"} or path.name.startswith("intermediate-"))
                ]
            )
            for evaluation in evaluations:
                for path in evaluation.rglob("*"):
                    if not path.is_file() or path.suffix not in {".json", ".csv"}:
                        continue
                    if environment == "merchant" and "base" in path.relative_to(evaluation).parts:
                        continue
                    paths.append(path)
            for path in paths:
                if path.is_symlink():
                    raise ValueError(f"Result files must not be symlinks: {path}")
                files[Path(identifier) / run.name / path.relative_to(run)] = path

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".retain-", dir=target.parent) as temporary:
        staging = Path(temporary) / environment
        staging.mkdir()
        hashes = {}
        for relative, source in sorted(files.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "environment": environment,
            "experiments": list(selected),
            "seeds": list(range(8)),
            "completed_training_runs": len(selected) * 8,
            "evaluation_name": evaluation_name,
            "files_sha256": hashes,
        }
        (staging / "archive.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.rename(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a retained paper dataset and print its complete-run counts."""
    parser = argparse.ArgumentParser(description="Validate complete paper results without modifying files.")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/results")
    args = parser.parse_args(argv)
    try:
        counts = validate_results(args.input)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
