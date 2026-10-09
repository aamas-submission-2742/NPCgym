import csv
import json

import pytest

from npc_gym.evaluation import (
    EpisodeCSVWriter,
    EpisodeResult,
    EvaluationJSONWriter,
    EvaluationSummary,
    LearningCurveCSVWriter,
    TerminationClass,
)


def make_summary() -> EvaluationSummary:
    return EvaluationSummary.from_episodes(
        (
            EpisodeResult(0, 1.0, 2, TerminationClass.TERMINATED, {"score": 2}, {"test/norm-v0": {"count": 0}}),
            EpisodeResult(1, 3.0, 4, TerminationClass.TRUNCATED, {"score": 6}, {"test/norm-v0": {"count": 2}}),
        )
    )


def test_writers_use_separate_versioned_schemas(tmp_path):
    summary = make_summary()
    episode_path = tmp_path / "episodes.csv"
    curve_path = tmp_path / "curve.csv"
    result_path = tmp_path / "result.json"

    with EpisodeCSVWriter(episode_path) as writer:
        writer.write_all(summary.episodes)
    with LearningCurveCSVWriter(curve_path) as writer:
        writer.write(100, summary)
    with EvaluationJSONWriter(result_path) as writer:
        writer.write(summary, metadata={"seed": 3})

    with episode_path.open(newline="") as stream:
        episode_rows = list(csv.DictReader(stream))
    with curve_path.open(newline="") as stream:
        curve_rows = list(csv.DictReader(stream))
    document = json.loads(result_path.read_text(encoding="utf-8"))

    assert episode_rows == [
        {
            "schema_version": "4",
            "episode": "0",
            "return": "1.0",
            "length": "2",
            "termination": "terminated",
            "count/test%2Fnorm-v0/count": "0",
            "metric/score": "2",
        },
        {
            "schema_version": "4",
            "episode": "1",
            "return": "3.0",
            "length": "4",
            "termination": "truncated",
            "count/test%2Fnorm-v0/count": "2",
            "metric/score": "6",
        },
    ]
    assert curve_rows == [
        {
            "schema_version": "4",
            "timesteps": "100",
            "episodes": "2",
            "return_mean": "2.0",
            "return_std": "1.0",
            "length_mean": "3.0",
            "length_std": "1.0",
            "count/test%2Fnorm-v0/count_mean": "1.0",
            "count/test%2Fnorm-v0/count_std": "1.0",
            "metric/score_mean": "4.0",
            "metric/score_std": "2.0",
        }
    ]
    assert document["schema_version"] == 5
    assert document["metadata"] == {"seed": 3}
    assert document["episodes"][1]["termination"] == "truncated"
    assert document["aggregate"]["return"] == {"mean": 2.0, "std": 1.0}


def test_writers_require_context_and_do_not_overwrite_by_default(tmp_path):
    path = tmp_path / "episodes.csv"
    path.write_text("keep me", encoding="utf-8")
    writer = EpisodeCSVWriter(path)

    with pytest.raises(RuntimeError, match="context manager"):
        writer.write(make_summary().episodes[0])
    with pytest.raises(FileExistsError), writer:
        pass
    assert path.read_text(encoding="utf-8") == "keep me"

    with EpisodeCSVWriter(path, overwrite=True) as overwrite_writer:
        overwrite_writer.write(make_summary().episodes[0])
    assert path.read_text(encoding="utf-8").startswith("schema_version,episode")


def test_reentering_a_writer_starts_a_new_file(tmp_path):
    path = tmp_path / "episodes.csv"
    writer = EpisodeCSVWriter(path, overwrite=True)

    with writer:
        writer.write(make_summary().episodes[0])
    with writer:
        writer.write(EpisodeResult(0, 0.0, 1, TerminationClass.TERMINATED, {"other": 0}))

    header, row = path.read_text(encoding="utf-8").splitlines()
    assert header == "schema_version,episode,return,length,termination,metric/other"
    assert row == "4,0,0.0,1,terminated,0"


def test_csv_writers_reject_schema_changes(tmp_path):
    first = make_summary().episodes[0]
    changed = EpisodeResult(1, 0.0, 1, TerminationClass.TERMINATED, {"other": 0})

    with EpisodeCSVWriter(tmp_path / "episodes.csv") as writer:
        writer.write(first)
        with pytest.raises(ValueError, match="established schema"):
            writer.write(changed)


def test_json_round_trip_retains_weighted_signals_and_separate_measurements(tmp_path):
    summary = EvaluationSummary.from_episodes(
        [
            EpisodeResult(
                episode=0,
                episode_return=2.5,
                length=3,
                termination=TerminationClass.TERMINATED_AND_TRUNCATED,
                metrics={"score": 7},
                monitor_counts={"checkpoint": {"inputs": 4}, "permission": {"allowed": 1}},
            )
        ]
    )
    path = tmp_path / "summary.json"
    with EvaluationJSONWriter(path) as writer:
        writer.write(summary)
    document = json.loads(path.read_text())
    assert document["schema_version"] == 5
    restored = EvaluationSummary.from_episodes(
        [
            EpisodeResult(
                episode=row["episode"],
                episode_return=row["return"],
                length=row["length"],
                termination=TerminationClass(row["termination"]),
                metrics=row["metrics"],
                monitor_counts=row["monitor_counts"],
            )
            for row in document["episodes"]
        ]
    )
    assert restored == summary
    assert set(document["episodes"][0]) == {
        "episode",
        "return",
        "length",
        "termination",
        "metrics",
        "monitor_counts",
    }
