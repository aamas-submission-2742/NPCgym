import csv
from dataclasses import replace

import pytest

from npc_gym.evaluation import (
    EpisodeCSVWriter,
    EpisodeResult,
    EvaluationSummary,
    LearningCurveCSVWriter,
    TerminationClass,
)


def episode_with(summaries):
    return EpisodeResult(0, 0.0, 1, TerminationClass.TERMINATED, {}, monitor_counts=summaries)


@pytest.mark.parametrize("changed", [{"test": {}}, {"test": {"other": 1}}, {"other": {"events": 1}}])
def test_aggregation_requires_stable_monitor_summary_schema(changed):
    with pytest.raises(ValueError, match="inconsistent|Inconsistent"):
        EvaluationSummary.from_episodes([episode_with({"test": {"events": 1}}), episode_with(changed)])


@pytest.mark.parametrize("writer_class", [EpisodeCSVWriter, LearningCurveCSVWriter])
def test_summary_csv_encoding_schema_failure_and_reentry(tmp_path, writer_class):
    first = episode_with({"a/b": {"c": 1, "c_std": 2}, "a": {"b/c": 3}, "a%2Fb": {"é%": 4}, "empty": {}})
    path = tmp_path / "output.csv"
    writer = writer_class(path, overwrite=True)

    def write(result):
        if isinstance(writer, EpisodeCSVWriter):
            writer.write(result)
        else:
            writer.write(0, EvaluationSummary.from_episodes([result]))

    with writer:
        write(first)
        content = path.read_bytes()
        for summaries in (
            {"a/b": {"different": 1}},
            {k: v for k, v in first.monitor_counts.items() if k != "empty"},
        ):
            with pytest.raises(ValueError, match="established schema"):
                write(replace(first, monitor_counts=summaries))
            assert path.read_bytes() == content
    with path.open() as stream:
        (row,) = csv.DictReader(stream)
    suffix = "_mean" if writer_class is LearningCurveCSVWriter else ""
    assert row[f"count/a%2Fb/c{suffix}"] == ("1.0" if suffix else "1")
    assert row[f"count/a/b%2Fc{suffix}"] == ("3.0" if suffix else "3")
    assert row[f"count/a%252Fb/%C3%A9%25{suffix}"] == ("4.0" if suffix else "4")
    with writer:
        write(episode_with({"new": {"field": 0}}))
    assert "count/new/field" in path.read_text()
    assert "count/a%2Fb/c" not in path.read_text()


@pytest.mark.parametrize(
    "values", [None, [], {"": 1}, {7: 1}, {"events": -1}, {"events": True}, {"events": 1.5}, {"events": "1"}]
)
def test_invalid_counts(values):
    with pytest.raises((TypeError, ValueError)):
        episode_with({"test": values})


def test_episode_count_snapshot_is_detached():
    source = {"test": {"events": 3}}
    result = episode_with(source)
    source["test"]["events"] = 99
    source.clear()
    assert result.monitor_counts == {"test": {"events": 3}}
