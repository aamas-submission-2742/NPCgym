"""Versioned output with named monitor counts and environment metrics.

CSV count columns use count/<escaped-monitor>/<escaped-member>; each component
is percent-encoded. Curves add mean/std suffixes. JSON retains nested counts.
The first row fixes the schema. Re-entering a writer starts a fresh file.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path
from typing import IO, Any, Self
from urllib.parse import quote

from npc_gym.evaluation._recording import monitor_count_schema
from npc_gym.evaluation.core import EpisodeResult, EvaluationSummary

SCHEMA_VERSION = 4
_JSON_SCHEMA_VERSION = 5


class _ContextWriter:
    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.overwrite = overwrite
        self._stream: IO[str] | None = None
        self._begin()

    def _begin(self) -> None:
        """Reset the state that describes a single output file."""

    def __enter__(self) -> Self:
        if self._stream is not None:
            raise RuntimeError("Writer context is already open")
        mode = "w" if self.overwrite else "x"
        self._stream = self.path.open(mode=mode, encoding="utf-8", newline="")
        self._begin()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _require_stream(self) -> IO[str]:
        if self._stream is None:
            raise RuntimeError("Use the writer as a context manager before writing")
        return self._stream


class EpisodeCSVWriter(_ContextWriter):
    """Write one version-4 episode row per :class:`EpisodeResult`."""

    def _begin(self) -> None:
        self._writer: csv.DictWriter[str] | None = None
        self._metric_keys: tuple[str, ...] | None = None
        self._summary_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    def write(self, result: EpisodeResult) -> None:
        """Write one episode, requiring stable dynamic columns after the first row."""
        stream = self._require_stream()
        metric_keys = tuple(sorted(result.metrics))
        summary_keys = monitor_count_schema(result.monitor_counts)
        summaries = _count_columns(result.monitor_counts)
        if self._writer is None:
            self._metric_keys = metric_keys
            self._summary_keys = summary_keys
            fields = [
                "schema_version",
                "episode",
                "return",
                "length",
                "termination",
                *(f"metric/{key}" for key in metric_keys),
                *summaries,
            ]
            self._writer = csv.DictWriter(stream, fieldnames=fields)
            self._writer.writeheader()
        elif metric_keys != self._metric_keys or summary_keys != self._summary_keys:
            raise ValueError("Episode result columns do not match the writer's established schema")

        row: dict[str, str | int | float] = {
            "schema_version": SCHEMA_VERSION,
            "episode": result.episode,
            "return": result.episode_return,
            "length": result.length,
            "termination": result.termination.value,
        }
        row.update({f"metric/{key}": result.metrics[key] for key in metric_keys})
        row.update(summaries)
        self._writer.writerow(row)
        stream.flush()

    def write_all(self, results: Iterable[EpisodeResult]) -> None:
        """Write each result in iteration order."""
        for result in results:
            self.write(result)


class LearningCurveCSVWriter(_ContextWriter):
    """Write version-4 aggregate evaluation points indexed by training transitions."""

    def _begin(self) -> None:
        self._writer: csv.DictWriter[str] | None = None
        self._metric_keys: tuple[str, ...] | None = None
        self._summary_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    def write(self, timesteps: int, summary: EvaluationSummary) -> None:
        """Write one learning-curve point using the summary's aggregate values."""
        if isinstance(timesteps, bool) or not isinstance(timesteps, Integral) or timesteps < 0:
            raise ValueError(f"timesteps must be a non-negative integer; got {timesteps!r}")
        stream = self._require_stream()
        metric_keys = tuple(sorted(summary.mean_metrics))
        summary_keys = monitor_count_schema(summary.mean_monitor_counts)
        summary_means = _count_columns(summary.mean_monitor_counts)
        summary_stds = _count_columns(summary.std_monitor_counts)
        if monitor_count_schema(summary.std_monitor_counts) != summary_keys:
            raise ValueError("Monitor count mean/std fields must match")
        if self._writer is None:
            self._metric_keys = metric_keys
            self._summary_keys = summary_keys
            fields = [
                "schema_version",
                "timesteps",
                "episodes",
                "return_mean",
                "return_std",
                "length_mean",
                "length_std",
                *(field for key in metric_keys for field in (f"metric/{key}_mean", f"metric/{key}_std")),
                *(field for key in summary_means for field in (f"{key}_mean", f"{key}_std")),
            ]
            self._writer = csv.DictWriter(stream, fieldnames=fields)
            self._writer.writeheader()
        elif metric_keys != self._metric_keys or summary_keys != self._summary_keys:
            raise ValueError("Evaluation summary columns do not match the writer's established schema")

        row: dict[str, int | float] = {
            "schema_version": SCHEMA_VERSION,
            "timesteps": int(timesteps),
            "episodes": len(summary.episodes),
            "return_mean": summary.mean_return,
            "return_std": summary.std_return,
            "length_mean": summary.mean_length,
            "length_std": summary.std_length,
        }
        for key in metric_keys:
            row[f"metric/{key}_mean"] = summary.mean_metrics[key]
            row[f"metric/{key}_std"] = summary.std_metrics[key]
        for key, mean in summary_means.items():
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = summary_stds[key]
        self._writer.writerow(row)
        stream.flush()


class EvaluationJSONWriter(_ContextWriter):
    """Write one version-5 evaluation document with nested named counts."""

    def _begin(self) -> None:
        self._written = False

    def write(self, summary: EvaluationSummary, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Write a summary and optional caller-owned provenance metadata."""
        stream = self._require_stream()
        if self._written:
            raise RuntimeError("EvaluationJSONWriter writes exactly one summary")
        document = {
            "schema_version": _JSON_SCHEMA_VERSION,
            "metadata": dict(metadata or {}),
            "episodes": [
                {
                    "episode": result.episode,
                    "return": result.episode_return,
                    "length": result.length,
                    "termination": result.termination.value,
                    "metrics": dict(result.metrics),
                    "monitor_counts": {key: dict(values) for key, values in result.monitor_counts.items()},
                }
                for result in summary.episodes
            ],
            "aggregate": {
                "episodes": len(summary.episodes),
                "return": {"mean": summary.mean_return, "std": summary.std_return},
                "length": {"mean": summary.mean_length, "std": summary.std_length},
                "metrics": _aggregate(summary.mean_metrics, summary.std_metrics),
                "monitor_counts": {
                    key: _aggregate(values, summary.std_monitor_counts[key])
                    for key, values in sorted(summary.mean_monitor_counts.items())
                },
            },
        }
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        self._written = True


def _aggregate(means: Mapping[str, float], standard_deviations: Mapping[str, float]) -> dict[str, dict[str, float]]:
    return {key: {"mean": means[key], "std": standard_deviations[key]} for key in sorted(means)}


def _count_columns(summaries: Mapping[str, Mapping[str, int | float]]) -> dict[str, int | float]:
    return {
        f"count/{quote(monitor_id, safe='')}/{quote(key, safe='')}": value
        for monitor_id, values in sorted(summaries.items())
        for key, value in sorted(values.items())
    }


__all__ = ["SCHEMA_VERSION", "EpisodeCSVWriter", "EvaluationJSONWriter", "LearningCurveCSVWriter"]
