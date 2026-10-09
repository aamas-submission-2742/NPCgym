"""Rules shared by everything that records one episode.

The environment-neutral evaluator and the Stable-Baselines3 callback build
``EpisodeResult`` values from the same public information. They read labels,
domain metrics, and counts through this module so the two paths cannot
drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Literal, TypeAlias

from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.monitors import MonitorFactory, MonitorInput
from npc_gym.monitors.collection import (
    advance_monitors,
    check_monitor_count_schema,
    copy_monitor_counts,
    labels_from,
    make_monitors,
    monitor_count_schema,
    monitor_counts,
    validate_monitor_factories,
)

MonitorSource: TypeAlias = Literal["factories", "wrapper"]


def validate_monitor_source(source: MonitorSource, factories: Mapping[str, MonitorFactory] | None) -> None:
    if source not in ("factories", "wrapper"):
        raise ValueError("monitor_source must be 'factories' or 'wrapper'")
    if source == "wrapper" and factories is not None:
        raise ValueError("monitor_source='wrapper' and monitors cannot be selected together")


class MonitorRecorder:
    """Snapshot counts from one selected source, including initial events."""

    def __init__(self, factories: Mapping[str, MonitorFactory], source: MonitorSource) -> None:
        self.monitors = make_monitors(factories)
        self.source = source
        self.counts: dict[str, dict[str, int]] = {}
        self._schema: dict[str, dict[str, int]] | None = None

    def record(
        self, info: Mapping[str, Any], *, reset: bool, terminated: bool = False, truncated: bool = False
    ) -> None:
        if self.source == "wrapper":
            if "monitors" not in info:
                raise RuntimeError("monitor_source='wrapper' requires info['monitors'] on reset and step")
            snapshot = info["monitors"]
            if not isinstance(snapshot, Mapping) or "counts" not in snapshot or "occurrences" not in snapshot:
                raise TypeError("info['monitors'] must contain occurrences and counts")
            counts = copy_monitor_counts(snapshot["counts"])
            occurrences = snapshot["occurrences"]
            if not isinstance(occurrences, Mapping) or set(occurrences) != set(counts):
                raise ValueError("occurrence monitor IDs must match count monitor IDs")
            for name, values in occurrences.items():
                if isinstance(values, bool):
                    if set(counts[name]) != {"count"}:
                        raise ValueError("Boolean occurrences require a single count field named 'count'")
                    continue
                if not isinstance(values, Mapping) or not set(values) <= set(counts[name]):
                    raise ValueError("occurrences must name member counts")
                if any(not isinstance(value, bool) for value in values.values()):
                    raise TypeError("occurrences must be Boolean")
        else:
            if self.monitors:
                advance_monitors(
                    self.monitors,
                    MonitorInput(labels_from(info, "reset" if reset else "step"), terminated, truncated),
                    reset=reset,
                )
            counts = monitor_counts(self.monitors)
        if self._schema is not None:
            check_monitor_count_schema(self._schema, counts)
        self._schema = copy_monitor_counts(counts)
        self.counts = counts


class TerminationClass(StrEnum):
    """How Gymnasium ended an episode."""

    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    TERMINATED_AND_TRUNCATED = "terminated_and_truncated"


def episode_metrics(info: Mapping[str, Any]) -> dict[str, int | float]:
    """Return the cumulative domain snapshot an environment reports, if any."""
    raw_metrics = info.get(EPISODE_METRICS_KEY, {})
    if not isinstance(raw_metrics, Mapping):
        raise TypeError(f"info[{EPISODE_METRICS_KEY!r}] must be a mapping")
    metrics: dict[str, int | float] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Episode metric names must be non-empty strings; got {name!r}")
        if isinstance(value, Integral):
            metrics[name] = int(value)
        elif isinstance(value, Real) and math.isfinite(float(value)):
            metrics[name] = float(value)
        else:
            raise TypeError(f"Episode metric {name!r} must be a finite real number; got {value!r}")
    return metrics


def termination_class(terminated: bool, truncated: bool) -> TerminationClass:
    """Classify how an episode ended, requiring at least one Gymnasium flag."""
    if terminated and truncated:
        return TerminationClass.TERMINATED_AND_TRUNCATED
    if terminated:
        return TerminationClass.TERMINATED
    if truncated:
        return TerminationClass.TRUNCATED
    raise ValueError("An episode result requires termination or truncation")


__all__ = [
    "MonitorRecorder",
    "MonitorSource",
    "TerminationClass",
    "advance_monitors",
    "check_monitor_count_schema",
    "copy_monitor_counts",
    "episode_metrics",
    "labels_from",
    "make_monitors",
    "monitor_count_schema",
    "monitor_counts",
    "termination_class",
    "validate_monitor_factories",
    "validate_monitor_source",
]
