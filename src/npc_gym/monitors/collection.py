"""Shared monitor consumer operations for wrappers and episode recorders.

Factories must create fresh graphs for each consumer. Occurrences are Boolean
for individual events and named Boolean mappings for MultiMonitors. Counts are
always nested mappings; individual events have the single count key ``count``.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from threading import Lock
from typing import Any, cast
from weakref import WeakValueDictionary

from npc_gym.labels import LabelSet
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.events import ComplexMonitor, Monitor
from npc_gym.monitors.registry import MonitorFactory

_claimed_monitors: WeakValueDictionary[int, Monitor] = WeakValueDictionary()
_claim_lock = Lock()


def validate_monitor_factories(factories: Mapping[str, MonitorFactory]) -> dict[str, MonitorFactory]:
    """Check configured identifiers and factories without creating monitors."""
    if not isinstance(factories, Mapping):
        raise TypeError("monitors must be a mapping of identifiers to factories")
    validated: dict[str, MonitorFactory] = {}
    for monitor_id, factory in factories.items():
        if not isinstance(monitor_id, str) or not monitor_id:
            raise ValueError(f"Monitor IDs must be non-empty strings; got {monitor_id!r}")
        if not callable(factory):
            raise TypeError(f"Monitor factory for {monitor_id!r} must be callable")
        validated[monitor_id] = factory
    return validated


def make_monitors(factories: Mapping[str, MonitorFactory]) -> dict[str, Monitor]:
    """Create one fresh monitor per validated factory."""
    monitors: dict[str, Monitor] = {}
    for monitor_id, factory in factories.items():
        monitor = factory()
        if not isinstance(monitor, Monitor):
            raise TypeError(f"Monitor factory for {monitor_id!r} returned {type(monitor).__name__}, not a Monitor")
        if any(monitor is existing for existing in monitors.values()):
            raise ValueError("Monitor factories must return fresh, independent instances")
        monitors[monitor_id] = monitor
    with _claim_lock:
        if any(id(monitor) in _claimed_monitors for monitor in monitors.values()):
            raise ValueError("Monitor factories must return fresh instances for each consumer")
        for monitor in monitors.values():
            _claimed_monitors[id(monitor)] = monitor
    return monitors


def labels_from(info: Mapping[str, Any], phase: str) -> LabelSet:
    """Freeze transition labels for all monitoring consumers; reject bare text."""
    if "labels" not in info:
        raise RuntimeError(f"Monitored environments must provide info['labels'] on {phase}()")
    if isinstance(info["labels"], (str, bytes, bytearray)):
        raise TypeError(
            f"Environment {phase} labels must be an iterable of hashable values, not a bare string or bytes"
        )
    try:
        return frozenset(info["labels"])
    except TypeError as error:
        raise TypeError(f"Environment {phase} labels must be an iterable of hashable values") from error


def advance_monitors(
    monitors: Mapping[str, Monitor],
    input: MonitorInput,
    *,
    reset: bool,
) -> dict[str, bool | dict[str, bool]]:
    """Return named occurrences without maintaining a second set of counts."""
    result: dict[str, bool | dict[str, bool]] = {}
    for name, monitor in monitors.items():
        occurred = monitor.reset(input) if reset else monitor.update(input)
        if isinstance(monitor, ComplexMonitor):
            if not isinstance(occurred, bool):
                raise TypeError("a monitor must return bool")
            result[name] = occurred
        else:
            result[name] = dict(cast(Mapping[str, bool], occurred))
    return result


def monitor_counts(monitors: Mapping[str, Monitor]) -> dict[str, dict[str, int]]:
    """Return detached named counts, using ``count`` for individual events."""
    return copy_monitor_counts(
        {
            name: {"count": monitor.count} if isinstance(monitor, ComplexMonitor) else monitor.counts
            for name, monitor in monitors.items()
        }
    )


def copy_monitor_counts(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    """Validate and detach episode measurements from monitor-owned mappings."""
    if not isinstance(counts, Mapping):
        raise TypeError("monitor_counts must be a mapping")
    copied: dict[str, dict[str, int]] = {}
    for monitor_id, values in counts.items():
        if not isinstance(monitor_id, str) or not monitor_id:
            raise ValueError(f"Monitor count IDs must be non-empty strings; got {monitor_id!r}")
        if not isinstance(values, Mapping):
            raise TypeError(f"Monitor {monitor_id!r} count must be a mapping")
        copied[monitor_id] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"Monitor {monitor_id!r} count key must be a non-empty string; got {key!r}")
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(
                    f"Monitor {monitor_id!r} count field {key!r} must be a nonnegative, "
                    f"non-Boolean integer; got {value!r}"
                )
            copied[monitor_id][key] = int(value)
    return copied


def monitor_count_schema(counts: Mapping[str, Mapping[str, object]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Describe nested keys, including monitors with empty counts."""
    return tuple((monitor_id, tuple(sorted(values))) for monitor_id, values in sorted(counts.items()))


def check_monitor_count_schema(
    expected: Mapping[str, Mapping[str, object]], actual: Mapping[str, Mapping[str, object]]
) -> None:
    """Require stable configured monitors and count fields across episodes."""
    if set(expected) != set(actual):
        raise ValueError(f"Inconsistent monitor_counts monitor IDs: expected {sorted(expected)}, got {sorted(actual)}")
    for monitor_id, values in expected.items():
        if set(values) != set(actual[monitor_id]):
            raise ValueError(
                f"Monitor {monitor_id!r} has inconsistent count fields: "
                f"expected {sorted(values)}, got {sorted(actual[monitor_id])}"
            )


__all__ = [
    "advance_monitors",
    "check_monitor_count_schema",
    "copy_monitor_counts",
    "labels_from",
    "make_monitors",
    "monitor_count_schema",
    "monitor_counts",
    "validate_monitor_factories",
]
