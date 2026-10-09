"""Observation-independent monitoring for ordinary Gymnasium environments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym

from npc_gym.monitors import MonitorFactory, MonitorInput
from npc_gym.monitors.collection import (
    advance_monitors,
    check_monitor_count_schema,
    copy_monitor_counts,
    labels_from,
    make_monitors,
    monitor_counts,
    validate_monitor_factories,
)

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class MonitorWrapper(gym.Wrapper[ObservationT, ActionT, ObservationT, ActionT]):
    """Publish named Boolean occurrences and count snapshots in info['monitors'].

    Factories create fresh monitor graphs. Labels are frozen once per input;
    observations and rewards pass through. Place labeling/time limits inside
    and vector auto-reset outside. Completion or processing failure requires
    another successful reset. Reset follows each monitor's initial-input policy.
    """

    def __init__(self, env: gym.Env[ObservationT, ActionT], *, monitors: Mapping[str, MonitorFactory]) -> None:
        super().__init__(env)
        current = env
        while isinstance(current, gym.Wrapper):
            if isinstance(current, MonitorWrapper):
                raise TypeError("Use one MonitorWrapper with a collection of monitors per environment")
            current = current.env
        configured = make_monitors(validate_monitor_factories(monitors))
        self._monitors = configured
        self._ready = False
        self._schema: dict[str, dict[str, int]] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationT, dict[str, Any]]:
        """Reset the environment and every monitor, counting initial events."""
        self._ready = False
        observation, info = self.env.reset(seed=seed, options=options)
        result = self._process(info, reset=True)
        self._ready = True
        return observation, result

    def step(self, action: ActionT) -> tuple[ObservationT, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process one transition, including the final transition exactly once."""
        if not self._ready:
            raise RuntimeError("MonitorWrapper.step() requires a successful reset after completion or failure")
        self._ready = False
        observation, reward, terminated, truncated, info = self.env.step(action)
        result = self._process(info, reset=False, terminated=terminated, truncated=truncated)
        self._ready = not (terminated or truncated)
        return observation, reward, terminated, truncated, result

    def _process(
        self, info: Mapping[str, Any], *, reset: bool, terminated: bool = False, truncated: bool = False
    ) -> dict[str, Any]:
        phase = "reset" if reset else "step"
        if not isinstance(info, Mapping):
            raise TypeError(f"Environment {phase} info must be a mapping")
        if "monitors" in info:
            raise ValueError("MonitorWrapper requires an unused info['monitors'] namespace")
        labels = labels_from(info, phase)
        occurrences = advance_monitors(
            self._monitors,
            MonitorInput(labels, terminated=terminated, truncated=truncated),
            reset=reset,
        )
        counts = monitor_counts(self._monitors)
        if self._schema is not None:
            check_monitor_count_schema(self._schema, counts)
        # Keep validation state separate from the caller's mutable diagnostics.
        self._schema = copy_monitor_counts(counts)
        return {**info, "labels": labels, "monitors": {"occurrences": occurrences, "counts": counts}}


__all__ = ["MonitorWrapper"]
