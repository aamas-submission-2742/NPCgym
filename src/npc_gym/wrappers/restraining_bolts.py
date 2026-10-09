"""Reward adjustment and retained DFA features for collections of bolts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np

from npc_gym.bolts import BoltSpec, _finite_reward
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.automaton import AutomatonMonitor
from npc_gym.monitors.collection import labels_from
from npc_gym.monitors.contracts import validate_monitor_name

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class BoltMetadata:
    """Feature layout for one bolt, exposed through immutable wrapper metadata.

    ``index`` is its position among sorted identifiers. ``state_indices`` maps
    DFA states to zero-based local feature indices. ``feature_slice`` selects
    this bolt's features in the chosen encoding (one index or a one-hot vector).
    """

    index: int
    state_indices: Mapping[int, int]
    feature_slice: slice

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_indices", MappingProxyType(dict(self.state_indices)))


class RestrainingBoltWrapper(gym.Wrapper[dict[str, Any], ActionT, ObservationT, ActionT]):
    """Add accepting-transition rewards and expose every bolt's retained state.

    Observations contain the untouched wrapped value under ``observation`` and
    concatenated float32 one-hot features under ``automata``. ``state_encoding=
    'index'`` instead supplies int64 local state indices. Bolt identifiers sort
    lexically; ``bolt_metadata`` describes the resulting layout.

    Reset consumes initial labels according to each spec and reports acceptance
    without paying or deferring reward. Detached ``info['restraining_bolts']``
    contains occurrences, retained states, wrapped_reward (None on reset), and
    reward_adjustments (all zero on reset). Iterable labels become frozensets. Place
    labeling/time limits inside and vector auto-reset outside this wrapper.
    Completion or any processing failure requires another successful reset.
    """

    def __init__(
        self,
        env: gym.Env[ObservationT, ActionT],
        *,
        bolts: Mapping[str, BoltSpec],
        state_encoding: Literal["one_hot", "index"] = "one_hot",
    ) -> None:
        # Gymnasium types Wrapper.env with its output observation type.
        super().__init__(cast(gym.Env[dict[str, Any], ActionT], env))
        if state_encoding not in ("one_hot", "index"):
            raise ValueError("state_encoding must be 'one_hot' or 'index'")
        if not isinstance(bolts, Mapping):
            raise TypeError("bolts must be a mapping of identifiers to BoltSpec values")
        for bolt_id, spec in bolts.items():
            validate_monitor_name(bolt_id)
            if not isinstance(spec, BoltSpec):
                raise TypeError(f"Bolt {bolt_id!r} must be a BoltSpec")
        current = env
        while isinstance(current, gym.Wrapper):
            if isinstance(current, RestrainingBoltWrapper):
                raise TypeError("Use one RestrainingBoltWrapper with a collection of bolts per environment")
            current = current.env
        self._specs = {key: bolts[key] for key in sorted(bolts)}
        self._monitors = {
            key: AutomatonMonitor(
                spec.definition,
                propositions=spec.propositions,
                reporting=spec.reporting,
                consume_initial=spec.consume_initial,
            )
            for key, spec in self._specs.items()
        }
        layout: dict[str, BoltMetadata] = {}
        offset = 0
        for index, (key, spec) in enumerate(self._specs.items()):
            indices = {state: position for position, state in enumerate(sorted(spec.definition.transitions))}
            width = len(indices) if state_encoding == "one_hot" else 1
            layout[key] = BoltMetadata(index, indices, slice(offset, offset + width))
            offset += width
        self._bolt_metadata = MappingProxyType(layout)
        self._encoding = state_encoding
        self._feature_size = offset
        feature_space: gym.Space[Any]
        if state_encoding == "one_hot":
            feature_space = gym.spaces.Box(0.0, 1.0, shape=(offset,), dtype=np.float32)
        else:
            feature_space = gym.spaces.MultiDiscrete(
                np.array([len(spec.definition.transitions) for spec in self._specs.values()], dtype=np.int64)
            )
        self.observation_space = gym.spaces.Dict({"observation": env.observation_space, "automata": feature_space})
        self._ready = False

    @property
    def bolt_metadata(self) -> Mapping[str, BoltMetadata]:
        """Immutable feature layout keyed by sorted bolt identifiers."""
        return self._bolt_metadata

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Clear histories, optionally consume initial labels, and pay no reward."""
        self._ready = False
        observation, info = cast(gym.Env[ObservationT, ActionT], self.env).reset(seed=seed, options=options)
        result = self._process(info, reset=True)
        augmented = self._observation(observation)
        self._ready = True
        return augmented, result

    def step(self, action: ActionT) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance each DFA once and pay for every accepting destination."""
        if not self._ready:
            raise RuntimeError("RestrainingBoltWrapper.step() requires a successful reset after completion or failure")
        self._ready = False
        observation, reward, terminated, truncated, info = cast(gym.Env[ObservationT, ActionT], self.env).step(action)
        wrapped_reward = _finite_reward(reward, "Wrapped reward")
        result = self._process(
            info, reset=False, terminated=terminated, truncated=truncated, wrapped_reward=wrapped_reward
        )
        try:
            adjusted_reward = math.fsum([wrapped_reward, *result["restraining_bolts"]["reward_adjustments"].values()])
        except OverflowError as error:
            raise ValueError("Adjusted reward must be finite") from error
        if not math.isfinite(adjusted_reward):
            raise ValueError("Adjusted reward must be finite")
        augmented = self._observation(observation)
        self._ready = not (terminated or truncated)
        return augmented, adjusted_reward, terminated, truncated, result

    def _process(
        self,
        info: Mapping[str, Any],
        *,
        reset: bool,
        terminated: bool = False,
        truncated: bool = False,
        wrapped_reward: float | None = None,
    ) -> dict[str, Any]:
        phase = "reset" if reset else "step"
        if not isinstance(info, Mapping):
            raise TypeError(f"Environment {phase} info must be a mapping")
        if "restraining_bolts" in info:
            raise ValueError("RestrainingBoltWrapper requires an unused info['restraining_bolts'] namespace")
        labels = labels_from(info, phase)
        input = MonitorInput(labels, terminated=terminated, truncated=truncated)
        occurrences: dict[str, bool] = {}
        adjustments: dict[str, float] = {}
        for key, monitor in self._monitors.items():
            occurred = monitor.reset(input) if reset else monitor.update(input)
            occurrences[key] = occurred
            adjustments[key] = self._specs[key].reward if occurred and not reset else 0.0
        return {
            **info,
            "labels": labels,
            "restraining_bolts": {
                "occurrences": occurrences,
                "states": {key: monitor.state for key, monitor in self._monitors.items()},
                "wrapped_reward": wrapped_reward,
                "reward_adjustments": adjustments,
            },
        }

    def _observation(self, observation: ObservationT) -> dict[str, Any]:
        if self._encoding == "index":
            features = np.array(
                [self._bolt_metadata[key].state_indices[monitor.state] for key, monitor in self._monitors.items()],
                dtype=np.int64,
            )
        else:
            features = np.zeros(self._feature_size, dtype=np.float32)
            for key, monitor in self._monitors.items():
                metadata = self._bolt_metadata[key]
                features[metadata.feature_slice.start + metadata.state_indices[monitor.state]] = 1.0
        return {"observation": observation, "automata": features}


__all__ = ["BoltMetadata", "RestrainingBoltWrapper"]
