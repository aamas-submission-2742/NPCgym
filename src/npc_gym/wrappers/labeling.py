"""Transition labeling for external Gymnasium environments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Generic, Literal, SupportsFloat, TypeAlias, TypeVar, cast

import gymnasium as gym

from npc_gym.labels import LabelingFunction, LabelSet, Transition

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
StateT = TypeVar("StateT")

StateExtractor: TypeAlias = Callable[[ObservationT, Mapping[str, Any]], StateT]
LabelMode: TypeAlias = Literal["extend", "replace"]

_UNINITIALIZED = object()


class LabelingWrapper(
    gym.Wrapper[ObservationT, ActionT, ObservationT, ActionT],
    Generic[ObservationT, ActionT, StateT],
):
    """Add transition labels to one ordinary Gymnasium environment.

    By default, states are detached copies of observations and generated labels
    extend an existing ``info["labels"]`` frozen set. A custom ``state_extractor``
    receives the observation and public information returned by the wrapped
    environment and must itself return a detached snapshot.

    Place this wrapper outside :class:`gymnasium.wrappers.TimeLimit` when the
    labeler needs to observe its truncation flag, and place it inside any vector
    environment. The transition action is the action presented at this wrapper's
    boundary, before an inner wrapper may remap it.
    """

    def __init__(
        self,
        env: gym.Env[ObservationT, ActionT],
        labeling_function: LabelingFunction[StateT, ActionT],
        *,
        state_extractor: StateExtractor[ObservationT, StateT] | None = None,
        mode: LabelMode = "extend",
    ) -> None:
        super().__init__(env)
        if not callable(labeling_function):
            raise TypeError("labeling_function must be callable")
        if state_extractor is not None and not callable(state_extractor):
            raise TypeError("state_extractor must be callable")
        if mode not in ("extend", "replace"):
            raise ValueError(f"mode must be 'extend' or 'replace'; got {mode!r}")
        self._labeling_function = labeling_function
        self._state_extractor = state_extractor
        self._mode: LabelMode = mode
        self._previous_state: StateT | object = _UNINITIALIZED

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationT, dict[str, Any]]:
        """Reset the environment and label its initial state."""
        self._previous_state = _UNINITIALIZED
        observation, info = self.env.reset(seed=seed, options=options)
        state = self._extract_state(observation, info)
        transition: Transition[StateT, ActionT] = Transition(
            previous_state=None,
            action=None,
            state=state,
            terminated=False,
            truncated=False,
        )
        labeled_info = self._labeled_info(info, transition, "reset")
        self._previous_state = state
        return observation, labeled_info

    def step(self, action: ActionT) -> tuple[ObservationT, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment and label the resulting transition."""
        if self._previous_state is _UNINITIALIZED:
            raise RuntimeError("LabelingWrapper.step() called before reset(); reset the environment first")
        previous_state = cast(StateT, self._previous_state)
        transition_action = deepcopy(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        state = self._extract_state(observation, info)
        transition: Transition[StateT, ActionT] = Transition(
            previous_state=previous_state,
            action=transition_action,
            state=state,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        labeled_info = self._labeled_info(info, transition, "step")
        self._previous_state = state
        return observation, reward, terminated, truncated, labeled_info

    def _extract_state(self, observation: ObservationT, info: Mapping[str, Any]) -> StateT:
        if self._state_extractor is None:
            return cast(StateT, deepcopy(observation))
        return self._state_extractor(observation, info)

    def _labeled_info(
        self,
        info: Mapping[str, Any],
        transition: Transition[StateT, ActionT],
        phase: str,
    ) -> dict[str, Any]:
        labels = _require_label_set(self._labeling_function(transition), f"labeling function output on {phase}")
        result = dict(info)
        if self._mode == "extend":
            existing = _require_label_set(result.get("labels", frozenset()), f"existing info['labels'] on {phase}")
            labels = existing | labels
        result["labels"] = labels
        return result


def _require_label_set(value: object, source: str) -> LabelSet:
    if not isinstance(value, frozenset):
        raise TypeError(f"{source} must be a frozenset of hashable labels, got {type(value).__name__}")
    return value


__all__ = ["LabelMode", "LabelingWrapper", "StateExtractor"]
