"""Shared transition-labeling contracts."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar

Label: TypeAlias = Hashable
LabelSet: TypeAlias = frozenset[Label]

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class Transition(Generic[StateT, ActionT]):
    """The authority-visible state change passed to a labeling function.

    A reset is represented by ``previous_state=None`` and ``action=None``.
    """

    previous_state: StateT | None
    action: ActionT | None
    state: StateT
    terminated: bool
    truncated: bool


class LabelingFunction(Protocol[StateT, ActionT]):
    """Translate an environment transition into normative labels."""

    def __call__(self, transition: Transition[StateT, ActionT]) -> LabelSet:
        """Return all labels that hold for ``transition``."""
        ...


__all__ = ["ActionT", "Label", "LabelSet", "LabelingFunction", "StateT", "Transition"]
