"""Immutable deterministic automata and their stateful runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from enum import StrEnum
from functools import wraps
from typing import Any, cast

Label = Hashable
State = Hashable


class Lifecycle(StrEnum):
    """How a runner retains state after reaching a final state."""

    PUNCTUAL = "punctual"
    ACHIEVEMENT = "achievement"
    MAINTENANCE = "maintenance"
    HELPER = "helper"

    def retained_state(self, previous: State, reached: State, initial: State, finals: frozenset[State]) -> State:
        if reached not in finals:
            return reached
        if self is Lifecycle.MAINTENANCE:
            return previous
        if self in {Lifecycle.PUNCTUAL, Lifecycle.ACHIEVEMENT}:
            return initial
        return reached


class AutomatonDefinition(ABC):
    """Immutable transition graph shared by one or more runners."""

    _sealed = False
    _initializing = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        initializer = cls.__dict__.get("__init__")
        if initializer is None:
            return

        @wraps(initializer)
        def initialize_and_seal(self: AutomatonDefinition, *args: Any, **inner_kwargs: Any) -> None:
            # A subclass initializer runs inside its parent's, so seal only once the outermost returns.
            object.__setattr__(self, "_initializing", self._initializing + 1)
            try:
                initializer(self, *args, **inner_kwargs)
            finally:
                object.__setattr__(self, "_initializing", self._initializing - 1)
            if not self._initializing:
                self._seal()

        cls.__init__ = initialize_and_seal  # type: ignore[method-assign]

    def __init__(
        self,
        alphabet: Iterable[Label],
        reward: float = 0,
        reset: Lifecycle | str = Lifecycle.PUNCTUAL,
        permission: bool = False,
    ) -> None:
        self.alphabet = frozenset(alphabet)
        self.reward = reward
        self.state0: State = 0
        self.states: Iterable[State] = ()
        self.final: Iterable[State] = ()
        self.type = Lifecycle(reset)
        self.permission = permission
        self.violation_definition: AutomatonDefinition | None = None
        self.reset_on_violation = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._sealed:
            raise AttributeError("automaton definitions are immutable")
        object.__setattr__(self, name, value)

    @property
    def initial_state(self) -> State:
        return self.state0

    @property
    def final_states(self) -> frozenset[State]:
        # _seal freezes final states before a completed definition is exposed.
        return cast(frozenset[State], self.final)

    @property
    def lifecycle(self) -> Lifecycle:
        return self.type

    @property
    def state(self) -> State:
        """Legacy default used by the hand-authored transition tables."""
        return self.initial_state

    def _seal(self) -> None:
        legacy_violation = getattr(self, "violDFA", None)
        if legacy_violation is not None:
            object.__setattr__(self, "violation_definition", legacy_violation)
            object.__delattr__(self, "violDFA")
        states = tuple(self.states)
        finals = frozenset(self.final)
        if self.state0 not in states:
            raise ValueError("initial state must be in states")
        if not finals <= set(states):
            raise ValueError("final states must be a subset of states")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "final", finals)
        object.__setattr__(self, "_sealed", True)

    @abstractmethod
    def transition(self, labels: frozenset[Label], state: State | None = None) -> State:
        """Return the state reached from ``state`` after consuming ``labels``."""


class AutomatonRunner:
    """Mutable execution state for an immutable automaton definition."""

    def __init__(self, definition: AutomatonDefinition) -> None:
        self.definition = definition
        self.state = definition.initial_state
        self.violation_runner = (
            AutomatonRunner(definition.violation_definition) if definition.violation_definition is not None else None
        )

    def step(self, labels: Iterable[Label]) -> State:
        """Consume one label set and return the reached state before lifecycle reset."""
        label_set = frozenset(labels)
        previous = self.state
        violation_reached = self.violation_runner.step(label_set) if self.violation_runner is not None else None
        reached = self.definition.transition(label_set, previous)
        if reached not in self.definition.states:
            raise ValueError(f"transition returned undeclared state {reached!r}")
        self.state = self.definition.lifecycle.retained_state(
            previous, reached, self.definition.initial_state, self.definition.final_states
        )
        if (
            self.definition.reset_on_violation
            and self.violation_runner is not None
            and (
                reached in self.definition.final_states
                or violation_reached in self.violation_runner.definition.final_states
            )
        ):
            self.state = self.definition.initial_state
            self.violation_runner.reset()
        return reached

    def reset(self) -> State:
        """Return this runner to its definition's initial state."""
        self.state = self.definition.initial_state
        if self.violation_runner is not None:
            self.violation_runner.reset()
        return self.state


# Short historical name retained for downstream definitions.
DFA = AutomatonDefinition

__all__ = ["DFA", "AutomatonDefinition", "AutomatonRunner", "Label", "Lifecycle", "State"]
