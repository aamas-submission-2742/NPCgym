"""Boolean event counters, compositions, and named count collections."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from numbers import Integral
from types import MappingProxyType

from npc_gym.monitors.contracts import MonitorInput, validate_monitor_name


def _graph(roots: tuple["ComplexMonitor", ...], owner: object) -> tuple["ComplexMonitor", ...]:
    ordered: list[ComplexMonitor] = []
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: ComplexMonitor) -> None:
        if id(node) in visiting:
            raise ValueError("monitor graph contains a cycle")
        if id(node) in visited:
            return
        if node._owner is not None and node._owner is not owner:
            raise ValueError("monitor already belongs to another live root; create fresh monitors")
        visiting.add(id(node))
        for child in node._children:
            visit(child)
        visiting.remove(id(node))
        visited.add(id(node))
        ordered.append(node)

    for root in roots:
        visit(root)
    # Claim only after validating the entire graph.
    for node in ordered:
        node._owner = owner
    return tuple(ordered)


def _advance(nodes: tuple["ComplexMonitor", ...], input: MonitorInput, *, reset: bool) -> dict[int, bool]:
    if not isinstance(input, MonitorInput):
        raise TypeError("input must be a MonitorInput")
    if reset:
        for node in nodes:
            node._count = 0
            node.reset_history()
    occurrences: dict[int, bool] = {}
    for node in nodes:
        occurred = False if reset and not node.consume_initial else node._detect(input, occurrences)
        if not isinstance(occurred, bool):
            raise TypeError("a monitor detector must return bool")
        occurrences[id(node)] = occurred
        node._count += int(occurred)
    return occurrences


class ComplexMonitor(ABC):
    """Count one Boolean event per input; combine occurrences with &, |, and ~.

    A graph is claimed by its root on first reset/update. Shared dependencies
    advance once, including operands that cannot affect the expression result.
    Owned children can be inspected but cannot be advanced independently.
    """

    def __init__(self, *, consume_initial: bool = True) -> None:
        if not isinstance(consume_initial, bool):
            raise TypeError("consume_initial must be bool")
        self._consume_initial = consume_initial
        self._count = 0
        self._owner: object | None = None
        self._children: tuple[ComplexMonitor, ...] = ()

    @property
    def consume_initial(self) -> bool:
        return self._consume_initial

    @property
    def count(self) -> int:
        """Cumulative event count since the last episode reset."""
        return self._count

    def reset(self, input: MonitorInput) -> bool:
        """Clear all dependency history/counts, optionally consuming initial labels."""
        return _advance(_graph((self,), self), input, reset=True)[id(self)]

    def update(self, input: MonitorInput) -> bool:
        """Advance every dependency once and return this event's occurrence."""
        return _advance(_graph((self,), self), input, reset=False)[id(self)]

    def reset_history(self) -> None:
        """Detector hook: clear history without changing counts or advancing children.

        Called by episode reset before optional initial-input detection.
        Stateful SimpleMonitor subclasses override this alongside detect.
        """

    @abstractmethod
    def _detect(self, input: MonitorInput, occurrences: Mapping[int, bool]) -> bool:
        """Recognize the current event after dependencies have advanced."""

    def __and__(self, other: "ComplexMonitor") -> "ComplexMonitor":
        return _Expression("and", self, other)

    def __or__(self, other: "ComplexMonitor") -> "ComplexMonitor":
        return _Expression("or", self, other)

    def __invert__(self) -> "ComplexMonitor":
        return _Expression("not", self)


class SimpleMonitor(ComplexMonitor):
    """A single event detector; override detect and optionally reset_history.

    Detectors return a bool, never a multiplicity. The runtime owns counting.
    Built-in detectors are generated from temporal expressions.
    """

    @abstractmethod
    def detect(self, input: MonitorInput) -> bool:
        """Consume one input and return whether the event occurred."""

    def _detect(self, input: MonitorInput, occurrences: Mapping[int, bool]) -> bool:
        return self.detect(input)


class _Expression(ComplexMonitor):
    def __init__(self, operation: str, *children: ComplexMonitor) -> None:
        if any(not isinstance(child, ComplexMonitor) for child in children):
            raise TypeError("Boolean operands must be ComplexMonitors")
        if len({child.consume_initial for child in children}) != 1:
            raise ValueError("Boolean operands must use the same initial-input convention")
        super().__init__(consume_initial=children[0].consume_initial)
        self._children = children
        self._operation = operation

    def _detect(self, input: MonitorInput, occurrences: Mapping[int, bool]) -> bool:
        values = [occurrences[id(child)] for child in self._children]
        if self._operation == "not":
            return not values[0]
        return all(values) if self._operation == "and" else any(values)


class MultiMonitor:
    """Named ComplexMonitors and pure derived counts, with no implicit total.

    Members must be distinct instances; their dependency graphs may overlap.
    Derived functions receive an immutable snapshot of member counts only and
    must return nonnegative, non-Boolean integers without retaining state.
    """

    def __init__(
        self,
        members: Mapping[str, ComplexMonitor],
        *,
        derived: Mapping[str, Callable[[Mapping[str, int]], int]] | None = None,
    ) -> None:
        for name, member in members.items():
            validate_monitor_name(name)
            if not isinstance(member, ComplexMonitor):
                raise TypeError("MultiMonitor members must be ComplexMonitors")
        if len({id(member) for member in members.values()}) != len(members):
            raise ValueError("MultiMonitor members must be distinct instances; repeated member instance")
        functions = dict(derived or {})
        for name, function in functions.items():
            validate_monitor_name(name)
            if name in members:
                raise ValueError(f"duplicate count name: {name!r}")
            if not callable(function):
                raise TypeError("derived counts must be callable")
        self._members = MappingProxyType(dict(members))
        self._derived = functions

    @property
    def members(self) -> Mapping[str, ComplexMonitor]:
        """Read-only named members, available for count/state inspection."""
        return self._members

    @property
    def counts(self) -> Mapping[str, int]:
        """Immutable snapshot of member and derived counts."""
        base = {name: member.count for name, member in self._members.items()}
        snapshot = MappingProxyType(base)
        result = dict(base)
        for name, function in self._derived.items():
            value = function(snapshot)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"derived count {name!r} must be a non-Boolean integer")
            if value < 0:
                raise ValueError(f"derived count {name!r} must be nonnegative")
            result[name] = int(value)
        return MappingProxyType(result)

    def reset(self, input: MonitorInput) -> Mapping[str, bool]:
        """Clear member graphs and return named initial occurrences."""
        return self._advance(input, reset=True)

    def update(self, input: MonitorInput) -> Mapping[str, bool]:
        """Advance shared dependencies once and return named occurrences."""
        return self._advance(input, reset=False)

    def _advance(self, input: MonitorInput, *, reset: bool) -> Mapping[str, bool]:
        nodes = _graph(tuple(self._members.values()), self)
        occurrences = _advance(nodes, input, reset=reset)
        return MappingProxyType({name: occurrences[id(member)] for name, member in self._members.items()})


Monitor = ComplexMonitor | MultiMonitor
