"""Bounded Boolean trace-regex compilation through a Thompson NFA."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, fields

from npc_gym.automata import CompiledDFA


class RegexSyntaxError(ValueError):
    """Malformed trace regex, with a zero-based offset and one-based line/column."""

    def __init__(self, message: str, expression: str, position: int) -> None:
        self.position = position
        self.line = expression.count("\n", 0, position) + 1
        self.column = position - expression.rfind("\n", 0, position)
        super().__init__(f"{message} at line {self.line}, column {self.column} (offset {position})")


class RegexCompilationError(RuntimeError):
    """Compilation exceeded a configured size limit or parser recursion capacity."""


@dataclass(frozen=True, slots=True)
class RegexLimits:
    """Positive integer compilation budgets; Boolean partitions use up to 2**max_atoms valuations."""

    max_length: int = 10_000
    max_nesting: int = 64
    max_atoms: int = 10
    max_nfa_states: int = 512
    max_dfa_states: int = 256
    max_transitions: int = 8_192

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive non-Boolean integer")


def _budget(value: int, limit: int, name: str) -> None:
    if value > limit:
        raise RegexCompilationError(f"Regex compilation exceeded {name}={limit}")


# Guards use postfix Boolean tokens; None denotes an epsilon edge.
_Guard = tuple[str, ...]
_Fragment = tuple[int, int]
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\[\]().|*+?!&]")


class _Parser:
    def __init__(self, expression: str, limits: RegexLimits) -> None:
        self.expression = expression
        self.limits = limits
        self.tokens: list[tuple[str, int]] = []
        position = 0
        while position < len(expression):
            if expression[position].isspace():
                position += 1
                continue
            match = _TOKEN.match(expression, position)
            if match is None:
                raise RegexSyntaxError("Unexpected character", expression, position)
            self.tokens.append((match.group(), position))
            position = match.end()
        self.tokens.append(("", len(expression)))
        self.index = 0
        self.depth = 0
        self.atoms: set[str] = set()
        self.edges: list[list[tuple[_Guard | None, int]]] = []

    @property
    def token(self) -> str:
        return self.tokens[self.index][0]

    def error(self, message: str) -> RegexSyntaxError:
        return RegexSyntaxError(message, self.expression, self.tokens[self.index][1])

    def take(self, token: str) -> bool:
        if self.token != token:
            return False
        self.index += 1
        return True

    def expect(self, token: str) -> None:
        if not self.take(token):
            raise self.error(f"Expected {token!r}")

    def enter(self) -> None:
        self.depth += 1
        _budget(self.depth, self.limits.max_nesting, "max_nesting")

    def fragment(self) -> _Fragment:
        start = len(self.edges)
        _budget(start + 2, self.limits.max_nfa_states, "max_nfa_states")
        self.edges.extend(([], []))
        return start, start + 1

    def regex(self) -> _Fragment:
        left = self.concatenation()
        while self.take("|"):
            right = self.concatenation()
            start, end = self.fragment()
            self.edges[start].extend(((None, left[0]), (None, right[0])))
            self.edges[left[1]].append((None, end))
            self.edges[right[1]].append((None, end))
            left = start, end
        return left

    def concatenation(self) -> _Fragment:
        left = self.repetition()
        while self.token in {"(", "[", ".", "eps"}:
            right = self.repetition()
            self.edges[left[1]].append((None, right[0]))
            left = left[0], right[1]
        return left

    def repetition(self) -> _Fragment:
        if self.take("("):
            self.enter()
            part = self.regex()
            self.expect(")")
            self.depth -= 1
        elif self.take("["):
            self.enter()
            guard = self.boolean_or()
            self.expect("]")
            self.depth -= 1
            part = self.fragment()
            self.edges[part[0]].append((tuple(guard), part[1]))
        elif self.take("."):
            part = self.fragment()
            self.edges[part[0]].append((("true",), part[1]))
        elif self.take("eps"):
            part = self.fragment()
            self.edges[part[0]].append((None, part[1]))
        else:
            raise self.error("Expected a test '[...]', '.', 'eps', or a regex group")
        if self.token in {"*", "+", "?"}:
            operator = self.token
            self.index += 1
            start, end = self.fragment()
            self.edges[start].append((None, part[0]))
            self.edges[part[1]].append((None, end))
            if operator in {"*", "?"}:
                self.edges[start].append((None, end))
            if operator in {"*", "+"}:
                self.edges[part[1]].append((None, part[0]))
            part = start, end
            if self.token in {"*", "+", "?"}:
                raise self.error("Group repeated postfix operators explicitly")
        return part

    def boolean_or(self) -> list[str]:
        result = self.boolean_and()
        while self.take("|"):
            result.extend(self.boolean_and())
            result.append("|")
        return result

    def boolean_and(self) -> list[str]:
        result = self.boolean_unary()
        while self.take("&"):
            result.extend(self.boolean_unary())
            result.append("&")
        return result

    def boolean_unary(self) -> list[str]:
        negations = 0
        while self.take("!"):
            negations += 1
        if self.take("("):
            self.enter()
            result = self.boolean_or()
            self.expect(")")
            self.depth -= 1
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.token):
            atom = self.token
            self.index += 1
            if atom not in {"true", "false"}:
                self.atoms.add(atom)
                _budget(len(self.atoms), self.limits.max_atoms, "max_atoms")
            result = [atom]
        else:
            raise self.error("Expected a Boolean identifier, constant, negation, or group")
        if negations % 2:
            result.append("!")
        return result


def _truth(guard: _Guard, atoms: dict[str, int], universe: int) -> int:
    stack: list[int] = []
    for token in guard:
        if token == "!":
            stack[-1] ^= universe
        elif token in {"&", "|"}:
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right if token == "&" else left | right)
        else:
            stack.append(universe if token == "true" else 0 if token == "false" else atoms[token])
    return stack[0]


def _cubes(mask: int, width: int, prefix: str = "") -> Iterator[str]:
    """Partition a valuation bitset into disjoint 0/1/X cubes in lexical order."""
    size = 1 << width
    full = (1 << size) - 1
    if not mask:
        return
    if mask == full:
        yield prefix + "X" * width
        return
    half = size // 2
    yield from _cubes(mask & ((1 << half) - 1), width - 1, prefix + "0")
    yield from _cubes(mask >> half, width - 1, prefix + "1")


def _determinize(parser: _Parser, start: int, end: int) -> CompiledDFA:
    atoms = tuple(sorted(parser.atoms))
    valuations = 1 << len(atoms)
    universe = (1 << valuations) - 1
    atom_masks = {
        atom: sum(1 << value for value in range(valuations) if value & (1 << (len(atoms) - index - 1)))
        for index, atom in enumerate(atoms)
    }
    edges = [
        [(_truth(guard, atom_masks, universe), target) for guard, target in outgoing if guard is not None]
        for outgoing in parser.edges
    ]

    def closure(states: frozenset[int]) -> frozenset[int]:
        reached = set(states)
        pending = list(states)
        while pending:
            for guard, target in parser.edges[pending.pop()]:
                if guard is None and target not in reached:
                    reached.add(target)
                    pending.append(target)
        return frozenset(reached)

    initial = closure(frozenset({start}))
    # Retain an explicit rejecting sink even when it is unreachable (e.g. '.*').
    subsets = [initial, frozenset()]
    _budget(len(subsets), parser.limits.max_dfa_states, "max_dfa_states")
    indices = {subset: index for index, subset in enumerate(subsets)}
    transitions: dict[int, list[tuple[str, int]]] = {}
    finals: set[int] = set()
    transition_count = 0
    for index, subset in enumerate(subsets):
        if end in subset:
            finals.add(index)
        # Each region maps valuations to the union of enabled NFA destinations.
        regions: dict[frozenset[int], int] = {frozenset(): universe}
        for state in sorted(subset):
            for guard, target in edges[state]:
                refined: dict[frozenset[int], int] = {}
                for destinations, mask in regions.items():
                    for targets, part in (
                        (destinations, mask & (universe ^ guard)),
                        (destinations | {target}, mask & guard),
                    ):
                        if part:
                            refined[targets] = refined.get(targets, 0) | part
                regions = refined
        closed: dict[frozenset[int], int] = {}
        for destinations, mask in regions.items():
            targets = closure(destinations)
            closed[targets] = closed.get(targets, 0) | mask
        outgoing: list[tuple[str, int]] = []
        for targets, mask in sorted(closed.items(), key=lambda item: item[1] & -item[1]):
            if targets not in indices:
                _budget(len(subsets) + 1, parser.limits.max_dfa_states, "max_dfa_states")
                indices[targets] = len(subsets)
                subsets.append(targets)
            for cube in _cubes(mask, len(atoms)):
                transition_count += 1
                _budget(transition_count, parser.limits.max_transitions, "max_transitions")
                outgoing.append((cube, indices[targets]))
        transitions[index] = sorted(outgoing)
    return CompiledDFA(atoms, 0, frozenset(finals), transitions)


def compile_regex(expression: str, *, limits: RegexLimits | None = None, minimize: bool = True) -> CompiledDFA:
    """Compile a whole-trace regex into an immutable complete Boolean DFA.

    Positions are Boolean tests ``[a & !b]`` or wildcard ``.``. Regex operators
    are grouping, alternation, concatenation, and postfix ``*``, ``+``, ``?``;
    ``eps`` accepts the empty word. See the trace-regex grammar in the monitor
    guide. Minimize by default; ``minimize=False`` retains subset-construction
    state IDs and the explicit sink. Limits apply before minimization.
    No proposition bindings or mutable execution state are compiled.
    """
    if not isinstance(minimize, bool):
        raise TypeError("minimize must be a bool")
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    if limits is None:
        limits = RegexLimits()
    if not isinstance(limits, RegexLimits):
        raise TypeError("limits must be a RegexLimits instance")
    _budget(len(expression), limits.max_length, "max_length")
    parser = _Parser(expression, limits)
    try:
        start, end = parser.regex()
    except RecursionError as error:
        raise RegexCompilationError("Regex parser recursion capacity exceeded; reduce expression nesting") from error
    if parser.token:
        raise parser.error("Unexpected token after regex")
    definition = _determinize(parser, start, end)
    return definition.minimized() if minimize else definition
