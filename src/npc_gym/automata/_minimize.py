"""Reachability and symbolic partition refinement for complete Boolean DFAs."""

from collections import deque
from typing import cast

from npc_gym.automata.compiled import CompiledDFA

_Edges = tuple[tuple[str, int], ...]
_Region = tuple[int, _Edges]


class _Signatures:
    """Canonical decision diagrams of the destination block for each valuation.

    Equal functions get equal IDs even when their guard partitions differ.
    Negative IDs are block terminals; nonnegative IDs are shared decision nodes.
    Wildcards stay symbolic, avoiding enumeration of all Boolean valuations.
    """

    def __init__(self) -> None:
        self._regions: dict[_Region, int] = {}
        self._nodes: dict[tuple[int, int, int], int] = {}

    def __call__(self, edges: _Edges) -> int:
        root = (0, tuple(sorted(edges)))
        pending = [root]
        while pending:
            region = pending[-1]
            if region in self._regions:
                pending.pop()
                continue
            offset, outgoing = region
            targets = {target for _, target in outgoing}
            if len(targets) == 1:
                self._regions[region] = -1 - next(iter(targets))
                continue
            # Skip variables on which this whole region is independent.
            index = min(i for guard, _ in outgoing for i, char in enumerate(guard) if char != "X")
            children = tuple(
                (offset + index + 1, tuple((g[index + 1 :], t) for g, t in outgoing if g[index] in (value, "X")))
                for value in "01"
            )
            missing = [child for child in children if child not in self._regions]
            if missing:
                pending.extend(missing)
                continue
            low, high = (self._regions[child] for child in children)
            self._regions[region] = (
                low if low == high else self._nodes.setdefault((offset + index, low, high), len(self._nodes))
            )
        return self._regions[root]


def minimize_dfa(definition: CompiledDFA) -> CompiledDFA:
    """Remove unreachable states and merge states accepting the same suffixes."""
    initial_state = cast(int, definition.initial_state)
    reachable = {initial_state}
    pending = [initial_state]
    while pending:
        for _, target in definition.transitions[pending.pop()]:
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    states = sorted(reachable)
    blocks = {state: int(state in definition.final_states) for state in states}
    while True:
        signatures = _Signatures()
        indices: dict[tuple[int, int], int] = {}
        refined = {}
        for state in states:
            edges = tuple((guard, blocks[target]) for guard, target in definition.transitions[state])
            signature = (blocks[state], signatures(edges))
            refined[state] = indices.setdefault(signature, len(indices))
        stable = len(indices) == len(set(blocks.values()))
        blocks = refined
        if stable:
            break

    representatives: dict[int, int] = {}
    for state in states:
        representatives.setdefault(blocks[state], state)
    initial = blocks[initial_state]
    numbering = {initial: 0}
    queue = deque([initial])
    transitions: dict[int, list[tuple[str, int]]] = {}
    finals = set()
    while queue:
        block = queue.popleft()
        state = representatives[block]
        if state in definition.final_states:
            finals.add(numbering[block])
        outgoing = []
        for guard, target in sorted(definition.transitions[state]):
            destination = blocks[target]
            if destination not in numbering:
                numbering[destination] = len(numbering)
                queue.append(destination)
            outgoing.append((guard, numbering[destination]))
        transitions[numbering[block]] = outgoing
    frozen_edges = {state: tuple(edges) for state, edges in transitions.items()}
    if definition.initial_state == 0 and definition.final_states == finals and definition.transitions == frozen_edges:
        return definition
    return CompiledDFA(definition.atoms, 0, frozenset(finals), transitions)
