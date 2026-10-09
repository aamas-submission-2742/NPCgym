"""Backend-independent compiled automata for Boolean propositions."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from npc_gym.automata.automaton import AutomatonDefinition, Lifecycle, State
from npc_gym.labels import Label


class CompiledDFA(AutomatonDefinition):
    """An immutable, complete DFA over simultaneous Boolean propositions.

    ``transitions[state]`` contains ``(guard, destination)`` pairs. Each guard
    has one character per atom: ``1`` requires it, ``0`` forbids it, and ``X``
    ignores it. Guards from a state must partition all valuations. Construction
    copies the supplied containers; runners can safely share the definition.
    Acceptance describes a finite trace, without any monitor reporting policy.
    """

    def __init__(
        self,
        atoms: Sequence[str],
        initial: int,
        finals: frozenset[int],
        transitions: Mapping[int, Sequence[tuple[str, int]]],
    ) -> None:
        self.atoms = tuple(atoms)
        if any(not isinstance(atom, str) or not atom for atom in self.atoms):
            raise ValueError("DFA atoms must be nonempty strings")
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("DFA atoms must be unique")
        super().__init__(self.atoms, reset=Lifecycle.HELPER)
        if not transitions or any(type(state) is not int for state in transitions):
            raise ValueError("DFA states must be integers and must not be empty")
        self.states = tuple(sorted(transitions))
        if type(initial) is not int or any(type(state) is not int for state in finals):
            raise ValueError("DFA initial and final states must be non-Boolean integers")
        self.state0 = initial
        self.final = frozenset(finals)
        self.transitions = MappingProxyType(
            {state: tuple((guard, target) for guard, target in edges) for state, edges in transitions.items()}
        )
        for edges in self.transitions.values():
            for index, (guard, target) in enumerate(edges):
                if len(guard) != len(self.atoms) or set(guard) - {"0", "1", "X"}:
                    raise ValueError("DFA guards must contain one 0, 1, or X per atom")
                if type(target) is not int or target not in transitions:
                    raise ValueError("DFA transition targets must be declared states")
                for other, _ in edges[:index]:
                    if all(a == b or "X" in (a, b) for a, b in zip(guard, other, strict=True)):
                        raise ValueError("DFA guards must not overlap")
            if sum(2 ** guard.count("X") for guard, _ in edges) != 2 ** len(self.atoms):
                raise ValueError("DFA transitions must cover all valuations")

    def minimized(self) -> "CompiledDFA":
        """Return a minimal reachable DFA accepting exactly the same finite traces.

        Preserve atom order and leave this definition unchanged. Equivalent
        states merge and unreachable states disappear; the initial state is 0
        and other states receive deterministic consecutive IDs. State IDs and
        bolt observation shapes can change. Guards remain symbolic.
        """
        from npc_gym.automata._minimize import minimize_dfa

        return minimize_dfa(self)

    def transition(self, labels: frozenset[Label], state: State | None = None) -> int:
        current = self.initial_state if state is None else state
        if type(current) is not int or current not in self.transitions:
            raise ValueError(f"Unknown DFA state: {current!r}")
        valuation = tuple("1" if atom in labels else "0" for atom in self.atoms)
        for guard, target in self.transitions[current]:
            if all(required == "X" or required == actual for required, actual in zip(guard, valuation, strict=True)):
                return target
        raise RuntimeError("Validated DFA has no matching transition")
