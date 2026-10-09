"""Simple monitors generated from finite-trace automata."""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import cast

from npc_gym.automata import CompiledDFA
from npc_gym.automata.automaton import AutomatonRunner
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.events import SimpleMonitor

Proposition = Callable[[MonitorInput], bool]


class Reporting(StrEnum):
    """Acceptance policy for counting events and retaining DFA history."""

    PREFIX = "prefix"
    RESTART = "restart"
    EPISODE_END = "episode_end"


class AutomatonMonitor(SimpleMonitor):
    """Count accepted inputs with an explicit acceptance policy.

    Prefix counts every accepted prefix. Restart clears DFA history after an
    occurrence without replaying the input or clearing the count. Episode-end
    counts only accepted terminated/truncated inputs. Reset clears history and
    counts, optionally consuming initial labels; no empty trace is counted.
    Proposition bindings must be stateless Boolean predicates.
    """

    def __init__(
        self,
        definition: CompiledDFA,
        *,
        propositions: Mapping[str, Proposition],
        reporting: Reporting | str = Reporting.PREFIX,
        consume_initial: bool = True,
    ) -> None:
        super().__init__(consume_initial=consume_initial)
        self._reporting = Reporting(reporting)
        missing = set(definition.atoms) - propositions.keys()
        if missing:
            raise ValueError(f"Missing proposition bindings: {', '.join(sorted(missing))}")
        self._bindings = {atom: propositions[atom] for atom in definition.atoms}
        if any(not callable(predicate) for predicate in self._bindings.values()):
            raise TypeError("Proposition bindings must be callable")
        self._runner = AutomatonRunner(definition)

    @property
    def state(self) -> int:
        """DFA state retained for the next input, after any acceptance restart."""
        return cast(int, self._runner.state)

    def reset_history(self) -> None:
        self._runner.reset()

    def detect(self, input: MonitorInput) -> bool:
        atoms: set[str] = set()
        for atom, predicate in self._bindings.items():
            value = predicate(input)
            if not isinstance(value, bool):
                raise TypeError(f"Proposition {atom!r} must return bool")
            if value:
                atoms.add(atom)
        reached = self._runner.step(atoms)
        if reached not in self._runner.definition.final_states:
            return False
        if self._reporting is Reporting.EPISODE_END and not (input.terminated or input.truncated):
            return False
        if self._reporting is Reporting.RESTART:
            self._runner.reset()
        return True
