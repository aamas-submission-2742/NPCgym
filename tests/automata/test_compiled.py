import pytest

from npc_gym.automata import CompiledDFA
from npc_gym.automata.automaton import AutomatonRunner


def test_definition_copies_containers_and_is_immutable():
    atoms = ["a"]
    edges = [["0", 0], ["1", 1]]
    transitions = {0: edges, 1: [("X", 1)]}
    definition = CompiledDFA(atoms, 0, frozenset({1}), transitions)
    atoms.clear()
    edges[1][1] = 0
    transitions.clear()
    assert AutomatonRunner(definition).step({"a"}) == 1
    with pytest.raises(AttributeError, match="immutable"):
        definition.atoms = ()
    with pytest.raises(TypeError):
        definition.transitions[0] = ()


@pytest.mark.parametrize(
    "atoms,initial,finals,transitions",
    [
        (["a", "a"], 0, set(), {0: [("XX", 0)]}),
        (["a"], True, set(), {0: [("X", 0)]}),
        (["a"], 0.0, set(), {0: [("X", 0)]}),
        (["a"], 0, {False}, {0: [("X", 0)]}),
        (["a"], 0, {0.0}, {0: [("X", 0)]}),
        (["a"], 1, set(), {0: [("X", 0)]}),
        (["a"], 0, {2}, {0: [("X", 0)]}),
        (["a"], 0, set(), {0: [("X", 1)]}),
        (["a"], 0, set(), {0: [("0", 0)]}),
        (["a"], 0, set(), {0: [("X", 0), ("0", 0)]}),
        (["a"], 0, set(), {0: [("XX", 0)]}),
        (["a"], 0, set(), {0: [("?", 0)]}),
    ],
)
def test_invalid_automata(atoms, initial, finals, transitions):
    with pytest.raises(ValueError):
        CompiledDFA(atoms, initial, frozenset(finals), transitions)
