"""Minimization preserves languages, removes unreachable states and merges equivalents."""

from itertools import combinations, product
from random import Random

import pytest

from npc_gym.automata import CompiledDFA
from npc_gym.monitors.builtins import RECIPES


def alphabet(definition):
    return tuple(
        frozenset(atom for atom, bit in zip(definition.atoms, bits, strict=True) if bit)
        for bits in product((False, True), repeat=len(definition.atoms))
    )


def assert_equivalent_and_minimal(original, minimal):
    assert original.atoms == minimal.atoms
    letters = alphabet(original)
    pending = [(original.initial_state, minimal.initial_state)]
    seen = set(pending)
    while pending:
        left, right = pending.pop()
        assert (left in original.final_states) == (right in minimal.final_states)
        for letter in letters:
            pair = (original.transition(letter, left), minimal.transition(letter, right))
            if pair not in seen:
                seen.add(pair)
                pending.append(pair)
    assert {right for _, right in seen} == set(minimal.states)
    # Independent table-filling check: every pair must have a distinguishing suffix.
    pairs = set(combinations(minimal.states, 2))
    distinct = {pair for pair in pairs if (pair[0] in minimal.final_states) != (pair[1] in minimal.final_states)}
    while True:
        new = {
            pair
            for pair in pairs - distinct
            if any(tuple(sorted(minimal.transition(letter, s) for s in pair)) in distinct for letter in letters)
        }
        if not new:
            break
        distinct |= new
    assert distinct == pairs
    assert minimal.minimized() is minimal


@pytest.mark.parametrize("domain,name", [(d, name) for d, recipes in RECIPES.items() for name in recipes])
def test_every_builtin_regex_language_and_minimality(domain, name):
    recipe = RECIPES[domain][name]
    original = recipe.compile(minimize=False)
    minimal = recipe.compile()
    assert len(minimal.states) <= len(original.states)
    assert_equivalent_and_minimal(original, minimal)
    assert recipe.compile() is minimal
    assert recipe.compile(minimize=False) is original


@pytest.mark.parametrize("seed", range(20))
def test_arbitrary_complete_dfas(seed):
    random = Random(seed)
    states = (3, 5, 8, 12, 15, 30)
    guards = ["".join(bits) for bits in product("01", repeat=2)]
    definition = CompiledDFA(
        ("a", "b"),
        12,
        frozenset(s for s in states if random.choice((False, True))),
        {s: [(g, random.choice(states)) for g in guards] for s in states},
    )
    assert_equivalent_and_minimal(definition, definition.minimized())


def test_equivalent_guard_partitions_unreachable_state_and_immutability():
    edges = {
        10: [("0X", 20), ("1X", 30)],
        20: [("X0", 40), ("X1", 50)],
        30: [("00", 40), ("10", 40), ("01", 50), ("11", 50)],
        40: [("XX", 40)],
        50: [("XX", 50)],
        99: [("XX", 99)],
    }
    definition = CompiledDFA(("a", "b"), 10, frozenset({50, 99}), edges)
    minimal = definition.minimized()
    assert len(minimal.states) == 4
    assert minimal.initial_state == 0
    assert definition.initial_state == 10 and len(definition.states) == 6
    assert definition.transitions == {s: tuple(outgoing) for s, outgoing in edges.items()}
    reordered = CompiledDFA(
        ("a", "b"), 10, frozenset({50, 99}), {s: list(reversed(e)) for s, e in reversed(list(edges.items()))}
    )
    assert reordered.minimized().transitions == minimal.transitions
    assert_equivalent_and_minimal(definition, minimal)


@pytest.mark.parametrize("accepting", [False, True])
def test_zero_atoms_and_uniform_acceptance(accepting):
    definition = CompiledDFA((), 4, frozenset({4, 9}) if accepting else frozenset(), {4: [("", 9)], 9: [("", 4)]})
    minimal = definition.minimized()
    assert minimal.states == (0,)
    assert_equivalent_and_minimal(definition, minimal)


def test_large_alphabet_is_not_enumerated():
    atoms = tuple(f"a{i}" for i in range(2000))
    wildcard = "X" * len(atoms)
    definition = CompiledDFA(
        atoms,
        0,
        frozenset({1, 2}),
        {
            0: [(wildcard[:-1] + "0", 1), (wildcard[:-1] + "1", 2)],
            1: [(wildcard, 1)],
            2: [(wildcard, 2)],
        },
    )
    minimal = definition.minimized()
    assert minimal.states == (0, 1)
    assert minimal.transition(frozenset()) in minimal.final_states
    assert minimal.transition(frozenset(atoms)) in minimal.final_states
    branching = CompiledDFA(
        atoms,
        0,
        frozenset({1}),
        {
            0: [(wildcard[:-1] + "0", 0), (wildcard[:-1] + "1", 1)],
            1: [(wildcard, 1)],
        },
    ).minimized()
    assert branching.transition(frozenset()) not in branching.final_states
    assert branching.transition(frozenset(atoms)) in branching.final_states
