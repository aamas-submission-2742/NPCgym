"""Compare whole-trace regex languages to independent real LTLf compilation."""

import itertools

import pytest

from npc_gym.automata.automaton import AutomatonRunner
from npc_gym.monitors.regex import compile_regex


@pytest.mark.parametrize(
    "expression,formula",
    [
        ("eps", "false"),
        ("[false]", "false"),
        ("[true]", "!(X true)"),
        ("[a & !b]", "a & !b & !(X true)"),
        ("[a][b]", "a & X(b & !(X true))"),
        ("[a]*", "G a"),
        ("[a]+", "G a"),
        ("[a]?", "a & !(X true)"),
        (".*[a][b]", "F(a & X(b & !(X true)))"),
        ("([a]|[b])*", "G(a | b)"),
    ],
)
def test_regex_matches_ltlf_on_nonempty_traces(expression, formula, compiler):
    regex, ltlf = compile_regex(expression), compiler.compile(formula)
    for length in range(1, 5):
        for word in itertools.product(
            (frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})), repeat=length
        ):
            runners = [AutomatonRunner(definition) for definition in (regex, ltlf)]
            for runner in runners:
                for valuation in word:
                    runner.step(valuation)
            assert (runners[0].state in regex.final_states) == (runners[1].state in ltlf.final_states), word
