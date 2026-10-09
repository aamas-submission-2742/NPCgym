"""Language checks use a separate expression-tree interpreter, not NFA simulation."""

import itertools
from dataclasses import FrozenInstanceError, replace
from functools import cache

import pytest

from npc_gym.automata.automaton import AutomatonRunner
from npc_gym.monitors.regex import RegexCompilationError, RegexLimits, RegexSyntaxError, compile_regex

VALUATIONS = tuple(frozenset(value) for value in ((), ("a",), ("b",), ("a", "b")))
WORDS = tuple(word for length in range(4) for word in itertools.product(VALUATIONS, repeat=length))
BASES = (
    ("eps",),
    ("test", ".", lambda v: True),
    ("test", "[a]", lambda v: "a" in v),
    ("test", "[!a & b]", lambda v: "a" not in v and "b" in v),
    ("test", "[a | b]", lambda v: "a" in v or "b" in v),
    ("test", "[false]", lambda v: False),
    ("test", "[true]", lambda v: True),
    ("test", "[!(a & b)]", lambda v: not ("a" in v and "b" in v)),
)


def render(tree):
    operator, *children = tree
    if operator == "eps":
        return "eps"
    if operator == "test":
        return children[0]
    if operator in {"*", "+", "?"}:
        return f"({render(children[0])}){operator}"
    return f"({render(children[0])}{operator or ' '}{render(children[1])})"


def interpret(tree, word):
    @cache
    def ends(node, start):
        operator, *children = node
        if operator == "eps":
            return frozenset({start})
        if operator == "test":
            return frozenset({start + 1}) if start < len(word) and children[1](word[start]) else frozenset()
        if operator == "|":
            return ends(children[0], start) | ends(children[1], start)
        if operator == "":
            return frozenset(end for middle in ends(children[0], start) for end in ends(children[1], middle))
        if operator == "?":
            return frozenset({start}) | ends(children[0], start)
        # Reach a fixed point of word positions; nullable children cannot loop forever.
        positions = {start} if operator == "*" else set(ends(children[0], start))
        pending = list(positions)
        while pending:
            for end in ends(children[0], pending.pop()):
                if end not in positions:
                    positions.add(end)
                    pending.append(end)
        return frozenset(positions)

    return len(word) in ends(tree, 0)


def accepts(definition, word):
    runner = AutomatonRunner(definition)
    for valuation in word:
        runner.step(valuation)
    return runner.state in definition.final_states


TREES = (
    list(BASES)
    + [(operator, left, right) for operator in ("|", "") for left in BASES for right in BASES]
    + [(operator, child) for operator in ("*", "+", "?") for child in BASES]
    + [("+", ("*", ("|", BASES[0], child))) for child in BASES]
    + [("", ("*", ("|", BASES[2], BASES[4])), ("", BASES[1], BASES[2]))]
)


@pytest.mark.parametrize("tree", TREES, ids=render)
def test_language_matches_independent_interpreter(tree):
    definition = compile_regex(render(tree))
    for word in WORDS:
        assert accepts(definition, word) == interpret(tree, word), word


@pytest.mark.parametrize(
    "expression,truth",
    [
        (
            "[a]|[b][a]*",
            lambda w: (len(w) == 1 and "a" in w[0]) or bool(w) and "b" in w[0] and all("a" in v for v in w[1:]),
        ),
        ("[!a & b | a & !b]", lambda w: len(w) == 1 and (("a" in w[0]) != ("b" in w[0]))),
        ("[a | b & false]", lambda w: len(w) == 1 and "a" in w[0]),
        ("[!!a & !(false | !b)]", lambda w: len(w) == 1 and {"a", "b"} <= w[0]),
        (".*[a][b]", lambda w: len(w) >= 2 and "a" in w[-2] and "b" in w[-1]),
        ("[a & !a]", lambda w: False),
    ],
)
def test_precedence_and_whole_trace_matching(expression, truth):
    definition = compile_regex(expression)
    for word in WORDS:
        assert accepts(definition, word) == truth(word), word


def test_whitespace_identifiers_and_reserved_words():
    definition = compile_regex(" ( [ _A2 & ! false ] \n [eps | true] ) ? ")
    assert definition.atoms == ("_A2", "eps")
    assert accepts(definition, ())
    assert accepts(definition, (frozenset({"_A2", "unused"}), frozenset()))
    assert not accepts(definition, (frozenset(), frozenset()))


def test_deterministic_complete_definitions_and_rejecting_sink():
    for expression in ("eps", "[false]", ".*", "([a]|[b])*[a][b]"):
        first, second = compile_regex(expression, minimize=False), compile_regex(expression, minimize=False)
        assert first.atoms == second.atoms
        assert first.states == second.states
        assert first.initial_state == second.initial_state == 0
        assert first.final_states == second.final_states
        assert first.transitions == second.transitions
        sinks = [
            state
            for state in first.states
            if state not in first.final_states and all(target == state for _, target in first.transitions[state])
        ]
        assert sinks
        for state in first.states:
            for valuation in VALUATIONS:
                assert first.transition(valuation, state) in first.states


@pytest.mark.parametrize(
    "expression,offset",
    [
        ("", 0),
        (" ", 1),
        ("a", 0),
        ("()", 1),
        ("[]", 1),
        ("[a", 2),
        ("[a &]", 4),
        ("[a b]", 3),
        ("[a || b]", 4),
        ("[!]", 2),
        ("[.]", 1),
        ("[a*]", 2),
        ("[a]|", 4),
        ("|[a]", 0),
        ("([a]", 4),
        ("eps)", 3),
        ("eps**", 4),
        (".+?", 2),
        ("[a] & [b]", 4),
        ("[a]\\", 3),
        ("[é]", 1),
        ("ep s", 0),
        ("[1a]", 1),
    ],
)
def test_malformed_expressions_report_locations(expression, offset):
    with pytest.raises(RegexSyntaxError) as caught:
        compile_regex(expression)
    assert caught.value.position == offset
    assert caught.value.line == 1
    assert caught.value.column == offset + 1
    assert f"offset {offset}" in str(caught.value)


def test_multiline_syntax_location():
    with pytest.raises(RegexSyntaxError) as caught:
        compile_regex("[a]\n[!]")
    assert (caught.value.position, caught.value.line, caught.value.column) == (6, 2, 3)


@pytest.mark.parametrize("field", RegexLimits.__dataclass_fields__)
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2", None])
def test_limits_validate_positive_non_boolean_integers(field, invalid):
    with pytest.raises(ValueError, match=field):
        RegexLimits(**{field: invalid})


@pytest.mark.parametrize(
    "expression,field,small,sufficient",
    [
        ("eps", "max_length", 2, 3),
        ("([a])", "max_nesting", 1, 2),
        ("[a & b]", "max_atoms", 1, 2),
        ("eps", "max_nfa_states", 1, 2),
        ("eps", "max_dfa_states", 1, 2),
        ("eps", "max_transitions", 1, 2),
    ],
)
def test_compilation_limits_are_enforced_and_configurable(expression, field, small, sufficient):
    with pytest.raises(RegexCompilationError, match=f"{field}={small}"):
        compile_regex(expression, limits=replace(RegexLimits(), **{field: small}))
    compile_regex(expression, limits=replace(RegexLimits(), **{field: sufficient}))


def test_exponential_and_nested_inputs_fail_explicitly():
    with pytest.raises(RegexCompilationError, match="max_dfa_states"):
        compile_regex(".*[a]" + "." * 10, limits=RegexLimits(max_dfa_states=32))
    with pytest.raises(RegexCompilationError, match="max_nesting"):
        compile_regex("(" * 100 + "eps" + ")" * 100)
    with pytest.raises(RegexCompilationError, match="max_nfa_states"):
        compile_regex("eps " * 300)
    with pytest.raises(RegexCompilationError, match="max_transitions"):
        compile_regex("[a]", limits=RegexLimits(max_transitions=2))


def test_input_types_and_immutable_limits():
    with pytest.raises(TypeError, match="expression"):
        compile_regex(None)
    with pytest.raises(TypeError, match="limits"):
        compile_regex("eps", limits={})
    with pytest.raises(FrozenInstanceError):
        RegexLimits().max_atoms = 2


def test_overlapping_guards_preserve_all_paths_with_three_simultaneous_atoms():
    definition = compile_regex("([a | b][c] | [b & !c][a] | [true][a & b & c])")
    valuations = [
        frozenset(atom for atom, active in zip("abc", flags, strict=True) if active)
        for flags in itertools.product((False, True), repeat=3)
    ]
    for first, second in itertools.product(valuations, repeat=2):
        expected = (
            bool({"a", "b"} & first)
            and "c" in second
            or "b" in first
            and "c" not in first
            and "a" in second
            or {"a", "b", "c"} <= second
        )
        assert accepts(definition, (first, second)) == expected


def test_depth_limit_applies_to_boolean_groups_and_recursion_errors_are_explicit():
    with pytest.raises(RegexCompilationError, match="max_nesting"):
        compile_regex("[" + "(" * 64 + "a" + ")" * 64 + "]")
    with pytest.raises(RegexCompilationError, match="recursion capacity"):
        compile_regex("(" * 2000 + "eps" + ")" * 2000, limits=RegexLimits(max_nesting=3000))
