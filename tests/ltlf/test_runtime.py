import itertools

import pytest

from npc_gym.automata import CompiledDFA
from npc_gym.automata.automaton import AutomatonRunner
from npc_gym.monitors import AutomatonMonitor, MonitorInput
from npc_gym.monitors.ltlf import from_ltlf


@pytest.mark.parametrize(
    "formula,truth",
    [
        ("true", lambda w: True),
        ("false", lambda w: False),
        ("a", lambda w: "a" in w[0]),
        ("!a", lambda w: "a" not in w[0]),
        ("X a", lambda w: len(w) > 1 and "a" in w[1]),
        ("WX a", lambda w: len(w) == 1 or "a" in w[1]),
        ("F a", lambda w: any("a" in v for v in w)),
        ("G a", lambda w: all("a" in v for v in w)),
        ("a U b", lambda w: any("b" in v and all("a" in x for x in w[:i]) for i, v in enumerate(w))),
        ("F(a & !(X true))", lambda w: "a" in w[-1]),
        ("G(a -> X b)", lambda w: all("a" not in v or (i + 1 < len(w) and "b" in w[i + 1]) for i, v in enumerate(w))),
    ],
)
def test_compiler_language_matches_finite_trace_semantics(formula, truth, compiler):
    definition = compiler.compile(formula)
    for length in range(1, 5):
        for word in itertools.product(
            (frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})), repeat=length
        ):
            runner = AutomatonRunner(definition)
            for valuation in word:
                reached = runner.step(valuation)
            assert (reached in definition.final_states) == truth(word), (formula, word)


@pytest.mark.parametrize(
    "policy,counts", [("prefix", [1, 2, 3, 4]), ("restart", [1, 1, 1, 1]), ("episode_end", [0, 0, 0, 1])]
)
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_reporting_and_reset(policy, counts, ending, compiler):
    monitor = from_ltlf("F a", propositions={"a": lambda i: "a" in i.labels}, reporting=policy, compiler=compiler)
    for index in range(4):
        input = MonitorInput(frozenset({"a"}) if index == 0 else frozenset(), **{ending: index == 3})
        result = monitor.reset(input) if index == 0 else monitor.update(input)
        delta = counts[index] - (counts[index - 1] if index else 0)
        assert result == bool(delta)
        assert monitor.count == counts[index]
    assert monitor.reset(MonitorInput(frozenset())) is False
    assert monitor.count == 0


def test_empty_trace_ignored_reset_and_independent_runners(compiler):
    definition = compiler.compile("X a")
    first, second = [
        AutomatonMonitor(
            definition,
            propositions={"a": lambda i: "a" in i.labels},
            reporting="restart",
            consume_initial=False,
        )
        for _ in range(2)
    ]
    a = MonitorInput(frozenset({"a"}))
    assert first.reset(a) is False
    assert first.update(a) is False
    assert first.update(a) == True
    assert first.update(a) is False  # Acceptance letter was not replayed.
    assert second.reset(a) is False
    assert second.update(a) is False
    assert second.count == 0
    truth = AutomatonMonitor(compiler.compile("true"), propositions={}, reporting="prefix")
    assert truth.count == 0
    assert truth.reset(a) == True


def test_alternative_compiler_and_unused_bindings(monkeypatch):
    from npc_gym.monitors.ltlf import MonaCompiler

    def forbidden(*args, **kwargs):
        pytest.fail("Default backend must not be used")

    monkeypatch.setattr(MonaCompiler, "compile", forbidden)

    class StaticCompiler:
        def compile(self, formula):
            assert formula == "custom expression"
            return CompiledDFA(("a",), 0, frozenset({1}), {0: [("0", 0), ("1", 1)], 1: [("X", 1)]})

    monitor = from_ltlf(
        "custom expression",
        reporting="prefix",
        propositions={"a": lambda input: True, "unused": forbidden},
        compiler=StaticCompiler(),
    )
    assert type(monitor) is AutomatonMonitor
    assert monitor.reset(MonitorInput(frozenset())) == True
