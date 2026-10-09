"""Factory defaults, opt-outs, and reporting semantics for minimal automata."""

from itertools import product

import pytest

from npc_gym.bolts import make_builtin_bolts, make_ltlf_bolt, make_regex_bolt
from npc_gym.monitors import MonitorInput, make_builtin_monitor
from npc_gym.monitors.builtins import RECIPES
from npc_gym.monitors.ltlf import MonaCompiler, from_ltlf
from npc_gym.monitors.regex import compile_regex, from_regex


@pytest.mark.parametrize("minimize,expected", [(True, (5, 18, 6, 3)), (False, (8, 47, 14, 13))])
def test_taxi_defaults_opt_out_and_shared_definitions(minimize, expected):
    monitor = make_builtin_monitor("taxi/emergency-v0", minimize=minimize)
    bolts = make_builtin_bolts("taxi/emergency-v0", reward=-1, minimize=minimize)
    names = ("Warn Violations", "Seven-Step Safety Violations", "Three-Step Safety Violations", "Stay Violations")
    assert tuple(len(bolts[f"taxi/emergency-v0/{name}"].definition.states) for name in names) == expected
    for name in names:
        assert monitor.members[name]._runner.definition is bolts[f"taxi/emergency-v0/{name}"].definition
    if minimize:
        assert (
            make_builtin_bolts("taxi/emergency-v0", reward=-1)["taxi/emergency-v0/Warn Violations"].definition
            is bolts["taxi/emergency-v0/Warn Violations"].definition
        )


@pytest.mark.parametrize("expression", [".*", "eps", "[false]", ".*[a]", "[a].?[a]"])
@pytest.mark.parametrize("policy", ["prefix", "restart", "episode_end"])
@pytest.mark.parametrize("consume_initial", [False, True])
def test_reset_reporting_and_endings_preserved(expression, policy, consume_initial):
    monitors = [
        from_regex(
            expression,
            propositions={"a": lambda i: "a" in i.labels},
            reporting=policy,
            consume_initial=consume_initial,
            minimize=value,
        )
        for value in (False, True)
    ]
    for bits in product((False, True), repeat=5):
        for ending in ("terminated", "truncated"):
            for index, bit in enumerate(bits):
                input = MonitorInput(frozenset({"a"}) if bit else frozenset(), **{ending: index == 4})
                events = [(m.reset if index == 0 else m.update)(input) for m in monitors]
                assert events[0] == events[1]
                assert monitors[0].count == monitors[1].count


class StaticCompiler:
    def __init__(self):
        self.definition = compile_regex(".*[a]", minimize=False)

    def compile(self, formula):
        return self.definition


@pytest.mark.parametrize("minimize,count", [(True, 2), (False, 4)])
def test_ltlf_factories_handle_injected_compiler(minimize, count):
    compiler = StaticCompiler()
    options = {"propositions": {"a": lambda i: "a" in i.labels}, "compiler": compiler, "minimize": minimize}
    monitor = from_ltlf("test", **options)
    bolt = make_ltlf_bolt("test", reward=-1, **options)
    assert len(bolt.definition.states) == len(monitor._runner.definition.states) == count
    if not minimize:
        assert bolt.definition is monitor._runner.definition is compiler.definition


@pytest.mark.parametrize("minimize", [None, 0, 1, "false"])
def test_invalid_option(minimize):
    factories = [
        lambda: compile_regex(".*", minimize=minimize),
        lambda: from_regex(".*", propositions={}, minimize=minimize),
        lambda: make_regex_bolt(".*", propositions={}, reward=-1, minimize=minimize),
        lambda: make_builtin_monitor("taxi/emergency-v0", minimize=minimize),
        lambda: make_builtin_bolts("taxi/emergency-v0", reward=-1, minimize=minimize),
        lambda: RECIPES["taxi"]["Warn Violations"].compile(minimize=minimize),
        lambda: from_ltlf("test", propositions={}, compiler=StaticCompiler(), minimize=minimize),
        lambda: make_ltlf_bolt("test", propositions={}, reward=-1, compiler=StaticCompiler(), minimize=minimize),
        lambda: MonaCompiler(minimize=minimize),
    ]
    for factory in factories:
        with pytest.raises(TypeError, match="minimize must be a bool"):
            factory()
