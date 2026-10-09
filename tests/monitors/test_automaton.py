import pytest

from npc_gym.automata import CompiledDFA
from npc_gym.monitors import AutomatonMonitor, MonitorInput


def test_binding_validation_and_snapshot():
    definition = CompiledDFA(["a"], 0, frozenset({1}), {0: [("0", 0), ("1", 1)], 1: [("X", 1)]})
    with pytest.raises(ValueError, match="Missing proposition bindings: a"):
        AutomatonMonitor(definition, propositions={}, reporting="prefix")
    with pytest.raises(TypeError, match="callable"):
        AutomatonMonitor(definition, propositions={"a": False}, reporting="prefix")
    with pytest.raises(ValueError):
        AutomatonMonitor(definition, propositions={"a": bool}, reporting="unknown")
    bindings = {"a": lambda i: True}
    monitor = AutomatonMonitor(definition, propositions=bindings, reporting="prefix")
    bindings.clear()
    assert monitor.reset(MonitorInput(frozenset())) == True
    monitor = AutomatonMonitor(definition, propositions={"a": lambda i: 1}, reporting="prefix")
    with pytest.raises(TypeError, match="must return bool"):
        monitor.reset(MonitorInput(frozenset()))


@pytest.mark.parametrize("policy", ["prefix", "restart", "episode_end"])
@pytest.mark.parametrize("consume_initial", [False, True])
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_accepting_empty_word_and_self_loop_require_consumed_inputs(policy, consume_initial, ending):
    # Unlike a compiled 'true' formula, this definition explicitly accepts the empty word.
    definition = CompiledDFA((), 0, frozenset({0}), {0: [("", 0)]})
    monitor = AutomatonMonitor(definition, propositions={}, reporting=policy, consume_initial=consume_initial)
    empty = MonitorInput(frozenset())
    end = MonitorInput(frozenset(), **{ending: True})
    event = True
    assert monitor.count == 0
    for _ in range(2):
        initial_count = int(consume_initial and policy != "episode_end")
        assert monitor.reset(empty) == bool(initial_count)
        assert monitor.count == initial_count
        assert monitor.update(empty) == (False if policy == "episode_end" else event)
        assert monitor.update(end) == event
        assert monitor.count == (1 if policy == "episode_end" else initial_count + 2)
        # Even terminal reset labels cannot emit when initial consumption is disabled.
        assert monitor.reset(end) == (event if consume_initial else False)
        assert monitor.count == int(consume_initial)


def test_restart_on_initial_acceptance_does_not_replay_the_initial_input():
    # A toggles between accepting and rejecting; B accepts only after one A.
    definition = CompiledDFA(
        ("a", "b"),
        0,
        frozenset({1}),
        {0: [("1X", 1), ("0X", 0)], 1: [("X1", 1), ("X0", 0)]},
    )
    monitor = AutomatonMonitor(
        definition,
        propositions={"a": lambda i: "a" in i.labels, "b": lambda i: "b" in i.labels},
        reporting="restart",
    )
    assert monitor.reset(MonitorInput(frozenset({"a"}))) == True
    assert monitor.update(MonitorInput(frozenset({"b"}))) is False
    assert monitor.update(MonitorInput(frozenset({"a"}))) == True
    assert monitor.update(MonitorInput(frozenset({"b"}), truncated=True)) is False
    assert monitor.count == 2


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_shared_definition_has_independent_bindings_policies_and_episode_state(ending):
    definition = CompiledDFA(("a",), 0, frozenset({1}), {0: [("0", 0), ("1", 1)], 1: [("X", 1)]})
    monitors = {
        policy: AutomatonMonitor(
            definition,
            propositions={"a": lambda input, label=policy: label in input.labels},
            reporting=policy,
            consume_initial=False,
        )
        for policy in ("prefix", "restart", "episode_end")
    }
    empty = MonitorInput(frozenset())
    for monitor in monitors.values():
        assert monitor.reset(empty) is False
    for policy, emissions in (("prefix", (1, 0, 0)), ("restart", (1, 1, 0)), ("episode_end", (1, 0, 0))):
        for (name, monitor), expected in zip(monitors.items(), emissions, strict=True):
            assert monitor.update(MonitorInput(frozenset({policy}))) == bool(expected)
    snapshot = monitors["prefix"].count
    assert snapshot == 3
    assert monitors["prefix"].reset(empty) is False
    for name, monitor in monitors.items():
        assert monitor.update(MonitorInput(frozenset(), **{ending: True})) == (name == "episode_end")
    assert snapshot == 3
    assert [monitor.count for monitor in monitors.values()] == [0, 1, 1]
