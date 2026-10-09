from itertools import product

import pytest

from npc_gym.monitors import MonitorInput, SimpleMonitor
from npc_gym.monitors.regex import from_regex


@pytest.mark.parametrize("policy,counts", [("prefix", [1, 2, 3]), ("restart", [1, 1, 1]), ("episode_end", [0, 0, 1])])
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_acceptance_policies(policy, counts, ending):
    monitor = from_regex("[a].*", propositions={"a": lambda i: "a" in i.labels}, reporting=policy)
    assert isinstance(monitor, SimpleMonitor)
    assert monitor.reset(MonitorInput(frozenset({"a"}))) == bool(counts[0])
    assert monitor.count == counts[0]
    assert monitor.update(MonitorInput(frozenset())) == bool(counts[1] - counts[0])
    assert monitor.update(MonitorInput(frozenset(), **{ending: True})) == bool(counts[2] - counts[1])
    assert monitor.count == counts[-1]
    monitor.reset(MonitorInput(frozenset()))
    assert monitor.count == 0


def test_restart_does_not_replay_and_composition_does_not_cancel_history():
    options = {"propositions": {"a": lambda i: "a" in i.labels}, "reporting": "restart"}
    violation = from_regex(".*[a][a]", **options)
    exception = from_regex(".*[a]", **options)
    combined = violation & ~exception
    a = MonitorInput(frozenset({"a"}))
    for _ in range(5):
        assert not combined.update(a)
    assert violation.count == 2
    assert exception.count == 5
    assert combined.count == 0


@pytest.mark.parametrize("policy", ["prefix", "restart", "episode_end"])
def test_skip_reset_and_no_empty_trace(policy):
    monitor = from_regex(".*", propositions={}, reporting=policy, consume_initial=False)
    end = MonitorInput(frozenset(), terminated=True)
    assert not monitor.reset(end)
    assert monitor.count == 0
    assert monitor.update(end)
    assert monitor.count == 1


@pytest.mark.parametrize("policy", ["prefix", "restart", "episode_end"])
def test_generic_ltlf_regex_equivalence(policy):
    pytest.importorskip("ltlf2dfa")
    from npc_gym.monitors.ltlf import from_ltlf

    bindings = {"a": lambda i: "a" in i.labels, "b": lambda i: "b" in i.labels}
    for trace in product([frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})], repeat=3):
        regex = from_regex(".*[a][b].*", propositions=bindings, reporting=policy)
        ltlf = from_ltlf("F(a & X b)", propositions=bindings, reporting=policy)
        assert regex.reset(MonitorInput(trace[0])) == ltlf.reset(MonitorInput(trace[0]))
        for index, labels in enumerate(trace[1:]):
            input = MonitorInput(labels, truncated=index == 1)
            assert regex.update(input) == ltlf.update(input)
        assert regex.count == ltlf.count
