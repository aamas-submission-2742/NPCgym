import pytest

from npc_gym.monitors import AutomatonMonitor, MonitorInput
from npc_gym.monitors.regex import RegexCompilationError, RegexLimits, from_regex


@pytest.mark.parametrize("policy,counts", [("prefix", [1, 2, 3]), ("restart", [1, 1, 1]), ("episode_end", [0, 0, 1])])
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_factory_uses_shared_reporting_and_initial_consumption(policy, counts, ending):
    monitor = from_regex("[a].*", propositions={"a": lambda i: "a" in i.labels}, reporting=policy)
    assert type(monitor) is AutomatonMonitor
    assert monitor.reset(MonitorInput(frozenset({"a"}))) == bool(counts[0])
    assert monitor.update(MonitorInput(frozenset())) == bool(counts[1] - counts[0])
    assert monitor.update(MonitorInput(frozenset(), **{ending: True})) == bool(counts[2] - counts[1])
    assert monitor.count == counts[-1]
    assert monitor.reset(MonitorInput(frozenset())) is False
    assert monitor.count == 0


@pytest.mark.parametrize("policy", ["prefix", "restart", "episode_end"])
def test_nullable_expression_emits_only_after_consumption(policy):
    monitor = from_regex("eps", propositions={}, reporting=policy, consume_initial=False)
    assert monitor.reset(MonitorInput(frozenset(), terminated=True)) is False
    assert monitor.update(MonitorInput(frozenset(), truncated=True)) is False
    assert monitor.count == 0


def test_independent_factories_and_binding_validation():
    options = {
        "propositions": {"a": lambda i: "a" in i.labels},
        "reporting": "prefix",
        "consume_initial": False,
    }
    first, second = [from_regex("[a][a]", **options) for _ in range(2)]
    a = MonitorInput(frozenset({"a"}))
    assert first.reset(a) is False
    assert second.reset(a) is False
    assert first.update(a) is False
    assert first.update(a) == True
    assert second.update(a) is False
    with pytest.raises(ValueError, match="Missing proposition"):
        from_regex("[a]", propositions={}, reporting="prefix")
    with pytest.raises(RegexCompilationError, match="max_length"):
        from_regex("eps", propositions={}, reporting="prefix", limits=RegexLimits(max_length=1))
