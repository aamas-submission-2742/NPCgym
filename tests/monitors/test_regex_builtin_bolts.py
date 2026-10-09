"""Built-in regex bolts preserve component rewards on characterized traces."""

import pytest

from npc_gym.bolts import make_builtin_bolts
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.automaton import AutomatonMonitor
from npc_gym.monitors.builtins import BUILTIN_IDS
from tests.monitor_characterization import CASES

SUPPORTED = [
    identifier
    for identifier in BUILTIN_IDS
    if identifier not in ("merchant/evolving-v0", "pacman/solution-guilt-maximum-v2")
]


@pytest.mark.parametrize("identifier", SUPPORTED)
def test_rewards_follow_characterized_occurrences(identifier):
    bolts = make_builtin_bolts(identifier, reward=-7)
    monitors = {
        key: AutomatonMonitor(
            spec.definition,
            propositions=spec.propositions,
            reporting=spec.reporting,
            consume_initial=spec.consume_initial,
        )
        for key, spec in bolts.items()
    }
    valuations, occurrences, _ = CASES[identifier]
    for index, labels in enumerate(valuations):
        input = MonitorInput(frozenset(labels), truncated=index == len(valuations) - 1)
        accepted = {key: (monitor.reset if index == 0 else monitor.update)(input) for key, monitor in monitors.items()}
        actual = sum(bolts[key].reward for key, occurred in accepted.items() if occurred) if index else 0
        assert actual == (-7 * len(occurrences[index]) if index else 0)


def test_component_reward_overrides_and_validation():
    identifier = "pacman/vegan-v0"
    blue = f"{identifier}/VegetarianBlue"
    specs = make_builtin_bolts(identifier, reward=-1, rewards={blue: -3})
    assert specs[blue].reward == -3
    assert specs[f"{identifier}/VegetarianOrange"].reward == -1
    with pytest.raises(ValueError, match="Unknown reward component"):
        make_builtin_bolts(identifier, reward=0, rewards={"missing": 1})
    with pytest.raises(TypeError, match="finite"):
        make_builtin_bolts(identifier, reward=True)
    with pytest.raises(ValueError, match="Boolean composition"):
        make_builtin_bolts("merchant/evolving-v0", reward=-1)


@pytest.mark.parametrize("identifier", SUPPORTED)
def test_collection_count_names_are_reward_override_ids(identifier):
    from npc_gym.monitors import MultiMonitor, make_builtin_monitor

    monitor = make_builtin_monitor(identifier)
    if not isinstance(monitor, MultiMonitor):
        return
    expected = {f"{identifier}/{name}": -index for index, name in enumerate(monitor.members, start=1)}
    bolts = make_builtin_bolts(identifier, reward=-100, rewards=expected)
    assert {key: spec.reward for key, spec in bolts.items()} == expected
    monitors = {
        key.rsplit("/", 1)[1]: AutomatonMonitor(
            spec.definition,
            propositions=spec.propositions,
            reporting=spec.reporting,
            consume_initial=spec.consume_initial,
        )
        for key, spec in bolts.items()
    }
    for index, labels in enumerate(CASES[identifier][0]):
        input = MonitorInput(frozenset(labels))
        actual = (monitor.reset if index == 0 else monitor.update)(input)
        expected_events = {
            name: (component.reset if index == 0 else component.update)(input) for name, component in monitors.items()
        }
        assert actual == expected_events
