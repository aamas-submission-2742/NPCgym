from dataclasses import dataclass

import pytest

from npc_gym.labels import Transition
from npc_gym.monitors import (
    DuplicateMonitorIDError,
    MonitorInput,
    MonitorRegistry,
    SimpleMonitor,
    UnknownMonitorIDError,
)


@dataclass(frozen=True)
class Collected:
    entity_id: int


def test_transition_and_monitor_input_accept_hashable_typed_labels():
    labels = frozenset({"move", Collected(2), 7})
    transition = Transition(previous_state=1, action="right", state=2, terminated=False, truncated=False)

    assert transition.previous_state == 1
    assert MonitorInput(labels).labels == labels


class NeverMonitor(SimpleMonitor):
    def detect(self, input):
        return False


class RepetitionMonitor(SimpleMonitor):
    def __init__(self):
        super().__init__()
        self.history = []

    def reset_history(self):
        self.history.clear()

    def detect(self, input):
        repeated = input.labels in self.history
        self.history.append(input.labels)
        return repeated


def test_custom_detectors_and_registry():
    registry = MonitorRegistry()
    registry.register("never-v0", NeverMonitor)
    registry.register("repeat-v0", RepetitionMonitor)
    first, second = registry.make("repeat-v0"), registry.make("repeat-v0")
    input = MonitorInput(frozenset({"a"}))
    assert not first.reset(input)
    assert first.update(input)
    assert first.count == 1 and second.count == 0
    assert not first.reset(input)
    assert first.count == 0
    with pytest.raises(DuplicateMonitorIDError):
        registry.register("repeat-v0", NeverMonitor)
    with pytest.raises(UnknownMonitorIDError):
        registry.make("missing")
    assert registry.ids == ("never-v0", "repeat-v0")
