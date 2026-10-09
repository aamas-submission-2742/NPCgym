from itertools import product

import pytest

from npc_gym.monitors import MonitorInput
from npc_gym.monitors.events import MultiMonitor, SimpleMonitor


class Event(SimpleMonitor):
    def __init__(self, label="a", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.calls = 0

    def detect(self, input):
        self.calls += 1
        return self.label in input.labels


def letter(*labels):
    return MonitorInput(frozenset(labels))


@pytest.mark.parametrize("a,b", list(product([False, True], repeat=2)))
def test_boolean_truth_and_shared_dependencies(a, b):
    left, right = Event("a"), Event("b")
    root = MultiMonitor({"and": left & right, "or": left | right, "exception": left & ~right})
    result = root.update(letter(*(["a"] if a else []), *(["b"] if b else [])))
    assert result == {"and": a and b, "or": a or b, "exception": a and not b}
    assert left.calls == right.calls == 1
    assert left.count == int(a)


def test_sum_preserves_simultaneous_events_and_snapshots_are_immutable():
    root = MultiMonitor({"a": Event("a"), "b": Event("b")}, derived={"total": lambda c: c["a"] + c["b"]})
    root.update(letter("a", "b"))
    snapshot = root.counts
    assert snapshot == {"a": 1, "b": 1, "total": 2}
    root.reset(letter())
    assert root.counts == {"a": 0, "b": 0, "total": 0}
    assert snapshot["total"] == 2
    with pytest.raises(TypeError):
        snapshot["a"] = 3


def test_exception_does_not_suppress_child_advancement_or_counts():
    violation, exception = Event("a"), Event("b")
    root = violation & ~exception
    assert not root.update(letter("a", "b"))
    assert root.update(letter("a"))
    assert (root.count, violation.count, exception.count) == (1, 2, 1)


def test_reset_conventions_and_unconditional_advancement():
    left, right = Event("a", consume_initial=False), Event("b", consume_initial=False)
    expression = ~(left & right)
    assert not expression.reset(letter())
    assert expression.update(letter())
    assert left.calls == right.calls == 1
    assert not expression.reset(letter())
    assert expression.count == left.count == 0
    with pytest.raises(ValueError, match="initial-input"):
        left & Event()
    root = MultiMonitor({"skip": Event(consume_initial=False), "consume": Event()})
    assert root.reset(letter("a")) == {"skip": False, "consume": True}


def test_live_graph_has_one_owner_and_failed_claim_is_atomic():
    child = Event()
    root = child | child
    root.update(letter("a"))
    assert child.count == 1
    with pytest.raises(ValueError, match="another live root"):
        child.update(letter())
    fresh = Event()
    with pytest.raises(ValueError, match="another live root"):
        MultiMonitor({"fresh": fresh, "taken": child}).reset(letter())
    assert fresh.update(letter("a"))


def test_cycle_is_rejected():
    root = ~Event()
    root._children = (root,)
    with pytest.raises(ValueError, match="cycle"):
        root.reset(letter())


@pytest.mark.parametrize("value", [True, 1.0, "1", None, -1])
def test_invalid_derived_counts(value):
    root = MultiMonitor({"a": Event()}, derived={"total": lambda counts: value})
    with pytest.raises((TypeError, ValueError), match="derived count"):
        _ = root.counts


def test_derived_counts_only_see_immutable_member_snapshot():
    root = MultiMonitor({"a": Event()}, derived={"first": lambda c: 1, "second": lambda c: c["first"]})
    with pytest.raises(KeyError):
        _ = root.counts
    root = MultiMonitor({"a": Event()}, derived={"total": lambda c: c.__setitem__("a", 2)})
    with pytest.raises(AttributeError):
        _ = root.counts


def test_invalid_members_and_names():
    with pytest.raises(ValueError, match="duplicate"):
        MultiMonitor({"a": Event()}, derived={"a": lambda c: 0})
    with pytest.raises(TypeError, match="ComplexMonitor"):
        MultiMonitor({"nested": MultiMonitor({})})
    with pytest.raises(ValueError, match="empty"):
        MultiMonitor({"": Event()})
    with pytest.raises(TypeError, match="operands"):
        Event() & True


def test_collection_rejects_repeated_member_without_claiming_it():
    event = Event()
    with pytest.raises(ValueError, match="repeated member instance"):
        MultiMonitor({"a": event, "b": event}, derived={"sum": lambda c: sum(c.values())})
    assert event.update(letter("a"))


def test_custom_history_hook_runs_before_initial_detection_for_shared_dependencies():
    class Repeated(Event):
        def reset_history(self):
            self.previous = None

        def detect(self, input):
            previous, self.previous = self.previous, input.labels
            return previous == input.labels

    event = Repeated()
    root = MultiMonitor({"a": event | event, "b": ~event})
    for _ in range(2):
        assert root.reset(letter("a")) == {"a": False, "b": True}
        assert event.count == 0
        assert root.update(letter("a")) == {"a": True, "b": False}
        assert event.count == 1
