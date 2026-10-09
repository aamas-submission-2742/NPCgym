from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError
from weakref import WeakValueDictionary, ref

import gymnasium as gym
import pytest

from npc_gym.evaluation import evaluate
from npc_gym.monitors import MonitorInput, SimpleMonitor
from npc_gym.monitors.regex import from_regex
from npc_gym.wrappers import MonitorWrapper


class TraceEnv(gym.Env):
    observation_space = gym.spaces.Dict({"value": gym.spaces.Discrete(3)})
    action_space = gym.spaces.Discrete(2)

    def __init__(self, ending=(True, False)):
        self.ending = ending
        self.steps = 0
        self.fail = False
        self.info = {"labels": frozenset({"a"}), "action_mask": [True, False], "other": {"kept": True}}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_args = seed, options
        self.steps = 0
        self.observation = {"value": 0}
        return self.observation, self.info

    def step(self, action):
        self.steps += 1
        self.action = action
        if self.fail:
            raise ValueError("environment failed")
        self.observation = {"value": min(self.steps, 2)}
        return self.observation, 1.25, *(self.ending if self.steps == 2 else (False, False)), self.info


class CountingMonitor(SimpleMonitor):
    def __init__(self, monitor_id="count"):
        super().__init__()
        self.inputs = []

    def reset_history(self):
        self.inputs.clear()

    def detect(self, input):
        self.inputs.append(input)
        return True


def factories():
    return {
        "count": CountingMonitor,
        "regex": lambda: from_regex(".*", propositions={}, reporting="episode_end"),
    }


@pytest.mark.parametrize("ending", [(True, False), (False, True), (True, True)])
def test_monitor_collection_preserves_environment_and_detaches_snapshots(ending):
    base = TraceEnv(ending)
    created = []

    def factory():
        monitor = CountingMonitor()
        created.append(monitor)
        return monitor

    env = MonitorWrapper(base, monitors={**factories(), "count": factory})
    assert env.observation_space is base.observation_space
    assert env.action_space is base.action_space
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 0
    options = {"option": 7}
    observation, initial = env.reset(seed=31, options=options)
    assert base.reset_args == (31, options)
    assert observation is base.observation
    assert "monitors" not in base.info
    assert initial["action_mask"] is base.info["action_mask"]
    assert initial["other"] == {"kept": True}
    assert initial["monitors"] == {
        "occurrences": {"count": True, "regex": False},
        "counts": {"count": {"count": 1}, "regex": {"count": 0}},
    }
    first = env.step(1)
    assert first[0] is base.observation
    assert first[1:4] == (1.25, False, False)
    assert base.action == 1
    terminal = env.step(0)
    assert terminal[1:4] == (1.25, *ending)
    snapshot = terminal[4]["monitors"]
    assert snapshot["occurrences"] == {"count": True, "regex": True}
    assert snapshot["counts"] == {"count": {"count": 3}, "regex": {"count": 1}}
    assert created[0].inputs == [
        MonitorInput(frozenset({"a"})),
        MonitorInput(frozenset({"a"})),
        MonitorInput(frozenset({"a"}), *ending),
    ]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 2
    initial["monitors"]["counts"]["count"]["caller"] = 99
    env.reset()
    assert snapshot["counts"]["count"] == {"count": 3}
    assert len(created) == 1


def test_independent_live_environments_and_configuration_copy():
    config = factories()
    left, right = [MonitorWrapper(TraceEnv(), monitors=config) for _ in range(2)]
    config.clear()
    left.reset()
    right.reset()
    left.step(0)
    assert right.step(0)[4]["monitors"]["counts"]["count"] == {"count": 2}
    left.reset()
    assert right.step(0)[4]["monitors"]["counts"]["count"] == {"count": 3}
    left.close()
    right.reset()
    assert right.step(0)[1] == 1.25
    right.close()


@pytest.mark.parametrize(
    "config,error",
    [
        ([], TypeError),
        ({"": CountingMonitor}, ValueError),
        ({1: CountingMonitor}, ValueError),
        ({"x": 1}, TypeError),
        ({"x": lambda: object()}, TypeError),
    ],
)
def test_invalid_configuration(config, error):
    with pytest.raises(error):
        MonitorWrapper(TraceEnv(), monitors=config)


def test_reused_instances_and_nested_collections_rejected():
    monitor = CountingMonitor()
    with pytest.raises(ValueError, match="fresh"):
        MonitorWrapper(TraceEnv(), monitors={"a": lambda: monitor, "b": lambda: monitor})
    left = MonitorWrapper(TraceEnv(), monitors={"count": lambda: monitor})
    with pytest.raises(ValueError, match="fresh"):
        MonitorWrapper(TraceEnv(), monitors={"count": lambda: monitor})
    with pytest.raises(TypeError, match="one MonitorWrapper"):
        MonitorWrapper(gym.wrappers.TimeLimit(left, 2), monitors={})


def test_concurrent_wrappers_cannot_claim_the_same_monitor(monkeypatch):
    from npc_gym.monitors import collection as monitoring

    starts = Barrier(4)
    checks = Barrier(4)

    class DelayedClaims(WeakValueDictionary):
        def __contains__(self, key):
            found = super().__contains__(key)
            if not found:
                # Without atomic claims, every constructor can observe "unclaimed".
                # With the lock, only one enters; the others wait for its claim.
                try:
                    checks.wait(timeout=0.2)
                except BrokenBarrierError:
                    pass
            return found

    monkeypatch.setattr(monitoring, "_claimed_monitors", DelayedClaims())
    monitor = CountingMonitor()

    def factory():
        starts.wait(timeout=5)
        return monitor

    def build(_):
        try:
            return MonitorWrapper(TraceEnv(), monitors={"count": factory})
        except ValueError as error:
            assert "fresh" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        wrappers = list(pool.map(build, range(4)))
    assert sum(wrapper is not None for wrapper in wrappers) == 1


def test_monitor_claims_use_identity_and_do_not_keep_monitors_alive():
    class UnhashableMonitor(CountingMonitor):
        def __eq__(self, other):
            return isinstance(other, UnhashableMonitor)

        __hash__ = None

    first, second = UnhashableMonitor(), UnhashableMonitor()
    left = MonitorWrapper(TraceEnv(), monitors={"count": lambda monitor=first: monitor})
    right = MonitorWrapper(TraceEnv(), monitors={"count": lambda: second})
    left.reset()
    right.reset()
    with pytest.raises(ValueError, match="fresh"):
        MonitorWrapper(TraceEnv(), monitors={"count": lambda monitor=first: monitor})
    monitor_ref = ref(first)
    del left, first
    gc.collect()
    assert monitor_ref() is None
    right.step(0)


@pytest.mark.parametrize("labels_factory", [frozenset, set, list, tuple, iter])
def test_monitor_sources_accept_the_same_iterable_labels(labels_factory):
    class IterableLabels(TraceEnv):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            return observation, {**info, "labels": labels_factory(("a", "b"))}

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            return observation, reward, terminated, truncated, {**info, "labels": labels_factory(("a", "b"))}

    class CheckLabels(CountingMonitor):
        def detect(self, input):
            assert isinstance(input.labels, frozenset)
            assert input.labels == {"a", "b"}
            return super().detect(input)

    factories = {"count": CheckLabels}
    direct = evaluate(IterableLabels(), lambda obs, info: 0, monitors=factories)
    env = MonitorWrapper(IterableLabels(), monitors=factories)
    wrapped = evaluate(env, lambda obs, info: 0, monitor_source="wrapper")
    assert wrapped == direct
    assert env.reset()[1]["labels"] == frozenset({"a", "b"})


@pytest.mark.parametrize("labels", ["a", b"a", bytearray(b"a"), None, 1, [["a"]]])
@pytest.mark.parametrize("phase", ["reset", "step"])
def test_factory_monitoring_rejects_malformed_labels(labels, phase):
    class BadLabels(TraceEnv):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            if phase == "reset":
                self.info["labels"] = labels
            return result

        def step(self, action):
            self.info["labels"] = labels
            return super().step(action)

    with pytest.raises(TypeError, match="labels"):
        evaluate(BadLabels(), lambda obs, info: 0, monitors={"count": CountingMonitor})


@pytest.mark.parametrize("phase", ["reset", "step"])
@pytest.mark.parametrize(
    "bad_info,error",
    [
        ({}, RuntimeError),
        ({"labels": "a"}, TypeError),
        ({"labels": b"a"}, TypeError),
        ({"labels": bytearray(b"a")}, TypeError),
        ({"labels": [["a"]]}, TypeError),
        ({"labels": None}, TypeError),
        ({"labels": frozenset(), "monitors": {}}, ValueError),
        (None, TypeError),
    ],
)
def test_bad_information_requires_reset(phase, bad_info, error):
    base = TraceEnv()
    env = MonitorWrapper(base, monitors={})
    valid = base.info
    if phase == "step":
        env.reset()
    base.info = bad_info
    with pytest.raises(error):
        env.reset() if phase == "reset" else env.step(0)
    steps = base.steps
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == steps
    base.info = valid
    env.reset()
    env.step(0)


@pytest.mark.parametrize("output", [[], (object(),), None, 1])
@pytest.mark.parametrize("phase", ["reset", "step"])
def test_invalid_monitor_output_requires_reset(output, phase):
    class Invalid(CountingMonitor):
        broken = False

        def detect(self, input):
            return output if self.broken else True

    monitor = Invalid()
    env = MonitorWrapper(TraceEnv(), monitors={"count": lambda: monitor})
    if phase == "step":
        env.reset()
    monitor.broken = True
    with pytest.raises(TypeError):
        env.reset() if phase == "reset" else env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    monitor.broken = False
    env.reset()
    env.step(0)


def test_environment_failure_requires_reset():
    base = TraceEnv()
    env = MonitorWrapper(base, monitors=factories())
    env.reset()
    base.fail = True
    with pytest.raises(ValueError, match="environment failed"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 1
    base.fail = False
    env.reset()
    env.step(0)


@pytest.mark.parametrize("ending", [(True, False), (False, True), (True, True)])
def test_evaluation_sources_have_identical_results_without_double_counting(ending):
    expected = evaluate(TraceEnv(ending), lambda obs, info: 0, monitors=factories(), episodes=2, seed=11)
    env = MonitorWrapper(TraceEnv(ending), monitors=factories())
    actual = evaluate(env, lambda obs, info: 0, monitor_source="wrapper", episodes=2, seed=11)
    assert actual == expected
    assert actual.mean_monitor_counts == {"count": {"count": 3}, "regex": {"count": 1}}
    # Neither monitor nor bolt diagnostics are inferred as a recording source.
    env.unwrapped.info["restraining_bolts"] = {"occurrences": {"bolt": True}}
    ignored = evaluate(env, lambda obs, info: 0)
    assert ignored.mean_monitor_counts == {}


@pytest.mark.parametrize("monitors", [{}, factories()])
def test_evaluation_rejects_both_sources_before_reset(monitors):
    base = TraceEnv()
    with pytest.raises(ValueError, match="cannot be selected together"):
        evaluate(base, lambda obs, info: 0, monitors=monitors, monitor_source="wrapper")
    assert not hasattr(base, "reset_args")


def test_evaluation_rejects_unknown_or_missing_source():
    with pytest.raises(ValueError, match="monitor_source"):
        evaluate(TraceEnv(), lambda obs, info: 0, monitor_source="auto")
    with pytest.raises(RuntimeError, match="requires info"):
        evaluate(TraceEnv(), lambda obs, info: 0, monitor_source="wrapper")


@pytest.mark.parametrize(
    "snapshot,error",
    [
        ({}, TypeError),
        ([], TypeError),
        ({"occurrences": [], "counts": {}}, ValueError),
        ({"occurrences": {"x": True}, "counts": {"x": {"member": 1}}}, ValueError),
        ({"occurrences": {"x": 1}, "counts": {"x": {"count": 1}}}, ValueError),
        ({"occurrences": {"x": {"count": 1}}, "counts": {"x": {"count": 1}}}, TypeError),
        ({"occurrences": {"unknown": {}}, "counts": {}}, ValueError),
        ({"occurrences": {}, "counts": {"x": {"count": True}}}, ValueError),
    ],
)
def test_evaluation_validates_wrapper_snapshots(snapshot, error):
    base = TraceEnv()
    base.info["monitors"] = snapshot
    with pytest.raises(error):
        evaluate(base, lambda obs, info: 0, monitor_source="wrapper")


def test_multiple_monitors_follow_configuration_order():
    env = MonitorWrapper(TraceEnv(), monitors={"z": CountingMonitor, "a": CountingMonitor})
    for info in (env.reset()[1], env.step(0)[4]):
        assert list(info["monitors"]["occurrences"]) == ["z", "a"]
        assert info["monitors"]["occurrences"] == {"z": True, "a": True}


def test_partial_monitor_failure_requires_reset_and_does_not_replay_transition():
    class Failing(CountingMonitor):
        def detect(self, input):
            if self.count == 1:
                raise LookupError("monitor failed")
            return super().detect(input)

    first, second = CountingMonitor(), Failing("failed")
    base = TraceEnv()
    env = MonitorWrapper(base, monitors={"count": lambda: first, "failed": lambda: second})
    env.reset()
    with pytest.raises(LookupError, match="monitor failed"):
        env.step(0)
    assert base.steps == 1
    assert first.count == 2
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 1
    env.reset()
    assert first.count == second.count == 1


def test_inner_time_limit_is_visible_to_episode_end_monitor():
    env = MonitorWrapper(gym.wrappers.TimeLimit(TraceEnv(), max_episode_steps=1), monitors=factories())
    result = evaluate(env, lambda obs, info: 0, monitor_source="wrapper").episodes[0]
    assert result.length == 1
    assert result.monitor_counts == {"count": {"count": 2}, "regex": {"count": 1}}
    assert result.termination == "truncated"


def test_empty_collection_still_publishes_and_records_snapshots():
    env = MonitorWrapper(TraceEnv(), monitors={})
    assert env.reset()[1]["monitors"] == {"occurrences": {}, "counts": {}}
    result = evaluate(env, lambda obs, info: 0, monitor_source="wrapper").episodes[0]
    assert result.monitor_counts == {}


def test_recorder_rejects_changing_diagnostic_schema():
    base = TraceEnv()
    base.info["monitors"] = {"occurrences": {"x": {}}, "counts": {"x": {"initial": 0}}}

    def policy(obs, info):
        base.info["monitors"]["counts"]["x"] = {"changed": 1}
        return 0

    with pytest.raises(ValueError, match="inconsistent count fields"):
        evaluate(base, policy, monitor_source="wrapper")


def test_direct_evaluation_cannot_reuse_a_live_factory_result():
    monitor = CountingMonitor()
    evaluate(TraceEnv(), lambda obs, info: 0, monitors={"count": lambda: monitor})
    with pytest.raises(ValueError, match="fresh"):
        evaluate(TraceEnv(), lambda obs, info: 0, monitors={"count": lambda: monitor})


def test_wrapper_records_individual_events_and_named_collection_members_together():
    from npc_gym.monitors import MultiMonitor

    monitors = {
        "single": CountingMonitor,
        "collection": lambda: MultiMonitor(
            {"count": CountingMonitor(), "second": CountingMonitor()},
            derived={"sum": lambda c: sum(c.values())},
        ),
    }
    env = MonitorWrapper(TraceEnv(), monitors=monitors)
    snapshot = env.reset()[1]["monitors"]
    assert snapshot["occurrences"] == {"single": True, "collection": {"count": True, "second": True}}
    assert snapshot["counts"] == {"single": {"count": 1}, "collection": {"count": 1, "second": 1, "sum": 2}}
    direct = evaluate(TraceEnv(), lambda obs, info: 0, monitors=monitors, episodes=2, seed=11)
    wrapped = evaluate(env, lambda obs, info: 0, monitor_source="wrapper", episodes=2, seed=11)
    assert wrapped == direct
    assert wrapped.mean_monitor_counts == {
        "single": {"count": 3},
        "collection": {"count": 3, "second": 3, "sum": 6},
    }
