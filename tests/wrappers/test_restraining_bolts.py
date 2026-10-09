from __future__ import annotations

from dataclasses import FrozenInstanceError

import gymnasium as gym
import numpy as np
import pytest

from npc_gym.algorithms import TabularQLearning
from npc_gym.automata import CompiledDFA
from npc_gym.bolts import BoltSpec, make_ltlf_bolt, make_regex_bolt
from npc_gym.evaluation import evaluate
from npc_gym.monitors.regex import RegexCompilationError, RegexLimits, from_regex
from npc_gym.wrappers import MonitorWrapper, RestrainingBoltWrapper


class BoltEnv(gym.Env):
    observation_space = gym.spaces.Dict({"value": gym.spaces.Discrete(4)})
    action_space = gym.spaces.Discrete(2)

    def __init__(self, ending=(True, False), labels=frozenset({"a", "b"})):
        self.ending = ending
        self.info = {"labels": labels, "action_mask": np.array([True, False]), "kept": {"x": 1}}
        self.reward = 2.0
        self.steps = 0
        self.fail = False

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
            raise LookupError("environment failure")
        self.observation = {"value": min(self.steps, 3)}
        return self.observation, self.reward, *(self.ending if self.steps == 3 else (False, False)), self.info


def spec(expression=".*", reward=3, **kwargs):
    return make_regex_bolt(expression, propositions={"a": lambda value: "a" in value.labels}, reward=reward, **kwargs)


@pytest.mark.parametrize("encoding", ["one_hot", "index"])
@pytest.mark.parametrize("ending", [(True, False), (False, True), (True, True)])
def test_rewards_self_loops_multiple_bolts_and_final_transition(encoding, ending):
    base = BoltEnv(ending)
    env = RestrainingBoltWrapper(
        base, bolts={"positive": spec(reward=3), "negative": spec(reward=-4)}, state_encoding=encoding
    )
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 0
    options = {"option": 1}
    initial, info = env.reset(seed=71, options=options)
    assert base.reset_args == (71, options)
    assert initial["observation"] is base.observation
    assert env.observation_space.contains(initial)
    assert env.action_space is base.action_space
    assert "restraining_bolts" not in base.info
    snapshot = info["restraining_bolts"]
    assert snapshot["occurrences"] == {"negative": True, "positive": True}
    assert snapshot["wrapped_reward"] is None
    assert snapshot["reward_adjustments"] == {"negative": 0.0, "positive": 0.0}
    for index in range(3):
        observation, reward, terminated, truncated, info = env.step(0)
        assert observation["observation"] is base.observation
        assert env.observation_space.contains(observation)
        assert reward == 1.0
        assert (terminated, truncated) == (ending if index == 2 else (False, False))
        assert info["labels"] == frozenset({"a", "b"})
        assert info["action_mask"] is base.info["action_mask"]
        assert info["kept"] == {"x": 1}
        assert info["restraining_bolts"]["occurrences"] == {"negative": True, "positive": True}
        assert info["restraining_bolts"]["wrapped_reward"] == 2.0
        assert info["restraining_bolts"]["reward_adjustments"] == {"negative": -4.0, "positive": 3.0}
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 3
    assert snapshot["reward_adjustments"] == {"negative": 0.0, "positive": 0.0}


@pytest.mark.parametrize("encoding", ["one_hot", "index"])
def test_noncontiguous_state_layout_is_sorted_immutable_and_detached(encoding):
    definition = CompiledDFA(("a",), 10, frozenset({40}), {40: (("X", 40),), 10: (("0", 10), ("1", 40))})
    bindings = {"a": lambda value: "a" in value.labels}
    shared = BoltSpec(definition, bindings, 4, consume_initial=False)
    config = {"z": shared, "a": shared}
    env = RestrainingBoltWrapper(BoltEnv(), bolts=config, state_encoding=encoding)
    config.clear()
    bindings.clear()
    assert tuple(env.bolt_metadata) == ("a", "z")
    width = 2 if encoding == "one_hot" else 1
    for index, key in enumerate(("a", "z")):
        metadata = env.bolt_metadata[key]
        assert metadata.index == index
        assert metadata.state_indices == {10: 0, 40: 1}
        assert metadata.feature_slice == slice(index * width, (index + 1) * width)
        with pytest.raises(TypeError):
            metadata.state_indices[10] = 1
        with pytest.raises(FrozenInstanceError):
            metadata.index = 7
    with pytest.raises(TypeError):
        env.bolt_metadata["new"] = env.bolt_metadata["a"]
    initial, info = env.reset()
    expected = [1, 0, 1, 0] if encoding == "one_hot" else [0, 0]
    np.testing.assert_array_equal(initial["automata"], expected)
    assert initial["automata"].dtype == (np.float32 if encoding == "one_hot" else np.int64)
    info["restraining_bolts"]["states"]["a"] = -100
    initial["automata"][:] = 0
    observation, reward, *_ = env.step(1)
    np.testing.assert_array_equal(observation["automata"], [0, 1, 0, 1] if encoding == "one_hot" else [1, 1])
    assert reward == 10
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("encoding", ["one_hot", "index"])
@pytest.mark.parametrize("consume_initial", [False, True])
def test_restart_retains_initial_state_without_replaying_accepting_input(encoding, consume_initial):
    bolt = spec("[a][a]", reporting="restart", consume_initial=consume_initial)
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"pair": bolt}, state_encoding=encoding)
    observation, info = env.reset()
    initial = bolt.definition.initial_state
    after_one = bolt.definition.transition(frozenset({"a"}), initial)
    assert info["restraining_bolts"]["states"] == {"pair": after_one if consume_initial else initial}
    for number in range(1, 4):
        observation, reward, _, _, info = env.step(0)
        accepted = (number + int(consume_initial)) % 2 == 0
        retained = initial if accepted else after_one
        assert reward == (5 if accepted else 2)
        assert info["restraining_bolts"]["states"] == {"pair": retained}
        position = env.bolt_metadata["pair"].state_indices[retained]
        assert (np.argmax(observation["automata"]) if encoding == "one_hot" else observation["automata"][0]) == position


@pytest.mark.parametrize("reporting", ["prefix", "restart"])
def test_reset_acceptance_never_pays_or_defers_reward(reporting):
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"once": spec("[a]", reporting=reporting)})
    _, info = env.reset()
    assert info["restraining_bolts"]["occurrences"] == {"once": True}
    env.unwrapped.info["labels"] = frozenset()
    assert env.step(0)[1] == 2
    assert env.step(0)[1] == 2


@pytest.mark.parametrize("expression", ["eps", "[false]", "eps|[a]"])
def test_unconsumed_empty_word_never_emits(expression):
    env = RestrainingBoltWrapper(BoltEnv(labels=frozenset()), bolts={"empty": spec(expression, consume_initial=False)})
    _, info = env.reset()
    assert info["restraining_bolts"]["occurrences"] == {"empty": False}
    assert env.step(0)[1] == 2


@pytest.mark.parametrize("bolt_outside", [False, True])
def test_monitor_composition_order_and_no_implicit_double_counting(bolt_outside):
    def monitor():
        return from_regex(".*", propositions={}, reporting="episode_end")

    base = BoltEnv()
    config = {"observed": monitor}
    env = (
        RestrainingBoltWrapper(MonitorWrapper(base, monitors=config), bolts={"bolt": spec()})
        if bolt_outside
        else MonitorWrapper(RestrainingBoltWrapper(base, bolts={"bolt": spec()}), monitors=config)
    )
    result = evaluate(env, lambda observation, info: 0, monitor_source="wrapper").episodes[0]
    assert result.episode_return == 15
    assert result.monitor_counts == {"observed": {"count": 1}}
    assert evaluate(env, lambda observation, info: 0).episodes[0].monitor_counts == {}


def test_shared_spec_has_independent_runtimes_and_episode_resets():
    shared = spec("[a][a]", consume_initial=False)
    left, right = [RestrainingBoltWrapper(BoltEnv(), bolts={"pair": shared}) for _ in range(2)]
    left.reset()
    right.reset()
    assert left.step(0)[1] == 2
    assert left.step(0)[1] == 5
    assert right.step(0)[1] == 2
    left.reset()
    left.close()
    assert right.step(0)[1] == 5
    right.reset()
    assert right.step(0)[1] == 2
    right.close()


@pytest.mark.parametrize("encoding", ["one_hot", "index"])
def test_tabular_structured_observation_training_save_load(tmp_path, encoding):
    env = RestrainingBoltWrapper(
        BoltEnv(), bolts={"pair": spec("[a][a]", consume_initial=False)}, state_encoding=encoding
    )
    model = TabularQLearning(env, learning_rate=1, gamma=0, seed=2, use_action_mask=True, log_interval=None)
    model.learn(6)
    observation, _ = env.reset()
    assert model.q_values(observation)[0] == 2
    observation, *_ = env.step(0)
    assert model.q_values(observation)[0] == 5
    assert model.explored_states == 3
    path = tmp_path / "augmented.zip"
    model.save(path)
    loaded = TabularQLearning.load(path, env=env)
    np.testing.assert_array_equal(loaded.q_values(observation), model.q_values(observation))
    assert evaluate(env, loaded).episodes[0].episode_return == 9


@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (complex(1), TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (-np.inf, ValueError),
        (10**400, ValueError),
    ],
)
def test_invalid_spec_rewards(value, error):
    with pytest.raises(error, match="reward"):
        spec(reward=value)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("definition", object(), TypeError),
        ("propositions", [], TypeError),
        ("propositions", {}, ValueError),
        ("propositions", {"a": 1}, TypeError),
        ("reporting", "episode_end", ValueError),
        ("reporting", "bad", ValueError),
        ("consume_initial", 1, TypeError),
    ],
)
def test_invalid_spec_configuration(field, value, error):
    args = {"definition": spec("[a]").definition, "propositions": {"a": lambda value: True}, "reward": 1}
    args[field] = value
    with pytest.raises(error):
        BoltSpec(**args)


def test_spec_immutability_and_zero_reward_signals():
    bolt = spec(reward=0)
    with pytest.raises(FrozenInstanceError):
        bolt.reward = -1
    with pytest.raises(TypeError):
        bolt.propositions["new"] = lambda value: True
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"zero": bolt})
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == 2
    assert info["restraining_bolts"]["occurrences"] == {"zero": True}


def test_specs_have_identity_equality_and_can_be_mapping_keys():
    first = spec("[a]")
    second = BoltSpec(first.definition, first.propositions, first.reward)
    separate = spec("[a]")
    assert first != second
    assert first != separate
    assert first != object()
    mapping = {first: "first", second: "second", separate: "separate"}
    assert len(mapping) == 3
    assert mapping[first] == "first"


@pytest.mark.parametrize("labels_factory", [frozenset, set, list, tuple, iter])
@pytest.mark.parametrize("monitor_outside", [False, True])
def test_iterable_labels_survive_composition_and_snapshot_mutation(labels_factory, monitor_outside):
    class IterableLabels(BoltEnv):
        def reset(self, **kwargs):
            self.info["labels"] = labels_factory(("a", "b"))
            return super().reset(**kwargs)

        def step(self, action):
            self.info["labels"] = labels_factory(("a", "b"))
            return super().step(action)

    def monitor():
        return from_regex("[a]*", propositions={"a": lambda value: "a" in value.labels}, reporting="prefix")

    base = IterableLabels()
    monitors = {"monitor": monitor}
    bolts = {"bolt": spec("[a]*")}
    if monitor_outside:
        env = MonitorWrapper(RestrainingBoltWrapper(base, bolts=bolts), monitors=monitors)
    else:
        env = RestrainingBoltWrapper(MonitorWrapper(base, monitors=monitors), bolts=bolts)
    initial = env.reset()[1]
    base.info["labels"] = frozenset()
    for _ in range(3):
        _, reward, _, _, info = env.step(0)
        assert reward == 5
        assert info["labels"] == frozenset({"a", "b"})
        assert info["monitors"]["occurrences"] == {"monitor": True}
        assert info["restraining_bolts"]["occurrences"] == {"bolt": True}
    assert initial["labels"] == frozenset({"a", "b"})
    assert initial["monitors"]["counts"] == {"monitor": {"count": 1}}


@pytest.mark.parametrize(
    "config,error",
    [([], TypeError), ({"": spec()}, ValueError), ({1: spec()}, TypeError), ({"a": object()}, TypeError)],
)
def test_invalid_wrapper_configuration(config, error):
    with pytest.raises(error):
        RestrainingBoltWrapper(BoltEnv(), bolts=config)


def test_encoding_and_duplicate_wrapper_validation():
    with pytest.raises(ValueError, match="state_encoding"):
        RestrainingBoltWrapper(BoltEnv(), bolts={}, state_encoding="flat")
    env = RestrainingBoltWrapper(BoltEnv(), bolts={})
    with pytest.raises(TypeError, match="one RestrainingBoltWrapper"):
        RestrainingBoltWrapper(MonitorWrapper(env, monitors={}), bolts={})


@pytest.mark.parametrize("encoding", ["one_hot", "index"])
def test_empty_collection(encoding):
    env = RestrainingBoltWrapper(BoltEnv(), bolts={}, state_encoding=encoding)
    observation, info = env.reset()
    assert env.observation_space.contains(observation)
    assert observation["automata"].shape == (0,)
    assert info["restraining_bolts"]["states"] == {}
    assert env.step(0)[1] == 2


@pytest.mark.parametrize("phase", ["reset", "step"])
@pytest.mark.parametrize(
    "info,error",
    [
        ({}, RuntimeError),
        ({"labels": "a"}, TypeError),
        ({"labels": b"a"}, TypeError),
        ({"labels": bytearray(b"a")}, TypeError),
        ({"labels": [["a"]]}, TypeError),
        ({"labels": None}, TypeError),
        ({"labels": frozenset(), "restraining_bolts": {}}, ValueError),
        (None, TypeError),
    ],
)
def test_malformed_labels_or_collision_requires_reset(phase, info, error):
    base = BoltEnv()
    env = RestrainingBoltWrapper(base, bolts={"one": spec()})
    original = base.info
    if phase == "step":
        env.reset()
    base.info = info
    with pytest.raises(error):
        env.reset() if phase == "reset" else env.step(0)
    previous_steps = base.steps
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == previous_steps
    base.info = original
    env.reset()
    env.step(0)


@pytest.mark.parametrize(
    "reward,error",
    [
        (True, TypeError),
        ("2", TypeError),
        (None, TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (10**400, ValueError),
    ],
)
def test_bad_wrapped_reward_requires_reset(reward, error):
    base = BoltEnv()
    env = RestrainingBoltWrapper(base, bolts={"one": spec()})
    env.reset()
    base.reward = reward
    with pytest.raises(error, match="reward"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert base.steps == 1
    base.reward = 2
    env.reset()
    assert env.step(0)[1] == 5


def test_overflowing_adjusted_reward_requires_reset():
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"a": spec(reward=1e308), "b": spec(reward=1e308)})
    env.reset()
    with pytest.raises(ValueError, match="Adjusted reward"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


@pytest.mark.parametrize("phase", ["reset", "step"])
def test_predicate_failure_and_invalid_return_require_reset(phase):
    broken = True

    def predicate(value):
        return 1 if broken else True

    bolt = make_regex_bolt("[a]", propositions={"a": predicate}, reward=2, consume_initial=phase == "reset")
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"b": bolt})
    if phase == "step":
        env.reset()
    with pytest.raises(TypeError, match="must return bool"):
        env.reset() if phase == "reset" else env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    broken = False
    env.reset()
    env.step(0)


def test_environment_failure_requires_reset():
    base = BoltEnv()
    env = RestrainingBoltWrapper(base, bolts={"one": spec()})
    env.reset()
    base.fail = True
    with pytest.raises(LookupError):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    base.fail = False
    env.reset()
    env.step(0)


def test_ltlf_injected_compiler_and_regex_limits():
    definition = spec("[a]").definition

    class Compiler:
        def compile(self, formula):
            assert formula == "a"
            return definition

    bolt = make_ltlf_bolt(
        "a",
        propositions={"a": lambda value: True},
        reward=-1,
        compiler=Compiler(),
        reporting="restart",
        consume_initial=False,
    )
    assert bolt.definition is definition
    assert bolt.reward == -1
    assert bolt.reporting == "restart"
    assert not bolt.consume_initial
    with pytest.raises(RegexCompilationError):
        make_regex_bolt(
            "[a][a]", propositions={"a": lambda value: True}, reward=1, limits=RegexLimits(max_nfa_states=2)
        )


@pytest.mark.parametrize("outside", [False, True])
def test_reward_scaling_order(outside):
    base = BoltEnv()
    env = (
        gym.wrappers.TransformReward(RestrainingBoltWrapper(base, bolts={"b": spec()}), lambda reward: reward * 10)
        if outside
        else RestrainingBoltWrapper(gym.wrappers.TransformReward(base, lambda reward: reward * 10), bolts={"b": spec()})
    )
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == (50 if outside else 23)
    assert info["restraining_bolts"]["wrapped_reward"] == (2 if outside else 20)
    assert info["restraining_bolts"]["reward_adjustments"] == {"b": 3}


def test_partial_bolt_failure_requires_reset_without_replaying_transition():
    broken = False

    def predicate(value):
        if broken:
            raise LookupError("predicate failed")
        return True

    second = make_regex_bolt("[a]*", propositions={"a": predicate}, reward=2)
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"first": spec(), "second": second})
    env.reset()
    broken = True
    with pytest.raises(LookupError, match="predicate failed"):
        env.step(0)
    assert env.unwrapped.steps == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    broken = False
    env.reset()
    assert env.step(0)[1] == 7


def test_time_limit_flags_reach_bindings_and_terminal_reward_is_paid():
    bolt = make_regex_bolt(".*[end]", propositions={"end": lambda value: value.truncated}, reward=-5)
    env = RestrainingBoltWrapper(gym.wrappers.TimeLimit(BoltEnv(), max_episode_steps=1), bolts={"end": bolt})
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (-3, False, True)
    assert info["restraining_bolts"]["occurrences"] == {"end": True}
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_reset_restart_applies_and_nonaccepting_followup_has_no_payment():
    bolt = spec("[a]", reporting="restart")
    env = RestrainingBoltWrapper(BoltEnv(), bolts={"b": bolt})
    observation, info = env.reset()
    assert info["restraining_bolts"]["states"] == {"b": bolt.definition.initial_state}
    assert np.argmax(observation["automata"]) == env.bolt_metadata["b"].state_indices[bolt.definition.initial_state]
    env.unwrapped.info["labels"] = frozenset()
    observation, reward, _, _, info = env.step(0)
    assert reward == 2
    assert info["restraining_bolts"]["reward_adjustments"] == {"b": 0}
