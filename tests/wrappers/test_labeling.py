from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import TimeLimit

from npc_gym.evaluation import EPISODE_METRICS_KEY, evaluate
from npc_gym.labels import Transition
from npc_gym.monitors import MonitorInput, SimpleMonitor
from npc_gym.wrappers import LabelingWrapper

NORM_ID = "external/ordered-actions-v0"


class ReusedArrayEnv(gym.Env[np.ndarray, int]):
    observation_space = gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.observation = np.zeros(1, dtype=np.int64)
        self.last_info: dict[str, Any] | None = None
        self.received_actions: list[int] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.observation[0] = 0
        self.last_info = {
            "labels": frozenset({"inner"}),
            "action_mask": np.array([1, 1], dtype=np.int8),
            EPISODE_METRICS_KEY: {"progress": 0},
        }
        return self.observation, self.last_info

    def step(self, action: int):
        self.received_actions.append(action)
        self.observation[0] += 1
        done = bool(self.observation[0] == 2)
        self.last_info = {
            "labels": frozenset({"inner"}),
            "action_mask": np.array([1, 1], dtype=np.int8),
            EPISODE_METRICS_KEY: {"progress": int(self.observation[0])},
        }
        return self.observation, float(action + 1), done, done, self.last_info


class HistoryMonitor(SimpleMonitor):
    """A deliberately stateful monitor not represented by a DFA."""

    def __init__(self) -> None:
        super().__init__(consume_initial=False)
        self.actions: list[int] = []

    def reset_history(self):
        self.actions.clear()

    def detect(self, input: MonitorInput) -> bool:
        self.actions.extend(label[1] for label in input.labels if isinstance(label, tuple) and label[0] == "action")
        return input.terminated and input.truncated and self.actions == [0, 0]


def transition_labels(transition: Transition[np.ndarray, int]) -> frozenset[object]:
    if transition.action is None:
        return frozenset({"reset", ("state", int(transition.state[0]))})
    return frozenset({("action", transition.action), ("state", int(transition.state[0]))})


def test_external_labeler_and_stateful_monitor_work_through_evaluation():
    env = LabelingWrapper(ReusedArrayEnv(), transition_labels)

    summary = evaluate(env, lambda observation, info: 0, episodes=2, monitors={NORM_ID: HistoryMonitor}, seed=3)

    assert [result.episode_return for result in summary.episodes] == [2.0, 2.0]
    assert [result.monitor_counts for result in summary.episodes] == [{NORM_ID: {"count": 1}}, {NORM_ID: {"count": 1}}]
    assert [result.metrics for result in summary.episodes] == [{"progress": 2}, {"progress": 2}]


def test_reset_and_step_build_stable_transitions_without_changing_other_results():
    transitions: list[Transition[np.ndarray, int]] = []

    def record(transition: Transition[np.ndarray, int]) -> frozenset[str]:
        transitions.append(transition)
        return frozenset({"external"})

    inner = ReusedArrayEnv()
    env = LabelingWrapper(inner, record)

    reset_observation, reset_info = env.reset(seed=7)
    reset_inner_info = inner.last_info
    observation, reward, terminated, truncated, info = env.step(1)

    assert reset_observation is observation is inner.observation
    assert reward == 2.0
    assert not terminated and not truncated
    assert np.array_equal(info["action_mask"], np.array([1, 1], dtype=np.int8))
    assert info[EPISODE_METRICS_KEY] == {"progress": 1}
    assert reset_info["labels"] == info["labels"] == frozenset({"inner", "external"})
    assert reset_inner_info is not None
    assert reset_inner_info["labels"] == frozenset({"inner"})
    assert np.array_equal(reset_inner_info["action_mask"], np.array([1, 1], dtype=np.int8))
    assert reset_inner_info[EPISODE_METRICS_KEY] == {"progress": 0}
    assert inner.last_info is not info
    assert inner.last_info["labels"] == frozenset({"inner"})

    reset_transition, step_transition = transitions
    assert reset_transition.previous_state is None
    assert reset_transition.action is None
    assert not reset_transition.terminated and not reset_transition.truncated
    assert step_transition.action == 1
    assert not step_transition.terminated and not step_transition.truncated
    assert np.array_equal(reset_transition.state, np.array([0]))
    assert np.array_equal(step_transition.previous_state, np.array([0]))
    assert np.array_equal(step_transition.state, np.array([1]))

    env.step(0)
    assert np.array_equal(step_transition.state, np.array([1]))


def test_state_extractor_receives_public_info_and_final_lifecycle_flags():
    seen: list[Transition[tuple[int, int], int]] = []

    def extract(observation: np.ndarray, info: Mapping[str, Any]) -> tuple[int, int]:
        return int(observation[0]), int(info[EPISODE_METRICS_KEY]["progress"])

    def label(transition: Transition[tuple[int, int], int]) -> frozenset[str]:
        seen.append(transition)
        return frozenset()

    env = LabelingWrapper(ReusedArrayEnv(), label, state_extractor=extract, mode="replace")
    env.reset()
    env.step(0)
    _, _, terminated, truncated, info = env.step(0)

    assert terminated and truncated
    assert info["labels"] == frozenset()
    assert seen[-1] == Transition(previous_state=(1, 1), action=0, state=(2, 2), terminated=True, truncated=True)


def test_replace_mode_discards_existing_labels():
    env = LabelingWrapper(ReusedArrayEnv(), lambda transition: frozenset({"external"}), mode="replace")

    _, reset_info = env.reset()
    _, _, _, _, step_info = env.step(0)

    assert reset_info["labels"] == step_info["labels"] == frozenset({"external"})


@pytest.mark.parametrize("output", [set(), [], ("label",), "label"])
def test_labeler_must_return_a_frozen_set(output):
    env = LabelingWrapper(ReusedArrayEnv(), lambda transition: output)

    with pytest.raises(TypeError, match="labeling function output on reset must be a frozenset"):
        env.reset()


def test_extend_mode_rejects_malformed_existing_labels():
    class MalformedInfoEnv(ReusedArrayEnv):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            info["labels"] = ["inner"]
            return observation, info

    env = LabelingWrapper(MalformedInfoEnv(), lambda transition: frozenset())

    with pytest.raises(TypeError, match=r"existing info\['labels'\] on reset must be a frozenset"):
        env.reset()


def test_step_before_reset_is_rejected_without_stepping_inner_environment():
    inner = ReusedArrayEnv()
    env = LabelingWrapper(inner, transition_labels)

    with pytest.raises(RuntimeError, match=r"step\(\) called before reset\(\)"):
        env.step(0)

    assert inner.received_actions == []


def test_failed_reset_clears_the_previous_episode_state():
    calls = 0

    def fail_second_reset(transition: Transition[np.ndarray, int]):
        nonlocal calls
        if transition.action is None:
            calls += 1
        return [] if calls == 2 else frozenset()

    env = LabelingWrapper(ReusedArrayEnv(), fail_second_reset)
    env.reset()
    env.step(0)

    with pytest.raises(TypeError, match="labeling function output on reset"):
        env.reset()
    with pytest.raises(RuntimeError, match=r"step\(\) called before reset\(\)"):
        env.step(0)


def test_wrapper_boundary_action_and_time_limit_flags_are_explicit():
    class RemapOneToZero(gym.ActionWrapper):
        def action(self, action):
            return 0

    seen: list[Transition[np.ndarray, int]] = []

    def record(transition: Transition[np.ndarray, int]) -> frozenset[str]:
        seen.append(transition)
        return frozenset()

    inner = ReusedArrayEnv()
    env = LabelingWrapper(TimeLimit(RemapOneToZero(inner), max_episode_steps=1), record)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(1)

    assert inner.received_actions == [0]
    assert reward == 1.0
    assert not terminated and truncated
    assert seen[-1].action == 1
    assert not seen[-1].terminated and seen[-1].truncated

    seen.clear()
    inside_time_limit = TimeLimit(LabelingWrapper(ReusedArrayEnv(), record), max_episode_steps=1)
    inside_time_limit.reset()
    _, _, _, truncated, _ = inside_time_limit.step(0)
    assert truncated
    assert not seen[-1].truncated


def test_mutable_action_is_snapshotted_at_the_wrapper_boundary():
    class MutatingActionEnv(gym.Env[np.ndarray, np.ndarray]):
        observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64)
        action_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.int64)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.int64), {}

        def step(self, action):
            action[0] = 9
            return np.ones(1, dtype=np.int64), 0.0, True, False, {}

    seen: list[Transition[np.ndarray, np.ndarray]] = []

    def record(transition: Transition[np.ndarray, np.ndarray]) -> frozenset[object]:
        seen.append(transition)
        return frozenset()

    env = LabelingWrapper(MutatingActionEnv(), record)
    env.reset()
    action = np.array([1], dtype=np.int64)
    env.step(action)

    assert np.array_equal(action, np.array([9]))
    assert np.array_equal(seen[-1].action, np.array([1]))


def test_vector_autoreset_keeps_final_transition_labels():
    def make_env():
        inner = TimeLimit(ReusedArrayEnv(), max_episode_steps=1)
        return LabelingWrapper(inner, transition_labels, mode="replace")

    env = SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)
    try:
        _, reset_info = env.reset(seed=11)
        _, _, terminated, truncated, info = env.step(np.array([0]))

        assert not terminated[0] and truncated[0]
        assert reset_info["labels"][0] == frozenset({"reset", ("state", 0)})
        assert info["labels"][0] == frozenset({"reset", ("state", 0)})
        assert info["final_info"]["labels"][0] == frozenset({("action", 0), ("state", 1)})
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["merge", "", True])
def test_invalid_label_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="mode must be 'extend' or 'replace'"):
        LabelingWrapper(ReusedArrayEnv(), transition_labels, mode=mode)
