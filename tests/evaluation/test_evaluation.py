from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit

from npc_gym.evaluation import EPISODE_METRICS_KEY, TerminationClass, evaluate
from npc_gym.monitors import MonitorInput, MultiMonitor, SimpleMonitor

NORM_ID = "test/deadline-v0"


class CountingEnv(gym.Env[int, int]):
    observation_space = gym.spaces.Discrete(4)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.steps = 0
        return 0, {
            "labels": frozenset({"reset"}),
            "action_mask": np.array([1, 0], dtype=np.int8),
            EPISODE_METRICS_KEY: {"progress": 0},
        }

    def step(self, action: int):
        assert action == 0
        self.steps += 1
        return (
            self.steps,
            float(self.steps),
            False,
            False,
            {
                "labels": frozenset({"step"}),
                "action_mask": np.array([1, 0], dtype=np.int8),
                EPISODE_METRICS_KEY: {"progress": self.steps},
            },
        )


class DeadlineMonitor(SimpleMonitor):
    def detect(self, input: MonitorInput) -> bool:
        return "reset" in input.labels or input.truncated


class MaskPolicy:
    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def __call__(self, observation: int, info: Mapping[str, Any]) -> int:
        self.seen.append((observation, int(info[EPISODE_METRICS_KEY]["progress"])))
        return int(np.flatnonzero(info["action_mask"])[0])


def test_evaluate_uses_public_info_and_preserves_time_limit_metrics():
    policy = MaskPolicy()
    env = TimeLimit(CountingEnv(), max_episode_steps=2)

    summary = evaluate(env, policy, episodes=2, monitors={NORM_ID: DeadlineMonitor}, seed=7)

    assert policy.seen == [(0, 0), (1, 1), (0, 0), (1, 1)]
    assert [result.episode_return for result in summary.episodes] == [3.0, 3.0]
    assert [result.length for result in summary.episodes] == [2, 2]
    assert [result.termination for result in summary.episodes] == [
        TerminationClass.TRUNCATED,
        TerminationClass.TRUNCATED,
    ]
    assert [result.monitor_counts for result in summary.episodes] == [{NORM_ID: {"count": 2}}, {NORM_ID: {"count": 2}}]
    assert [result.metrics for result in summary.episodes] == [{"progress": 2}, {"progress": 2}]
    assert summary.mean_return == 3.0
    assert summary.std_return == 0.0
    assert summary.mean_length == 2.0
    assert summary.std_length == 0.0
    assert summary.mean_monitor_counts == {NORM_ID: {"count": 2.0}}
    assert summary.std_monitor_counts == {NORM_ID: {"count": 0.0}}
    assert summary.mean_metrics == {"progress": 2.0}
    assert summary.std_metrics == {"progress": 0.0}


def test_unmonitored_environment_does_not_need_labels():
    class NoLabelsEnv(CountingEnv):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            del info["labels"]
            return observation, info

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            del info["labels"]
            return observation, reward, terminated, truncated, info

    summary = evaluate(TimeLimit(NoLabelsEnv(), max_episode_steps=1), lambda observation, info: 0)

    assert summary.episodes[0].monitor_counts == {}


def test_monitored_environment_requires_reset_and_transition_labels():
    class MissingLabelsEnv(CountingEnv):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            del info["labels"]
            return observation, info

    with pytest.raises(RuntimeError, match=r"info\['labels'\].*reset"):
        evaluate(
            TimeLimit(MissingLabelsEnv(), max_episode_steps=1),
            lambda observation, info: 0,
            monitors={NORM_ID: DeadlineMonitor},
        )


def test_configured_monitors_are_reported_when_they_emit_no_signals():
    class QuietMonitor(DeadlineMonitor):
        def detect(self, input):
            return False

    summary = evaluate(
        TimeLimit(CountingEnv(), max_episode_steps=1),
        lambda observation, info: 0,
        monitors={NORM_ID: QuietMonitor},
    )

    assert summary.episodes[0].monitor_counts == {NORM_ID: {"count": 0}}
    assert summary.mean_monitor_counts == {NORM_ID: {"count": 0.0}}


def test_evaluate_rejects_non_numeric_episode_metrics():
    class InvalidMetricEnv(CountingEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info[EPISODE_METRICS_KEY] = {"progress": "one"}
            return observation, reward, terminated, truncated, info

    with pytest.raises(TypeError, match="finite real number"):
        evaluate(TimeLimit(InvalidMetricEnv(), max_episode_steps=1), lambda observation, info: 0)


def test_evaluate_reports_simultaneous_termination_and_truncation():
    class FinishedEnv(CountingEnv):
        def step(self, action):
            observation, reward, _, _, info = super().step(action)
            return observation, reward, True, True, info

    summary = evaluate(FinishedEnv(), lambda observation, info: 0)

    assert summary.episodes[0].termination is TerminationClass.TERMINATED_AND_TRUNCATED


@pytest.mark.parametrize("episodes", [True, 0, -1, 1.5])
def test_evaluate_rejects_invalid_episode_counts(episodes):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate(CountingEnv(), lambda observation, info: 0, episodes=episodes)


def test_independent_events_and_explicit_total_survive_evaluation():
    from npc_gym.monitors.regex import from_regex

    def factory():
        return MultiMonitor(
            {name: from_regex(".*", propositions={}, consume_initial=False) for name in ("warn", "stay")},
            derived={"total": lambda c: sum(c.values())},
        )

    result = evaluate(TimeLimit(CountingEnv(), 1), lambda obs, info: 0, monitors={NORM_ID: factory})
    assert result.episodes[0].monitor_counts == {NORM_ID: {"warn": 1, "stay": 1, "total": 2}}
