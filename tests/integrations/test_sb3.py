from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from npc_gym.evaluation import EpisodeResult, EvaluationSummary, TerminationClass, evaluate
from npc_gym.integrations.sb3 import SB3EvaluationCallback, SB3Policy
from npc_gym.monitors import SimpleMonitor
from npc_gym.wrappers import MonitorWrapper

NORM_ID = "test/labels-v0"


class LabelMonitor(SimpleMonitor):
    def detect(self, input):
        return True


class VectorEpisodeEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(low=-1.0, high=100.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, slot: int, *, horizon: int, truncate: bool) -> None:
        self.slot = slot
        self.horizon = horizon
        self.truncate = truncate
        self.episode = -1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return np.array([0.0], dtype=np.float32), {
            "labels": frozenset({f"reset-{self.slot}-{self.episode}"}),
            "episode_metrics": {"slot": self.slot, "steps": 0},
        }

    def step(self, action: np.ndarray):
        assert action.shape == (1,)
        self.steps += 1
        done = self.steps == self.horizon
        return (
            np.array([self.steps], dtype=np.float32),
            float(self.slot + 1),
            done and not self.truncate,
            done and self.truncate,
            {
                "labels": frozenset({f"step-{self.slot}-{self.episode}-{self.steps}"}),
                "episode_metrics": {"slot": self.slot, "steps": self.steps},
            },
        )


class FakeModel:
    def __init__(self, env) -> None:
        self.env = env
        self.num_timesteps = 0

    def get_env(self):
        return self.env

    def predict(self, observation, *, deterministic=True):
        assert deterministic
        return np.array([0.25], dtype=np.float32), None


def one_episode_summary(monitor_id: str = "test/evaluation-v0") -> EvaluationSummary:
    return EvaluationSummary.from_episodes(
        (
            EpisodeResult(
                0,
                10.0,
                1,
                TerminationClass.TERMINATED,
                {"score": 10},
                {monitor_id: {"count": 4}},
            ),
        )
    )


def run_vector_steps(callback, model, count: int) -> None:
    for _ in range(count):
        _observations, rewards, dones, infos = model.env.step(np.array([[0.1], [0.2]], dtype=np.float32))
        model.num_timesteps += model.env.num_envs
        callback.update_locals({"rewards": rewards, "dones": dones, "infos": infos})
        assert callback.on_step()


@pytest.mark.parametrize("source", ["factories", "wrapper"])
def test_callback_tracks_parallel_autoresets_terminal_info_and_truncation(source):
    def make_env(slot, horizon, truncate):
        env = VectorEpisodeEnv(slot, horizon=horizon, truncate=truncate)
        return MonitorWrapper(env, monitors={NORM_ID: LabelMonitor}) if source == "wrapper" else env

    env = VecMonitor(
        DummyVecEnv(
            [
                lambda: make_env(0, 1, False),
                lambda: make_env(1, 2, True),
            ]
        )
    )
    model = FakeModel(env)
    callback = SB3EvaluationCallback(
        monitors={NORM_ID: LabelMonitor} if source == "factories" else None, monitor_source=source
    )
    try:
        env.reset()
        callback.init_callback(model)
        callback.on_training_start({}, {})
        run_vector_steps(callback, model, 2)

        assert [result.episode for result in callback.training_episodes] == [0, 1, 2]
        assert [result.termination for result in callback.training_episodes] == [
            TerminationClass.TERMINATED,
            TerminationClass.TERMINATED,
            TerminationClass.TRUNCATED,
        ]
        assert [result.episode_return for result in callback.training_episodes] == [1.0, 1.0, 4.0]
        assert [result.metrics for result in callback.training_episodes] == [
            {"slot": 0, "steps": 1},
            {"slot": 0, "steps": 1},
            {"slot": 1, "steps": 2},
        ]
        assert [result.monitor_counts for result in callback.training_episodes] == [
            {NORM_ID: {"count": 2}},
            {NORM_ID: {"count": 2}},
            {NORM_ID: {"count": 3}},
        ]
        assert callback.training_summary().mean_monitor_counts == {NORM_ID: {"count": pytest.approx(7 / 3)}}
        assert callback.training_summary().std_monitor_counts == {NORM_ID: {"count": pytest.approx(2**0.5 / 3)}}
    finally:
        env.close()


def test_callback_runs_in_an_sb3_training_loop():
    env = DummyVecEnv(
        [
            lambda: VectorEpisodeEnv(0, horizon=1, truncate=False),
            lambda: VectorEpisodeEnv(1, horizon=2, truncate=True),
        ]
    )
    try:
        model = A2C("MlpPolicy", env, n_steps=1, seed=0, verbose=0)
        callback = SB3EvaluationCallback(monitors={NORM_ID: LabelMonitor})

        model.learn(total_timesteps=4, callback=callback)

        assert model.num_timesteps == 4
        assert [result.termination for result in callback.training_episodes] == [
            TerminationClass.TERMINATED,
            TerminationClass.TERMINATED,
            TerminationClass.TRUNCATED,
        ]
    finally:
        env.close()


def test_intermediate_frequency_counts_transitions_and_stays_out_of_training_results():
    env = DummyVecEnv(
        [
            lambda: VectorEpisodeEnv(0, horizon=1, truncate=False),
            lambda: VectorEpisodeEnv(1, horizon=2, truncate=True),
        ]
    )
    model = FakeModel(env)
    evaluation_summary = one_episode_summary()
    evaluated_at: list[int] = []
    hook_calls: list[tuple[int, EvaluationSummary]] = []

    def evaluator(current_model):
        evaluated_at.append(current_model.num_timesteps)
        return evaluation_summary

    callback = SB3EvaluationCallback(
        monitors={NORM_ID: LabelMonitor},
        evaluator=evaluator,
        evaluation_frequency=3,
        on_evaluation=lambda step, summary: hook_calls.append((step, summary)),
    )
    try:
        env.reset()
        callback.init_callback(model)
        callback.on_training_start({}, {})
        run_vector_steps(callback, model, 3)

        assert evaluated_at == [0, 4, 6]
        assert [point.thresholds for point in callback.evaluations] == [(0,), (3,), (6,)]
        assert [point.timesteps for point in callback.evaluations] == [0, 4, 6]
        assert hook_calls == [(0, evaluation_summary), (4, evaluation_summary), (6, evaluation_summary)]
        assert all(
            result.monitor_counts == {NORM_ID: {"count": result.length + 1}} for result in callback.training_episodes
        )
    finally:
        env.close()


def test_training_aggregation_can_be_disabled_for_intermediate_only_callback():
    factory_called = False

    def unused_factory():
        nonlocal factory_called
        factory_called = True
        return LabelMonitor()

    env = DummyVecEnv(
        [
            lambda: VectorEpisodeEnv(0, horizon=1, truncate=False),
            lambda: VectorEpisodeEnv(1, horizon=2, truncate=True),
        ]
    )
    model = FakeModel(env)
    callback = SB3EvaluationCallback(
        monitors={NORM_ID: unused_factory},
        collect_training_episodes=False,
        evaluator=lambda model: one_episode_summary(),
        evaluation_frequency=1,
    )
    try:
        env.reset()
        callback.init_callback(model)
        callback.on_training_start({}, {})
        run_vector_steps(callback, model, 1)

        assert not factory_called
        assert callback.training_episodes == ()
        assert [point.thresholds for point in callback.evaluations] == [(0,), (1, 2)]
        assert [point.timesteps for point in callback.evaluations] == [0, 2]
        with pytest.raises(RuntimeError, match="disabled"):
            callback.training_summary()
    finally:
        env.close()


def test_callback_refuses_to_record_a_second_learn_call():
    env = DummyVecEnv([lambda: VectorEpisodeEnv(0, horizon=1, truncate=False)])
    model = FakeModel(env)
    callback = SB3EvaluationCallback(monitors={NORM_ID: LabelMonitor})
    try:
        env.reset()
        callback.init_callback(model)
        callback.on_training_start({}, {})

        with pytest.raises(RuntimeError, match="already recorded a learn"):
            callback.on_training_start({}, {})
    finally:
        env.close()


def test_training_episode_collector_refuses_to_attach_to_a_resumed_learn_call():
    env = DummyVecEnv([lambda: VectorEpisodeEnv(0, horizon=5, truncate=False)])
    model = A2C("MlpPolicy", env, n_steps=1, seed=0, verbose=0)
    callback = SB3EvaluationCallback()
    try:
        model.learn(total_timesteps=3)

        with pytest.raises(RuntimeError, match="must start at timestep zero"):
            model.learn(total_timesteps=3, callback=callback, reset_num_timesteps=False)

        assert callback.training_episodes == ()
    finally:
        env.close()


def test_intermediate_only_callback_can_attach_to_a_resumed_learn_call():
    env = DummyVecEnv([lambda: VectorEpisodeEnv(0, horizon=5, truncate=False)])
    model = A2C("MlpPolicy", env, n_steps=1, seed=0, verbose=0)
    callback = SB3EvaluationCallback(
        collect_training_episodes=False,
        evaluator=lambda model: one_episode_summary(),
        evaluation_frequency=2,
    )
    try:
        model.learn(total_timesteps=3)
        model.learn(total_timesteps=1, callback=callback, reset_num_timesteps=False)

        assert [(point.thresholds, point.timesteps) for point in callback.evaluations] == [((4,), 4)]
    finally:
        env.close()


def test_training_episodes_match_environment_neutral_evaluation():
    monitors = {NORM_ID: LabelMonitor}
    env = DummyVecEnv([lambda: VectorEpisodeEnv(0, horizon=3, truncate=True)])
    model = FakeModel(env)
    callback = SB3EvaluationCallback(monitors=monitors)
    try:
        env.reset()
        callback.init_callback(model)
        callback.on_training_start({}, {})
        for _ in range(3):
            _observations, rewards, dones, infos = env.step(np.array([[0.1]], dtype=np.float32))
            model.num_timesteps += env.num_envs
            callback.update_locals({"rewards": rewards, "dones": dones, "infos": infos})
            assert callback.on_step()
        (training,) = callback.training_episodes
    finally:
        env.close()

    evaluated = evaluate(
        TimeLimit(VectorEpisodeEnv(0, horizon=4, truncate=True), max_episode_steps=3),
        lambda observation, info: np.array([0.1], dtype=np.float32),
        monitors=monitors,
    ).episodes[0]

    assert training.episode_return == evaluated.episode_return
    assert training.length == evaluated.length
    assert training.termination == evaluated.termination
    assert training.monitor_counts == evaluated.monitor_counts
    assert training.metrics == evaluated.metrics


def test_sb3_policy_preserves_a_size_one_box_action():
    env = VectorEpisodeEnv(0, horizon=1, truncate=False)
    model = FakeModel(None)

    action = SB3Policy(model)(np.array([0.0], dtype=np.float32), {})

    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert env.action_space.contains(action)


@pytest.mark.parametrize("frequency", [True, 0, -1, 1.5])
def test_callback_rejects_invalid_evaluation_frequencies(frequency):
    with pytest.raises(ValueError, match="positive integer"):
        SB3EvaluationCallback(evaluator=lambda model: one_episode_summary(), evaluation_frequency=frequency)


def test_callback_requires_complete_intermediate_evaluation_configuration():
    with pytest.raises(ValueError, match="provided together"):
        SB3EvaluationCallback(evaluator=lambda model: one_episode_summary())
    with pytest.raises(ValueError, match="provided together"):
        SB3EvaluationCallback(evaluation_frequency=10)


def test_callback_copies_derived_counts_before_reset():
    from npc_gym.monitors import MultiMonitor

    def factory():
        return MultiMonitor(
            {"event": LabelMonitor(consume_initial=False)}, derived={"double": lambda c: 2 * c["event"]}
        )

    env = DummyVecEnv(
        [lambda: VectorEpisodeEnv(0, horizon=1, truncate=False), lambda: VectorEpisodeEnv(1, horizon=2, truncate=True)]
    )
    model = FakeModel(env)
    callback = SB3EvaluationCallback(monitors={NORM_ID: factory})
    try:
        env.reset()
        callback.init_callback(model)
        callback.on_training_start({}, {})
        run_vector_steps(callback, model, 2)
        assert [episode.monitor_counts for episode in callback.training_episodes] == [
            {NORM_ID: {"event": 1, "double": 2}},
            {NORM_ID: {"event": 1, "double": 2}},
            {NORM_ID: {"event": 2, "double": 4}},
        ]
    finally:
        env.close()


@pytest.mark.parametrize("monitors", [{}, {NORM_ID: LabelMonitor}])
def test_callback_rejects_both_monitor_sources(monitors):
    with pytest.raises(ValueError, match="cannot be selected together"):
        SB3EvaluationCallback(monitors=monitors, monitor_source="wrapper")


def test_callback_rejects_unknown_monitor_source():
    with pytest.raises(ValueError, match="monitor_source"):
        SB3EvaluationCallback(monitor_source="auto")


def test_wrapper_recording_in_real_sb3_training():
    env = DummyVecEnv(
        [
            lambda: MonitorWrapper(VectorEpisodeEnv(0, horizon=1, truncate=False), monitors={NORM_ID: LabelMonitor}),
            lambda: MonitorWrapper(VectorEpisodeEnv(1, horizon=2, truncate=True), monitors={NORM_ID: LabelMonitor}),
        ]
    )
    try:
        model = A2C("MlpPolicy", env, n_steps=1, seed=0, verbose=0)
        callback = SB3EvaluationCallback(monitor_source="wrapper")
        model.learn(total_timesteps=4, callback=callback)
        assert [result.monitor_counts for result in callback.training_episodes] == [
            {NORM_ID: {"count": 2}},
            {NORM_ID: {"count": 2}},
            {NORM_ID: {"count": 3}},
        ]
    finally:
        env.close()
