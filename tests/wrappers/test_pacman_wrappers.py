"""Trapped history is observable without changing task dynamics or rewards."""

import gymnasium as gym
import numpy as np
import pytest

from npc_gym.envs import PacmanEnv
from npc_gym.envs.pacman.labels import PacmanLabel as P
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.pacman_monitors import TRAPPED_NORM_ID, make_pacman_monitor
from npc_gym.wrappers.pacman_wrappers import TrappedObservation


class LabelTrace(gym.Env):
    action_space = gym.spaces.Discrete(1)

    def __init__(self, dtype):
        self.observation_space = gym.spaces.Box(-1, 1, (2,), dtype=dtype)
        self.obs = np.array([0.25, -0.5], dtype=dtype)
        self.labels = frozenset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.obs, {"labels": self.labels}

    def step(self, action):
        return self.obs, 7.0, False, True, {"labels": self.labels}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_history_dtype_bounds_final_observations_and_reset(dtype):
    env = LabelTrace(dtype)
    wrapped = TrappedObservation(env)
    monitor = make_pacman_monitor(TRAPPED_NORM_ID)
    env.labels = frozenset({P.SCORE_0})
    obs, info = wrapped.reset(seed=7)
    assert obs[-1] == 1 and monitor.reset(MonitorInput(info["labels"]))
    assert wrapped.observation_space.shape == (3,)
    for labels in (
        (),
        (P.SCORE_GREATER_100,),
        (P.SCORE_GREATER_400,),
        (),
        (P.SCORE_0,),
        (P.SCORE_0, P.SCORE_GREATER_400),
    ):
        env.labels = frozenset(labels)
        obs, reward, terminated, truncated, info = wrapped.step(0)
        assert obs.dtype == dtype and wrapped.observation_space.contains(obs)
        np.testing.assert_array_equal(obs[:-1], env.obs)
        assert reward == 7.0 and not terminated and truncated
        assert info == {"labels": env.labels}
        assert obs[-1] == int(monitor.update(MonitorInput(info["labels"])))
    env.labels = frozenset({P.SCORE_0})
    assert wrapped.reset()[0][-1] == 1
    env.labels = frozenset()
    assert wrapped.reset()[0][-1] == 0


@pytest.mark.parametrize(
    "space", [gym.spaces.Discrete(5), gym.spaces.Box(0, 255, (2,), dtype=np.uint8), gym.spaces.Box(0, 1, (2, 2))]
)
def test_invalid_observation_spaces(space):
    env = LabelTrace(np.float64)
    env.observation_space = space
    with pytest.raises(TypeError, match="one-dimensional floating-point Box"):
        TrappedObservation(env)


def test_real_pacman_trajectories_are_unchanged():
    with TrappedObservation(PacmanEnv(features="complete")) as wrapped, PacmanEnv(features="complete") as raw:
        obs, info = wrapped.reset(seed=7)
        original, original_info = raw.reset(seed=7)
        np.testing.assert_array_equal(obs[:-1], original)
        assert info == original_info and obs[-1] == 1
        for action in [0, 1, 2, 3, 4] * 3:
            observed = wrapped.step(action)
            expected = raw.step(action)
            np.testing.assert_array_equal(observed[0][:-1], expected[0])
            assert observed[1:] == expected[1:]
            if observed[2] or observed[3]:
                break


@pytest.mark.parametrize("behavior", ["random", "deterministic", "partly-deterministic"])
@pytest.mark.parametrize("with_bolts", [False, True])
def test_pixels_keep_numeric_schedule_and_bolt_fields(behavior, with_bolts):
    from npc_gym.bolts import make_builtin_bolts
    from npc_gym.envs import PacmanEnv
    from npc_gym.wrappers import PacmanPixelObservation, RestrainingBoltWrapper

    base = PacmanEnv(features="image-full", ghost_behavior=behavior)
    wrapped = (
        RestrainingBoltWrapper(base, bolts=make_builtin_bolts("pacman/vegan-v0", reward=-1)) if with_bolts else base
    )
    with PacmanPixelObservation(wrapped) as env:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        for _ in range(3):
            observation, *_ = env.step(0)
            assert env.observation_space.contains(observation)
            if behavior != "random":
                assert observation["ghost_mode"].tolist() == [0, base.state().phase_turns_remaining, 0]
            if with_bolts:
                assert observation["automata"].ndim == 1
        pixels = observation["observation"] if isinstance(observation, dict) else observation
        assert pixels.shape == (80, 210, 2)
