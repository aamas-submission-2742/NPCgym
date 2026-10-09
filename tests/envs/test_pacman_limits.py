"""Registered map limits and explicit Gymnasium overrides."""

import gymnasium as gym
import pytest

from npc_gym.envs import PacmanEnv, PacmanLayout

CONFIGURATIONS = (
    ("npc_gym/Pacman-v1", "small", 300),
    ("npc_gym/PacmanMedium-v1", "medium", 500),
    ("npc_gym/PacmanLarge-v1", "large", 800),
)


@pytest.mark.parametrize(("environment_id", "name", "limit"), CONFIGURATIONS)
@pytest.mark.parametrize("override", [None, 2, 901])
def test_registered_limits_truncate_exactly_and_can_be_shortened_or_extended(environment_id, name, limit, override):
    layout = PacmanLayout.from_text("%%%%%\n%P .%\n%%%%%", name=name)
    kwargs = {} if override is None else {"max_episode_steps": override}
    with gym.make(environment_id, layout=layout, **kwargs) as env:
        assert env.unwrapped.layout.default_episode_steps == limit
        expected = limit if override is None else override
        assert env.spec.max_episode_steps == expected
        env.reset(seed=0)
        for turn in range(1, expected + 1):
            _, _, terminated, truncated, _ = env.step(0)
            assert not terminated
            assert truncated == (turn == expected)


@pytest.mark.parametrize(("environment_id", "name", "limit"), CONFIGURATIONS)
def test_registered_limit_can_be_disabled_and_direct_environment_is_unbounded(environment_id, name, limit):
    layout = PacmanLayout.from_text("%%%%%\n%P .%\n%%%%%", name=name)
    with gym.make(environment_id, layout=layout, max_episode_steps=-1) as registered, PacmanEnv(layout) as direct:
        for env in (registered, direct):
            env.reset(seed=0)
            for _ in range(limit + 1):
                _, _, terminated, truncated, _ = env.step(0)
                assert not terminated and not truncated
