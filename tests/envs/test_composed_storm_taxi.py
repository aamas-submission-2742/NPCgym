"""Storm Taxi contracts, state isolation and bounded lazy transitions."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from npc_gym.envs import StormTaxiEnv
from npc_gym.envs.taxi._storm_weather import weather_outcomes


def assert_info_equal(first, second):
    np.testing.assert_array_equal(first["action_mask"], second["action_mask"])
    assert {key: value for key, value in first.items() if key != "action_mask"} == {
        key: value for key, value in second.items() if key != "action_mask"
    }


def test_spaces_checker_and_registration():
    with StormTaxiEnv() as env, gym.make("npc_gym/StormTaxi-v0") as registered:
        check_env(env, skip_render_check=True)
        assert env.unwrapped is env
        assert env.action_space.n == 7
        assert env.observation_space.n == 352000
        assert type(registered.unwrapped) is StormTaxiEnv
        assert registered.spec.max_episode_steps == 50
        assert not hasattr(env, "P")


def test_snapshots_require_reset_and_masks_are_detached():
    with StormTaxiEnv() as env:
        with pytest.raises(ResetNeeded):
            env.labeling_state()
        state, info = env.reset(seed=7)
        original = env.action_mask(state)
        info["action_mask"][:] = 0
        np.testing.assert_array_equal(env.action_mask(), original)
        snapshot = env.labeling_state()
        env.step(0)
        assert snapshot != env.labeling_state()


@pytest.mark.parametrize("action", [-1, 7, 1.5, "north"])
def test_invalid_actions(action):
    with StormTaxiEnv() as env:
        state, _ = env.reset(seed=7)
        before = env.np_random.bit_generator.state
        with pytest.raises(ValueError, match="Action.*outside"):
            env.step(action)
        with pytest.raises(ValueError, match="Action.*outside"):
            env.transition_distribution(state, action)
        assert env.get_state() == state
        assert env.np_random.bit_generator.state == before


@pytest.mark.parametrize("state", [-1, 352000, 1.5, "clear"])
def test_invalid_state_queries(state):
    with StormTaxiEnv() as env:
        with pytest.raises(ValueError, match="State.*outside"):
            env.transition_distribution(state, 6)
        with pytest.raises(ValueError, match="State.*outside"):
            env.action_mask(state)


def test_encoding_defaults_and_single_use_decoder():
    with StormTaxiEnv() as env:
        with pytest.raises(TypeError, match="home and shelter"):
            env.encode(0, 0, 0, 1)
        env.is_raining, env.hurricane, env.home, env.flood, env.shelter = True, 7, 2, True, 3
        state = env.encode(3, 2, 4, 1)
        decoded = env.decode(state)
        assert iter(decoded) is decoded
        assert tuple(decoded) == (3, 2, 4, 1, 1, 7, 2, 1, 3)
        assert tuple(decoded) == ()


def test_distribution_queries_do_not_mutate_state_and_cache_is_bounded():
    weather_outcomes.cache_clear()
    with StormTaxiEnv() as env:
        state, _ = env.reset(seed=7)
        before = env.np_random.bit_generator.state
        for value in range(5000):
            env.transition_distribution(value, value % 7)
        assert env.s == state
        assert env.np_random.bit_generator.state == before
        assert weather_outcomes.cache_info().maxsize == 4096
        assert weather_outcomes.cache_info().currsize <= 4096


def test_no_plain_weather_switches():
    for kwargs in ({"storm_risk": True}, {"is_rainy": True}):
        with pytest.raises(TypeError):
            StormTaxiEnv(**kwargs)


def test_two_composed_instances_have_independent_state_and_rng():
    with (
        StormTaxiEnv(fickle_passenger=True) as first,
        StormTaxiEnv(fickle_passenger=True) as second,
        StormTaxiEnv(fickle_passenger=True) as reference,
    ):
        first.reset(seed=7)
        second.reset(seed=2)
        reference.reset(seed=2)
        for action in range(7):
            first.reset(seed=action)
            first.step(6)
            actual, expected = second.step(action), reference.step(action)
            assert actual[:4] == expected[:4]
            assert_info_equal(actual[4], expected[4])
            assert second.np_random.bit_generator.state == reference.np_random.bit_generator.state
        first.close()
        assert second.step(0)[:4] == reference.step(0)[:4]
