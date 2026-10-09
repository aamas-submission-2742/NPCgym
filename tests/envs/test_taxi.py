import gymnasium as gym
import numpy as np
import pytest

import npc_gym  # noqa: F401 - registers the environment
from npc_gym.envs import StormTaxiEnv
from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.envs.taxi.labels import (
    TaxiActionLabel,
    TaxiLabelingFunction,
    TaxiLocationLabel,
    TaxiPassengerLabel,
    TaxiWeatherLabel,
)
from npc_gym.labels import Transition
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant


def test_storm_state_encoding_round_trip():
    env = StormTaxiEnv()
    values = (3, 2, 4, 1, True, 7, 2, False, 3)
    assert tuple(env.decode(env.encode(*values))) == values


def test_storm_kernel_advances_hurricane():
    env = StormTaxiEnv()
    state = env.encode(2, 2, 0, 1, True, 4, 0, False, 3)
    transitions = env.transition_distribution(state, 0)
    assert len(transitions) == 1
    assert transitions[0][0] == 1.0
    assert tuple(env.decode(transitions[0][1]))[5] == 5


def test_action_mask_marks_pickup_and_dropoff():
    env = StormTaxiEnv()
    pickup_mask = env.action_mask(env.encode(0, 0, 0, 1, False, 0, 0, False, 2))
    assert pickup_mask[4] == 1
    assert pickup_mask[5] == 0
    dropoff_mask = env.action_mask(env.encode(0, 4, 4, 1, False, 0, 0, False, 2))
    assert dropoff_mask[5] == 1


def test_storm_action_mask_matches_action_space():
    env = StormTaxiEnv()
    state = env.encode(2, 2, 0, 1, False, 0, 0, False, 3)
    mask = env.action_mask(state)
    assert mask.shape == (env.action_space.n,)
    assert mask[6] == 1


def test_weather_wrapper_drops_only_norm_specific_fields():
    env = StormTaxiEnv()
    wrapper = IgnoreWeatherRelevant(env)
    state_a = env.encode(3, 2, 4, 1, True, 7, 0, False, 2)
    state_b = env.encode(3, 2, 4, 1, True, 7, 3, True, 0)
    assert wrapper.observation(state_a) == wrapper.observation(state_b)
    assert wrapper.observation_space.contains(wrapper.observation(state_a))


def test_taxi_labels_decode_storm_state():
    env = StormTaxiEnv()
    state = env.encode(0, 0, 4, 1, True, 3, 0, False, 2)
    labels = TaxiLabelingFunction()(
        Transition(previous_state=state, action=6, state=state, terminated=False, truncated=False)
    )
    assert {
        TaxiWeatherLabel.HURRICANE,
        TaxiWeatherLabel.RAIN,
        TaxiLocationLabel.AT_HOME,
        TaxiPassengerLabel.HAS_PASSENGER,
        TaxiActionLabel.WARN,
    } <= labels


def test_taxi_warning_action_uses_consistent_label():
    env = StormTaxiEnv()
    state = env.encode(0, 0, 4, 1, True, 3, 0, False, 2)
    labels = TaxiLabelingFunction()(
        Transition(previous_state=state, action=6, state=state, terminated=False, truncated=False)
    )
    assert TaxiActionLabel.WARN in labels


def test_registered_environment_has_time_limit():
    env = gym.make("npc_gym/StormTaxi-v0")
    try:
        observation, _ = env.reset(seed=4)
        assert env.observation_space.contains(observation)
        assert env.spec.max_episode_steps == 50
    finally:
        env.close()


def test_fickle_passenger_changes_destination_once_after_pickup():
    expected_states, expected_probabilities = [12323, 83427, 97507, 167907], [0.2, 0.8, 0.8, 0.8]
    env = StormTaxiEnv(fickle_passenger=True)
    try:
        env.reset(seed=2)
        assert env.fickle_step
        env.s = env.encode(0, 0, 0, 1, False, 0, 0, False, 3)
        for index, (action, expected_state, probability) in enumerate(
            zip((4, 0, 2, 0), expected_states, expected_probabilities, strict=True)
        ):
            observation, reward, terminated, truncated, info = env.step(np.int64(action))
            assert observation == env.get_state() == expected_state
            assert (reward, terminated, truncated) == (-1, False, False)
            assert info["prob"] == probability
            assert env.fickle_step == (index == 0)
            snapshot = env.labeling_state()
            assert snapshot.passenger == 4
            assert snapshot.destination == (1 if index == 0 else 2)
            assert "hasPassenger" in info["labels"]
            assert "rain" in info["labels"]
            assert info[EPISODE_METRICS_KEY] == {"success": 0}
    finally:
        env.close()


def test_storm_encode_uses_explicit_or_configured_weather_and_locations():
    env = StormTaxiEnv()
    try:
        # Unspecified locations have always failed, even before reset.
        with pytest.raises(TypeError):
            env.encode(0, 0, 0, 1)
        env.is_raining, env.hurricane, env.home, env.flood, env.shelter = True, 7, 2, True, 3
        state = env.encode(3, 2, 4, 1)
        decoded = env.decode(state)
        assert iter(decoded) is decoded
        assert tuple(decoded) == (3, 2, 4, 1, 1, 7, 2, 1, 3)
        assert tuple(decoded) == ()
        assert state == env.encode(3, 2, 4, 1, True, 7, 2, True, 3)
        explicit = env.encode(3, 2, 4, 1, False, 0, 0, False, 1)
        assert tuple(env.decode(explicit)) == (3, 2, 4, 1, 0, 0, 0, 0, 1)
    finally:
        env.close()
