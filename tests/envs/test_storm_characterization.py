"""Durable fingerprints for unchanged dynamics, weather labels and warning behavior."""

import pytest
from storm_taxi_characterization import delivery_trajectory_fingerprint, dynamics_fingerprints, trajectory_fingerprint

from npc_gym.envs import StormTaxiEnv

# Dynamics captured at d7adb91ae251cea88403eaecc74ca8e0af09f192; never regenerate from a replacement.
DYNAMICS = {
    "transitions": "5584d22efa34babb9d850e10d34c71fc18aa527cb358edb59edc55fa36623d90",
    "masks": "69e30710dec9c9a399c9777f8479eb926121b90b9efe8d6a6714f6faab9fd602",
    "initial": "e42516e8aaca98a4f0a730837f0d6728174427938c1d5933ff803d4f16f8ea3f",
}
# Unchanged trajectory fields captured with the pre-fix labels from dc4fba33.
TRAJECTORIES = {
    False: "2b8d30ae8ef590f71f60f44cbf24c87c91ce8c0fbdf639c032004fa3ad542b41",
    True: "a9426baac7937bfb887afa25205da6d239134b9d15405fa610b8a717305b17b0",
}


@pytest.fixture
def factory():
    return StormTaxiEnv


@pytest.fixture
def storm(factory):
    env = factory()
    yield env
    env.close()


def test_full_dynamics_fingerprint(storm):
    assert dynamics_fingerprints(storm) == DYNAMICS


@pytest.mark.parametrize("fickle", [False, True])
def test_seeded_behavior_fingerprint(factory, fickle):
    env = factory(fickle_passenger=fickle)
    try:
        assert trajectory_fingerprint(env) == TRAJECTORIES[fickle]
    finally:
        env.close()


@pytest.mark.parametrize("rain", [False, True])
@pytest.mark.parametrize("hurricane", range(11))
@pytest.mark.parametrize("action, passenger, ordinary_reward, delivered", [(4, 1, -10, False), (5, 4, 20, True)])
def test_weather_specific_pickup_delivery_rewards(
    storm, rain, hurricane, action, passenger, ordinary_reward, delivered
):
    state = storm.encode(0, 0, passenger, 0, rain, hurricane, 1, False, 3)
    outcomes = storm.transition_distribution(state, action)
    for _, successor, reward, terminated in outcomes:
        assert reward == (ordinary_reward if rain and hurricane == 0 else -1)
        assert terminated is delivered
        assert tuple(storm.decode(successor))[:4] == (0, 0, 0 if delivered else passenger, 0)


@pytest.mark.parametrize("hurricane", range(1, 11))
def test_warning_advances_hurricane_without_movement(storm, hurricane):
    state = storm.encode(2, 3, 4, 1, True, hurricane, 0, True, 2)
    expected = storm.encode(2, 3, 4, 1, True, hurricane % 10 + (hurricane != 10), 0, True, 2)
    assert storm.transition_distribution(state, 6) == ((1.0, expected, -1, False),)


def test_clear_warning_preserves_order_of_weather_outcomes(storm):
    state = storm.encode(2, 3, 4, 1, False, 0, 0, True, 2)
    rainy = storm.encode(2, 3, 4, 1, True, 0, 0, True, 2)
    hurricane = storm.encode(2, 3, 4, 1, True, 1, 0, True, 2)
    assert storm.transition_distribution(state, 6) == (
        (0.75, state, -1, False),
        (0.2, rainy, -1, False),
        (0.05, hurricane, -1, False),
    )


def test_fickle_destination_changes_once_after_successful_movement(storm):
    storm.fickle_passenger = True
    storm.reset(seed=2)
    assert storm.fickle_step
    storm.s = storm.encode(0, 0, 0, 1, False, 0, 0, False, 3)
    for index, (action, state) in enumerate(zip((4, 0, 2, 0), (12323, 83427, 97507, 167907), strict=True)):
        result = storm.step(action)
        assert result[:4] == (state, -1, False, False)
        assert storm.fickle_step == (index == 0)


@pytest.mark.parametrize("fickle", [False, True])
def test_goal_directed_seeded_behavior(factory, fickle):
    # Both trajectory fingerprints exclude location labels and safety counts.
    # Captured using the pre-fix labeling function from dc4fba33, then matched against the correction.
    expected = {
        False: "7b962718dd5c3644c6a327ce233feb9c9659db444f31f6c9fb9d23e4a1dc85ed",
        True: "1f5f48bd69af2156df71c2676e6cc82d4037db4ff6a452f12accd3e91f596b22",
    }
    with factory(fickle_passenger=fickle) as env:
        digest, coverage = delivery_trajectory_fingerprint(env)
    assert digest == expected[fickle]
    assert coverage["deliveries"] == 48
    assert coverage["pickups"] >= 48
    assert coverage["rain_deliveries"] > 0
    assert (coverage["destination_changes"] > 0) is fickle
