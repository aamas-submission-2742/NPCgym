"""Storm Taxi's lazy transition and trajectory contracts."""

import numpy as np
import pytest


def test_construction_declares_the_storm_spaces(storm_taxi):
    assert storm_taxi.action_space.n == 7
    assert storm_taxi.observation_space.n == 352000


def test_seeded_reset_is_reproducible_and_valid(storm_taxi):
    first, info = storm_taxi.reset(seed=17)
    second, _ = storm_taxi.reset(seed=17)

    assert first == second
    assert storm_taxi.observation_space.contains(first)
    assert info["prob"] == 1.0
    assert info["action_mask"].shape == (storm_taxi.action_space.n,)
    assert info["action_mask"][6] == 1
    _, _, passenger, destination, *_ = storm_taxi.decode(first)
    assert passenger < 4
    assert passenger != destination


def test_every_kernel_result_is_a_complete_probability_distribution(storm_taxi):
    states = storm_taxi.observation_space.n
    malformed = []
    for state in range(states):
        for action in range(storm_taxi.action_space.n):
            transitions = storm_taxi.transition_distribution(state, action)
            total = sum(probability for probability, *_ in transitions)
            if not transitions or abs(total - 1.0) > 1e-9:
                malformed.append((state, action, total))
            elif any(not 0 <= next_state < states for _, next_state, _, _ in transitions):
                malformed.append((state, action, "successor out of range"))
    assert not malformed, f"{len(malformed)} malformed entries, first: {malformed[:5]}"


def test_every_storm_state_encoding_round_trip(storm_taxi):
    malformed = []
    for state in range(storm_taxi.observation_space.n):
        decoded = tuple(storm_taxi.decode(state))
        if storm_taxi.encode(*decoded) != state:
            malformed.append((state, "encoding round trip"))
    assert not malformed, f"{len(malformed)} encoding mismatches, first: {malformed[:5]}"


def test_step_accepts_the_seventh_action(storm_taxi):
    storm_taxi.reset(seed=3)
    state, reward, terminated, truncated, info = storm_taxi.step(6)
    assert storm_taxi.observation_space.contains(state)
    assert reward == -1
    assert not terminated
    assert not truncated
    assert info["action_mask"].shape == (storm_taxi.action_space.n,)
    assert info["action_mask"][6] == 1


def test_storm_startup_has_no_eager_transition_mapping(storm_taxi):
    assert not hasattr(storm_taxi, "P")
    assert storm_taxi.initial_state_distrib.shape == (storm_taxi.observation_space.n,)
    assert np.count_nonzero(storm_taxi.initial_state_distrib) == 3600
    assert storm_taxi.initial_state_distrib.sum() == pytest.approx(1.0)


def test_step_rejects_actions_outside_the_declared_space(storm_taxi):
    storm_taxi.reset(seed=5)
    with pytest.raises(ValueError, match="outside"):
        storm_taxi.step(storm_taxi.action_space.n)


def test_seeded_trajectory_matches_the_eager_kernel_baseline(storm_taxi):
    state, info = storm_taxi.reset(seed=7)
    trajectory = [(state, info["prob"])]
    label_trace = [info["labels"]]
    for action in (0, 2, 4, 1, 6, 3, 5, 0, 6, 2, 1, 5):
        state, reward, terminated, truncated, info = storm_taxi.step(action)
        trajectory.append((state, reward, terminated, truncated, info["prob"]))
        label_trace.append(info["labels"])

    assert trajectory == [
        (217552, 1.0),
        (288304, -1, False, False, 0.2),
        (288304, -1, False, False, 0.8),
        (293936, -1, False, False, 0.9),
        (223536, -1, False, False, 0.8),
        (223536, -1, False, False, 0.9),
        (223536, -1, False, False, 0.8),
        (223536, -10, False, False, 0.9),
        (293936, -1, False, False, 0.8),
        (293936, -1, False, False, 0.9),
        (293936, -1, False, False, 0.8),
        (223536, -1, False, False, 0.8),
        (223536, -10, False, False, 0.9),
    ]
    assert label_trace == [
        set(),
        {"south", "rain", "atHome", "toward(atHome)"},
        {"east", "rain", "atHome"},
        {"pickup", "rain", "hasPassenger", "atHome"},
        {"north", "rain", "hasPassenger", "toward(atDestination)", "toward(atShelter)"},
        {"warn", "rain", "hasPassenger"},
        {"west", "rain", "hasPassenger"},
        {"dropoff", "rain", "hasPassenger"},
        {"south", "rain", "hasPassenger", "atHome", "toward(atHome)"},
        {"warn", "rain", "hasPassenger", "atHome"},
        {"east", "rain", "hasPassenger", "atHome"},
        {"north", "rain", "hasPassenger", "toward(atDestination)", "toward(atShelter)"},
        {"dropoff", "rain", "hasPassenger"},
    ]
