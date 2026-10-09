from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from pacman_fixtures import TERMINAL_LAYOUT

import npc_gym  # noqa: F401 - registers the environments
from npc_gym.envs import (
    GardenerEnv,
    GardenerObservation,
    MerchantAuthorityState,
    MerchantEnv,
    PacmanAuthorityState,
    PacmanEnv,
    StormTaxiEnv,
    TaxiAuthorityState,
)
from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.labels import Transition
from npc_gym.wrappers import LabelingWrapper
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant

REGISTERED_ENVIRONMENTS = (
    ("npc_gym/StormTaxi-v0", TaxiAuthorityState),
    ("npc_gym/Merchant-v2", MerchantAuthorityState),
    ("npc_gym/Gardener-v0", GardenerObservation),
    ("npc_gym/Pacman-v1", PacmanAuthorityState),
)


@pytest.mark.parametrize(
    "factory",
    [
        StormTaxiEnv,
        MerchantEnv,
        lambda: GardenerEnv(size=8),
        lambda: PacmanEnv(layout="small", features="essential"),
    ],
)
def test_labeling_state_requires_an_explicit_reset(factory):
    env = factory()
    try:
        with pytest.raises(ResetNeeded, match=r"reset before labeling_state\(\)"):
            env.labeling_state()
    finally:
        env.close()


@pytest.mark.parametrize(("environment_id", "state_type"), REGISTERED_ENVIRONMENTS)
def test_registered_environments_expose_snapshots_on_reset_and_step(environment_id, state_type):
    env = gym.make(environment_id)
    try:
        _, reset_info = env.reset(seed=5)
        reset_state = env.unwrapped.labeling_state()
        assert isinstance(reset_state, state_type)
        assert "labeling_state" not in reset_info

        _, _, _, _, step_info = env.step(0)
        step_state = env.unwrapped.labeling_state()
        assert isinstance(step_state, state_type)
        assert "labeling_state" not in step_info

        reset_hash = hash(reset_state)
        env.reset(seed=6)
        assert hash(reset_state) == reset_hash
        assert reset_state is not env.unwrapped.labeling_state()
    finally:
        env.close()


def test_taxi_snapshot_decodes_storm_state():
    storm = StormTaxiEnv()
    try:
        storm.reset(seed=0)
        storm.s = storm.encode(1, 4, 2, 3, True, 7, 1, True, 0)
        snapshot = storm.labeling_state()
        assert snapshot.taxi_position == (4, 1)
        assert (snapshot.passenger, snapshot.destination) == (2, 3)
        assert (snapshot.raining, snapshot.hurricane, snapshot.flood_risk) == (True, 7, True)
        assert (snapshot.home, snapshot.shelter) == (1, 0)
    finally:
        storm.close()


def test_merchant_snapshot_contains_layout_inventory_time_and_resources():
    env = MerchantEnv(layout="basic")
    try:
        env.reset(seed=0)
        snapshot = env.labeling_state()

        assert snapshot.position == tuple(env.home)
        assert snapshot.cell == "H"
        assert (snapshot.carried_wood, snapshot.carried_ore, snapshot.time) == (0, 0, 0)
        assert snapshot.wood_available == (True,) * len(env.wood_positions)
        assert snapshot.ore_available == (True,) * len(env.ore_positions)
        assert snapshot.layout_name == "basic"
        assert snapshot.layout == tuple("".join(row) for row in env.map)
    finally:
        env.close()


def test_gardener_snapshot_reuses_the_decoded_observation_value():
    env = GardenerEnv(size=8)
    try:
        observation, _ = env.reset(seed=4)
        snapshot = env.labeling_state()

        assert snapshot == env.observation_codec.decode(observation)
        assert snapshot.size == 8
        assert len(snapshot.frogs) == len(snapshot.frog_timer) == len(snapshot.frog_collected)
        assert len(snapshot.puddles) == len(snapshot.puddle_timer) == len(snapshot.puddle_drained)
        assert len(snapshot.grass) == len(snapshot.grass_timer) == len(snapshot.grass_active)
    finally:
        env.close()


def test_pacman_snapshot_contains_geometry_entities_status_and_events():
    env = PacmanEnv(layout="small", features="essential")
    try:
        env.reset(seed=9)
        snapshot = env.labeling_state()

        assert snapshot.layout.name == "small"
        assert (snapshot.layout.width, snapshot.layout.height) == (13, 13)
        assert (0, 0) in snapshot.layout.walls
        assert snapshot.food == env.layout.food
        assert snapshot.capsules == env.layout.capsules
        assert snapshot.player.position == env.layout.player
        assert snapshot.player.alive
        assert snapshot.player.direction == 0
        assert len(snapshot.ghosts) == 2
        assert snapshot.ghosts[0].ghost_id == 1
        assert not snapshot.ghosts[0].scared
        assert snapshot.score == 0.0
        assert not snapshot.terminated and not snapshot.won and not snapshot.lost
        assert not snapshot.killed_blue and not snapshot.killed_orange and not snapshot.ate_capsule
        assert snapshot.last_action is None

        env.step(0)
        stepped = env.labeling_state()
        assert stepped.last_action == 0
        assert stepped.score == -1.0
    finally:
        env.close()


def test_pacman_completed_episode_remains_available_for_inspection():
    env = PacmanEnv(layout=TERMINAL_LAYOUT, features="essential")
    try:
        env.reset(seed=0)
        env.step(3)
        _, _, terminated, _, _ = env.step(3)

        snapshot = env.labeling_state()
        assert terminated
        assert snapshot.terminated and snapshot.lost and not snapshot.won
        assert not snapshot.player.alive
    finally:
        env.close()


def test_snapshots_are_deeply_immutable_public_values():
    environments = [
        StormTaxiEnv(),
        MerchantEnv(),
        GardenerEnv(size=8),
        PacmanEnv(layout="small", features="essential"),
    ]
    try:
        for env in environments:
            env.reset(seed=2)
            snapshot = env.labeling_state()
            _assert_public_immutable_value(snapshot)
            first_field = fields(snapshot)[0].name
            with pytest.raises(FrozenInstanceError):
                setattr(snapshot, first_field, None)
    finally:
        for env in environments:
            env.close()


@pytest.mark.parametrize(("environment_id", "steps"), [(item[0], 3) for item in REGISTERED_ENVIRONMENTS])
def test_requesting_snapshots_does_not_change_seeded_trajectories(environment_id, steps):
    inspected = gym.make(environment_id)
    control = gym.make(environment_id)
    try:
        inspected_observation, inspected_info = inspected.reset(seed=17)
        control_observation, control_info = control.reset(seed=17)
        _assert_transition_equal(inspected_observation, inspected_info, control_observation, control_info)
        initial_snapshot = inspected.unwrapped.labeling_state()
        initial_hash = hash(initial_snapshot)

        for _ in range(steps):
            inspected.unwrapped.labeling_state()
            inspected_result = inspected.step(0)
            control_result = control.step(0)
            _assert_step_equal(inspected_result, control_result)
            inspected.unwrapped.labeling_state()
            if inspected_result[2] or inspected_result[3]:
                break

        assert hash(initial_snapshot) == initial_hash
    finally:
        inspected.close()
        control.close()


def test_external_labels_can_use_hidden_taxi_authority_state():
    base = StormTaxiEnv()
    env = LabelingWrapper(
        IgnoreWeatherRelevant(base),
        lambda transition: frozenset({("home", transition.state.home), ("shelter", transition.state.shelter)}),
        state_extractor=lambda observation, info: base.labeling_state(),
        mode="replace",
    )
    try:
        observation, info = env.reset(seed=3)

        assert env.observation_space.contains(observation)
        assert info["labels"] == frozenset(
            {("home", base.labeling_state().home), ("shelter", base.labeling_state().shelter)}
        )
    finally:
        env.close()


def test_external_labels_can_use_pacman_state_hidden_by_observation_features():
    base = PacmanEnv(layout="small", features="hungry")

    def label(transition: Transition[PacmanAuthorityState, int]) -> frozenset[tuple[str, int]]:
        return frozenset({("food", len(transition.state.food)), ("walls", len(transition.state.layout.walls))})

    env = LabelingWrapper(
        base,
        label,
        state_extractor=lambda observation, info: base.labeling_state(),
        mode="replace",
    )
    try:
        observation, info = env.reset(seed=3)

        assert env.observation_space.contains(observation)
        assert info["labels"] == frozenset(
            {("food", len(base.labeling_state().food)), ("walls", len(base.labeling_state().layout.walls))}
        )
    finally:
        env.close()


def _assert_public_immutable_value(value: Any) -> None:
    assert "._engine" not in f"{type(value).__module__}.{type(value).__qualname__}"
    if is_dataclass(value):
        for field in fields(value):
            _assert_public_immutable_value(getattr(value, field.name))
    elif isinstance(value, (tuple, frozenset)):
        for item in value:
            _assert_public_immutable_value(item)
    else:
        assert value is None or isinstance(value, (bool, int, float, str))


def _assert_transition_equal(observation, info, expected_observation, expected_info):
    np.testing.assert_array_equal(observation, expected_observation)
    assert info["labels"] == expected_info["labels"]
    assert info[EPISODE_METRICS_KEY] == expected_info[EPISODE_METRICS_KEY]
    if "action_mask" in info:
        np.testing.assert_array_equal(info["action_mask"], expected_info["action_mask"])


def _assert_step_equal(actual, expected):
    _assert_transition_equal(actual[0], actual[4], expected[0], expected[4])
    assert actual[1:4] == expected[1:4]
