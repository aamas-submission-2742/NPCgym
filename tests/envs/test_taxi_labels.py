"""Storm Taxi labels describe completed moves and actual safety locations."""

from collections import deque

import pytest

from npc_gym.envs import StormTaxiEnv
from npc_gym.envs.taxi.labels import TaxiLabelingFunction
from npc_gym.labels import Transition
from npc_gym.monitors import MonitorInput, make_builtin_monitor


@pytest.mark.parametrize(
    ("row", "column", "action", "expected"),
    [
        (0, 0, None, {"atHome", "hasPassenger"}),
        (0, 4, None, {"atDestination", "hasPassenger"}),
        (4, 0, None, {"atShelter", "hasPassenger"}),
        (0, 3, 2, {"east", "hasPassenger"}),
        (1, 0, 1, {"north", "hasPassenger"}),
        (3, 0, 0, {"south", "hasPassenger"}),
        (4, 1, 3, {"west", "hasPassenger"}),
    ],
)
def test_location_labels_use_current_position_without_inventing_previous_movement(row, column, action, expected):
    env = StormTaxiEnv()
    try:
        state = env.encode(row, column, 4, 1, False, 0, 0, False, 2)
        transition = Transition(previous_state=None, action=action, state=state, terminated=False, truncated=False)
        assert env.labeling_function(transition) == expected
    finally:
        env.close()


def test_destination_labels_report_arrival_on_the_actual_arrival_step():
    env = StormTaxiEnv()
    try:
        env.reset(seed=7)
        env.s = env.encode(0, 2, 4, 1, False, 0, 0, False, 2)
        state, _, _, _, info = env.step(2)
        assert tuple(env.decode(state))[:4] == (0, 3, 4, 1)
        assert info["labels"] == {"rain", "east", "toward(atDestination)", "hasPassenger"}
        assert env.labeling_state().taxi_position == (3, 0)
        state, _, _, _, info = env.step(2)
        assert tuple(env.decode(state))[:2] == (0, 4)
        assert info["labels"] == {"rain", "east", "atDestination", "toward(atDestination)", "hasPassenger"}
    finally:
        env.close()


@pytest.mark.parametrize("home", range(4))
def test_location_and_progress_labels_match_backend_routes_for_every_cell_and_action(home):
    with StormTaxiEnv() as env:
        destination, shelter = (home + 1) % 4, (home + 2) % 4
        targets = {"Home": env.locs[home], "Destination": env.locs[destination], "Shelter": env.locs[shelter]}
        successors = {}
        for row in range(5):
            for column in range(5):
                previous = env.encode(row, column, 4, destination, True, 2, home, False, shelter)
                successors[row, column] = []
                for action in range(7):
                    _, state, _, _ = env.transition_distribution(previous, action)[0]
                    successors[row, column].append((state, tuple(env.decode(state))[:2]))
        # Independent distances from the actual transition graph, including its walls.
        distances = {}
        for name, target in targets.items():
            distances[name] = {target: 0}
            queue = deque([target])
            while queue:
                position = queue.popleft()
                for _, successor in successors[position]:
                    if successor not in distances[name]:
                        distances[name][successor] = distances[name][position] + 1
                        queue.append(successor)
        for position, outcomes in successors.items():
            previous = env.encode(*position, 4, destination, True, 2, home, False, shelter)
            for action, (state, successor) in enumerate(outcomes):
                labels = env.labeling_function(Transition(previous, action, state, False, False))
                for name, target in targets.items():
                    assert (f"at{name}" in labels) == (successor == target)
                    assert (f"toward(at{name})" in labels) == (distances[name][successor] < distances[name][position])


def test_arriving_at_home_on_deadline_then_staying_is_safe():
    with StormTaxiEnv() as env:
        env.reset(seed=7)
        env.s = env.encode(1, 0, 4, 1, True, 1, 0, False, 2)
        monitor = make_builtin_monitor("taxi/emergency-v0")
        monitor.reset(MonitorInput(frozenset()))
        monitor.update(MonitorInput(frozenset({"rain"})))
        monitor.update(MonitorInput(env.labeling_function(Transition(None, None, env.s, False, False))))
        # Two stationary steps, physical arrival on the third, then a boundary-blocked move and waiting.
        for action in (6, 6, 1, 1, 6):
            _, _, terminated, truncated, info = env.step(action)
            monitor.update(MonitorInput(info["labels"], terminated, truncated))
        assert env.labeling_state().taxi_position == (0, 0)
        assert "atHome" in info["labels"]
        assert monitor.counts["Three-Step Safety Violations"] == 0
        assert monitor.counts["Stay Violations"] == 0
        _, _, terminated, truncated, info = env.step(0)
        monitor.update(MonitorInput(info["labels"], terminated, truncated))
        assert "atHome" not in info["labels"]
        assert monitor.counts["Stay Violations"] == 1


def test_blocked_move_cannot_clear_a_shelter_deadline():
    with StormTaxiEnv() as env:
        env.reset(seed=7)
        env.s = env.encode(4, 1, 4, 1, True, 1, 0, True, 2)
        monitor = make_builtin_monitor("taxi/emergency-v0")
        monitor.reset(MonitorInput(frozenset()))
        monitor.update(MonitorInput(frozenset({"rain"})))
        monitor.update(MonitorInput(env.labeling_function(Transition(None, None, env.s, False, False))))
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(3)
            assert env.labeling_state().taxi_position == (1, 4)
            assert "atShelter" not in info["labels"]
            assert "toward(atShelter)" not in info["labels"]
            monitor.update(MonitorInput(info["labels"], terminated, truncated))
        assert monitor.counts["Three-Step Safety Violations"] == 1


@pytest.mark.parametrize(
    ("rain", "hurricane", "flood", "expected"),
    [
        (False, 0, False, set()),
        (True, 0, False, {"rain"}),
        (True, 1, True, {"rain", "hurricane", "newHurricane", "floodrisk"}),
        (True, 7, False, {"rain", "hurricane"}),
    ],
)
def test_storm_weather_labels_come_from_encoded_state(rain, hurricane, flood, expected):
    env = StormTaxiEnv()
    try:
        state = env.encode(2, 2, 0, 1, rain, hurricane, 0, flood, 2)
        transition = Transition(previous_state=None, action=None, state=state, terminated=False, truncated=False)
        assert TaxiLabelingFunction()(transition) == expected
    finally:
        env.close()
