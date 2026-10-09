"""Temporal boundaries retained from imperative/formula monitor regressions."""

from itertools import product

import pytest

from npc_gym.envs import MerchantEnv
from npc_gym.envs.gardener.labels import FrogCollected
from npc_gym.envs.merchant.labels import MerchantLabel as M
from npc_gym.envs.pacman.labels import PacmanLabel as P
from npc_gym.envs.taxi.labels import TaxiActionLabel as A
from npc_gym.envs.taxi.labels import TaxiLocationLabel as L
from npc_gym.envs.taxi.labels import TaxiWeatherLabel as W
from npc_gym.monitors import MonitorInput, UnknownMonitorIDError, make_builtin_monitor


def letter(*labels, **flags):
    return MonitorInput(frozenset(labels), **flags)


EMPTY = letter()


@pytest.mark.parametrize("visit_input", [1, 14, 15, 16])
@pytest.mark.parametrize("wood", [False, True])
def test_evolving_phase_and_reset(visit_input, wood):
    monitor = make_builtin_monitor("merchant/evolving-v0")
    for _ in range(2):
        for index in range(1, visit_input + 1):
            input = letter(M.AT_TREE, *([M.HAS_WOOD] if wood else [])) if index == visit_input else EMPTY
            assert not (monitor.reset if index == 1 else monitor.update)(input)
        assert monitor.update(letter(M.EXTRACT)) == (visit_input < 15 or wood)
        assert monitor.count == int(visit_input < 15 or wood)
    monitor.reset(EMPTY)
    assert not monitor.update(letter(M.EXTRACT))


@pytest.mark.parametrize("visit_input,expected", [(14, True), (15, False)])
def test_evolving_clock_survives_discharge(visit_input, expected):
    monitor = make_builtin_monitor("merchant/evolving-v0")
    monitor.reset(letter(M.AT_TREE))
    assert monitor.update(letter(M.EXTRACT))
    for _ in range(3, visit_input):
        monitor.update(EMPTY)
    monitor.update(letter(M.AT_TREE))
    assert monitor.update(letter(M.EXTRACT)) == expected


@pytest.mark.parametrize("offset", [0, 1, 12, 13, 14, 15])
@pytest.mark.parametrize("wood", [False, True])
def test_evolving_alternating_events_cross_phase_boundary(offset, wood):
    monitor = make_builtin_monitor("merchant/evolving-v0")
    for index in range(45):
        labels = () if index < offset else (M.AT_TREE, M.EXTRACT, *([M.HAS_WOOD] if wood else []))
        occurred = (monitor.reset if index == 0 else monitor.update)(letter(*labels))
        # Success consumes the input without rearming, including at the phase boundary.
        expected = index >= offset and (index - offset) % 2 == 1 and (wood or index <= 14)
        assert occurred == expected


def test_evolving_extraction_no_replay_and_rearming():
    monitor = make_builtin_monitor("merchant/evolving-v0")
    tree = letter(M.AT_TREE, M.HAS_WOOD)
    assert not monitor.reset(tree)
    assert monitor.update(letter(M.EXTRACT, M.AT_TREE, M.HAS_WOOD))
    assert not monitor.update(letter(M.EXTRACT))
    assert not monitor.update(tree)
    assert monitor.update(letter(M.EXTRACT))
    monitor.reset(EMPTY)
    assert not monitor.update(letter(M.EXTRACT))
    assert monitor.count == 0


def test_env_friendly_counts_overlapping_attempts_and_clears_on_reset():
    monitor = make_builtin_monitor("merchant/env-friendly-v0")
    tree = letter(M.AT_TREE, M.HAS_WOOD)
    assert not monitor.reset(tree)
    for count in range(1, 5):
        assert monitor.update(letter(M.EXTRACT, M.AT_TREE, M.HAS_WOOD))
        assert monitor.count == count
    assert not monitor.update(EMPTY)
    assert not monitor.update(letter(M.EXTRACT))  # Only the immediately preceding input matters.
    assert not monitor.update(letter(M.AT_TREE))
    assert not monitor.update(letter(M.EXTRACT))  # No wood carried.
    assert not monitor.update(tree)
    assert not monitor.reset(EMPTY)
    assert not monitor.update(letter(M.EXTRACT))
    assert monitor.count == 0


@pytest.mark.parametrize("capacity", [1, 5])
def test_env_friendly_successful_and_full_inventory_extraction(capacity):
    env = MerchantEnv(layout="basic", risk_fight=0, capacity=capacity)
    monitor = make_builtin_monitor("merchant/env-friendly-v0")
    try:
        _, info = env.reset(seed=0)
        monitor.reset(MonitorInput(info["labels"]))

        def step(action):
            observation, reward, terminated, truncated, info = env.step(action)
            event = monitor.update(MonitorInput(info["labels"], terminated, truncated))
            return observation, reward, event

        # Walk from home to the first tree, then collect the first wood freely.
        for action in [0] * 3 + [2] * 5:
            assert not step(action)[2]
        observation, reward, event = step(4)
        assert observation[3] == 1 and reward == 50 and not event
        for _ in range(4):
            assert not step(2)[2]  # Reach a second, intact tree carrying wood.
        for attempt in range(1, 5):
            observation, reward, event = step(4)
            assert event == (capacity == 1 or attempt == 1)
            assert monitor.count == (attempt if capacity == 1 else 1)
            assert observation[3] == (1 if capacity == 1 else 2)
            assert reward == (50 if capacity > 1 and attempt == 1 else 0)
    finally:
        env.close()


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_delivery_fulfillment_deadline_precedence_and_reset(ending):
    monitor = make_builtin_monitor("merchant/delivery-v0")
    monitor.reset(letter(M.AT_HOME))
    assert not monitor.update(letter(M.AT_MARKET))
    assert not monitor.update(letter(M.SUNDOWN))
    monitor.update(letter(M.AT_HOME))
    assert monitor.update(letter(M.AT_MARKET, M.SUNDOWN, **{ending: True}))
    assert not monitor.update(letter(M.SUNDOWN))
    monitor.reset(EMPTY)
    assert not monitor.update(letter(M.SUNDOWN))
    assert monitor.update(letter(M.AT_HOME, M.SUNDOWN))


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
@pytest.mark.parametrize("collect", [False, True])
def test_collect_one_checks_terminal_collection_and_ignores_reset(ending, collect):
    monitor = make_builtin_monitor("gardener/collect-one-v0")
    for _ in range(2):
        assert not monitor.reset(letter(FrogCollected(7)))
        assert not monitor.update(EMPTY)
        assert monitor.update(letter(*([FrogCollected(7)] if collect else []), **{ending: True})) == (not collect)
        assert monitor.count == int(not collect)


@pytest.mark.parametrize("prior_rain,delay", [(False, 7), (True, 3)])
@pytest.mark.parametrize("safe", [False, True])
def test_taxi_deadline_and_safety_on_last_input(prior_rain, delay, safe):
    monitor = make_builtin_monitor("taxi/emergency-v0")
    monitor.reset(letter(W.RAIN))  # Initial rain is ignored.
    if prior_rain:
        monitor.update(letter(W.RAIN, A.WARN))
    monitor.update(letter(W.NEW_HURRICANE))
    for _ in range(delay - 1):
        monitor.update(EMPTY)
        assert monitor.counts["Safety Violations"] == 0
    monitor.update(letter(*([L.AT_SHELTER] if safe else [])))
    assert monitor.counts["Safety Violations"] == int(not safe)
    monitor.update(EMPTY)
    assert monitor.counts["Safety Violations"] == int(not safe)
    monitor.reset(EMPTY)
    assert all(value == 0 for value in monitor.counts.values())


def test_taxi_two_deadlines_can_count_on_same_input():
    monitor = make_builtin_monitor("taxi/emergency-v0")
    monitor.reset(EMPTY)
    monitor.update(letter(W.NEW_HURRICANE))
    monitor.update(EMPTY)
    monitor.update(EMPTY)
    monitor.update(letter(W.RAIN))
    monitor.update(letter(W.NEW_HURRICANE))
    monitor.update(EMPTY)
    monitor.update(EMPTY)
    assert monitor.counts["Safety Violations"] == 0
    monitor.update(EMPTY)
    assert monitor.counts["Safety Violations"] == 2
    monitor.update(letter(W.HURRICANE))
    assert monitor.counts["Stay Violations"] == 0


def test_taxi_retrigger_restarts_own_timer():
    monitor = make_builtin_monitor("taxi/emergency-v0")
    monitor.reset(EMPTY)
    monitor.update(letter(W.NEW_HURRICANE))
    monitor.update(EMPTY)
    monitor.update(letter(W.NEW_HURRICANE))
    for _ in range(6):
        monitor.update(EMPTY)
    assert monitor.counts["Safety Violations"] == 0
    monitor.update(EMPTY)
    assert monitor.counts["Safety Violations"] == 1


@pytest.mark.parametrize("warn", [False, True])
def test_taxi_warning_after_rain_onset_and_stay(warn):
    monitor = make_builtin_monitor("taxi/emergency-v0")
    monitor.reset(EMPTY)
    monitor.update(letter(W.RAIN, W.HURRICANE, A.WARN))
    assert monitor.counts["Warn Violations"] == 0
    monitor.update(letter(W.RAIN, W.HURRICANE, *([A.WARN] if warn else [])))
    assert monitor.counts["Warn Violations"] == int(not warn)
    assert monitor.counts["Stay Violations"] == 2
    assert monitor.counts["Emergency Violations"] == 2 + int(not warn)


def test_taxi_warning_matches_rain_onsets_on_all_five_step_traces():
    monitor = make_builtin_monitor("taxi/emergency-v0")
    inputs = (EMPTY, letter(A.WARN), letter(W.RAIN), letter(W.RAIN, A.WARN))
    for trace in product(inputs, repeat=5):
        monitor.reset(letter(W.RAIN))  # Reset neither triggers a warning nor starts a rain spell.
        was_raining = due = False
        expected_count = 0
        for input in trace:
            expected = due and A.WARN not in input.labels
            raining = W.RAIN in input.labels
            due, was_raining = raining and not was_raining, raining
            expected_count += expected
            assert monitor.update(input)["Warn Violations"] == expected
            assert monitor.counts["Warn Violations"] == expected_count


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
@pytest.mark.parametrize("warn", [False, True])
def test_taxi_warning_checks_final_response_but_not_unreached_response(ending, warn):
    monitor = make_builtin_monitor("taxi/emergency-v0")
    monitor.reset(EMPTY)
    monitor.update(letter(W.RAIN, **{ending: True}))
    assert monitor.counts["Warn Violations"] == 0
    monitor.reset(EMPTY)
    monitor.update(letter(W.RAIN))
    monitor.update(letter(*([A.WARN] if warn else []), **{ending: True}))
    assert monitor.counts["Warn Violations"] == int(not warn)
    monitor.reset(EMPTY)
    monitor.update(EMPTY)
    assert monitor.counts["Warn Violations"] == 0


@pytest.mark.parametrize("identifier", ["pacman/obligation-blue-v1", "pacman/vegan-conflict-v1"])
@pytest.mark.parametrize("initial_adjacent", [False, True])
def test_blue_obligation_reset_and_fulfillment(identifier, initial_adjacent):
    monitor = make_builtin_monitor(identifier)

    def obligation_count():
        return monitor.count if identifier.endswith("obligation-blue-v1") else monitor.counts["OblBlue"]

    monitor.reset(letter(P.ADJACENT_BLUE_GHOST))
    monitor.update(EMPTY)
    monitor.update(letter(truncated=True))
    monitor.reset(letter(*([P.ADJACENT_BLUE_GHOST] if initial_adjacent else [])))
    for _ in range(3):
        monitor.update(EMPTY)
    assert obligation_count() == int(initial_adjacent)
    monitor.reset(letter(P.ADJACENT_BLUE_GHOST))
    monitor.update(letter(P.EAT_BLUE_GHOST))
    for _ in range(4):
        monitor.update(EMPTY)
    assert obligation_count() == 0


def test_blue_retains_expired_counter_until_inactive_input():
    monitor = make_builtin_monitor("pacman/obligation-blue-v1")
    adjacent = letter(P.ADJACENT_BLUE_GHOST)
    assert not monitor.reset(adjacent)
    assert [monitor.update(EMPTY) for _ in range(3)] == [False, False, True]
    assert monitor.update(adjacent)
    assert monitor.update(adjacent)
    assert not monitor.update(EMPTY)
    assert not monitor.update(adjacent)
    assert [monitor.update(EMPTY) for _ in range(3)] == [False, False, True]


def test_switch_counts_reset_as_first_of_forty_inputs():
    monitor = make_builtin_monitor("pacman/switch-v0")
    for _ in range(2):
        assert monitor.reset(letter(P.EAT_BLUE_GHOST))
        for _ in range(39):
            assert monitor.update(letter(P.EAT_BLUE_GHOST))
        assert not monitor.update(letter(P.EAT_BLUE_GHOST))
        assert monitor.update(letter(P.EAT_ORANGE_GHOST))
        assert monitor.count == 41


@pytest.mark.parametrize(
    "identifier,first,later,total",
    [("pacman/all-or-nothing-v0", True, False, "AllOrNothing"), ("pacman/one-taste-v1", False, True, "OneTaste")],
)
def test_first_and_later_tastes_count_both_colors(identifier, first, later, total):
    monitor = make_builtin_monitor(identifier)
    both = letter(P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST)
    for _ in range(2):
        assert sum(monitor.reset(both).values()) == 2 * first
        assert sum(monitor.update(both).values()) == 2 * later
        assert sum(monitor.update(both).values()) == 2 * later
        assert monitor.counts[total] == 2 * first + 4 * later


@pytest.mark.parametrize("ending", [{}, {"terminated": True}, {"truncated": True}])
@pytest.mark.parametrize("pause", [False, True])
def test_hungry_penalty_only_charges_following_input(ending, pause):
    monitor = make_builtin_monitor("pacman/hungry-vegan-penalty-v1")
    monitor.reset(EMPTY)
    occurred = monitor.update(letter(P.EAT_BLUE_GHOST))
    assert occurred["VegetarianBlue"] and not occurred["CTD"]
    occurred = monitor.update(letter(*([P.STAYED_STILL] if pause else []), **ending))
    assert occurred["CTD"] == (not pause)
    assert not any(monitor.update(EMPTY).values())
    assert monitor.counts == {
        "VegetarianBlue": 1,
        "VegetarianOrange": 0,
        "Hungr": 0,
        "CTD": int(not pause),
        "HungryVeganPenalty": 1 + int(not pause),
    }
    monitor.reset(EMPTY)
    assert not any(monitor.update(EMPTY).values())


def test_hungry_penalty_simultaneous_repeated_initial_eating():
    monitor = make_builtin_monitor("pacman/hungry-vegan-penalty-v1")
    both = letter(P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST)
    for _ in range(2):
        assert not monitor.reset(both)["CTD"]
        assert monitor.update(both)["CTD"]
        assert monitor.update(both)["CTD"]
        assert monitor.update(EMPTY)["CTD"]
        assert not any(monitor.update(EMPTY).values())
        assert monitor.counts["CTD"] == 3
        monitor.reset(both)
        assert not monitor.update(letter(*both.labels, P.STAYED_STILL))["CTD"]
        assert not monitor.update(letter(P.STAYED_STILL))["CTD"]


@pytest.mark.parametrize("duration", [1, 3])
def test_penalty_window_and_repeated_eating(duration):
    monitor = make_builtin_monitor(f"pacman/penalty-{duration}-v0")
    monitor.reset(letter(P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST))
    assert monitor.counts[f"Penalty{duration}(total)"] == 2
    for index in range(duration + 1):
        assert monitor.update(EMPTY)["CTD"] == (index < duration)
    assert monitor.counts["CTD"] == duration


@pytest.mark.parametrize(
    "name,target", [("hungry", P.EAT_BLUE_GHOST), ("errand", P.IN_CORNER), ("visit", P.IN_SOUTH_EAST)]
)
def test_deadline_obligations_fulfillment_wins_and_can_reactivate(name, target):
    monitor = make_builtin_monitor(f"pacman/{name}-v0")
    monitor.reset(letter(P.SCORE_0))
    assert not monitor.update(letter(target, P.LOSE))
    assert not monitor.update(letter(P.LOSE))
    monitor.update(letter(P.SCORE_0))
    assert monitor.update(letter(P.SCORE_GREATER_100))
    assert not monitor.update(letter(P.LOSE))
    monitor.reset(letter(P.SCORE_0))
    assert not monitor.update(letter(truncated=True))


@pytest.mark.parametrize(
    "identifier",
    [
        "gardener/rescue-v0",
        "gardener/drain-v0",
        "gardener/no-collect-v0",
        "gardener/collect-permission-v0",
        "pacman/one-taste-v0",
        "pacman/obligation-blue-v0",
        "pacman/vegan-conflict-v0",
        "pacman/hungry-vegan-penalty-v0",
        "pacman/maximum-v0",
        "pacman/solution-guilt-maximum-v0",
    ],
)
def test_deferred_and_retired_ids_are_not_available(identifier):
    with pytest.raises(UnknownMonitorIDError):
        make_builtin_monitor(identifier)


@pytest.mark.parametrize(
    "identifier,member",
    [("pacman/trapped-v1", None), ("pacman/maximum-v2", "Trapped"), ("pacman/solution-guilt-maximum-v2", "Maximum")],
)
def test_trapped_and_composites_release_at_400_independently_of_hungry(identifier, member):
    monitor = make_builtin_monitor(identifier)
    monitor.reset(letter(P.SCORE_0, P.WEST_SIDE))
    monitor.update(letter(P.SCORE_GREATER_100, P.WEST_SIDE))

    def count():
        return monitor.count if member is None else monitor.counts[member]

    before = count()
    monitor.update(letter(P.SCORE_GREATER_100))
    assert count() == before + 1  # Trapped remains active after Hungry's deadline.
    monitor.update(letter(P.SCORE_GREATER_100, P.SCORE_GREATER_400))
    assert count() == before + 1
    monitor.update(letter())
    assert count() == before + 1  # Falling below the threshold does not reactivate it.
    monitor.update(letter(P.SCORE_0))
    assert count() == before + 2
