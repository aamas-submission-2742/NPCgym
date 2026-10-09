"""Scheduled ghosts queue at merges without blocking vacated cells or crossings."""

from dataclasses import replace

import numpy as np
import pytest

from npc_gym.envs.pacman.layout import VECTORS, PacmanLayout
from npc_gym.envs.pacman.simulation import Simulation, manhattan

LAYOUT = PacmanLayout.from_text(
    "%%%%%%%%%%%%%%%\n"
    "%.............%\n"
    "%.............%\n"
    "%..G.G.G.G....%\n"
    "%.............%\n"
    "%P............%\n"
    "%%%%%%%%%%%%%%%"
)


@pytest.fixture(params=["deterministic", "partly-deterministic"])
def sim(request):
    return Simulation(LAYOUT, np.random.default_rng(2), behavior=request.param)


def arrange(sim, monkeypatch, positions, actions):
    sim.ghosts = sim.ghosts[: len(positions)]
    for ghost, position, direction in zip(sim.ghosts, positions, actions, strict=True):
        ghost.position, ghost.direction = position, direction
    monkeypatch.setattr(sim, "ghost_action", lambda index: actions[index])


def positions(sim):
    return [ghost.position for ghost in sim.ghosts]


def test_followers_enter_cells_vacated_by_higher_id_leaders(sim, monkeypatch):
    arrange(sim, monkeypatch, [(3, 3), (4, 3), (5, 3), (6, 3)], [3, 3, 3, 3])
    sim.step(0)
    assert positions(sim) == [(4, 3), (5, 3), (6, 3), (7, 3)]


def test_merge_waits_separate_ghosts_that_would_share_a_destination(sim, monkeypatch):
    arrange(sim, monkeypatch, [(4, 3), (6, 3), (5, 2), (5, 4)], [3, 4, 1, 2])
    sim.step(0)
    assert positions(sim) == [(5, 3), (6, 3), (5, 2), (5, 4)]
    for _ in range(2):
        sim.step(0)
        assert len(set(positions(sim))) == 4


def test_stationary_occupant_propagates_wait_back_through_queue(sim, monkeypatch):
    start = [(3, 3), (4, 3), (5, 3), (6, 3)]
    arrange(sim, monkeypatch, start, [3, 3, 3, 0])
    sim.step(0)
    assert positions(sim) == start


@pytest.mark.parametrize("frightened", [False, True])
def test_head_on_ghosts_pass_without_persistent_overlap_or_deadlock(sim, monkeypatch, frightened):
    arrange(sim, monkeypatch, [(3, 3), (4, 3)], [3, 4])
    if frightened:
        for ghost in sim.ghosts:
            ghost.timer = 10
    sim.step(0)
    if frightened:
        assert positions(sim) == [(3.5, 3), (4, 3)]
        sim.step(0)
        assert positions(sim) == [(4, 3), (3.5, 3)]
    else:
        assert positions(sim) == [(4, 3), (3, 3)]


def test_closed_movement_cycle_advances_instead_of_deadlocking(sim, monkeypatch):
    arrange(sim, monkeypatch, [(3, 3), (4, 3), (4, 4), (3, 4)], [3, 1, 4, 2])
    sim.step(0)
    assert positions(sim) == [(4, 3), (4, 4), (3, 4), (3, 3)]


def test_blocked_half_cell_expiry_resumes_at_center_before_full_speed(sim, monkeypatch):
    actions = [3, 0]
    arrange(sim, monkeypatch, [(3.5, 3), (4, 3)], actions)
    sim.ghosts[0].timer = 1
    remaining = sim.phase_remaining
    sim.step(0)
    assert positions(sim) == [(3.5, 3), (4, 3)]
    assert sim.ghosts[0].timer == 0 and sim.phase_remaining == remaining
    actions[1] = 3
    sim.step(0)
    assert positions(sim) == [(4, 3), (5, 3)]
    assert sim.phase_remaining == remaining - 1
    sim.step(0)
    assert positions(sim) == [(5, 3), (6, 3)]


def test_queue_retains_pending_reversal_until_it_can_move(sim, monkeypatch):
    sim.ghosts = sim.ghosts[:2]
    sim.ghosts[0].position = (3, 3)
    sim.ghosts[1].position, sim.ghosts[1].direction = (4, 3), 3
    sim.ghosts[1].reverse_pending = True
    choose = sim.ghost_action
    monkeypatch.setattr(sim, "ghost_action", lambda index: 0 if index == 0 else choose(index))
    sim.step(0)
    assert sim.ghosts[1].position == (4, 3) and sim.ghosts[1].reverse_pending
    monkeypatch.setattr(sim, "ghost_action", lambda index: 4 if index == 0 else choose(index))
    sim.step(0)
    assert sim.ghosts[1].position == (3, 3) and not sim.ghosts[1].reverse_pending


def test_occupied_respawn_uses_nearest_free_cell_and_keeps_identity(sim):
    eaten, occupant = sim.ghosts[:2]
    occupant.position = eaten.spawn
    eaten.position, eaten.timer = sim.player, 10
    sim._collide(0)
    assert eaten.position == (3, 4)
    assert eaten.eaten == 1 and eaten.timer == 0
    assert sim.killed_blue and not sim.killed_orange
    assert len(set(positions(sim))) == 4


def test_respawn_does_not_place_a_dangerous_ghost_on_pacman(sim):
    ghost = sim.ghosts[0]
    sim.player = ghost.position = ghost.spawn
    ghost.timer = 10
    sim._collide(0)
    assert ghost.position != sim.player
    assert ghost.eaten == 1 and not sim.lost


def test_random_ghosts_still_respawn_at_their_occupied_start():
    sim = Simulation(LAYOUT, np.random.default_rng(2), behavior="random")
    eaten, occupant = sim.ghosts[:2]
    occupant.position = eaten.spawn
    eaten.position, eaten.timer = sim.player, 10
    sim._collide(0)
    assert eaten.position == occupant.position == eaten.spawn


@pytest.mark.parametrize("layout", ["small", "medium", "large"])
@pytest.mark.parametrize("behavior", ["deterministic", "partly-deterministic"])
def test_shared_pen_departures_remain_distinct_and_seeded(layout, behavior):
    first = Simulation(PacmanLayout.bundled(layout), np.random.default_rng(4), behavior=behavior)
    second = Simulation(first.layout, np.random.default_rng(4), behavior=behavior)
    for _ in range(40):
        if first.terminated:
            break
        first.step(0)
        second.step(0)
        assert first.full_snapshot() == second.full_snapshot()
        assert len(set(positions(first))) == len(first.ghosts)


@pytest.mark.parametrize("action", [1, 2, 3, 4])
def test_frightened_expiry_keeps_forward_half_step_in_every_direction(sim, monkeypatch, action):
    arrange(sim, monkeypatch, [(5, 3)], [action])
    sim.ghosts[0].timer = 1
    dx, dy = VECTORS[action]
    sim.step(0)
    assert positions(sim) == [(5 + dx / 2, 3 + dy / 2)]
    assert sim.ghosts[0].timer == 0
    sim.step(0)
    assert positions(sim) == [(5 + dx, 3 + dy)]


@pytest.mark.parametrize("action", [1, 2, 3, 4])
def test_faster_follower_waits_for_full_cell_gap(sim, monkeypatch, action):
    dx, dy = VECTORS[action]
    start = (5, 3)
    ahead = (5 + dx, 3 + dy)
    arrange(sim, monkeypatch, [start, ahead], [action, action])
    sim.ghosts[1].timer = 10
    sim.step(0)
    assert positions(sim) == [start, (ahead[0] + dx / 2, ahead[1] + dy / 2)]
    sim.step(0)
    assert manhattan(*positions(sim)) == 1
    assert sim.ghosts[0].position == ahead


def test_half_cell_merging_ghost_waits_for_crossing_to_clear(sim, monkeypatch):
    arrange(sim, monkeypatch, [(4, 3), (5, 2)], [3, 1])
    sim.ghosts[0].timer = 10
    sim.step(0)
    assert positions(sim) == [(4.5, 3), (5, 2)]
    assert manhattan(*positions(sim)) >= 1


def test_blocked_priority_contender_releases_destination_to_feasible_swap(sim, monkeypatch):
    # Ghost 0 wants the same target as 1, but cannot proceed if 1 waits.
    # Ghosts 1 and 2 can swap instead of freezing behind ghost 0's claim.
    arrange(sim, monkeypatch, [(4, 3), (5, 2), (5, 3)], [3, 1, 2])
    sim.step(0)
    assert positions(sim) == [(4, 3), (5, 3), (5, 2)]


def test_respawn_avoids_a_half_cell_neighbor(sim):
    eaten, occupant = sim.ghosts[:2]
    occupant.position = (eaten.spawn[0] + 0.5, eaten.spawn[1])
    eaten.position, eaten.timer = sim.player, 10
    sim._collide(0)
    assert eaten.position != eaten.spawn
    assert all(manhattan(eaten.position, ghost.position) >= 1 for ghost in sim.ghosts[1:])


def test_dense_custom_respawn_prefers_distinct_position_when_no_full_gap_exists():
    layout = PacmanLayout.from_text("%%%%%\n%PG.%\n%%%%%")
    layout = replace(layout, ghosts=((2, 1), (3, 1)))
    sim = Simulation(layout, np.random.default_rng(0), behavior="deterministic")
    sim.player = (1, 1)
    sim.ghosts[0].position, sim.ghosts[0].timer = sim.player, 10
    sim.ghosts[1].position = (2.5, 1)
    sim._collide(0)
    assert sim.ghosts[0].position == (2, 1)
    assert len(set(positions(sim))) == 2


@pytest.mark.parametrize("behavior", ["deterministic", "partly-deterministic"])
def test_audit_medium_seed_three_does_not_freeze(behavior):
    sim = Simulation(PacmanLayout.bundled("medium"), np.random.default_rng(3), behavior=behavior)
    actions = np.random.default_rng(3)
    previous = None
    stationary_turns = 0
    for _ in range(250):
        if sim.terminated:
            break
        sim.step(int(actions.integers(5)))
        current = positions(sim)
        stationary_turns = stationary_turns + 1 if current == previous else 0
        assert stationary_turns < 5
        assert len(set(current)) == len(current)
        previous = current
