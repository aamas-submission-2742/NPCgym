"""Focused transition rules on original, small synthetic mazes."""

import numpy as np

from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.simulation import Simulation


def game(text="%%%%%%%%%\n%P o G .%\n%%%%%%%%%"):
    return Simulation(PacmanLayout.from_text(text), np.random.default_rng(4))


def test_corridor_random_ghost_can_stop_and_consumes_one_draw():
    sim = game()
    ghost = sim.ghosts[0]
    ghost.direction = 3
    assert sim.legal_ghost_actions(ghost) == (0, 3)
    expected = np.random.default_rng(4)
    assert sim.ghost_action(0) == (0 if expected.random() <= 0.5 else 3)
    assert sim.rng.bit_generator.state == expected.bit_generator.state


def test_blocked_action_is_stop_and_preserves_heading():
    sim = game()
    sim.direction = 3
    reward, blocked = sim.step(1)
    assert (reward, blocked, sim.last_action, sim.direction) == (-1, True, 0, 3)
    assert sim.player == (1, 1)


def test_capsule_eaten_before_collision_and_respawn_can_move_in_same_turn():
    sim = game()
    sim.player = (2, 1)
    sim.ghosts[0].position = (3, 1)
    reward, _ = sim.step(3)
    assert reward == 199
    assert sim.ate_capsule and sim.killed_blue
    assert sim.ghosts[0].eaten == 1
    assert sim.ghosts[0].timer == 0
    assert sim.ghosts[0].position != sim.ghosts[0].spawn


def test_timer_expiry_snaps_before_collision():
    sim = game()
    sim.player = (4, 1)
    ghost = sim.ghosts[0]
    ghost.position, ghost.direction, ghost.timer = (3, 1), 3, 1
    sim.step(0)
    assert ghost.timer == 0
    assert ghost.position == (4, 1)
    assert sim.lost


def test_final_food_wins_even_with_dangerous_ghost_on_destination():
    sim = game("%%%%%%%\n%P. G %\n%%%%%%%")
    sim.ghosts[0].position = (2, 1)
    before = sim.rng.bit_generator.state
    reward, _ = sim.step(3)
    assert sim.won and not sim.lost
    assert reward == 509
    assert sim.rng.bit_generator.state == before


def test_snapshot_is_detached_from_subsequent_turns():
    sim = game()
    initial = sim.snapshot()
    sim.step(3)
    assert initial.player.position == (1, 1)
    assert initial.score == 0


def test_junction_and_half_cell_sampling_preserve_legal_actions_and_draws():
    layout = PacmanLayout.from_text("%%%%%%%\n%%%G%%%\n%P....%\n%%%.%%%\n%%%%%%%")
    sim = Simulation(layout, np.random.default_rng(47))
    ghost = sim.ghosts[0]
    ghost.position, ghost.direction = (3, 2), 1
    assert sim.legal_ghost_actions(ghost) == (0, 1, 3, 4)
    ghost.position = (3, 2.5)
    expected = np.random.default_rng(47)
    expected.random()
    assert sim.ghost_action(0) == 1
    assert sim.rng.bit_generator.state == expected.bit_generator.state
