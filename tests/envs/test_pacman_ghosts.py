"""Fixed scheduled behavior, interruption and seeded noise."""

import numpy as np
import pytest

from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.observations import PacmanObservation
from npc_gym.envs.pacman.simulation import Simulation
from npc_gym.envs.pacman.state import GhostConfig


def controller_layout():
    """Fixed junctions and corridors, independent of bundled-map playtesting."""
    return PacmanLayout.from_text(
        "%%%%%%%%%%%\n"
        "%G...G...G%\n"
        "%.%%%.%%%.%\n"
        "%.........%\n"
        "%.%%%.%%%.%\n"
        "%.%%%.%%%.%\n"
        "%.%%%.%%%.%\n"
        "%........P%\n"
        "%%%%%%%%%%%\n"
    )


def simulation(behavior="deterministic", seed=4):
    return Simulation(
        controller_layout(),
        np.random.default_rng(seed),
        behavior=behavior,
        config=GhostConfig(scatter_turns=2, chase_turns=3),
    )


def test_fixed_schedule_switches_after_exact_number_of_turns():
    sim = simulation()
    assert (sim.phase, sim.phase_remaining) == ("scatter", 2)
    sim.step(0)
    assert (sim.phase, sim.phase_remaining) == ("scatter", 1)
    sim.step(0)
    assert (sim.phase, sim.phase_remaining) == ("chase", 3)
    assert all(g.reverse_pending for g in sim.ghosts)
    for _ in range(3):
        sim.step(0)
    assert (sim.phase, sim.phase_remaining) == ("scatter", 2)


def test_frightened_timer_pauses_schedule_including_expiry_turn():
    sim = simulation()
    for ghost in sim.ghosts:
        ghost.timer = 2
    sim.step(0)
    assert sim.full_snapshot().schedule_paused
    assert sim.phase_remaining == 2
    sim.step(0)
    assert not sim.full_snapshot().schedule_paused
    assert sim.phase_remaining == 2
    sim.step(0)
    assert sim.phase_remaining == 1


def test_mode_reversal_is_deferred_until_cell_center():
    sim = simulation()
    ghost = sim.ghosts[0]
    ghost.position, ghost.direction = (5.5, 7), 3
    ghost.reverse_pending = True
    assert sim.ghost_action(0) == 3
    assert ghost.reverse_pending
    ghost.position = (6, 7)
    assert sim.ghost_action(0) == 4
    assert not ghost.reverse_pending


def test_deterministic_mode_never_consumes_rng_even_when_frightened():
    first, second = simulation(seed=1), simulation(seed=99)
    initial = first.rng.bit_generator.state
    for sim in (first, second):
        for ghost in sim.ghosts:
            ghost.timer = 10
    for _ in range(12):
        first.step(0)
        second.step(0)
        assert first.snapshot() == second.snapshot()
    assert first.rng.bit_generator.state == initial


@pytest.mark.parametrize("behavior", ["random", "partly-deterministic"])
def test_seeded_stochastic_modes_replay_when_interleaved(behavior):
    first, second = simulation(behavior), simulation(behavior)
    for _ in range(12):
        first.step(0)
        second.step(0)
        assert first.full_snapshot() == second.full_snapshot()
        assert first.rng.bit_generator.state == second.rng.bit_generator.state
        if first.terminated:
            break


class Frames:
    frame_shape = (100, 100)

    def frame(self, *, crop=False):
        return np.zeros((84, 84, 3) if crop else (100, 100, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", ["essential", "image-full", "image-crop", "image-full+dfa"])
def test_schedule_observations_have_numeric_fields_and_correct_space(mode):
    sim = simulation()
    observations = PacmanObservation(mode, sim.layout, {}, Frames(), max_phase_turns=3)
    observation = observations.observe(sim, None)
    assert observations.space.contains(observation)
    mode_values = observation["ghost_mode"] if isinstance(observation, dict) else observation[-3:]
    assert mode_values.tolist() == [0, 2, 0]


def test_random_observation_has_no_schedule_features():
    sim = simulation("random")
    observations = PacmanObservation("essential", sim.layout, {}, Frames())
    assert observations.observe(sim, None).shape == (19 + 13 * 3,)
    assert sim.full_snapshot().phase is None


def test_scheduled_junction_noise_is_uniform_and_never_reverses():
    sim = Simulation(
        controller_layout(),
        np.random.default_rng(29),
        behavior="partly-deterministic",
        config=GhostConfig(2, 3, 1.0),
    )
    ghost = sim.ghosts[0]
    ghost.position, ghost.direction = (5, 5), 1
    legal = sim.legal_ghost_actions(ghost, scheduled=True)
    assert 2 not in legal and 0 not in legal
    choices = [sim.ghost_action(0) for _ in range(3000)]
    assert set(choices) == set(legal)
    for action in legal:
        assert abs(choices.count(action) / len(choices) - 1 / len(legal)) < 0.04


def test_direct_pursuit_and_frightened_flee_choose_opposite_targets():
    sim = simulation()
    sim.phase = "chase"
    sim.player = (9, 5)
    ghost = sim.ghosts[0]
    ghost.position, ghost.direction = (5, 5), 0
    assert sim.ghost_action(0) == 3
    ghost.timer = 10
    assert sim.ghost_action(0) == 4


def test_full_state_is_detached_and_exposes_configuration_and_spawns():
    sim = simulation()
    snapshot = sim.full_snapshot()
    sim.step(0)
    assert snapshot.step_count == 0
    assert snapshot.game.player.position == sim.layout.player
    assert snapshot.ghost_spawns == sim.layout.ghosts
    assert snapshot.config.scatter_turns == 2
    assert snapshot.phase_turns_remaining == 2


@pytest.mark.parametrize(("name", "scatter", "chase"), [("small", 14, 40), ("medium", 18, 50), ("large", 28, 80)])
def test_bundled_schedule_defaults_and_individual_overrides(name, scatter, chase):
    layout = PacmanLayout.bundled(name)
    for config, expected in [
        (None, (scatter, chase)),
        (GhostConfig(scatter_turns=3), (3, chase)),
        (GhostConfig(chase_turns=9), (scatter, 9)),
    ]:
        sim = Simulation(layout, np.random.default_rng(0), behavior="deterministic", config=config)
        assert (sim.config.scatter_turns, sim.config.chase_turns) == expected
        assert sim.full_snapshot().phase_turns_remaining == expected[0]


def test_random_config_is_retained_but_does_not_change_transitions_or_rng():
    layout = controller_layout()
    plain = Simulation(layout, np.random.default_rng(7))
    configured = Simulation(layout, np.random.default_rng(7), config=GhostConfig(1, 1, 1.0))
    for _ in range(50):
        assert configured.full_snapshot().config == GhostConfig(1, 1, 1.0)
        assert configured.full_snapshot().phase is None
        assert configured.full_snapshot().phase_turns_remaining == 0
        assert not configured.full_snapshot().schedule_paused
        assert plain.snapshot() == configured.snapshot()
        assert plain.rng.bit_generator.state == configured.rng.bit_generator.state
        if plain.terminated:
            break
        assert plain.step(0) == configured.step(0)
