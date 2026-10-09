"""Preserved detour, fractional-distance and unreachable-target contracts."""

import numpy as np
import pytest

from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.observations import Routes, VectorFeatures
from npc_gym.envs.pacman.simulation import Simulation


@pytest.fixture
def routes():
    return Routes(PacmanLayout.from_text("%%%%%\n%   %\n%   %\n%P%.%\n%%%%%"))


@pytest.mark.parametrize("cached", [False, True])
def test_food_search_preserves_detours_first_direction_and_empty_result(routes, cached):
    if cached:
        routes.from_cell((1, 1))
    assert routes.nearest((1, 1), {(3, 1)}) == (4, 1)
    assert routes.nearest((3, 1), {(3, 1)}) == (0, 0)
    assert routes.nearest((1, 1), set()) is None
    assert routes.nearest((1, 1), {(0, 0)}) is None


@pytest.mark.parametrize("cached", [False, True])
def test_ghost_search_preserves_fractional_positions_and_collision_offset(routes, cached):
    if cached:
        routes.from_cell((1, 1))
    assert routes.ghost((1, 1), (3, 1.5)) == (3.5, 1)
    assert routes.ghost((1, 1), (1, 1.5)) == (0.5, 0)
    with pytest.raises(RuntimeError, match="No traversable path"):
        routes.ghost((1, 1), (0, 0))


def test_unreachable_capsule_retains_specific_error(routes):
    sim = Simulation(routes.layout, np.random.default_rng(0))
    sim.capsules = {(0, 0)}
    with pytest.raises(TypeError, match="No traversable path to a remaining Pacman capsule"):
        VectorFeatures("hungry", routes.layout, {}).values(sim, None)
