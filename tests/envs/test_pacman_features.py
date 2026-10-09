"""Search and feature semantics on original geometry."""

import numpy as np
import pytest

from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.observations import SUPPORTED_VECTOR_FEATURES, Routes, VectorFeatures, feature_array
from npc_gym.envs.pacman.simulation import Simulation


def maze():
    return PacmanLayout.from_text("%%%%%%%\n%   G %\n% %%% %\n%P . o%\n%%%%%%%")


def test_search_ties_wall_probes_and_fractional_ghost_distance():
    routes = Routes(maze())
    assert routes.nearest((1, 1), {(3, 1)}) == (2, 3)
    assert routes.nearest((3, 1), {(3, 1)}) == (0, 0)
    assert routes.nearest((1, 1), set()) is None
    assert routes.ghost((1, 1), (3.5, 1)) == (2.5, 3)
    assert routes.ghost((3, 1), (3.5, 1)) == (0.5, 0)
    with pytest.raises(RuntimeError, match="No traversable path"):
        routes.ghost((1, 1), (0, 0))


@pytest.mark.parametrize("mode", SUPPORTED_VECTOR_FEATURES)
@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_features_are_finite_fixed_size_and_in_space(mode, name):
    layout = PacmanLayout.bundled(name)
    sim = Simulation(layout, np.random.default_rng(4))
    features = VectorFeatures(mode, layout, {})
    for _ in range(10):
        values = feature_array(features.values(sim, 0))
        assert np.isfinite(values).all()
        assert features.space().contains(values)
        if sim.terminated:
            break
        sim.step(0)


def test_distinguish_retains_documented_west_food_probe():
    sim = Simulation(maze(), np.random.default_rng(1))
    values = VectorFeatures("complete-distinguish", sim.layout, {}).values(sim, 0)
    assert len({values[f"closest-food-{action}"] for action in range(1, 5)}) == 1


def test_feature_sort_moves_only_the_two_directional_count_groups_last():
    assert feature_array({"z": 2, "a": 1, "#-of-ghosts-1-step-away-0": 3}).tolist() == [1, 2, 3]
