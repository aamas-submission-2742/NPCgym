"""Original maze and explicit external format contracts."""

from collections import deque
from dataclasses import replace

import pytest

from npc_gym.envs.pacman.layout import BUNDLED_LAYOUTS, PacmanLayout
from npc_gym.envs.pacman.state import GhostConfig


@pytest.mark.parametrize(("name", "size", "ghosts"), [("small", 13, 2), ("medium", 15, 3), ("large", 19, 4)])
def test_bundled_maps_are_square_connected_and_have_reachable_corners(name, size, ghosts):
    layout = PacmanLayout.bundled(name)
    assert (layout.width, layout.height) == (size, size)
    assert len(layout.ghosts) == ghosts
    assert len(layout.capsules) == ghosts
    assert all(cell not in layout.walls for cell in [(1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2)])
    assert name in BUNDLED_LAYOUTS


def test_explicit_file_loading_preserves_coordinates_and_marker_order(tmp_path):
    path = tmp_path / "own.lay"
    path.write_text("%%%%%%%\n%2.G1.%\n%P o  %\n%%%%%%%\n")
    layout = PacmanLayout.from_file(path)
    assert layout.player == (1, 1)
    assert layout.ghosts == ((3, 2), (4, 2), (1, 2))
    assert layout.capsules == {(3, 1)}
    assert layout.name == "own"
    assert layout.default_episode_steps == 300


@pytest.mark.parametrize(("name", "limit"), [("small", 300), ("medium", 500), ("large", 800)])
def test_layout_episode_limit_recommendations(name, limit):
    assert PacmanLayout.bundled(name).default_episode_steps == limit


@pytest.mark.parametrize(
    "text",
    [
        "",
        "%%%\n%P%\n%%",
        "%%%%%\n%P? %\n%%%%%",
        "%%%%%\n%PP.%\n%%%%%",
        "%%%%%\n%P  %\n%%%%%",
        "%%%%%%%\n%P %. %\n%%%%%%%",
    ],
)
def test_malformed_layouts_are_rejected(text):
    with pytest.raises(ValueError):
        PacmanLayout.from_text(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scatter_turns": 0},
        {"chase_turns": True},
        {"random_turn_probability": float("nan")},
        {"random_turn_probability": 1.1},
    ],
)
def test_invalid_ghost_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GhostConfig(**kwargs)


@pytest.mark.parametrize(
    ("name", "cell_limit", "cycle_range"),
    [
        ("small", 85, (4, 8)),
        ("medium", 115, (5, 9)),
        ("large", 180, (7, 12)),
    ],
)
def test_bundled_maps_mix_escape_loops_with_dead_ends_and_long_corridors(name, cell_limit, cycle_range):
    layout = PacmanLayout.bundled(name)
    cells = {(x, y) for x in range(layout.width) for y in range(layout.height)} - layout.walls
    degrees = [len(layout.neighbors(cell)) for cell in cells]
    cycles = sum(degrees) // 2 - len(cells) + 1
    assert len(cells) <= cell_limit
    assert cycle_range[0] <= cycles <= cycle_range[1]
    dead_ends = {cell for cell in cells if len(layout.neighbors(cell)) == 1}
    assert 2 <= len(dead_ends - set(layout.ghosts)) <= 10
    assert len(dead_ends & layout.capsules) >= 2
    # Wide floor blocks provide accidental shortcuts around intended corridors.
    assert not any({(x + 1, y), (x, y + 1), (x + 1, y + 1)} <= cells for x, y in cells)
    longest_corridor = 0
    for start in cells:
        if len(layout.neighbors(start)) == 2:
            continue
        for current in layout.neighbors(start):
            previous, length = start, 1
            while len(layout.neighbors(current)) == 2:
                following = next(cell for cell in layout.neighbors(current) if cell != previous)
                previous, current = current, following
                length += 1
            longest_corridor = max(longest_corridor, length)
    assert longest_corridor >= 8


@pytest.mark.parametrize("name", BUNDLED_LAYOUTS)
def test_bundled_geometry_and_entities_are_mirrored_left_to_right(name):
    layout = PacmanLayout.bundled(name)
    for cells in (layout.walls, layout.food, layout.capsules, {layout.player}, set(layout.ghosts)):
        assert {(layout.width - 1 - x, y) for x, y in cells} == cells


@pytest.mark.parametrize("name", BUNDLED_LAYOUTS)
def test_ghosts_share_a_food_free_central_pen_opening_onto_a_junction(name):
    layout = PacmanLayout.bundled(name)
    middle = layout.width // 2
    rows = {y for _, y in layout.ghosts}
    assert len(rows) == 1
    row = rows.pop()
    assert abs(row - layout.height // 2) <= 1
    assert all(abs(x - middle) <= 2 for x, _ in layout.ghosts)
    pen = {(x, row) for x in range(min(x for x, _ in layout.ghosts), max(x for x, _ in layout.ghosts) + 1)}
    assert not pen & (layout.walls | layout.food | layout.capsules)
    exits = {neighbor for cell in pen for neighbor in layout.neighbors(cell)} - pen
    assert exits == {(middle, row + 1)}
    assert len(layout.neighbors((middle, row + 2))) >= 3


@pytest.mark.parametrize("name", BUNDLED_LAYOUTS)
def test_dead_ends_are_short_and_pellet_escape_junctions_are_far_from_respawns(name):
    layout = PacmanLayout.bundled(name)
    cells = {(x, y) for x in range(layout.width) for y in range(layout.height)} - layout.walls
    for start in cells:
        if len(layout.neighbors(start)) != 1:
            continue
        previous, current, length = start, layout.neighbors(start)[0], 1
        while len(layout.neighbors(current)) == 2:
            following = next(cell for cell in layout.neighbors(current) if cell != previous)
            previous, current, length = current, following, length + 1
        assert length <= 4
        if start not in layout.capsules:
            continue
        distances = {current: 0}
        pending = deque([current])
        while pending:
            cell = pending.popleft()
            for neighbor in layout.neighbors(cell):
                if neighbor not in distances:
                    distances[neighbor] = distances[cell] + 1
                    pending.append(neighbor)
        # Even a full-speed respawn needs longer to reach the branch entrance
        # than Pacman needs to leave the pellet, with two moves of clearance.
        assert all(distances[spawn] >= length + 2 for spawn in layout.ghosts)


@pytest.mark.parametrize("overlap_player", [False, True])
def test_direct_layout_constructor_rejects_overlapping_actor_spawns(overlap_player):
    layout = PacmanLayout.bundled("small")
    ghosts = (layout.player, *layout.ghosts) if overlap_player else (layout.ghosts[0], layout.ghosts[0])
    with pytest.raises(ValueError, match="spawns must occupy distinct cells"):
        replace(layout, ghosts=ghosts)
