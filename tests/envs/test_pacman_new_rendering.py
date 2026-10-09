"""Renderer lifecycle and image contracts with two live environments."""

import numpy as np
import pytest
from gymnasium.wrappers import GrayscaleObservation
from PIL import Image

from npc_gym.envs import PacmanEnv, PacmanLayout, pygame_display
from npc_gym.wrappers import PacmanPixelObservation

# Separate sprites at the largest bundled image size: palette checks must not
# measure occlusion when ghosts converge inside a shared spawning enclosure.
PALETTE_LAYOUT = PacmanLayout.from_text(
    "\n".join(["%" * 19] * 9 + ["%..G...G...G...G..%", "%%%%%%%%%P%%%%%%%%%"] + ["%" * 19] * 8)
)


@pytest.mark.parametrize("features", ["image-full", "image-crop", "image-full+dfa"])
@pytest.mark.parametrize("behavior", ["random", "deterministic", "partly-deterministic"])
def test_image_spaces_and_returned_buffers_are_independent(features, behavior):
    with PacmanEnv(features=features, ghost_behavior=behavior, render_mode="rgb_array") as env:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        pixels = observation["observation"] if isinstance(observation, dict) else observation
        expected = env.render()
        pixels[:] = 0
        np.testing.assert_array_equal(env.render(), expected)
        assert env.render().shape == (420, 420, 3)
        if features == "image-crop":
            assert pixels.shape == (84, 84, 3)


def test_close_one_renderer_keeps_other_alive():
    with PacmanEnv(features="image-full") as first, PacmanEnv(features="image-full") as second:
        first.reset(seed=1)
        second.reset(seed=1)
        assert pygame_display.open_displays() == 2
        first.close()
        first.close()
        observation, *_ = second.step(0)
        assert second.observation_space.contains(observation)
        assert pygame_display.open_displays() == 1
    assert pygame_display.open_displays() == 0


def test_drawing_error_releases_resources_and_reset_recovers(monkeypatch):
    with PacmanEnv(features="image-full") as env:
        env.reset(seed=0)
        with monkeypatch.context() as patch:

            def fail():
                raise RuntimeError("drawing failed")

            patch.setattr(env._renderer, "_draw", fail)
            with pytest.raises(RuntimeError, match="drawing failed"):
                env.step(0)
        assert pygame_display.open_displays() == 0
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)


def ghost_patches(frame, snapshot):
    """Interior sprite pixels, scaled to the actual observation dimensions."""
    layout = snapshot.game.layout
    sx, sy = frame.shape[1] / (layout.width + 1), frame.shape[0] / (layout.height + 1)
    return [
        frame[
            round((layout.height - ghost.position[1] - 1 / 3) * sy) : round(
                (layout.height - ghost.position[1] + 1 / 3) * sy
            ),
            round((ghost.position[0] + 1 - 1 / 3) * sx) : round((ghost.position[0] + 1 + 1 / 3) * sx),
        ]
        for ghost in snapshot.game.ghosts
    ]


def test_frightened_ghost_identity_survives_both_grayscale_conversions():
    with PacmanEnv(layout=PALETTE_LAYOUT, features="image-full", render_mode="rgb_array") as env:
        normal, _ = env.reset(seed=0)
        snapshot = env.state()
        # Hold positions fixed to isolate the visual frightened-state signal.
        for ghost in env._sim.ghosts:
            ghost.timer = 40
        env._renderer.update(env._sim)
        frightened = env.render()
        grayscale = GrayscaleObservation(env)
        for convert in (
            lambda frame: frame,
            grayscale.observation,
            lambda frame: np.asarray(Image.fromarray(frame).convert("L")),
        ):
            ordinary = ghost_patches(convert(normal), snapshot)
            scared = ghost_patches(convert(frightened), snapshot)
            colors = []
            for before, after in zip(ordinary, scared, strict=True):
                # The body below the eyes keeps its identity color/brightness.
                color = before[11, 10]
                np.testing.assert_array_equal(after[11, 10], color)
                assert np.any(before != after)  # Mouth conveys frightened state.
                colors.append(color)
            assert len({tuple(np.atleast_1d(color)) for color in colors}) == 4
            if ordinary[0].ndim == 2:
                assert np.min(np.diff(np.sort(colors))) >= 30
        for ghost in env._sim.ghosts:
            ghost.timer = 0
        env._renderer.update(env._sim)
        np.testing.assert_array_equal(env.render(), normal)


@pytest.mark.parametrize("behavior", ["random", "deterministic"])
@pytest.mark.parametrize("frightened", [False, True])
def test_ghost_brightness_remains_separated_in_default_resized_observations(behavior, frightened):
    raw = PacmanEnv(layout=PALETTE_LAYOUT, features="image-full", ghost_behavior=behavior)
    with PacmanPixelObservation(raw, frames=1) as env:
        env.reset(seed=0)
        if frightened:
            for ghost in raw._sim.ghosts:
                ghost.timer = 40
        observation, *_ = env.step(0)
        assert env.observation_space.contains(observation)
        pixels = observation["observation"] if isinstance(observation, dict) else observation
        brightness = [patch.mean() for patch in ghost_patches(pixels[..., 0], raw.state())]
        assert np.argsort(brightness).tolist() == [0, 2, 1, 3]
        assert np.min(np.diff(np.sort(brightness))) >= 20
