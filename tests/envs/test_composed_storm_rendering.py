"""Rendering delegates to upstream and adds only NPC Gym's storm information."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium
import numpy as np
import pygame
import pytest
from gymnasium.envs.toy_text.taxi import TaxiEnv as GymnasiumTaxi

from npc_gym.envs import StormTaxiEnv, pygame_display


@pytest.mark.parametrize("action", [None, *range(7)])
def test_ansi_delegates_map_and_actions_to_gymnasium(action):
    with StormTaxiEnv(render_mode="ansi") as composed, GymnasiumTaxi(render_mode="ansi") as upstream:
        for state in (0, 217552, 351999):
            composed.reset(seed=7)
            upstream.reset(seed=7)
            composed.s, upstream.s = state, state // 704
            composed.lastaction = action
            upstream.lastaction = action if action != 6 else None
            expected = upstream.render()
            if action == 6:
                expected = expected.removesuffix("\n") + "  (Warn)\n"
            assert composed.render() == expected
    assert pygame_display.open_displays() == 0


def test_graphics_use_the_installed_upstream_renderer_and_assets(monkeypatch):
    rendered, loaded = [], []
    upstream_render, load = GymnasiumTaxi.render, pygame.image.load

    def record_render(env):
        rendered.append((env.s, env.lastaction, env.render_mode))
        return upstream_render(env)

    def record_load(path):
        loaded.append(Path(path).resolve())
        return load(path)

    monkeypatch.setattr(GymnasiumTaxi, "render", record_render)
    monkeypatch.setattr(pygame.image, "load", record_load)
    with StormTaxiEnv(render_mode="rgb_array") as env:
        state, _ = env.reset(seed=7)
        rng = env.np_random.bit_generator.state
        pixels = env.render()
        assert rendered == [(state // 704, None, "rgb_array")]
        assert loaded and all(path.is_relative_to(Path(gymnasium.__file__).parent) for path in loaded)
        assert pixels.shape == (350, 550, 3)
        assert pixels.dtype == np.uint8
        assert env.s == state and env.np_random.bit_generator.state == rng
        pixels[:] = 0
        assert env.render().any()


def test_weather_home_and_shelter_are_visible():
    with StormTaxiEnv(render_mode="rgb_array") as env:
        env.reset(seed=7)
        env.s = env.encode(2, 2, 4, 1, False, 0, 0, False, 2)
        clear = env.render()
        for rain, hurricane, flood, home, shelter in (
            (True, 0, False, 0, 2),
            (True, 4, False, 0, 2),
            (False, 0, True, 0, 2),
            (False, 0, False, 1, 3),
        ):
            env.s = env.encode(2, 2, 4, 1, rain, hurricane, home, flood, shelter)
            assert not np.array_equal(clear, env.render())


def test_two_composed_renderers_survive_close_and_reset(monkeypatch):
    ticks = []
    monkeypatch.setattr(pygame.time, "Clock", lambda: SimpleNamespace(tick=ticks.append))
    with StormTaxiEnv(render_mode="human") as first, StormTaxiEnv(render_mode="human") as second:
        first.reset(seed=7)
        second.reset(seed=2)
        assert ticks == [4, 4]
        first.close()
        first.close()
        assert pygame_display.open_displays() == 1
        second.step(6)
        first.reset(seed=7)
        assert pygame_display.open_displays() == 2
        assert ticks == [4, 4, 4, 4]
    assert pygame_display.open_displays() == 0


def test_asset_failure_releases_resources_and_can_be_retried(monkeypatch):
    with StormTaxiEnv(render_mode="rgb_array") as survivor, StormTaxiEnv(render_mode="rgb_array") as failed:
        survivor.reset(seed=7)
        survivor.render()
        failed.reset(seed=7)
        with monkeypatch.context() as patch:

            def missing_asset(*args, **kwargs):
                raise FileNotFoundError("upstream asset unavailable")

            patch.setattr(pygame.image, "load", missing_asset)
            with pytest.raises(FileNotFoundError, match="upstream asset"):
                failed.render()
        assert pygame_display.open_displays() == 1
        survivor.render()
        assert failed.render().shape == (350, 550, 3)
    assert pygame_display.open_displays() == 0


def test_upstream_private_attributes_exist():
    # Reset creates s and lastaction; the other attributes exist at construction.
    upstream = GymnasiumTaxi(render_mode="rgb_array")
    upstream.reset(seed=7)
    for name in (
        "s",
        "lastaction",
        "taxi_orientation",
        "window",
        "clock",
        "taxi_imgs",
        "passenger_img",
        "destination_img",
        "median_horiz",
        "median_vert",
        "background_img",
    ):
        assert hasattr(upstream, name), f"Gymnasium Taxi contract changed: missing {name}"


def test_close_releases_all_upstream_pygame_objects():
    def contains_pygame(value):
        if type(value).__module__.startswith("pygame"):
            return True
        if isinstance(value, dict):
            return any(contains_pygame(item) for pair in value.items() for item in pair)
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(contains_pygame(item) for item in value)
        return False

    with StormTaxiEnv(render_mode="rgb_array") as env:
        env.reset(seed=7)
        env.render()
        assert any(contains_pygame(value) for value in vars(env._backend._env).values())
        env.close()
        retained = [name for name, value in vars(env._backend._env).items() if contains_pygame(value)]
        assert retained == []
