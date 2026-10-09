"""Pygame's display is process-wide, so one environment's `close()` must not disturb another."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from npc_gym.envs import pygame_display
from npc_gym.envs.gardener.gardener import GardenerEnv
from npc_gym.envs.merchant.merchant import MerchantEnv
from npc_gym.envs.pacman.pacman_env import PacmanEnv
from npc_gym.envs.taxi.storm_taxi import StormTaxiEnv


@pytest.fixture(autouse=True)
def frame_ticks(monkeypatch):
    """Observe frame pacing without sleeping in renderer tests."""
    ticks = []
    monkeypatch.setattr(pygame.time, "Clock", lambda: SimpleNamespace(tick=lambda fps: ticks.append(fps)))
    return ticks


@pytest.mark.parametrize("domain, fps", [("storm", 4), ("gardener", 10), ("pacman", 10)])
def test_human_reset_and_step_present_and_pace_once(domain, fps, frame_ticks, monkeypatch):
    env = make_human_env(domain)
    frames = []
    monkeypatch.setattr(pygame.display, "flip", lambda: frames.append(1))
    frame_ticks.clear()  # Merchant and Gardener currently reset during construction.
    try:
        env.reset(seed=5)
        env.step(0)
        assert len(frames) == 2
        assert frame_ticks == [fps, fps]
        assert env.metadata["render_fps"] == fps
    finally:
        env.close()


@pytest.mark.parametrize("experiment_id", ["pacman-ppo-unconstrained-v2", "gardener-dqn-unconstrained-v0"])
def test_runner_evaluation_presents_every_reset_and_step(experiment_id, frame_ticks, monkeypatch):
    from experiments.execution import _evaluate_model
    from experiments.specifications import REGISTRY

    resolved = REGISTRY.resolve(experiment_id)
    resolved = replace(
        resolved,
        specification=replace(resolved.specification, run=replace(resolved.specification.run, max_episode_steps=2)),
    )
    frames = []
    monkeypatch.setattr(pygame.display, "flip", lambda: frames.append(1))

    class StopPolicy:
        def predict(self, observation, *, deterministic):
            return 0, None

    summary = _evaluate_model(resolved, StopPolicy(), episodes=2, seed=5, render=True)
    assert [episode.length for episode in summary.episodes] == [2, 2]
    expected_frames = 7 if resolved.environment.family == "gardener" else 6  # Gardener also resets in __init__.
    assert len(frames) == expected_frames
    assert frame_ticks == [10] * expected_frames
    assert pygame_display.open_displays() == 0


@pytest.mark.parametrize(
    "domain, mode",
    [
        ("pacman", None),
        ("pacman", "rgb_array"),
        ("storm", "rgb_array"),
        ("merchant", "rgb_array"),
    ],
)
def test_headless_images_never_present_or_pace(domain, mode, frame_ticks, monkeypatch):
    if domain == "pacman":
        env = PacmanEnv(layout="small", features="image-full", render_mode=mode)
    elif domain == "storm":
        env = StormTaxiEnv(render_mode=mode)
    else:
        env = MerchantEnv(render_mode=mode)
    frames = []
    monkeypatch.setattr(pygame.display, "flip", lambda: frames.append(1))
    try:
        observation, _ = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        observation, *_ = env.step(0)
        assert env.observation_space.contains(observation)
        if mode == "rgb_array":
            frame = env.render()
            assert frame.ndim == 3 and frame.shape[2] == 3
            if domain == "pacman":
                np.testing.assert_array_equal(frame, observation)
        assert frames == []
        assert frame_ticks == []
    finally:
        env.close()


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
def test_constructor_selected_mode_has_no_public_setter(domain):
    env = make_human_env(domain)
    try:
        assert not hasattr(env, "set_render_mode")
    finally:
        env.close()


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
def test_failed_presentation_releases_only_the_failed_environment(domain, monkeypatch):
    survivor = PacmanEnv(layout="small", features="image-full")
    failed = None
    try:
        survivor.reset(seed=5)

        def fail_to_present(*args, **kwargs):
            raise RuntimeError("display unavailable")

        monkeypatch.setattr(pygame_display, "present", fail_to_present)
        with pytest.raises(RuntimeError, match="display unavailable"):
            failed = make_human_env(domain)
            failed.reset(seed=5)
        assert pygame_display.open_displays() == 1
        observation, *_ = survivor.step(0)
        assert survivor.observation_space.contains(observation)
    finally:
        if failed is not None:
            failed.close()
            failed.close()
        survivor.close()
    assert pygame_display.open_displays() == 0


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
def test_failed_surface_allocation_releases_ownership(domain, monkeypatch):
    survivor = PacmanEnv(layout="small", features="image-full")
    failed = None
    try:
        survivor.reset(seed=5)

        def fail_to_allocate(*args, **kwargs):
            raise pygame.error("surface allocation failed")

        with monkeypatch.context() as patch:
            patch.setattr(pygame, "Surface", fail_to_allocate)
            with pytest.raises(pygame.error, match="surface allocation failed"):
                failed = make_human_env(domain)
                failed.reset(seed=5)
        assert pygame_display.open_displays() == 1
        observation, *_ = survivor.step(0)
        assert survivor.observation_space.contains(observation)
    finally:
        if failed is not None:
            failed.close()
        survivor.close()
    assert pygame_display.open_displays() == 0


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
def test_reset_after_close_reacquires_only_one_share(domain):
    env = make_human_env(domain)
    try:
        for _ in range(2):
            env.reset(seed=5)
            env.step(0)
            assert pygame_display.open_displays() == 1
            env.close()
            env.close()
            assert pygame_display.open_displays() == 0
    finally:
        env.close()


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
def test_headless_vector_environments_never_initialize_pygame(domain, monkeypatch):
    def unexpected_pygame(*args, **kwargs):
        pytest.fail("headless vector environment initialized Pygame")

    monkeypatch.setattr(pygame, "init", unexpected_pygame)
    monkeypatch.setattr(pygame.time, "Clock", unexpected_pygame)
    constructors = {
        "storm": StormTaxiEnv,
        "gardener": lambda: GardenerEnv(size=8),
        "merchant": MerchantEnv,
        "pacman": lambda: PacmanEnv(layout="small", features="essential"),
    }
    env = constructors[domain]()
    try:
        env.reset(seed=5)
        observation, *_ = env.step(0)
        assert env.observation_space.contains(observation)
    finally:
        env.close()
    assert pygame_display.open_displays() == 0


@pytest.mark.parametrize("render_mode", ["human", "rgb_array"])
def test_merchant_delay_applies_only_to_human_frames(render_mode, frame_ticks, monkeypatch):
    from npc_gym.envs.merchant import py_game_merchant_display

    delays = []
    monkeypatch.setattr(py_game_merchant_display.time, "sleep", delays.append)
    env = MerchantEnv(render_mode=render_mode, step_delay_ms=500)
    delays.clear()
    try:
        env.reset(seed=5)
        env.step(0)
        env.render()
        assert delays == ([0.5] * 3 if render_mode == "human" else [])
        assert frame_ticks == []
    finally:
        env.close()


def test_pacman_drawing_failure_can_be_closed_and_reset(monkeypatch):
    env = PacmanEnv(layout="small", features="image-full")
    try:
        env.reset(seed=5)

        def fail_to_draw(*args, **kwargs):
            raise RuntimeError("drawing failed")

        with monkeypatch.context() as patch:
            patch.setattr(env._renderer, "_draw", fail_to_draw)
            with pytest.raises(RuntimeError, match="drawing failed"):
                env.step(0)
        assert pygame_display.open_displays() == 0
        observation, _ = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        assert pygame_display.open_displays() == 1
    finally:
        env.close()


def test_storm_taxi_render_without_a_mode_warns_instead_of_asserting():
    env = StormTaxiEnv()
    try:
        env.reset(seed=5)
        with pytest.warns(UserWarning, match="render_mode at construction"):
            assert env.render() is None
    finally:
        env.close()


def test_storm_taxi_can_render_the_warning_action_as_text():
    env = StormTaxiEnv(render_mode="ansi")
    try:
        env.reset(seed=5)
        env.step(6)
        assert "(Warn)" in env.render()
    finally:
        env.close()


@pytest.fixture
def live_environments():
    environments = [
        StormTaxiEnv(render_mode="human"),
        GardenerEnv(size=8, render_mode="human"),
        MerchantEnv(layout="basic", render_mode="human"),
        PacmanEnv(layout="small", features="essential", render_mode="human"),
    ]
    for index, env in enumerate(environments):
        env.reset(seed=index)
        env.render()
    yield environments
    for env in environments:
        env.close()


def test_closing_one_environment_keeps_the_others_rendering(live_environments):
    while len(live_environments) > 1:
        live_environments.pop(0).close()
        for env in live_environments:
            env.render()


def test_the_last_close_releases_the_shared_display(live_environments):
    assert pygame_display.open_displays() == len(live_environments)
    for env in live_environments:
        env.close()
    assert pygame_display.open_displays() == 0
    # a released display can be taken again
    live_environments[0].render()
    assert pygame_display.open_displays() == 1


def make_human_env(domain):
    if domain == "storm":
        return StormTaxiEnv(render_mode="human")
    if domain == "gardener":
        return GardenerEnv(size=8, render_mode="human")
    if domain == "merchant":
        return MerchantEnv(render_mode="human")
    return PacmanEnv(layout="small", features="image-full", render_mode="human")


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
def test_human_frames_survive_interleaved_rendering_and_close(domain):
    first = make_human_env(domain)
    second = None
    try:
        first.reset(seed=5)
        first.render()
        expected = pygame.surfarray.array3d(pygame.display.get_surface())
        second = make_human_env("pacman" if domain == "gardener" else "gardener")
        second.reset(seed=7)
        for _ in range(2):
            second.render()
            first.render()
            np.testing.assert_array_equal(pygame.surfarray.array3d(pygame.display.get_surface()), expected)
        second.close()
        first.render()
        np.testing.assert_array_equal(pygame.surfarray.array3d(pygame.display.get_surface()), expected)
        observation, *_ = first.step(0)
        assert first.observation_space.contains(observation)
    finally:
        first.close()
        if second is not None:
            second.close()


@pytest.mark.parametrize("domain", ["storm", "gardener", "merchant", "pacman"])
@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_headless_pacman_keeps_rendering_after_a_human_window_closes(domain, render_mode):
    first = PacmanEnv(layout="small", features="image-full", render_mode=render_mode)
    second = None
    try:
        first.reset(seed=5)
        second = make_human_env(domain)
        second.reset(seed=7)
        second.render()
        second.close()
        observation, *_ = first.step(0)
        assert first.observation_space.contains(observation)
        if render_mode == "rgb_array":
            np.testing.assert_array_equal(first.render(), observation)
    finally:
        first.close()
        if second is not None:
            second.close()


@pytest.mark.parametrize("domain", ["storm", "merchant", "pacman"])
def test_headless_renderers_release_pygame_after_repeated_resets(domain):
    if domain == "storm":
        env = StormTaxiEnv(render_mode="rgb_array")
    elif domain == "merchant":
        env = MerchantEnv(render_mode="rgb_array")
    else:
        env = PacmanEnv(layout="small", features="image-full")
    try:
        for seed in range(2):
            env.reset(seed=seed)
            if domain != "pacman":
                env.render()
            assert pygame_display.open_displays() == 1
    finally:
        env.close()
        env.close()
    assert pygame_display.open_displays() == 0
    assert not pygame.get_init()
