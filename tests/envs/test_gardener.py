import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from npc_gym.envs.gardener.gardener import (
    UNREACHABLE,
    GardenerEnv,
    GardenerObservation,
    GardenerObservationCodec,
    action_dict,
    target_routes,
)
from npc_gym.envs.gardener.labels import FrogCollected, GardenerLabel, PuddleDrained
from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.wrappers import gardener_wrappers
from npc_gym.wrappers.gardener_wrappers import (
    IllegalActionPenaltyWrapper,
    LocalGridObsWrapper,
    StateFeatureObsWrapper,
)


def test_seeded_reset_is_reproducible_valid_and_non_overlapping():
    env = GardenerEnv(size=8)
    first, info = env.reset(seed=23)
    second, _ = env.reset(seed=23)
    assert first == second
    assert info["action_mask"].shape == (env.action_space.n,)
    assert info["labels"] == frozenset()
    assert info[EPISODE_METRICS_KEY] == {"score": 0}
    assert env.observation_space.contains(first)
    occupied = [
        tuple(env.agent),
        *map(tuple, env.frogs),
        *map(tuple, env.puddles),
        *map(tuple, env.grass),
        *map(tuple, env.walls),
    ]
    assert len(occupied) == len(set(occupied))


def test_minimum_size_handles_layout_retries_reproducibly():
    env = GardenerEnv(size=3)
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert first == second
    assert env.observation_space.contains(first)


def test_seeded_trajectory_is_reproducible():
    first = GardenerEnv(size=8)
    second = GardenerEnv(size=8)
    actions = [0, 1, 4, 2, 3, 0, 4]
    first.reset(seed=13)
    second.reset(seed=13)
    assert [first.step(action)[:4] for action in actions] == [second.step(action)[:4] for action in actions]


@pytest.mark.parametrize("container", [tuple, list, np.asarray])
def test_codec_preserves_entity_order_timers_and_step_events(container):
    codec = GardenerObservationCodec(size=5, num_grass=2, num_puddles=1, num_frogs=2, num_walls=1)
    observation = container(
        (
            4,
            3,  # agent
            0,
            1,
            2,
            3,
            1,
            0,
            0,
            7,  # grass positions, active flags, timers
            1,
            4,
            0,
            5,  # puddle position, full flag, timer
            3,
            1,
            4,
            0,
            1,
            0,
            0,
            1,
            6,
            0,  # frog positions, collected/captured flags, timers
            2,
            2,  # wall
            0,
            1,
            1,  # collection and drainage events
        )
    )
    assert codec.decode(observation) == GardenerObservation(
        size=5,
        agent=(4, 3),
        grass=((0, 1), (2, 3)),
        grass_active=(True, False),
        grass_timer=(0, 7),
        puddles=((1, 4),),
        puddles_full=(False,),
        puddle_timer=(5,),
        frogs=((3, 1), (4, 0)),
        collected_frogs=(True, False),
        captured_frogs=(False, True),
        frog_timer=(6, 0),
        walls=((2, 2),),
        frog_collected=(False, True),
        puddle_drained=(True,),
    )


@pytest.mark.parametrize("length", [0, 28, 30])
def test_codec_rejects_incorrect_observation_lengths(length):
    codec = GardenerObservationCodec(size=5, num_grass=2, num_puddles=1, num_frogs=2, num_walls=1)
    with pytest.raises(ValueError, match=f"length {length}; expected 29"):
        codec.decode((0,) * length)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 2}, "size"),
        ({"grass_respawn": 0}, "grass_respawn"),
        ({"puddle_respawn": 0}, "puddle_respawn"),
        ({"score_limit": 0}, "score_limit"),
        ({"frog_freeze": -1}, "frog_freeze"),
    ],
)
def test_invalid_constructor_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GardenerEnv(**kwargs)


def test_blocked_action_executes_the_stay_transition():
    blocked_env = GardenerEnv(size=8)
    stay_env = GardenerEnv(size=8)
    blocked_env.reset(seed=2)
    stay_env.reset(seed=2)
    blocked = np.flatnonzero(blocked_env.action_mask() == 0)[0]
    assert stay_env.action_mask()[blocked] == 0

    blocked_result = blocked_env.step(blocked)
    stay_result = stay_env.step(action_dict["stay"])

    assert blocked_result[:4] == stay_result[:4]
    assert np.array_equal(blocked_result[4]["action_mask"], stay_result[4]["action_mask"])
    assert blocked_env.action == action_dict["stay"]


def test_step_accepts_every_action_the_declared_space_contains():
    for action in range(len(action_dict)):
        env = GardenerEnv(size=8)
        env.reset(seed=2)
        assert env.action_space.contains(action)
        env.step(action)


@pytest.mark.parametrize("action", [-1, len(action_dict), 1.5])
def test_step_rejects_actions_outside_the_declared_space(action):
    env = GardenerEnv(size=8)
    env.reset(seed=2)
    with pytest.raises(ValueError, match="outside"):
        env.step(action)


def test_passes_gymnasium_environment_checker():
    check_env(GardenerEnv(size=8), skip_render_check=True)


def test_grass_cutting_reward_regrowth_and_terminal_score():
    env = GardenerEnv(size=8, grass_respawn=2, score_limit=10)
    env.reset(seed=3)
    gx, gy = env.grass[0]
    candidates = [
        (gx - 1, gy, "right"),
        (gx + 1, gy, "left"),
        (gx, gy - 1, "up"),
        (gx, gy + 1, "down"),
    ]
    x, y, action = next(
        (x, y, a)
        for x, y, a in candidates
        if 0 <= x < env.size and 0 <= y < env.size and env.pos_actions[x, y, action_dict[a]]
    )
    env.agent = np.array([x, y])
    _, reward, terminated, _, info = env.step(action_dict[action])
    assert reward == pytest.approx(9.9)
    assert terminated is True
    assert env.score_delta == 10
    assert env.grass_active[0] == 0
    assert env.grass_timer[0] == 2
    assert info[EPISODE_METRICS_KEY] == {"score": 10}


def test_puddle_drain_records_event_and_freezes_nearby_frog():
    env = GardenerEnv(size=8, frog_freeze=5)
    env.reset(seed=5)
    px, py = env.puddles[0]
    adjacent = [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)]
    target = next(
        (x, y) for x, y in adjacent if 0 <= x < env.size and 0 <= y < env.size and (x, y) not in env._walls_set
    )
    env.agent = np.array(target)
    env.frogs[0] = np.array(
        [
            px + 1 if px + 1 < env.size else px - 1,
            py + 1 if py + 1 < env.size else py - 1,
        ]
    )
    env.collected_frogs[0] = False
    env._move_frogs = lambda: None
    _, reward, _, _, info = env.step(action_dict["stay"])
    assert reward == pytest.approx(4.9)
    assert env.puddle_drained[0]
    assert env.frog_timer[0] == 5
    assert PuddleDrained(0, (0,)) in info["labels"]
    assert GardenerLabel.STAY in info["labels"]


def test_collection_labels_carry_frog_identity():
    env = GardenerEnv(size=15)
    env.reset(seed=6)
    env.agent = env.frogs[1].copy()
    env._move_frogs = lambda: None

    _, _, _, _, info = env.step(action_dict["stay"])

    assert FrogCollected(1) in info["labels"]
    assert GardenerLabel.STAY in info["labels"]


def test_state_feature_wrapper_reports_nearest_targets():
    base = GardenerEnv(size=8)
    base.reset(seed=7)
    wrapper = StateFeatureObsWrapper(base)
    observation, _ = wrapper.reset(seed=7)
    assert wrapper.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert 0 <= observation[0] <= 1
    assert 0 <= observation[1] <= 1
    assert observation[2:7].sum() <= 1
    assert observation[7:].sum() <= 1


def test_puddle_routes_treat_other_puddles_as_obstacles():
    """Puddles are impassable, so a route to one may not cut through another."""
    routes = target_routes(3, [(0, 0), (2, 0)], impassable={(0, 0), (1, 0), (2, 0)})
    assert (1, 0) not in routes
    assert routes[(2, 1)] == [(1, 1, action_dict["down"]), (0, 3, action_dict["left"])]


def test_state_feature_wrapper_agrees_with_the_environment_navigation():
    """The distance the agent learns from must match the one the environment steers frogs by.

    A drained puddle still blocks movement, so the route to a full one has to go around it.
    """
    base = GardenerEnv(size=10)
    wrapper = StateFeatureObsWrapper(base)
    for seed in range(5):
        base.reset(seed=seed)
        for only_full in range(base.num_puddles):
            base.puddles_full = np.arange(base.num_puddles) == only_full
            for cell, entries in base.puddle_dict.items():
                base.agent = np.array(cell)
                expected = min(
                    (distance for index, distance, _ in entries if base.puddles_full[index]), default=UNREACHABLE
                )
                feature = wrapper.observation(base.get_state())[1]
                assert feature == pytest.approx(1.0 / (expected + 1.0) if expected < UNREACHABLE else 0.0)


def test_state_feature_wrapper_uses_the_supplied_observation():
    base = GardenerEnv(size=8)
    observation, _ = base.reset(seed=7)
    wrapper = StateFeatureObsWrapper(base)
    expected = wrapper.observation(observation)
    base.reset(seed=8)
    assert np.array_equal(wrapper.observation(observation), expected)


def test_state_feature_route_caches_are_independent(monkeypatch):
    original_target_routes = gardener_wrappers.target_routes
    calls = 0

    def counting_target_routes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_target_routes(*args, **kwargs)

    monkeypatch.setattr(gardener_wrappers, "target_routes", counting_target_routes)
    wrapped_observations = []
    for seed in range(20):
        base = GardenerEnv(size=8)
        observation, _ = base.reset(seed=seed)
        wrapped_observations.append((StateFeatureObsWrapper(base), observation))

    for _ in range(2):
        for wrapper, observation in wrapped_observations:
            wrapper.observation(observation)
    assert calls == 2 * len(wrapped_observations)

    wrapper, _ = wrapped_observations[0]
    changed_observation, _ = wrapper.env.reset(seed=100)
    wrapper.observation(changed_observation)
    assert calls == 2 * len(wrapped_observations) + 2


def test_local_grid_marks_out_of_bounds_and_entities():
    base = GardenerEnv(size=8)
    base.reset(seed=9)
    base.agent = np.array([0, 0])
    wrapper = LocalGridObsWrapper(base, radius=1)
    observation = wrapper.observation(base.get_state())
    grid = observation[: wrapper.n_grid_features].reshape(wrapper.N_CHANNELS, 3, 3)
    assert np.all(grid[wrapper.CH_WALL, 0, :])
    assert np.all(grid[wrapper.CH_WALL, :, 0])
    assert wrapper.observation_space.contains(observation)


@pytest.mark.parametrize("radius", [-1, 1.5, True])
def test_local_grid_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        LocalGridObsWrapper(GardenerEnv(size=8), radius=radius)


def test_illegal_action_penalty_remaps_to_stay():
    base = GardenerEnv(size=8)
    base.reset(seed=12)
    illegal = np.flatnonzero(base.action_mask() == 0)[0]
    before = base.agent.copy()
    wrapper = IllegalActionPenaltyWrapper(base, penalty=-2.0)
    _, reward, _, _, _ = wrapper.step(illegal)
    assert np.array_equal(base.agent, before)
    assert reward == pytest.approx(-2.1)
    assert base.action == action_dict["stay"]


@pytest.mark.parametrize("penalty", [float("inf"), float("nan"), True])
def test_illegal_action_wrapper_rejects_invalid_penalty(penalty):
    with pytest.raises(ValueError, match="penalty"):
        IllegalActionPenaltyWrapper(GardenerEnv(size=8), penalty=penalty)


def test_closing_one_renderer_keeps_the_other_environment_live():
    first = GardenerEnv(size=8, render_mode="human")
    second = GardenerEnv(size=8, render_mode="human")
    try:
        first.reset(seed=1)
        second.reset(seed=2)
        first.close()
        second.render()
        assert second.display is not None
    finally:
        first.close()
        second.close()


def test_screenshot_saves_current_frame_only_while_renderer_is_live(tmp_path, monkeypatch):
    import pygame

    from npc_gym.envs.gardener.gardener_rendering import GardenerRenderer

    # Drawing still exercises presentation; skip only the real-time delay.
    class UnpacedClock:
        def tick(self, fps):
            pass

    monkeypatch.setattr(pygame.time, "Clock", UnpacedClock)
    env = GardenerEnv(size=5)
    renderer = GardenerRenderer()
    path = tmp_path / "frame.png"
    try:
        renderer.save_screenshot(path)
        assert not path.exists()
        env.reset(seed=1)
        renderer.draw(env)
        renderer.save_screenshot(path)
        np.testing.assert_array_equal(
            pygame.surfarray.array3d(pygame.image.load(path)), pygame.surfarray.array3d(renderer.screen)
        )
        first_frame = path.read_bytes()
        env.reset(seed=2)
        renderer.draw(env)
        renderer.save_screenshot(path)
        assert path.read_bytes() != first_frame
        np.testing.assert_array_equal(
            pygame.surfarray.array3d(pygame.image.load(path)), pygame.surfarray.array3d(renderer.screen)
        )
        renderer.close()
        saved_frame = path.read_bytes()
        renderer.save_screenshot(path)
        assert path.read_bytes() == saved_frame
        renderer.save_screenshot(tmp_path / "closed.png")
        assert not (tmp_path / "closed.png").exists()
    finally:
        renderer.close()
        env.close()


@pytest.mark.parametrize("freeze", [0, 5])
def test_local_grid_frog_channels_preserve_counts_timers_and_hide_collected(freeze):
    env = GardenerEnv(size=15, frog_freeze=freeze)
    try:
        env.reset(seed=4)
        env.agent = np.array([7, 7])
        env.frogs[:] = [7, 7]
        env.frog_timer[:] = freeze
        env.frog_timer[0] = 0
        wrapped = LocalGridObsWrapper(env, radius=4, include_frogs=True)
        plain = LocalGridObsWrapper(env, radius=4)
        observation = wrapped.observation(env.get_state())
        grid = observation[: wrapped.n_grid_features].reshape(wrapped.n_channels, 9, 9)
        assert wrapped.observation_space.contains(observation)
        assert grid[5:, 4, 4].sum() == pytest.approx(1)
        assert grid[5, 4, 4] == pytest.approx(1 if freeze == 0 else 1 / env.num_frogs)
        if freeze:
            assert grid[5 + freeze, 4, 4] == pytest.approx((env.num_frogs - 1) / env.num_frogs)
        original = plain.observation(env.get_state())
        np.testing.assert_array_equal(observation[: plain.n_grid_features], original[: plain.n_grid_features])
        np.testing.assert_array_equal(observation[-8:], original[-8:])
        # Collection and leaving the square remove only the affected frogs.
        env.collected_frogs[0] = True
        env.frogs[1] = [0, 0]
        observation = wrapped.observation(env.get_state())
        grid = observation[: wrapped.n_grid_features].reshape(wrapped.n_channels, 9, 9)
        assert grid[5:].sum() == pytest.approx((env.num_frogs - 2) / env.num_frogs)
        assert wrapped.observation_space.contains(observation)
    finally:
        env.close()


def test_local_grid_frog_channels_are_empty_after_all_collections():
    env = GardenerEnv(size=3)
    try:
        env.reset(seed=0)
        env.collected_frogs[:] = True
        observation = env.get_state()
        wrapper = LocalGridObsWrapper(env, include_frogs=True)
        encoded = wrapper.observation(observation)
        grid = encoded[: wrapper.n_grid_features].reshape(wrapper.n_channels, 5, 5)
        assert not grid[5:].any()
        assert wrapper.observation_space.contains(encoded)
        with pytest.raises(TypeError, match="include_frogs"):
            LocalGridObsWrapper(env, include_frogs=1)
    finally:
        env.close()


def test_state_features_with_frogs_preserve_task_features_and_use_only_observation():
    env = GardenerEnv(size=15)
    try:
        original, _ = env.reset(seed=0)
        plain = StateFeatureObsWrapper(env)
        wrapped = StateFeatureObsWrapper(env, include_frogs=True)
        assert plain.observation_space.shape == (12,)
        assert wrapped.observation_space.shape == (27,)
        saved = wrapped.observation(original)
        for action in [0, 1, 2, 3, 4] * 3:
            obs = env.get_state()
            before = env.labeling_state()
            features = wrapped.observation(obs)
            np.testing.assert_array_equal(features[:12], plain.observation(obs))
            np.testing.assert_array_equal(features[22:], env.action_mask())
            assert wrapped.observation_space.contains(features)
            assert env.labeling_state() == before
            env.step(action)
        # A detached old observation must not acquire the live environment's risks.
        np.testing.assert_array_equal(wrapped.observation(original), saved)
    finally:
        env.close()


def test_state_features_distinguish_mobile_frozen_and_collected_frogs():
    env = GardenerEnv(size=15)
    try:
        env.reset(seed=4)
        env.frogs[:] = env.agent
        wrapped = StateFeatureObsWrapper(env, include_frogs=True)
        assert env.action_mask()[:4].any()
        env.frog_timer[:] = 0
        assert wrapped.observation(env.get_state())[16] == 0  # Mobile frogs leave before Stay collects.
        env.frog_timer[:] = 1
        assert wrapped.observation(env.get_state())[16] == 1  # One frozen step is sufficient.
        env.collected_frogs[0] = True
        assert wrapped.observation(env.get_state())[16] == pytest.approx((env.num_frogs - 1) / env.num_frogs)
        env.collected_frogs[:] = True
        assert not wrapped.observation(env.get_state())[12:22].any()
    finally:
        env.close()


def test_state_features_handle_empty_object_lists_and_reject_nonboolean_option():
    env = GardenerEnv(size=5)
    try:
        env.observation_codec = GardenerObservationCodec(5, 0, 0, 0, 0)
        wrapped = StateFeatureObsWrapper(env, include_frogs=True)
        features = wrapped.observation((1, 1))
        np.testing.assert_array_equal(features[12:22], np.zeros(10))
        assert wrapped.observation_space.contains(features)
        with pytest.raises(TypeError, match="include_frogs"):
            StateFeatureObsWrapper(env, include_frogs=1)
    finally:
        env.close()
