from pacman_fixtures import TERMINAL_LAYOUT

"""Boundary tests for the Pacman environment."""

import os
import random
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest
from gymnasium.envs.registration import registry
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit

from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.envs.pacman.labels import PacmanLabel, PacmanLabelingFunction
from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.pacman_env import SUPPORTED_FEATURES, SUPPORTED_VECTOR_FEATURES, PacmanEnv
from npc_gym.envs.pacman.simulation import Simulation
from npc_gym.labels import Transition


def make_env(features="essential", layout="small"):
    return PacmanEnv(layout=layout, features=features, render_mode=None)


PACMAN_LAYOUTS = PacmanEnv.layouts
IMAGE_FEATURES = tuple(sorted(SUPPORTED_FEATURES - set(SUPPORTED_VECTOR_FEATURES)))


def test_namespaced_pacman_environment_is_registered():
    assert "npc_gym/Pacman-v1" in registry
    assert "NormativePacman-v0" not in registry
    assert "NormativePacmanPO-v0" not in registry


def test_headless_reset_and_step_return_gymnasium_tuples():
    env = make_env()
    try:
        observation, info = env.reset(seed=13)
        assert isinstance(observation, np.ndarray)
        assert info["step_counter"] == [[0]]
        assert PacmanLabel.SCORE_0 in info["labels"]
        assert info[EPISODE_METRICS_KEY]["score"] == 0.0
        assert info[EPISODE_METRICS_KEY]["won"] == 0
        assert info[EPISODE_METRICS_KEY]["lost"] == 0
        result = env.step(0)
        assert len(result) == 5
        assert isinstance(result[1], float)
        assert isinstance(result[2], bool)
        assert isinstance(result[3], bool)
        assert result[4]["step_counter"] == [[1]]
        assert PacmanLabel.STOP in result[4]["labels"]
        assert isinstance(result[4]["agent_eaten"], list)
        assert all(count >= 0 for count in result[4]["agent_eaten"])
        snapshot = env.labeling_state()
        assert result[4][EPISODE_METRICS_KEY] == {
            "score": snapshot.score,
            "blue_eaten": snapshot.ghosts[0].eaten_count,
            "orange_eaten": 0,
            "food_remaining": len(snapshot.food),
            "won": int(snapshot.won),
            "lost": int(snapshot.lost),
        }
    finally:
        env.close()


def test_gymnasium_checker_accepts_pacman_adapter():
    check_env(make_env(), skip_render_check=True)


def test_action_meanings_and_illegal_action_falls_back_to_stop():
    env = make_env()
    try:
        env.reset(seed=2)
        assert env.get_action_meanings() == [0, 1, 2, 3, 4]
        snapshot = env.labeling_state()
        illegal = next(
            action
            for action, delta in {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}.items()
            if (snapshot.player.position[0] + delta[0], snapshot.player.position[1] + delta[1]) in snapshot.layout.walls
        )
        position = snapshot.player.position
        before = env.illegal_move_counter
        env.step(illegal)
        assert env.illegal_move_counter == before + 1
        assert env.labeling_state().player.position == position
    finally:
        env.close()


def test_legal_action_moves_pacman():
    env = make_env()
    try:
        env.reset(seed=2)
        snapshot = env.labeling_state()
        legal = next(
            action
            for action, delta in {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}.items()
            if (snapshot.player.position[0] + delta[0], snapshot.player.position[1] + delta[1])
            not in snapshot.layout.walls
        )
        env.step(legal)
        assert env.labeling_state().player.position != snapshot.player.position
        assert env.illegal_move_counter == 0
    finally:
        env.close()


def test_terminal_transition_is_reported_with_a_valid_observation():
    env = make_env(layout=TERMINAL_LAYOUT)
    try:
        env.reset(seed=0)
        first = env.step(3)
        observation, _, terminated, truncated, info = env.step(3)
        assert not first[2]
        assert terminated
        assert not truncated
        assert env.labeling_state().lost
        assert env.observation_space.contains(observation)
        assert info["episode"] == [{"r": env.cum_reward, "l": 2, "w": False}]
        assert info["labels"]
    finally:
        env.close()


def test_time_limit_wrapper_owns_episode_truncation():
    env = TimeLimit(make_env(), max_episode_steps=1)
    try:
        env.reset(seed=3)
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert reward == -1.0
        assert not terminated
        assert truncated
        assert info["episode"] is None
        assert "labels" in info
    finally:
        env.close()


def test_direct_environment_has_no_hidden_episode_limit():
    env = make_env()
    try:
        env.reset(seed=3)
        env.step_counter = 10_000
        _, _, _, truncated, _ = env.step(0)
        assert not truncated
    finally:
        env.close()


def test_supported_layouts_match_bundled_resources():
    layout_dir = files("npc_gym.envs.pacman").joinpath("layouts")
    bundled = {
        resource.name.removesuffix(".lay") for resource in layout_dir.iterdir() if resource.name.endswith(".lay")
    }
    assert set(PACMAN_LAYOUTS) == bundled


def assert_rollout_stays_in_space(env, dtype, steps=40):
    """Drive a seeded random rollout, since a single Stop step visits very few features."""
    observation, _ = env.reset(seed=5)
    assert env.observation_space.contains(observation)
    assert observation.dtype == dtype
    rng = np.random.default_rng(5)
    for _ in range(steps):
        observation, _, terminated, _, _ = env.step(int(rng.integers(0, 5)))
        assert env.observation_space.contains(observation)
        assert observation.dtype == dtype
        if terminated:
            observation, _ = env.reset(seed=6)
            assert env.observation_space.contains(observation)


@pytest.mark.parametrize("features", SUPPORTED_VECTOR_FEATURES)
@pytest.mark.parametrize("layout", PACMAN_LAYOUTS)
def test_vector_observation_belongs_to_declared_space_for_every_mode_and_layout(features, layout):
    env = make_env(features, layout)
    try:
        assert_rollout_stays_in_space(env, env.observation_space.dtype)
    finally:
        env.close()


@pytest.mark.parametrize("features", IMAGE_FEATURES)
@pytest.mark.parametrize("layout", PACMAN_LAYOUTS)
def test_image_observation_belongs_to_declared_space_for_every_mode_and_layout(features, layout):
    env = make_env(features, layout)
    try:
        assert_rollout_stays_in_space(env, np.uint8, steps=5)
    finally:
        env.close()


def test_vector_observations_stack_into_a_batched_buffer():
    """A varying observation length only fails once observations share one batched buffer."""
    envs = SyncVectorEnv([lambda: TimeLimit(make_env("hungry", "small"), max_episode_steps=60)] * 2)
    try:
        observations, _ = envs.reset(seed=5)
        assert observations.shape == (2, *envs.single_observation_space.shape)
        for _ in range(60):
            observations, *_ = envs.step(envs.action_space.sample())
            assert observations.shape == (2, *envs.single_observation_space.shape)
    finally:
        envs.close()


@pytest.mark.parametrize("features", ["dfa", "dfa-distinguish", "image-full+dfa", "image-full+dfa-distinguish"])
def test_dfa_observation_modes_include_configured_state(features):
    env = PacmanEnv(layout="small", features=features, dfas={"VegBlueDFA": 1.0})
    try:
        observation, _ = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        observation, *_ = env.step(0)
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_image_crop_observation_is_84_pixels_square():
    env = make_env(features="image-crop")
    try:
        observation, _ = env.reset(seed=5)
        assert observation.shape == (84, 84, 3)
    finally:
        env.close()


def test_rgb_array_rendering_is_headless_and_returns_full_frame():
    env = PacmanEnv(layout="small", features="essential", render_mode="rgb_array")
    try:
        env.reset(seed=5)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (420, 420, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_render_requires_reset_when_a_mode_is_configured():
    env = PacmanEnv(layout="small", features="essential", render_mode="rgb_array")
    try:
        with pytest.raises(ResetNeeded, match="reset before"):
            env.render()
    finally:
        env.close()


def test_render_has_no_legacy_mode_argument():
    env = PacmanEnv(layout="small", features="essential", render_mode="rgb_array")
    try:
        env.reset(seed=5)
        with pytest.raises(TypeError):
            env.render(mode="human")
    finally:
        env.close()


def test_unsupported_feature_name_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Pacman features 'unknown'"):
        make_env(features="unknown")


@pytest.mark.parametrize("features", [None, 1, ["essential"]])
def test_non_string_feature_name_is_rejected(features):
    with pytest.raises(TypeError, match="features must be a string"):
        make_env(features=features)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="Unknown Pacman layout 'missing'"):
        make_env(layout="missing")


def test_non_string_layout_is_rejected():
    with pytest.raises(TypeError, match="layout must be a string"):
        make_env(layout=3)


def test_layout_outside_the_packaged_resources_is_rejected(tmp_path):
    (tmp_path / "external.lay").write_text("%%%%%\n%P .%\n%%%%%\n", encoding="utf-8")
    escape = os.path.relpath(tmp_path / "external", files("npc_gym.envs.pacman").joinpath("layouts"))

    for name in (escape, str(tmp_path / "external"), r"..\elsewhere\external"):
        with pytest.raises(ValueError, match="Unknown Pacman layout"):
            make_env(layout=name)


def test_invalid_dfa_configuration_is_rejected():
    with pytest.raises(TypeError, match="dfas must be a mapping"):
        PacmanEnv(dfas=["VegBlueDFA"])
    with pytest.raises(ValueError, match="Unknown Pacman DFA"):
        PacmanEnv(dfas={"MissingDFA": 1.0})
    for name in ("np", "DFA"):
        with pytest.raises(ValueError, match="Unknown Pacman DFA"):
            PacmanEnv(dfas={name: 1.0})


@pytest.mark.parametrize("reward", ["bad", None, True])
def test_non_numeric_dfa_rewards_are_rejected(reward):
    with pytest.raises(TypeError, match="DFA rewards must be finite real numbers"):
        PacmanEnv(features="dfa", dfas={"VegBlueDFA": reward})


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_dfa_rewards_are_rejected(reward):
    with pytest.raises(ValueError, match="DFA rewards must be finite real numbers"):
        PacmanEnv(features="dfa", dfas={"VegBlueDFA": reward})


def test_dfas_are_rejected_for_feature_modes_that_ignore_them():
    with pytest.raises(ValueError, match="ignore dfas"):
        PacmanEnv(features="essential", dfas={"VegBlueDFA": 1.0})
    with pytest.raises(ValueError, match="ignore dfas"):
        PacmanEnv(features="image-full", dfas={"VegBlueDFA": 1.0})


@pytest.mark.parametrize("action", range(5))
def test_every_declared_action_returns_a_valid_transition(action):
    env = make_env()
    try:
        env.reset(seed=9)
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["labels"]
    finally:
        env.close()


@pytest.mark.parametrize("action", [-1, 5, 1.5, "North", np.array([1])])
def test_invalid_actions_are_rejected(action):
    env = make_env()
    try:
        env.reset(seed=9)
        with pytest.raises(ValueError, match="Invalid Pacman action"):
            env.step(action)
    finally:
        env.close()


def test_step_before_reset_is_rejected():
    env = make_env()
    try:
        with pytest.raises(ResetNeeded, match="reset before step"):
            env.step(0)
    finally:
        env.close()


def test_step_after_termination_is_rejected():
    env = make_env(layout=TERMINAL_LAYOUT)
    try:
        env.reset(seed=0)
        env.step(3)
        env.step(3)
        with pytest.raises(ResetNeeded, match="reset after"):
            env.step(0)
    finally:
        env.close()


def test_nonempty_reset_options_are_rejected():
    env = make_env()
    try:
        with pytest.raises(ValueError, match="does not support reset options"):
            env.reset(options={"layout": "smallGrid"})
    finally:
        env.close()


def test_render_mode_may_be_omitted_for_headless_use():
    env = PacmanEnv(layout="small", features="essential")
    try:
        observation, _ = env.reset()
        assert env.render_mode is None
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_close_is_safe_after_partial_construction():
    env = PacmanEnv.__new__(PacmanEnv)
    env.close()
    env.close()


def test_close_is_idempotent_after_normal_construction():
    env = make_env()
    env.reset(seed=1)
    env.close()
    env.close()


def test_seeded_environments_remain_identical_when_stepped_interleaved():
    first = make_env()
    second = make_env()
    try:
        first_observation, first_info = first.reset(seed=37)
        second_observation, second_info = second.reset(seed=37)
        assert np.array_equal(first_observation, second_observation)
        assert first_info == second_info

        for action in [0, 1, 2, 3, 4] * 3:
            first_result = first.step(action)
            second_result = second.step(action)
            assert np.array_equal(first_result[0], second_result[0])
            assert first_result[1:] == second_result[1:]
            assert first.labeling_state().ghosts == second.labeling_state().ghosts
            if first_result[2] or first_result[3]:
                break
    finally:
        first.close()
        second.close()


def test_resetting_with_the_same_seed_repeats_the_trajectory():
    env = make_env()

    def trajectory():
        observation, _ = env.reset(seed=19)
        result = [observation.copy()]
        for action in [0, 1, 0, 2, 0, 3, 0, 4]:
            observation, reward, terminated, truncated, _ = env.step(action)
            result.append((observation.copy(), reward, terminated, truncated))
            if terminated or truncated:
                break
        return result

    try:
        first = trajectory()
        second = trajectory()
        assert len(first) == len(second)
        assert np.array_equal(first[0], second[0])
        for first_step, second_step in zip(first[1:], second[1:]):
            assert np.array_equal(first_step[0], second_step[0])
            assert first_step[1:] == second_step[1:]
    finally:
        env.close()


def engine_state(env):
    state = env.labeling_state()
    return (
        state.player.position,
        state.player.direction,
        tuple(ghost.position for ghost in state.ghosts),
        tuple(ghost.direction for ghost in state.ghosts),
        tuple(ghost.scared_timer for ghost in state.ghosts),
        state.score,
        len(state.food),
        tuple(sorted(state.capsules)),
        state.won,
        state.lost,
    )


def test_seeded_random_engine_retains_captured_trajectory_on_original_fixture():
    with make_env(layout=TERMINAL_LAYOUT) as env:
        env.reset(seed=0)
        env.step(3)
        state = env.labeling_state()
        assert state.player.position == (2, 1)
        assert state.ghosts[0].position == (3, 1)
        _, reward, terminated, _, _ = env.step(3)
        assert (reward, terminated) == (-501, True)


def test_layout_lookup_ignores_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "external.lay").write_text("%%%%%\n%P .%\n%%%%%\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    env = make_env(layout="small")
    try:
        observation, _ = env.reset(seed=3)
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_seeded_pacman_does_not_consume_python_global_randomness():
    random.seed(73)
    state = random.getstate()
    expected = [random.random() for _ in range(5)]
    random.setstate(state)

    env = make_env()
    try:
        env.reset(seed=11)
        for action in [0, 1, 2, 3, 4]:
            env.step(action)
        actual = [random.random() for _ in range(5)]
    finally:
        env.close()

    assert actual == expected


def label_state():
    sim = Simulation(PacmanLayout.from_text("%%%%%%%%%%\n%P  G G .%\n%        %\n%%%%%%%%%%"), np.random.default_rng(0))
    state = sim.snapshot()
    return replace(
        state,
        player=replace(state.player, position=(1, 1)),
        score=101,
        killed_blue=True,
        ate_capsule=True,
        last_action=0,
        ghosts=(
            replace(state.ghosts[0], position=(2, 2), scared=True, scared_timer=8),
            replace(state.ghosts[1], position=(7, 7)),
        ),
    )


@pytest.mark.parametrize("label", [PacmanLabel.ADJACENT_BLUE_GHOST, PacmanLabel.ADJACENT_ORANGE_GHOST])
@pytest.mark.parametrize("scared", [False, True])
@pytest.mark.parametrize(
    "offset, adjacent",
    [
        ((-1, -1), True),
        ((-1, 0), True),
        ((-1, 1), True),
        ((0, -1), True),
        ((0, 1), True),
        ((1, -1), True),
        ((1, 0), True),
        ((1, 1), True),
        ((0.5, 0), True),
        ((0, -0.5), True),
        ((0, 0), False),
        ((1.5, 0), False),
        ((0, -1.5), False),
        ((1, 1.5), False),
    ],
)
def test_ghost_adjacency_labels_cover_neighbors_and_require_scared_ghost(label, scared, offset, adjacent):
    state = label_state()
    state = replace(
        state,
        ghosts=tuple(
            replace(g, position=(1 + offset[0], 1 + offset[1]), scared=scared, scared_timer=8 if scared else 0)
            for g in state.ghosts
        ),
    )
    labels = PacmanLabelingFunction()(
        Transition(previous_state=state, action=0, state=state, terminated=False, truncated=False)
    )
    assert (label in labels) == (adjacent and scared)


def test_pacman_labels_cover_events_scores_positions_and_action():
    state = label_state()
    labels = PacmanLabelingFunction()(
        Transition(previous_state=state, action=0, state=state, terminated=False, truncated=False)
    )
    assert {
        PacmanLabel.STOP,
        PacmanLabel.EAT_BLUE_GHOST,
        PacmanLabel.ADJACENT_BLUE_GHOST,
        PacmanLabel.STAYED_STILL,
        PacmanLabel.SCORE_GREATER_70,
        PacmanLabel.SCORE_GREATER_80,
        PacmanLabel.SCORE_GREATER_90,
        PacmanLabel.SCORE_GREATER_100,
        PacmanLabel.WEST_SIDE,
        PacmanLabel.EAT_POWER_PELLET,
        PacmanLabel.IN_CORNER,
    } <= labels
    assert PacmanLabel.EAT_ORANGE_GHOST not in labels
    assert PacmanLabel.IN_SOUTH_EAST not in labels


@pytest.mark.parametrize("score", [0, 100, 101, 399, 400, 401, 500])
def test_trapped_release_label_uses_actual_score_strictly_above_400(score):
    state = label_state()
    state = replace(state, score=score)
    labels = PacmanLabelingFunction()(
        Transition(previous_state=None, action=None, state=state, terminated=False, truncated=False)
    )
    assert (PacmanLabel.SCORE_GREATER_400 in labels) == (score > 400)
    assert (PacmanLabel.SCORE_GREATER_100 in labels) == (score > 100)
