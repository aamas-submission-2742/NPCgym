import os
from importlib.resources import files

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import TimeLimit

from npc_gym.algorithms import TabularQLearning
from npc_gym.envs.merchant.labels import MerchantLabel
from npc_gym.envs.merchant.merchant import MerchantEnv, action_dict, labels_dict
from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.wrappers.merchant_wrappers import IgnoreTimeObservation


@pytest.mark.parametrize(
    "layout",
    ["basic", "cycle", "dangerous", "possibledanger", "possibledanger2", "twist"],
)
def test_layouts_reset_to_valid_observations(layout):
    env = MerchantEnv(layout=layout)
    observation, info = env.reset(seed=11)
    assert env.observation_space.contains(observation)
    assert info["action_mask"].shape == (env.action_space.n,)
    assert info["labels"] == frozenset({MerchantLabel.AT_HOME})
    assert info[EPISODE_METRICS_KEY] == {
        "score": 0,
        "unload_danger": 0,
        "unload_market": 0,
        "death": 0,
    }
    assert observation[:2] == tuple(env.home)
    assert observation[2] == labels_dict["H"]


def test_passes_gymnasium_environment_checker():
    check_env(MerchantEnv(layout="basic"), skip_render_check=True)


def test_seeded_danger_outcome_is_reproducible():
    def rollout():
        env = MerchantEnv(layout="basic", risk_fight=0.5)
        env.reset(seed=29)
        env.pos = np.argwhere(env.map == "D")[0][::-1]
        env.label = "."
        return env.step(action_dict["north"])[0:3]

    assert rollout() == rollout()


def test_seeded_trajectory_is_reproducible():
    first = MerchantEnv(layout="dangerous", risk_fight=0.5, risk_death=0.5)
    second = MerchantEnv(layout="dangerous", risk_fight=0.5, risk_death=0.5)
    actions = [2, 2, 1, 1, 3, 6, 0]
    first.reset(seed=31)
    second.reset(seed=31)
    assert [first.step(action)[:4] for action in actions] == [second.step(action)[:4] for action in actions]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layout": "missing"}, "Unknown Merchant layout"),
        ({"risk_fight": -0.1}, "risk_fight"),
        ({"risk_death": 1.1}, "risk_death"),
        ({"capacity": 0}, "capacity"),
        ({"sunset": -1}, "sunset"),
        ({"step_delay_ms": -1}, "step_delay_ms"),
    ],
)
def test_invalid_constructor_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MerchantEnv(**kwargs)


def test_layout_paths_outside_the_bundled_catalog_are_rejected(tmp_path):
    external = tmp_path / "external"
    external.with_suffix(".txt").write_text("XXXXX\nXH.MX\nXXXXX\n", encoding="utf-8")
    layouts = files("npc_gym.envs.merchant").joinpath("merchant_layouts")
    for name in (str(external), os.path.relpath(external, layouts), "../merchant_layouts/basic", r"..\external"):
        with pytest.raises(ValueError, match="Unknown Merchant layout"):
            MerchantEnv(layout=name)


@pytest.mark.parametrize("action", [-1, len(action_dict), 1.5])
def test_step_rejects_actions_outside_the_declared_space(action):
    env = MerchantEnv()
    with pytest.raises(ValueError, match="outside"):
        env.step(action)


def test_wall_and_reverse_actions_are_excluded():
    env = MerchantEnv(layout="basic")
    env.reset(seed=1)
    assert env.action_mask()[action_dict["west"]] == 0
    env.step(action_dict["east"])
    assert env.action_mask()[action_dict["west"]] == 0


def test_empty_handed_cycle_market_allows_only_reversal():
    env = IgnoreTimeObservation(MerchantEnv(layout="cycle"))
    try:
        env.reset(seed=7)
        route = ["south"] * 2 + ["east"] * 4 + ["north"] * 2 + ["east"] * 3
        for name in route:
            observation, reward, terminated, truncated, info = env.step(action_dict[name])
            assert (reward, terminated, truncated) == (0, False, False)
        assert observation[:5] == (8, 3, labels_dict["M"], 0, 0)
        np.testing.assert_array_equal(info["action_mask"], [0, 0, 0, 1, 0, 0, 0])
        assert info["labels"] == frozenset({MerchantLabel.AT_MARKET, MerchantLabel.EAST})

        model = TabularQLearning(env, seed=7, use_action_mask=True, log_interval=None)
        action = model.predict(observation, info)
        assert action == action_dict["west"]
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation[:5] == (7, 3, labels_dict["."], 0, 0)
        assert (reward, terminated, truncated) == (0, False, False)
        np.testing.assert_array_equal(info["action_mask"], [0, 0, 0, 1, 0, 0, 0])
    finally:
        env.close()


def test_masked_learning_bootstraps_into_and_leaves_cycle_dead_end(monkeypatch):
    base = MerchantEnv(layout="cycle")
    original_reset = base.reset

    def reset_near_market(**kwargs):
        original_reset(**kwargs)
        route = ["south"] * 2 + ["east"] * 4 + ["north"] * 2 + ["east"] * 2
        for name in route:
            observation, _, _, _, info = base.step(action_dict[name])
        return observation, info

    monkeypatch.setattr(base, "reset", reset_near_market)
    env = TimeLimit(IgnoreTimeObservation(base), max_episode_steps=150)
    try:
        model = TabularQLearning(
            env,
            exploration_initial_eps=0,
            exploration_final_eps=0,
            seed=7,
            use_action_mask=True,
            log_interval=None,
        )
        chosen = []
        model.learn(3, callback=lambda model, step: chosen.append(step.action))
        assert chosen == [action_dict[name] for name in ("east", "west", "west")]
        assert model.num_timesteps == 3
        assert tuple(base.pos) == (6, 3)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("cell", "carried", "previous", "expected"),
    [
        ("M", 0, "east", ["west"]),
        ("M", 1, "east", ["unload"]),
        ("D", 0, "east", ["fight"]),
        ("D", 1, "east", ["unload", "fight"]),
        ("T", 0, "east", ["extract"]),
        ("R", 0, "east", ["extract"]),
        ("T", 5, "east", ["west"]),
        ("R", 5, "east", ["west"]),
        ("C", 1, "east", ["west"]),
        ("M", 0, "west", ["west"]),  # Reversal points into the outer wall.
    ],
)
def test_dead_end_fallback_preserves_other_action_restrictions(cell, carried, previous, expected):
    env = MerchantEnv(layout="cycle")
    try:
        # Use the market's one-exit geometry to isolate each cell/inventory rule.
        env.pos = env.market.copy()
        env.label, env.carried_wood, env.action = cell, carried, previous
        assert np.flatnonzero(env.action_mask()).tolist() == [action_dict[name] for name in expected]
    finally:
        env.close()


@pytest.mark.parametrize("layout", ["basic", "cycle", "dangerous", "possibledanger", "possibledanger2", "twist"])
def test_bundled_layout_masks_preserve_choices_except_for_forced_reversal(layout):
    env = MerchantEnv(layout=layout)
    try:
        for y, x in np.argwhere(env.map != "X"):
            cell = str(env.map[y, x])
            cells = [cell] + (["C"] if cell in {"T", "R"} else ["."] if cell == "D" else [])
            for cell in cells:
                for carried in (0, 1, env.capacity):
                    env.pos = np.array([x, y])
                    env.label, env.carried_wood, env.action = cell, carried, None
                    unrestricted = set(np.flatnonzero(env.action_mask()))
                    assert unrestricted
                    for previous, reverse in (
                        ("north", "south"),
                        ("south", "north"),
                        ("east", "west"),
                        ("west", "east"),
                    ):
                        env.action = previous
                        allowed = set(np.flatnonzero(env.action_mask()))
                        assert allowed == (unrestricted - {action_dict[reverse]} or unrestricted)
                        for name, (dx, dy) in {
                            "north": (0, -1),
                            "south": (0, 1),
                            "east": (1, 0),
                            "west": (-1, 0),
                        }.items():
                            if env.map[y + dy, x + dx] == "X":
                                assert action_dict[name] not in allowed
    finally:
        env.close()


def test_extract_depletes_resource_and_pays_once():
    env = MerchantEnv(layout="basic")
    y, x = np.argwhere(env.map == "T")[0]
    env.pos = np.array([x, y])
    env.label = "T"
    _, reward, terminated, _, _ = env.step(action_dict["extract"])
    assert (reward, terminated, env.carried_wood, env.label) == (50, False, 1, "C")
    _, reward, _, _, _ = env.step(action_dict["extract"])
    assert (reward, env.carried_wood) == (0, 1)


@pytest.mark.parametrize("wrapped", [False, True])
def test_unmasked_route_cannot_extract_beyond_capacity(wrapped):
    base = MerchantEnv(layout="basic", capacity=1, risk_fight=0.0)
    env = IgnoreTimeObservation(base) if wrapped else base
    try:
        env.reset(seed=0)
        for action in [0, 0, 0, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2]:
            observation, _, _, _, info = env.step(action)
            assert env.observation_space.contains(observation)
        assert base.carried_wood == 1
        assert info["action_mask"][action_dict["extract"]] == 0
        observation, reward, terminated, truncated, info = env.step(action_dict["extract"])
        assert (reward, terminated, truncated) == (0, False, False)
        assert env.observation_space.contains(observation)
        assert info[EPISODE_METRICS_KEY]["score"] == 50
    finally:
        env.close()


@pytest.mark.parametrize("layout", ["basic", "cycle", "dangerous", "possibledanger", "possibledanger2", "twist"])
@pytest.mark.parametrize("resource", ["T", "R"])
@pytest.mark.parametrize("inventory", [(1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_full_inventory_preserves_resource_until_unloaded(layout, resource, inventory, wrapped):
    base = MerchantEnv(layout=layout, capacity=sum(inventory))
    env = IgnoreTimeObservation(base) if wrapped else base
    try:
        env.reset(seed=0)
        # Isolate the capacity boundary with same-resource and mixed inventories.
        y, x = np.argwhere(base.map == resource)[0]
        base.pos = np.array([x, y])
        base.label = resource
        base.carried_wood, base.carried_ore = inventory
        wood, ore = base.wood.copy(), base.ore.copy()
        assert base.action_mask()[action_dict["extract"]] == 0
        for clock in (1, 2):
            observation, reward, terminated, truncated, info = env.step(action_dict["extract"])
            assert (reward, terminated, truncated) == (0, False, False)
            assert env.observation_space.contains(observation)
            assert (base.carried_wood, base.carried_ore) == inventory
            assert base.clock == clock
            assert base.label == resource
            np.testing.assert_array_equal(base.wood, wood)
            np.testing.assert_array_equal(base.ore, ore)
            assert MerchantLabel.EXTRACT in info["labels"]
            assert info[EPISODE_METRICS_KEY]["score"] == 0

        observation, reward, _, _, info = env.step(action_dict["unload"])
        assert reward == -50 * sum(inventory)
        assert env.observation_space.contains(observation)
        assert info["action_mask"][action_dict["extract"]] == 1
        observation, reward, _, _, _ = env.step(action_dict["extract"])
        assert reward == 50
        assert env.observation_space.contains(observation)
        assert base.carried_wood + base.carried_ore == 1
        assert base.label == "C"
    finally:
        env.close()


def test_fight_death_and_survival_are_controllable():
    fatal = MerchantEnv(layout="basic", risk_death=1.0)
    fatal.label = "D"
    fatal_result = fatal.step(action_dict["fight"])
    assert fatal_result[1:4] == (-100, True, False)
    assert fatal_result[4][EPISODE_METRICS_KEY] == {
        "score": -100,
        "unload_danger": 0,
        "unload_market": 0,
        "death": 1,
    }

    safe = MerchantEnv(layout="basic", risk_death=0.0)
    safe.label = "D"
    assert safe.step(action_dict["fight"])[1:4] == (0, False, False)
    assert safe.label == "."


def test_unload_rewards_market_and_penalizes_elsewhere():
    env = MerchantEnv(layout="basic")
    env.carried_wood = 2
    env.label = "."
    first = env.step(action_dict["unload"])
    assert first[1:4] == (-100, False, False)
    assert env.carried_wood == 0

    env.carried_ore = 2
    env.label = "M"
    second = env.step(action_dict["unload"])
    assert second[1:4] == (200, True, False)
    assert second[4][EPISODE_METRICS_KEY] == {
        "score": 100,
        "unload_danger": 0,
        "unload_market": 1,
        "death": 0,
    }


def test_unloading_under_attack_counts_as_a_danger_unload():
    env = MerchantEnv(layout="basic")
    env.carried_wood = 1
    env.label = "D"

    _, reward, terminated, _, info = env.step(action_dict["unload"])

    assert (reward, terminated) == (-50, False)
    assert info[EPISODE_METRICS_KEY]["unload_danger"] == 1


def test_clock_saturates_and_labels_reflect_state():
    env = MerchantEnv(layout="basic", sunset=2)
    env.step(action_dict["east"])
    env.step(action_dict["east"])
    _, _, _, _, info = env.step(action_dict["east"])
    assert env.clock == 2
    assert MerchantLabel.SUNDOWN in info["labels"]


def test_step_info_contains_typed_state_and_action_labels():
    env = MerchantEnv(layout="basic")
    _, reset_info = env.reset(seed=2)
    assert reset_info["labels"] == frozenset({MerchantLabel.AT_HOME})

    _, _, _, _, step_info = env.step(action_dict["east"])
    assert MerchantLabel.EAST in step_info["labels"]
    assert all(isinstance(label, MerchantLabel) for label in step_info["labels"])


def test_ignore_time_wrapper_preserves_other_fields():
    env = IgnoreTimeObservation(MerchantEnv(layout="basic"))
    observation, _ = env.reset(seed=7)
    raw = env.unwrapped.get_state()
    assert observation == raw[:5] + raw[6:]
    assert env.observation_space.contains(observation)
    stepped = env.step(action_dict["east"])[0]
    assert env.observation_space.contains(stepped)


def test_rgb_array_render_is_headless_and_closes_cleanly():
    env = MerchantEnv(render_mode="rgb_array")
    try:
        env.reset(seed=4)
        image = env.render()
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
    finally:
        env.close()
    assert env.display is None


def test_masked_tabular_q_learning_bootstraps_across_time_limit():
    env = TimeLimit(IgnoreTimeObservation(MerchantEnv(layout="basic")), max_episode_steps=2)
    try:
        model = TabularQLearning(
            env,
            exploration_initial_eps=0,
            exploration_final_eps=0,
            seed=3,
            use_action_mask=True,
            log_interval=None,
        )
        model.learn(3)
        assert model.num_timesteps == 3
    finally:
        env.close()
