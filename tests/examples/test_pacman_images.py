"""Headless CPU checks for the example's real image and bolt input path."""

from contextlib import closing
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("pygame")

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import preprocess_obs

from examples import pacman_images as example
from npc_gym.bolts import make_builtin_bolts
from npc_gym.envs.pacman.labels import PacmanLabel


@pytest.fixture(scope="module")
def bolts():
    return make_builtin_bolts(example.VEGAN_NORM_ID, reward=-300)


@pytest.fixture(autouse=True)
def small_cpu_run(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(example, "TRAINING_STEPS", 16)
    monkeypatch.setattr(example, "LEARNING_STARTS", 4)
    monkeypatch.setattr(example, "EVALUATION_EPISODES", 2)
    monkeypatch.setattr(example, "MAX_EPISODE_STEPS", 6)
    monkeypatch.setattr(example, "DEVICE", "cpu")
    yield
    torch.set_num_threads(previous)


def test_two_live_environments_are_seeded_and_close_independently(bolts):
    with closing(example.make_env(bolts)) as first, closing(example.make_env(bolts)) as second:
        left, li = first.reset(seed=12)
        right, ri = second.reset(seed=12)
        np.testing.assert_equal(left, right)
        assert li == ri
        assert first.observation_space.contains(left)
        assert left["observation"].shape == (80, 210, 2)
        assert left["observation"].dtype == np.uint8
        assert left["automata"].sum() == len(bolts) == 2
        assert all(value == 0 for value in li["restraining_bolts"]["reward_adjustments"].values())
        np.testing.assert_array_equal(left["observation"][..., 0], left["observation"][..., 1])
        next_obs, *_ = first.step(0)
        np.testing.assert_array_equal(next_obs["observation"][..., 0], left["observation"][..., 1])
        first.close()
        for _ in range(6):
            obs, reward, terminated, truncated, info = second.step(0)
            assert second.observation_space.contains(obs)
            diagnostics = info["restraining_bolts"]
            assert reward == pytest.approx(
                (diagnostics["wrapped_reward"] + sum(diagnostics["reward_adjustments"].values())) / 100
            )
            if terminated or truncated:
                break
        assert terminated or truncated
        second.reset(seed=13)


def test_ghost_penalties_and_task_reward_are_scaled_together(bolts, monkeypatch):
    class GhostMeal(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (240, 630, 3), dtype=np.uint8)
        action_space = gym.spaces.Discrete(5)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {"labels": frozenset()}

        def step(self, action):
            return (
                np.zeros(self.observation_space.shape, dtype=np.uint8),
                399,  # Two ghosts at +200 each, minus the living cost.
                True,
                False,
                {"labels": frozenset({PacmanLabel.EAT_BLUE_GHOST, PacmanLabel.EAT_ORANGE_GHOST})},
            )

    monkeypatch.setattr(example, "PacmanEnv", lambda **kwargs: GhostMeal())
    with closing(example.make_env(bolts)) as env:
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)
        assert terminated and not truncated
        assert reward == pytest.approx(-2.01)
        assert info["restraining_bolts"]["wrapped_reward"] == 399
        assert list(info["restraining_bolts"]["reward_adjustments"].values()) == [-300, -300]


def test_normalization_routing_and_gradients(bolts):
    with closing(example.make_env(bolts)) as env:
        obs, _ = env.reset(seed=12)
        model = DQN(
            "MultiInputPolicy",
            env,
            seed=3,
            device="cpu",
            buffer_size=8,
            policy_kwargs={"features_extractor_class": example.PixelBoltFeatures},
        )
        tensor_obs, _ = model.policy.obs_to_tensor(obs)
        prepared = preprocess_obs(tensor_obs, model.observation_space)
        pixels = prepared["observation"]
        np.testing.assert_allclose(pixels.numpy()[0], np.moveaxis(obs["observation"], -1, 0) / 255, atol=1e-7)
        np.testing.assert_array_equal(prepared["automata"].numpy()[0], obs["automata"])
        extractor = model.q_net.features_extractor
        features = extractor(prepared)
        assert features.shape == (1, 128 + len(obs["automata"]))
        torch.testing.assert_close(features[:, 128:], prepared["automata"])
        changed = {**prepared, "automata": torch.roll(prepared["automata"], 1, dims=1)}
        alternate = extractor(changed)
        torch.testing.assert_close(alternate[:, :128], features[:, :128])
        torch.testing.assert_close(alternate[:, 128:], changed["automata"])
        blank = extractor({**prepared, "observation": torch.zeros_like(pixels)})
        assert not torch.equal(blank[:, :128], features[:, :128])
        torch.testing.assert_close(blank[:, 128:], features[:, 128:])
        model.q_net(tensor_obs).square().sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in extractor.cnn.parameters())


@pytest.mark.parametrize("fail", [False, True])
def test_tiny_training_reload_evaluation_and_cleanup(bolts, monkeypatch, tmp_path, capsys, fail):
    monkeypatch.setattr(example, "make_builtin_bolts", lambda *args, **kwargs: bolts)
    monkeypatch.setattr(example, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path))
    captured = {}
    closed = []
    original_make, original_learn, original_evaluate = example.make_env, DQN.learn, example.evaluate_model

    def make_env(specs):
        env = original_make(specs)
        close = env.close

        def record_close():
            closed.append(env)
            close()

        env.close = record_close
        return env

    def learn(model, **kwargs):
        captured["trained"] = model
        captured["before"] = {key: value.clone() for key, value in model.policy.state_dict().items()}
        return original_learn(model, **kwargs)

    def evaluate(model, env):
        trained = captured["trained"]
        assert model is not trained
        assert model.num_timesteps == trained.num_timesteps == 16
        assert model._n_updates > 0
        assert model.batch_size == 32 and model.learning_rate == 1e-4
        assert model.policy.net_arch == [64, 64]
        assert any(not torch.equal(value, captured["before"][key]) for key, value in model.policy.state_dict().items())
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, trained.policy.state_dict()[key], rtol=0, atol=0)
        assert list(tmp_path.rglob("*.zip"))
        if fail:
            raise RuntimeError("evaluation failed")
        result = original_evaluate(model, env)
        assert result["augmented"] == pytest.approx(result["wrapped"] + sum(result[key] for key in bolts))
        return result

    monkeypatch.setattr(example, "make_env", make_env)
    monkeypatch.setattr(DQN, "learn", learn)
    monkeypatch.setattr(example, "evaluate_model", evaluate)
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            example.main()
    else:
        result = example.main()
        assert set(result) == {
            "augmented",
            "wrapped",
            "steps",
            "score",
            "blue_eaten",
            "orange_eaten",
            "food_remaining",
            "won",
            "lost",
            *bolts,
        }
        output = capsys.readouterr().out
        assert "Mean wrapped return" in output and "Mean bolt adjustments" in output
    assert len(closed) == 2
    assert list(tmp_path.iterdir()) == []


def test_evaluation_counts_final_adjustments_and_uses_independent_seed(monkeypatch):
    class Model:
        def predict(self, observation, *, deterministic):
            assert deterministic
            return np.array(2), None

    class Environment:
        def __init__(self):
            self.seeds = []

        def reset(self, *, seed):
            self.seeds.append(seed)
            return None, {}

        def step(self, action):
            assert action == 2
            return (
                None,
                -0.05,
                False,
                True,
                {
                    "restraining_bolts": {"wrapped_reward": 2, "reward_adjustments": {"one": -3, "two": -4}},
                    "episode_metrics": {"won": 0, "blue_eaten": 1, "orange_eaten": 2},
                },
            )

    env = Environment()
    assert example.evaluate_model(Model(), env) == {
        "augmented": -5,
        "wrapped": 2,
        "one": -3,
        "two": -4,
        "steps": 1,
        "won": 0,
        "blue_eaten": 1,
        "orange_eaten": 2,
    }
    assert env.seeds == [example.SEED + 10_000, None]


def test_script_keeps_formulas_in_library():
    source = Path(example.__file__).read_text()
    assert "make_regex" not in source and "make_ltlf_bolt(" not in source
    assert "make_builtin_bolts(VEGAN_NORM_ID" in source
    assert len(source.splitlines()) < 115


def test_scheduled_pixels_and_bolts_train_through_numeric_side_channels():
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3 import DQN

    from examples.pacman_images import PixelBoltFeatures
    from npc_gym.bolts import make_builtin_bolts
    from npc_gym.envs import PacmanEnv
    from npc_gym.wrappers import PacmanPixelObservation, RestrainingBoltWrapper

    with PacmanPixelObservation(
        RestrainingBoltWrapper(
            TimeLimit(PacmanEnv(features="image-full", ghost_behavior="deterministic"), max_episode_steps=8),
            bolts=make_builtin_bolts("pacman/vegan-v0", reward=-1),
        )
    ) as env:
        model = DQN(
            "MultiInputPolicy",
            env,
            device="cpu",
            buffer_size=32,
            learning_starts=0,
            batch_size=4,
            policy_kwargs={"features_extractor_class": PixelBoltFeatures},
            seed=0,
        )
        model.learn(8)
        assert model.num_timesteps == 8
