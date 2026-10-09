"""Defaults, pixel preprocessing, and saved-model execution for added baselines."""

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments import execution
from experiments.specifications import (
    ALGORITHMS,
    ENVIRONMENTS,
    GARDENER_PPO_EXPERIMENT_ID,
    PACMAN_DQN_EXPERIMENT_ID,
    PACMAN_IMAGES_EXPERIMENT_ID,
    REGISTRY,
    SCENARIOS,
    TECHNIQUES,
    WRAPPERS,
    ExperimentRegistry,
    KeywordArguments,
)

NEW_BASELINES = (GARDENER_PPO_EXPERIMENT_ID, PACMAN_DQN_EXPERIMENT_ID, PACMAN_IMAGES_EXPERIMENT_ID)


@pytest.fixture
def backend():
    sb3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield sb3
    torch.set_num_threads(before)


@pytest.mark.parametrize("identifier", NEW_BASELINES)
def test_learning_parameters_match_defaults_and_image_example(identifier, backend):
    resolved = REGISTRY.resolve(identifier)
    assert resolved.technique.id == "technique/unconstrained-v0" and resolved.specification.often is None
    gardener = identifier == GARDENER_PPO_EXPERIMENT_ID
    images = identifier == PACMAN_IMAGES_EXPERIMENT_ID
    assert resolved.specification.run.training_steps == (100_000 if gardener else 5_000_000)
    assert resolved.specification.run.max_episode_steps == (1_000 if gardener else 300)
    constructor = backend.PPO if gardener else backend.DQN
    defaults = inspect.signature(constructor).parameters
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    exceptions = {"device", "verbose"} | ({"buffer_size", "policy_kwargs"} if images else set())
    for key, value in kwargs.items():
        if key not in exceptions:
            assert value == defaults[key].default, key
    assert kwargs["gamma"] == 0.99
    if images:
        assert kwargs["buffer_size"] == 100_000
        assert kwargs["policy_kwargs"] == {"features_extractor_kwargs": {"features_dim": 128}, "net_arch": [64, 64]}
        assert resolved.algorithm.policy == "CnnPolicy"
    assert execution.resolved_configuration(
        execution.resolved_from_configuration(execution.resolved_configuration(resolved))
    ) == execution.resolved_configuration(resolved)


def test_images_match_example_pixels_and_rewards_without_bolts():
    pytest.importorskip("pygame")
    Image = pytest.importorskip("PIL.Image")
    from npc_gym.envs import PacmanEnv

    resolved = REGISTRY.resolve(PACMAN_IMAGES_EXPERIMENT_ID)
    with (
        resolved.make_env(training=True) as train,
        resolved.make_env(training=False) as evaluation,
        PacmanEnv(features="image-full") as raw,
    ):
        frame, raw_info = raw.reset(seed=7)
        obs, info = train.reset(seed=7)
        other, other_info = evaluation.reset(seed=7)
        expected = np.asarray(Image.fromarray(frame).convert("L").resize((210, 80), Image.Resampling.BOX))
        assert obs.shape == (80, 210, 2) and obs.dtype == np.uint8
        assert train.observation_space.contains(obs)
        np.testing.assert_array_equal(obs[..., 0], expected)
        np.testing.assert_array_equal(obs[..., 1], expected)
        np.testing.assert_array_equal(other, obs)
        assert info == other_info == raw_info and "restraining_bolts" not in info
        for action in (0, 1, 3):
            current = train.step(action)
            unscaled = evaluation.step(action)
            original = raw.step(action)
            np.testing.assert_array_equal(current[0][..., 0], expected)
            expected = np.asarray(Image.fromarray(original[0]).convert("L").resize((210, 80), Image.Resampling.BOX))
            np.testing.assert_array_equal(current[0][..., 1], expected)
            np.testing.assert_array_equal(current[0], unscaled[0])
            assert current[1] == original[1] / 100 and unscaled[1] == original[1]
            assert current[2:] == unscaled[2:] == original[2:]


@pytest.mark.parametrize("module", ["pygame", "PIL"])
def test_missing_image_dependency_fails_before_writing(tmp_path, monkeypatch, module):
    (plan,) = execution.plan_training_batch(tmp_path, experiment_ids=[PACMAN_IMAGES_EXPERIMENT_ID], seeds=[0])
    original = execution.importlib.util.find_spec
    monkeypatch.setattr(execution.importlib.util, "find_spec", lambda name: None if name == module else original(name))
    with pytest.raises(execution.PreflightError, match="sb3,render"):
        execution.preflight_training([plan])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("identifier", NEW_BASELINES)
def test_small_train_save_reload_and_evaluation(tmp_path, backend, identifier):
    if identifier == PACMAN_IMAGES_EXPERIMENT_ID:
        pytest.importorskip("pygame")
    resolved = REGISTRY.resolve(identifier)
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    kwargs["verbose"] = 0
    if identifier == GARDENER_PPO_EXPERIMENT_ID:
        kwargs.update(n_steps=8, batch_size=4, n_epochs=1)
    else:
        kwargs.update(buffer_size=32, learning_starts=0, batch_size=4)
    algorithm = replace(resolved.algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs))
    specification = replace(
        resolved.specification,
        run=replace(
            resolved.specification.run,
            training_steps=16,
            max_episode_steps=4,
            intermediate_evaluation_frequency=8,
            intermediate_evaluation_episodes=1,
            final_evaluation_episodes=2,
        ),
    )
    registry = ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=tuple(algorithm if item.id == algorithm.id else item for item in ALGORITHMS),
        techniques=TECHNIQUES,
        experiments=(specification,),
    )
    (plan,) = execution.plan_training_batch(tmp_path, registry=registry, seeds=[7])
    (result,) = execution.execute_training_batch([plan])
    assert result.actual_timesteps == 16
    metadata = json.loads(plan.run_path.read_text())
    assert metadata["status"] == "complete"
    (again,) = execution.execute_evaluation_batch(
        execution.plan_evaluation_batch(tmp_path, registry=registry, name="repeat", seeds=[7])
    )
    assert again.summary == result.final_evaluation
    model = (backend.PPO if identifier == GARDENER_PPO_EXPERIMENT_ID else backend.DQN).load(plan.model_path)
    assert model._n_updates > 0
    if identifier == PACMAN_IMAGES_EXPERIMENT_ID:
        assert model.observation_space.shape == (2, 80, 210)
        assert model.policy.q_net.features_extractor.features_dim == 128
        assert model.policy.net_arch == [64, 64]
