"""Keep examples short while checking baseline settings and real train/save/load/evaluate flows."""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
from contextlib import closing
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from experiments.specifications import get_experiment

EXAMPLES = {
    "taxi": "taxi-tabular-unconstrained-v1",
    "gardener": "gardener-dqn-unconstrained-v0",
    "pacman": "pacman-ppo-unconstrained-v2",
    "merchant": "merchant-tabular-unconstrained-v2",
}
SMOKE = {
    "NPC_GYM_TRAINING_STEPS": "16",
    "NPC_GYM_EVALUATION_EPISODES": "2",
    "NPC_GYM_MAX_EPISODE_STEPS": "40",
    "NPC_GYM_SEED": "3",
}


def load_example(name, monkeypatch, *, smoke=False):
    for key in (*SMOKE, "NPC_GYM_ALGORITHM", "NPC_GYM_MODEL_PATH"):
        monkeypatch.delenv(key, raising=False)
    if smoke:
        for key, value in SMOKE.items():
            monkeypatch.setenv(key, value)
        if name == "pacman":
            monkeypatch.setenv("NPC_GYM_TRAINING_STEPS", "128")
        if name in {"merchant", "taxi"}:
            monkeypatch.setenv("NPC_GYM_TRAINING_STEPS", "500")
    path = Path(__file__).resolve().parents[2] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", EXAMPLES)
def test_task_settings_and_original_algorithm_match_baseline(name, monkeypatch):
    example = load_example(name, monkeypatch)
    baseline = get_experiment(EXAMPLES[name])
    assert len(Path(example.__file__).read_text().splitlines()) < 100
    assert example.TRAINING_STEPS == baseline.specification.run.training_steps
    assert example.EVALUATION_EPISODES == 100
    assert example.MAX_EPISODE_STEPS == baseline.specification.run.max_episode_steps
    for training in (False, True):
        with (
            closing(example.make_env(**({"training": training} if name == "gardener" else {}))) as actual,
            closing(baseline.make_env(training=training and name != "pacman")) as expected,
        ):
            assert actual.observation_space == expected.observation_space
            assert actual.action_space == expected.action_space
            np.testing.assert_equal(actual.reset(seed=19)[0], expected.reset(seed=19)[0])
            for index in range(40):
                action = index % int(actual.action_space.n)
                left, right = actual.step(action), expected.step(action)
                np.testing.assert_equal(left[0], right[0])
                assert left[1:4] == right[1:4]
                assert left[4]["labels"] == right[4]["labels"]
                assert left[4]["episode_metrics"] == right[4]["episode_metrics"]
                if left[2] or left[3]:
                    actual.reset(seed=index)
                    expected.reset(seed=index)

    class Configured(Exception):
        pass

    if name in {"pacman", "gardener"}:
        assert example.ALGORITHM == "dqn"
    if name == "pacman":
        monkeypatch.setattr(example, "ALGORITHM", "ppo")
    model_name = baseline.algorithm.implementation.rsplit(".", 1)[-1]
    constructor = getattr(example, model_name)

    def inspect_constructor(*args, **kwargs):
        bound = inspect.signature(constructor).bind(*args, **kwargs)
        bound.apply_defaults()
        for key, value in baseline.algorithm.constructor_kwargs.to_dict().items():
            assert bound.arguments[key] == value, key
        assert bound.arguments["seed"] == 0
        if baseline.algorithm.policy:
            assert bound.arguments["policy"] == baseline.algorithm.policy
        raise Configured

    monkeypatch.setattr(example, model_name, inspect_constructor)
    with pytest.raises(Configured):
        example.main()


@pytest.mark.parametrize("name", EXAMPLES)
@pytest.mark.parametrize("fail_evaluation", [False, True])
def test_smoke_updates_reloads_evaluates_and_cleans_up(name, fail_evaluation, monkeypatch, tmp_path, capsys):
    example = load_example(name, monkeypatch, smoke=True)
    if name in {"pacman", "gardener"}:
        monkeypatch.setenv("NPC_GYM_MODEL_PATH", str(tmp_path / "model.zip"))
    baseline = get_experiment(EXAMPLES[name])
    constructor = (
        example.DQN if name == "pacman" else getattr(example, baseline.algorithm.implementation.rsplit(".", 1)[-1])
    )
    original_learn = constructor.learn
    original_save = constructor.save
    original_evaluate = example.evaluate
    original_make = example.make_env
    captured = {}
    environments = []

    def make_env(**kwargs):
        env = original_make(**kwargs)
        closed = []
        environments.append((env, kwargs, closed))
        close = env.close

        def close_and_record():
            closed.append(True)
            close()

        env.close = close_and_record
        return env

    def learn(model, *args, **kwargs):
        captured["trained"] = model
        if hasattr(model, "policy"):
            captured["initial"] = {key: value.clone() for key, value in model.policy.state_dict().items()}
        assert kwargs.get("log_interval") == baseline.specification.run.learning_kwargs.to_dict().get("log_interval")
        return original_learn(model, *args, **kwargs)

    def save(model, path):
        original_save(model, path)
        captured["path"] = Path(str(path) + ("" if str(path).endswith(".zip") else ".zip"))
        assert captured["path"].is_file()

    def evaluate(env, policy, **kwargs):
        loaded = policy.model if hasattr(policy, "model") else policy
        trained = captured["trained"]
        assert env is environments[1][0]
        assert loaded is not trained
        assert loaded.num_timesteps == trained.num_timesteps == example.TRAINING_STEPS
        if hasattr(loaded, "policy"):
            assert loaded._n_updates > 0
            assert any(
                not np.array_equal(value.numpy(), captured["initial"][key].numpy())
                for key, value in loaded.policy.state_dict().items()
            )
            for key, value in loaded.policy.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), trained.policy.state_dict()[key].numpy())
        else:
            assert any(np.any(row) for row in loaded.q_table.values())
            assert loaded.q_table.keys() == trained.q_table.keys()
            for key, value in loaded.q_table.items():
                np.testing.assert_array_equal(value, trained.q_table[key])
        assert kwargs["seed"] == 10_003
        assert kwargs["episodes"] == 2
        if fail_evaluation:
            raise RuntimeError("evaluation failed")

        def checked_policy(observation, info):
            action = policy(observation, info)
            if name == "merchant":
                assert info["action_mask"][action]
            return action

        summary = original_evaluate(env, checked_policy, **kwargs)
        assert summary.mean_metrics
        assert summary.mean_monitor_counts
        return summary

    if name in {"taxi", "merchant"}:
        monkeypatch.setattr(example, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path))
    monkeypatch.setattr(example, "make_env", make_env)
    monkeypatch.setattr(constructor, "learn", learn)
    monkeypatch.setattr(constructor, "save", save)
    monkeypatch.setattr(example, "evaluate", evaluate)
    if fail_evaluation:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            example.main()
    else:
        example.main()
        assert "mean task return:" in capsys.readouterr().out
    assert len(environments) == 2
    assert environments[0][1] == ({"training": True} if name in {"gardener", "pacman"} else {})
    assert environments[1][1] == ({"training": False} if name == "gardener" else {})
    assert all(closed for _, _, closed in environments)
    if name in {"pacman", "gardener"}:
        assert list(tmp_path.iterdir()) == [captured["path"]]
    else:
        assert not captured["path"].exists()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", EXAMPLES)
def test_environment_overrides_run_without_cli_arguments(name, tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples" / f"{name}.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env={**os.environ, **SMOKE},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "mean task return:" in result.stdout
    assert "Mean task outcomes:" in result.stdout
    assert "Mean monitor measurements:" in result.stdout
    assert {path.name for path in tmp_path.iterdir()} == (
        {f"{name}-dqn.zip"} if name in {"pacman", "gardener"} else set()
    )


def test_gardener_evaluation_omits_training_penalty(monkeypatch):
    example = load_example("gardener", monkeypatch)
    with closing(example.make_env(training=True)) as train, closing(example.make_env(training=False)) as evaluation:
        _, info = train.reset(seed=19)
        evaluation.reset(seed=19)
        illegal = int(np.flatnonzero(np.asarray(info["action_mask"]) == 0)[0])
        trained, evaluated = train.step(illegal), evaluation.step(illegal)
        np.testing.assert_array_equal(trained[0], evaluated[0])
        assert trained[1] == pytest.approx(evaluated[1] - 1)
        assert trained[4]["episode_metrics"] == evaluated[4]["episode_metrics"]


def test_pacman_training_scales_rewards_but_evaluation_reports_raw_rewards(monkeypatch):
    example = load_example("pacman", monkeypatch)
    with closing(example.make_env(training=True)) as train, closing(example.make_env()) as evaluation:
        _, previous = train.reset(seed=19)
        evaluation.reset(seed=19)
        for action in range(5):
            trained, evaluated = train.step(action), evaluation.step(action)
            np.testing.assert_array_equal(trained[0], evaluated[0])
            assert trained[1] == pytest.approx(evaluated[1] / 100)
            assert evaluated[1] == trained[4]["episode_metrics"]["score"] - previous["episode_metrics"]["score"]
            assert trained[4]["episode_metrics"] == evaluated[4]["episode_metrics"]
            previous = trained[4]
            if trained[2] or trained[3]:
                break
