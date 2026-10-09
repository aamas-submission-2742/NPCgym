from __future__ import annotations

import copy
import json
import logging
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile

import gymnasium as gym
import numpy as np
import pytest

from npc_gym.algorithms import TabularQLearning, TrainingStep
from npc_gym.evaluation import evaluate


class RecordingEnv(gym.Env[int, int]):
    metadata: ClassVar = {}

    def __init__(
        self,
        *,
        horizon: int = 3,
        truncate: bool = False,
        masked: bool = False,
        action_start: int = 0,
        n_actions: int = 2,
    ) -> None:
        self.action_space = gym.spaces.Discrete(n_actions, start=action_start)
        self.observation_space = gym.spaces.Discrete(horizon + 1)
        self.horizon = horizon
        self.truncate = truncate
        self.masked = masked
        self.actions: list[int] = []
        self.seeds: list[int | None] = []
        self.state = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.state = 0
        return self.state, self._info()

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.actions.append(action)
        self.state += 1
        done = self.state == self.horizon
        return (
            self.state,
            float(action - self.action_space.start + 1),
            done and not self.truncate,
            (done and self.truncate),
            self._info(),
        )

    def _info(self) -> dict[str, Any]:
        if not self.masked:
            return {}
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[self.state % self.action_space.n] = 1
        return {"action_mask": mask}


class BoxActionEnv(RecordingEnv):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("learning_rate", -0.1),
        ("learning_rate", 1.1),
        ("gamma", -0.1),
        ("gamma", 1.1),
        ("exploration_fraction", -0.1),
        ("exploration_initial_eps", 1.1),
        ("exploration_final_eps", -0.1),
    ],
)
def test_constructor_validates_probability_parameters(argument, value):
    with pytest.raises((TypeError, ValueError), match=argument):
        TabularQLearning(RecordingEnv(), **{argument: value})


def test_constructor_requires_an_ordinary_environment_with_discrete_actions():
    with pytest.raises(TypeError, match="ordinary Gymnasium Env"):
        TabularQLearning(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Discrete"):
        TabularQLearning(BoxActionEnv())  # type: ignore[arg-type]


def test_learning_uses_action_masks_from_public_info():
    env = RecordingEnv(masked=True)
    model = TabularQLearning(
        env,
        use_action_mask=True,
        exploration_initial_eps=1.0,
        exploration_final_eps=1.0,
        seed=4,
    )

    model.learn(6)

    assert env.actions == [0, 1, 0, 0, 1, 0]


def test_masked_learning_and_prediction_require_valid_explicit_masks():
    model = TabularQLearning(RecordingEnv(), use_action_mask=True)
    with pytest.raises(RuntimeError, match=r"info\['action_mask'\]"):
        model.learn(1)
    with pytest.raises(RuntimeError, match=r"info\['action_mask'\]"):
        model.predict(0)
    with pytest.raises(ValueError, match="shape"):
        model.predict(0, {"action_mask": np.array([1])})
    with pytest.raises(ValueError, match="exclude all"):
        model.predict(0, {"action_mask": np.array([0, 0])})


def test_truncation_bootstraps_from_the_terminal_observation():
    env = RecordingEnv(horizon=1, truncate=True)
    model = TabularQLearning(
        env,
        learning_rate=1.0,
        gamma=1.0,
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        seed=1,
    )
    model._q_values_for_update(0)[:] = [1.0, 0.0]
    model._q_values_for_update(1)[:] = [5.0, 0.0]

    model.learn(1)

    assert model.q_values(0).tolist() == [6.0, 0.0]


def test_termination_does_not_bootstrap():
    env = RecordingEnv(horizon=1)
    model = TabularQLearning(
        env,
        learning_rate=1.0,
        gamma=1.0,
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        seed=1,
    )
    model._q_values_for_update(0)[:] = [1.0, 0.0]
    model._q_values_for_update(1)[:] = [5.0, 0.0]

    model.learn(1)

    assert model.q_values(0).tolist() == [1.0, 0.0]


def test_truncation_bootstrap_uses_the_terminal_action_mask():
    env = RecordingEnv(horizon=1, truncate=True, masked=True)
    model = TabularQLearning(
        env,
        learning_rate=1.0,
        gamma=1.0,
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        use_action_mask=True,
        seed=1,
    )
    model._q_values_for_update(0)[:] = [1.0, 0.0]
    model._q_values_for_update(1)[:] = [100.0, 5.0]

    model.learn(1)

    assert model.q_values(0).tolist() == [6.0, 0.0]


def test_deterministic_prediction_is_stable_read_only_and_rng_neutral():
    model = TabularQLearning(RecordingEnv(), seed=12)
    model._q_values_for_update(0)[:] = [4.0, 4.0]
    training_state = copy.deepcopy(model._training_rng.bit_generator.state)
    prediction_state = copy.deepcopy(model._prediction_rng.bit_generator.state)

    assert [model.predict(0) for _ in range(5)] == [0] * 5
    assert model.q_values(99).tolist() == [0.0, 0.0]
    assert model.explored_states == 1
    assert model._training_rng.bit_generator.state == training_state
    assert model._prediction_rng.bit_generator.state == prediction_state


def test_q_table_property_is_detached_from_the_learner():
    model = TabularQLearning(RecordingEnv())
    model._q_values_for_update(0)[:] = [2.0, 3.0]

    detached = model.q_table
    next(iter(detached.values()))[:] = 100.0

    assert model.q_values(0).tolist() == [2.0, 3.0]
    with pytest.raises(TypeError):
        detached["new"] = np.zeros(2)  # type: ignore[index]


def test_offset_discrete_actions_are_returned_as_environment_values():
    env = RecordingEnv(action_start=4)
    model = TabularQLearning(env, exploration_initial_eps=0.0, exploration_final_eps=0.0, seed=1)
    model._q_values_for_update(0)[:] = [0.0, 3.0]

    assert model.predict(0) == 5
    model.learn(1)
    assert env.actions == [5]


def test_typed_callback_observes_steps_and_can_stop_learning():
    seen: list[TrainingStep] = []

    def callback(model: TabularQLearning[Any], step: TrainingStep) -> bool:
        assert model.num_timesteps == step.timesteps
        seen.append(step)
        return step.timesteps < 2

    model = TabularQLearning(RecordingEnv(horizon=4), seed=2)

    returned = model.learn(10, callback=callback)

    assert returned is model
    assert model.num_timesteps == 2
    assert [step.timesteps for step in seen] == [1, 2]
    assert all(isinstance(step.action, int) for step in seen)


def test_callback_return_type_is_validated():
    model = TabularQLearning(RecordingEnv(), seed=2)
    with pytest.raises(TypeError, match="bool or None"):
        model.learn(1, callback=lambda model, step: 1)  # type: ignore[arg-type,return-value]


def test_progress_uses_standard_logging(caplog):
    caplog.set_level(logging.INFO, logger="npc_gym.algorithms.tabular_q_learning")
    model = TabularQLearning(RecordingEnv(horizon=1), log_interval=1, seed=2)

    model.learn(1)

    assert "tabular q-learning: timesteps=1 episodes=1" in caplog.text
    assert "mean_abs_td_error=" in caplog.text


def test_save_load_round_trip_uses_safe_archive_and_restores_state(tmp_path):
    path = tmp_path / "agent.zip"
    observation = {"position": np.array([1, 2], dtype=np.int16), "flags": (True, None)}
    source = TabularQLearning(
        RecordingEnv(),
        learning_rate=0.4,
        gamma=0.8,
        exploration_fraction=0.6,
        exploration_initial_eps=0.7,
        exploration_final_eps=0.2,
        seed=33,
        use_action_mask=True,
        log_interval=None,
    )
    source._q_values_for_update(observation)[:] = [1.5, 2.5]
    source.num_timesteps = 12
    source.num_episodes = 4
    source.epsilon = 0.3
    source.predict(0, {"action_mask": np.array([1, 1])}, deterministic=False)
    source.save(path)

    loaded = TabularQLearning.load(path, env=RecordingEnv())

    assert is_zipfile(path)
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == {"metadata.json", "state_keys.json", "q_values.npy"}
        assert json.loads(archive.read("metadata.json"))["schema_version"] == 1
        assert json.loads(archive.read("state_keys.json"))[0][0] == "mapping"
        values = np.load(BytesIO(archive.read("q_values.npy")), allow_pickle=False)
        assert values.dtype == np.float64
    assert loaded.q_values(observation).tolist() == [1.5, 2.5]
    assert (loaded.num_timesteps, loaded.num_episodes, loaded.epsilon) == (12, 4, 0.3)
    assert loaded.use_action_mask
    assert loaded.log_interval is None
    assert loaded._training_rng.bit_generator.state == source._training_rng.bit_generator.state
    assert loaded._prediction_rng.bit_generator.state == source._prediction_rng.bit_generator.state


def test_prediction_and_q_inspection_do_not_enlarge_a_saved_table(tmp_path):
    path = tmp_path / "agent.zip"
    model = TabularQLearning(RecordingEnv(), seed=3)
    model._q_values_for_update(0)[:] = [1.0, 2.0]

    model.q_values(999)
    model.predict(999)
    model.predict(999, deterministic=False)
    model.save(path)

    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        states = json.loads(archive.read("state_keys.json"))
    assert metadata["state_count"] == 1
    assert len(states) == 1


def test_save_cleanly_overwrites_an_existing_model(tmp_path):
    path = tmp_path / "agent.zip"
    model = TabularQLearning(RecordingEnv())
    model._q_values_for_update(0)[:] = [1.0, 2.0]
    model.save(path)
    model._q_values_for_update(0)[:] = [3.0, 4.0]

    model.save(path)

    loaded = TabularQLearning.load(path, env=RecordingEnv())
    assert loaded.q_values(0).tolist() == [3.0, 4.0]
    with ZipFile(path) as archive:
        assert len(archive.namelist()) == 3


def test_load_rejects_pickle_schema_and_action_space_mismatches(tmp_path):
    pickle_path = tmp_path / "legacy.model"
    pickle_path.write_bytes(pickle.dumps({"q_table": {}}))
    with pytest.raises(ValueError, match="pickle models are unsupported"):
        TabularQLearning.load(pickle_path, env=RecordingEnv())

    source_path = tmp_path / "source.zip"
    TabularQLearning(RecordingEnv()).save(source_path)
    schema_path = _rewrite_archive(source_path, tmp_path / "schema.zip", schema_version=99)
    with pytest.raises(ValueError, match="schema version 99"):
        TabularQLearning.load(schema_path, env=RecordingEnv())
    with pytest.raises(ValueError, match="does not match"):
        TabularQLearning.load(source_path, env=RecordingEnv(n_actions=3))
    with pytest.raises(ValueError, match="does not match"):
        TabularQLearning.load(source_path, env=RecordingEnv(action_start=1))


def test_load_never_enables_object_array_loading(tmp_path):
    source_path = tmp_path / "source.zip"
    TabularQLearning(RecordingEnv()).save(source_path)
    with ZipFile(source_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    values = BytesIO()
    np.save(values, np.array([[object()]], dtype=object), allow_pickle=True)
    members["q_values.npy"] = values.getvalue()
    unsafe_path = tmp_path / "unsafe.zip"
    with ZipFile(unsafe_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    with pytest.raises(ValueError, match="non-pickled NumPy array"):
        TabularQLearning.load(unsafe_path, env=RecordingEnv())


def test_intermediate_evaluations_cannot_change_training_or_training_rng(tmp_path):
    baseline = _train_with_intermediate_evaluations(tmp_path, None)
    frequent = _train_with_intermediate_evaluations(tmp_path, 2)
    sparse = _train_with_intermediate_evaluations(tmp_path, 5)

    for candidate in (frequent, sparse):
        assert candidate[0] == baseline[0]
        assert candidate[1].keys() == baseline[1].keys()
        for key in baseline[1]:
            assert np.array_equal(candidate[1][key], baseline[1][key])
        assert candidate[2] == baseline[2]


def _train_with_intermediate_evaluations(tmp_path: Path, frequency: int | None):
    train_env = RecordingEnv(horizon=4)
    model = TabularQLearning(
        train_env,
        exploration_initial_eps=0.65,
        exploration_final_eps=0.65,
        seed=91,
    )
    callback = None
    if frequency is not None:
        eval_env = RecordingEnv(horizon=3)

        def intermediate(current: TabularQLearning[Any], step: TrainingStep) -> None:
            if step.timesteps % frequency == 0:
                evaluate(eval_env, current, episodes=1 + frequency % 2, seed=700 + frequency)
                for observation in range(frequency):
                    current.predict(
                        100 + observation,
                        {"action_mask": np.array([1, 1], dtype=np.int8)},
                        deterministic=False,
                    )

        callback = intermediate

    model.learn(40, callback=callback)
    path = tmp_path / f"model-{frequency}.zip"
    model.save(path)
    with ZipFile(path) as archive:
        training_rng_state = json.loads(archive.read("metadata.json"))["training"]["training_rng_state"]
    return list(train_env.actions), model.q_table, training_rng_state


def _rewrite_archive(source: Path, target: Path, *, schema_version: int) -> Path:
    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(members["metadata.json"])
    metadata["schema_version"] = schema_version
    members["metadata.json"] = json.dumps(metadata).encode()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return target


def test_disabled_logging_retains_no_progress_history():
    model = TabularQLearning(RecordingEnv(horizon=2), log_interval=None, seed=5)

    model.learn(20)

    assert (model._logged_returns, model._logged_lengths, model._logged_td_errors) == ([], [], [])


def test_a_loaded_model_does_not_reseed_the_environment_it_resumes_on(tmp_path):
    path = tmp_path / "agent.zip"
    source = TabularQLearning(RecordingEnv(horizon=2), seed=7)
    source.learn(4)
    source.save(path)

    resumed_env = RecordingEnv(horizon=2)
    loaded = TabularQLearning.load(path, env=resumed_env)
    loaded.learn(8)

    assert resumed_env.seeds == [None, None, None]
