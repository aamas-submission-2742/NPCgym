"""Independent tabular Q-learning for ordinary Gymnasium environments."""

from __future__ import annotations

import base64
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Protocol, Self, TypeVar
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

_FORMAT = "npc_gym.tabular_q_learning"
_SCHEMA_VERSION = 1
_ARCHIVE_MEMBERS = frozenset({"metadata.json", "state_keys.json", "q_values.npy"})
_LOGGER = logging.getLogger(__name__)

ObservationT = TypeVar("ObservationT")
StatePayload = list[Any]


@dataclass(frozen=True, slots=True)
class TrainingStep:
    """Public progress snapshot passed to a training callback after one Q update."""

    timesteps: int
    episodes: int
    epsilon: float
    action: int
    reward: float
    terminated: bool
    truncated: bool


class TrainingCallback(Protocol):
    """A callback invoked after each completed environment transition."""

    def __call__(self, learner: TabularQLearning[Any], step: TrainingStep, /) -> bool | None:
        """Return ``False`` to stop learning after the reported transition."""
        ...


class TabularQLearning(Generic[ObservationT]):
    """Tabular Q-learning with isolated prediction randomness and safe persistence.

    The environment must have a zero-based or offset :class:`~gymnasium.spaces.Discrete`
    action space and hashable or structurally encodable observations. When
    ``use_action_mask`` is enabled, every reset and step ``info`` used to choose
    or bootstrap an action must contain ``info["action_mask"]``.

    ``learn(total_timesteps)`` treats its argument as a cumulative target. The
    environment is reset at the start of each call; only the first call on a
    learner that has not trained yet applies the constructor's seed, so a loaded
    model never reseeds the environment it resumes on and its caller owns that
    environment's random stream. Evaluation should use a separate environment.
    """

    def __init__(
        self,
        env: gym.Env[ObservationT, int],
        *,
        learning_rate: float = 0.2,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        seed: int | None = None,
        use_action_mask: bool = False,
        log_interval: int | None = 10_000,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(env, gym.Env):
            raise TypeError("env must be an ordinary Gymnasium Env")
        if not isinstance(env.action_space, Discrete):
            raise TypeError("Action space must be Discrete")
        if not isinstance(use_action_mask, bool):
            raise TypeError("use_action_mask must be a bool")
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger")

        self.learning_rate = _probability("learning_rate", learning_rate)
        self.gamma = _probability("gamma", gamma)
        self.exploration_fraction = _probability("exploration_fraction", exploration_fraction)
        self.exploration_initial_eps = _probability("exploration_initial_eps", exploration_initial_eps)
        self.exploration_final_eps = _probability("exploration_final_eps", exploration_final_eps)
        self.seed = _seed(seed)
        self.log_interval = _log_interval(log_interval)
        self.env = env
        self.use_action_mask = use_action_mask
        self.n_actions = int(env.action_space.n)
        self.action_start = int(env.action_space.start)
        self.num_timesteps = 0
        self.num_episodes = 0
        self.epsilon = self.exploration_initial_eps

        streams = np.random.SeedSequence(self.seed).spawn(2)
        self._training_rng = np.random.default_rng(streams[0])
        self._prediction_rng = np.random.default_rng(streams[1])
        self._q: dict[str, NDArray[np.float64]] = {}
        self._environment_seeded = False
        self._logger = logger if logger is not None else _LOGGER
        self._next_log_step = self.log_interval if self.log_interval is not None else 0
        self._logged_returns: list[float] = []
        self._logged_lengths: list[int] = []
        self._logged_td_errors: list[float] = []

    @property
    def explored_states(self) -> int:
        """Number of observations currently stored in the Q-table."""
        return len(self._q)

    @property
    def q_table(self) -> Mapping[str, NDArray[np.float64]]:
        """Return a detached, read-only mapping of tagged state keys to Q rows."""
        return MappingProxyType({key: values.copy() for key, values in self._q.items()})

    def q_values(self, observation: ObservationT) -> NDArray[np.float64]:
        """Return a copy of one Q row without adding an unseen observation."""
        values = self._q.get(_state_key(observation))
        return np.zeros(self.n_actions, dtype=np.float64) if values is None else values.copy()

    def __call__(self, observation: ObservationT, info: Mapping[str, Any], /) -> int:
        """Choose a deterministic action for environment-neutral evaluation."""
        return self.predict(observation, info, deterministic=True)

    def predict(
        self,
        observation: ObservationT,
        info: Mapping[str, Any] | None = None,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose one action without changing the Q-table.

        Deterministic prediction uses the first maximum and consumes no random
        state. Stochastic prediction uses a generator independent from training.
        """
        allowed = self._allowed_actions(info, phase="prediction")
        values = self.q_values(observation)
        if deterministic:
            return self._action_value(_first_maximum(values, allowed))
        return self._action_value(self._epsilon_greedy(values, allowed, self._prediction_rng, self.epsilon))

    def learn(
        self,
        total_timesteps: int,
        *,
        callback: TrainingCallback | None = None,
    ) -> Self:
        """Train until the cumulative transition count reaches ``total_timesteps``."""
        target = _timestep_target(total_timesteps)
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        if target <= self.num_timesteps:
            return self

        reset_seed = self.seed if not self._environment_seeded else None
        observation, raw_info = self.env.reset(seed=reset_seed)
        self._environment_seeded = True
        info = _info(raw_info, "reset")
        episode_return = 0.0
        episode_length = 0

        while self.num_timesteps < target:
            self.epsilon = self._current_epsilon(target)
            allowed = self._allowed_actions(info, phase="action selection")
            values = self._q_values_for_update(observation)
            action_index = self._epsilon_greedy(values, allowed, self._training_rng, self.epsilon)
            action = self._action_value(action_index)
            next_observation, raw_reward, terminated, truncated, raw_next_info = self.env.step(action)
            next_info = _info(raw_next_info, "step")
            reward = _reward(raw_reward)
            terminated = bool(terminated)
            truncated = bool(truncated)

            next_value = 0.0
            if not terminated:
                next_allowed = self._allowed_actions(next_info, phase="Q-value bootstrapping")
                next_values = self._q_values_for_update(next_observation)
                next_value = float(np.max(next_values[next_allowed]))
            temporal_difference = reward + self.gamma * next_value - values[action_index]
            values[action_index] += self.learning_rate * temporal_difference

            self.num_timesteps += 1
            episode_return += reward
            episode_length += 1
            if terminated or truncated:
                self.num_episodes += 1
                self._record_episode(episode_return, episode_length)
            self._log_progress(float(temporal_difference))

            if callback is not None:
                decision = callback(
                    self,
                    TrainingStep(
                        timesteps=self.num_timesteps,
                        episodes=self.num_episodes,
                        epsilon=self.epsilon,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                    ),
                )
                if decision is not None and not isinstance(decision, bool):
                    raise TypeError("callback must return bool or None")
                if decision is False:
                    break

            if terminated or truncated:
                observation, raw_info = self.env.reset()
                info = _info(raw_info, "reset")
                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation
                info = next_info
        return self

    def save(self, path: str | PathLike[str]) -> None:
        """Write a versioned ZIP model without pickle or object arrays."""
        ordered = sorted(self._q.items())
        states = [json.loads(key) for key, _values in ordered]
        q_values = (
            np.stack([values for _key, values in ordered]).astype(np.float64, copy=False)
            if ordered
            else np.empty((0, self.n_actions), dtype=np.float64)
        )
        if not np.isfinite(q_values).all():
            raise ValueError("Q-values must be finite before they can be saved")
        metadata = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "action_space": {"type": "Discrete", "n": self.n_actions, "start": self.action_start},
            "algorithm": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "exploration_fraction": self.exploration_fraction,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_final_eps": self.exploration_final_eps,
                "seed": self.seed,
                "use_action_mask": self.use_action_mask,
                "log_interval": self.log_interval,
            },
            "training": {
                "epsilon": self.epsilon,
                "num_timesteps": self.num_timesteps,
                "num_episodes": self.num_episodes,
                "training_rng_state": self._training_rng.bit_generator.state,
                "prediction_rng_state": self._prediction_rng.bit_generator.state,
            },
            "state_count": len(states),
        }
        values_buffer = BytesIO()
        np.save(values_buffer, q_values, allow_pickle=False)
        with ZipFile(Path(path), "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", _json_bytes(metadata))
            archive.writestr("state_keys.json", _json_bytes(states))
            archive.writestr("q_values.npy", values_buffer.getvalue())

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        env: gym.Env[ObservationT, int],
        logger: logging.Logger | None = None,
    ) -> TabularQLearning[ObservationT]:
        """Load the safe ZIP format and bind it to a compatible environment."""
        metadata, states, q_values = _read_archive(path)
        action_space = _mapping(metadata.get("action_space"), "metadata.action_space")
        if not isinstance(env, gym.Env) or not isinstance(env.action_space, Discrete):
            raise TypeError("env must be an ordinary Gymnasium Env with a Discrete action space")
        expected_action_space = {"type": "Discrete", "n": int(env.action_space.n), "start": int(env.action_space.start)}
        if action_space != expected_action_space:
            raise ValueError(
                f"Saved action space {dict(action_space)!r} does not match environment action space "
                f"{expected_action_space!r}"
            )

        algorithm = _mapping(metadata.get("algorithm"), "metadata.algorithm")
        training = _mapping(metadata.get("training"), "metadata.training")
        model = cls(
            env,
            learning_rate=_real(algorithm.get("learning_rate"), "metadata.algorithm.learning_rate"),
            gamma=_real(algorithm.get("gamma"), "metadata.algorithm.gamma"),
            exploration_fraction=_real(
                algorithm.get("exploration_fraction"), "metadata.algorithm.exploration_fraction"
            ),
            exploration_initial_eps=_real(
                algorithm.get("exploration_initial_eps"), "metadata.algorithm.exploration_initial_eps"
            ),
            exploration_final_eps=_real(
                algorithm.get("exploration_final_eps"), "metadata.algorithm.exploration_final_eps"
            ),
            seed=_optional_integer(algorithm.get("seed"), "metadata.algorithm.seed"),
            use_action_mask=_boolean(algorithm.get("use_action_mask"), "metadata.algorithm.use_action_mask"),
            log_interval=_optional_integer(algorithm.get("log_interval"), "metadata.algorithm.log_interval"),
            logger=logger,
        )
        model.epsilon = _probability("metadata.training.epsilon", training.get("epsilon"))
        model.num_timesteps = _nonnegative_integer(training.get("num_timesteps"), "metadata.training.num_timesteps")
        model.num_episodes = _nonnegative_integer(training.get("num_episodes"), "metadata.training.num_episodes")
        _restore_rng(model._training_rng, training.get("training_rng_state"), "training_rng_state")
        _restore_rng(model._prediction_rng, training.get("prediction_rng_state"), "prediction_rng_state")
        model._q = {key: values.copy() for key, values in zip(states, q_values, strict=True)}
        model._environment_seeded = True
        if model.log_interval is not None:
            model._next_log_step = (model.num_timesteps // model.log_interval + 1) * model.log_interval
        return model

    def _allowed_actions(self, info: Mapping[str, Any] | None, *, phase: str) -> NDArray[np.intp]:
        if not self.use_action_mask:
            return np.arange(self.n_actions, dtype=np.intp)
        if info is None or "action_mask" not in info:
            raise RuntimeError(f"Masked {phase} requires info['action_mask']")
        mask = np.asarray(info["action_mask"])
        if mask.shape != (self.n_actions,):
            raise ValueError(f"action_mask must have shape ({self.n_actions},), got {mask.shape}")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("action_mask entries must be 0 or 1")
        allowed = np.flatnonzero(mask).astype(np.intp, copy=False)
        if allowed.size == 0:
            raise ValueError("action_mask cannot exclude all available actions")
        return allowed

    def _q_values_for_update(self, observation: ObservationT) -> NDArray[np.float64]:
        key = _state_key(observation)
        values = self._q.get(key)
        if values is None:
            values = np.zeros(self.n_actions, dtype=np.float64)
            self._q[key] = values
        return values

    def _epsilon_greedy(
        self,
        values: NDArray[np.float64],
        allowed: NDArray[np.intp],
        rng: np.random.Generator,
        epsilon: float,
    ) -> int:
        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.choice(allowed))
        best = allowed[values[allowed] == np.max(values[allowed])]
        return int(rng.choice(best))

    def _action_value(self, index: int) -> int:
        return index + self.action_start

    def _current_epsilon(self, total_timesteps: int) -> float:
        if self.exploration_fraction == 0:
            return self.exploration_final_eps
        decay_steps = self.exploration_fraction * total_timesteps
        fraction = float(np.clip(self.num_timesteps / decay_steps, 0.0, 1.0))
        return self.exploration_initial_eps + fraction * (self.exploration_final_eps - self.exploration_initial_eps)

    def _record_episode(self, episode_return: float, episode_length: int) -> None:
        if self.log_interval is None:
            return
        self._logged_returns.append(episode_return)
        self._logged_lengths.append(episode_length)

    def _log_progress(self, temporal_difference: float) -> None:
        if self.log_interval is None:
            return
        self._logged_td_errors.append(temporal_difference)
        if self.num_timesteps < self._next_log_step:
            return
        mean_return = float(np.mean(self._logged_returns)) if self._logged_returns else 0.0
        mean_length = float(np.mean(self._logged_lengths)) if self._logged_lengths else 0.0
        mean_absolute_td_error = float(np.mean(np.abs(self._logged_td_errors))) if self._logged_td_errors else 0.0
        self._logger.info(
            "tabular q-learning: timesteps=%d episodes=%d epsilon=%.6f states=%d "
            "mean_return=%.6f mean_length=%.6f mean_abs_td_error=%.6f",
            self.num_timesteps,
            self.num_episodes,
            self.epsilon,
            self.explored_states,
            mean_return,
            mean_length,
            mean_absolute_td_error,
        )
        self._logged_returns.clear()
        self._logged_lengths.clear()
        self._logged_td_errors.clear()
        self._next_log_step += self.log_interval


def _first_maximum(values: NDArray[np.float64], allowed: NDArray[np.intp]) -> int:
    allowed_values = values[allowed]
    return int(allowed[int(np.argmax(allowed_values))])


def _state_key(observation: Any) -> str:
    return json.dumps(_encode_state(observation), ensure_ascii=False, separators=(",", ":"))


def _encode_state(value: Any) -> StatePayload:
    if isinstance(value, np.generic):
        return _encode_state(value.item())
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, Integral):
        return ["int", str(int(value))]
    if isinstance(value, Real):
        return ["float", float(value).hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError("Observations cannot contain object or structured NumPy arrays")
        array = np.ascontiguousarray(value)
        return [
            "ndarray",
            array.dtype.str,
            list(array.shape),
            base64.b64encode(array.tobytes()).decode("ascii"),
        ]
    if isinstance(value, tuple):
        return ["tuple", [_encode_state(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_encode_state(item) for item in value]]
    if isinstance(value, Mapping):
        items = [[_encode_state(key), _encode_state(item)] for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], ensure_ascii=False, separators=(",", ":")))
        return ["mapping", items]
    raise TypeError(f"Observation values must be structurally encodable; got {type(value).__name__}")


def _read_archive(path: str | PathLike[str]) -> tuple[Mapping[str, Any], list[str], NDArray[np.float64]]:
    try:
        with ZipFile(Path(path), "r") as archive:
            members = archive.namelist()
            if len(members) != len(_ARCHIVE_MEMBERS) or set(members) != _ARCHIVE_MEMBERS:
                raise ValueError(f"Model archive must contain exactly {sorted(_ARCHIVE_MEMBERS)!r}")
            metadata_value = _read_json(archive.read("metadata.json"), "metadata.json")
            states_value = _read_json(archive.read("state_keys.json"), "state_keys.json")
            try:
                loaded_values = np.load(BytesIO(archive.read("q_values.npy")), allow_pickle=False)
            except (EOFError, OSError, TypeError, ValueError) as error:
                raise ValueError("q_values.npy must be a non-pickled NumPy array") from error
    except BadZipFile as error:
        raise ValueError("Model must be a TabularQLearning ZIP archive; pickle models are unsupported") from error

    metadata = _mapping(metadata_value, "metadata.json")
    if metadata.get("format") != _FORMAT:
        raise ValueError(f"Unsupported model format {metadata.get('format')!r}")
    schema_version = metadata.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported model schema version {schema_version!r}")
    if not isinstance(states_value, list):
        raise TypeError("state_keys.json must contain a list")
    states = [_validated_state_key(payload) for payload in states_value]
    if len(set(states)) != len(states):
        raise ValueError("state_keys.json contains duplicate states")
    state_count = _nonnegative_integer(metadata.get("state_count"), "metadata.state_count")
    if state_count != len(states):
        raise ValueError(f"metadata.state_count is {state_count}, but state_keys.json contains {len(states)} states")
    if (
        not isinstance(loaded_values, np.ndarray)
        or loaded_values.dtype != np.dtype(np.float64)
        or loaded_values.ndim != 2
    ):
        raise ValueError("q_values.npy must be a two-dimensional float64 array")
    q_values = np.asarray(loaded_values, dtype=np.float64)
    if q_values.shape[0] != len(states):
        raise ValueError("q_values.npy row count does not match state_keys.json")
    if not np.isfinite(q_values).all():
        raise ValueError("q_values.npy must contain only finite values")
    action_space = _mapping(metadata.get("action_space"), "metadata.action_space")
    if action_space.get("type") != "Discrete":
        raise ValueError(f"Unsupported saved action space type {action_space.get('type')!r}")
    n_actions = _positive_integer(action_space.get("n"), "metadata.action_space.n")
    _integer(action_space.get("start"), "metadata.action_space.start")
    if q_values.shape[1] != n_actions:
        raise ValueError("q_values.npy column count does not match metadata.action_space.n")
    return metadata, states, q_values


def _validated_state_key(payload: Any) -> str:
    _validate_state_payload(payload)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validate_state_payload(payload: Any) -> None:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], str):
        raise ValueError("Every state key must be a tagged JSON list")
    tag = payload[0]
    if tag == "none" and len(payload) == 1:
        return
    if tag == "bool" and len(payload) == 2 and isinstance(payload[1], bool):
        return
    if tag == "int" and len(payload) == 2 and isinstance(payload[1], str):
        try:
            if str(int(payload[1])) == payload[1]:
                return
        except ValueError:
            pass
    if tag == "float" and len(payload) == 2 and isinstance(payload[1], str):
        try:
            if float.fromhex(payload[1]).hex() == payload[1]:
                return
        except ValueError:
            pass
    if tag == "str" and len(payload) == 2 and isinstance(payload[1], str):
        return
    if tag == "bytes" and len(payload) == 2 and isinstance(payload[1], str):
        try:
            base64.b64decode(payload[1], validate=True)
            return
        except ValueError:
            pass
    if tag == "ndarray" and len(payload) == 4:
        _validate_array_payload(payload)
        return
    if tag in {"tuple", "list"} and len(payload) == 2 and isinstance(payload[1], list):
        for item in payload[1]:
            _validate_state_payload(item)
        return
    if tag == "mapping" and len(payload) == 2 and isinstance(payload[1], list):
        canonical_keys: list[str] = []
        for pair in payload[1]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("A tagged mapping state must contain key-value pairs")
            _validate_state_payload(pair[0])
            _validate_state_payload(pair[1])
            canonical_keys.append(json.dumps(pair[0], ensure_ascii=False, separators=(",", ":")))
        if canonical_keys != sorted(canonical_keys) or len(set(canonical_keys)) != len(canonical_keys):
            raise ValueError("Tagged mapping state keys must be unique and sorted")
        return
    raise ValueError(f"Invalid tagged state key {payload!r}")


def _validate_array_payload(payload: StatePayload) -> None:
    dtype_value, shape_value, data_value = payload[1:]
    if not isinstance(dtype_value, str) or not isinstance(shape_value, list) or not isinstance(data_value, str):
        raise TypeError("Invalid tagged ndarray state")
    try:
        dtype = np.dtype(dtype_value)
    except TypeError as error:
        raise ValueError(f"Invalid tagged ndarray dtype {dtype_value!r}") from error
    if dtype.str != dtype_value or dtype.hasobject or dtype.fields is not None:
        raise ValueError(f"Unsupported tagged ndarray dtype {dtype_value!r}")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape_value):
        raise ValueError("Tagged ndarray dimensions must be non-negative integers")
    try:
        data = base64.b64decode(data_value, validate=True)
    except ValueError as error:
        raise ValueError("Tagged ndarray data must be base64") from error
    expected_bytes = math.prod(shape_value) * dtype.itemsize
    if len(data) != expected_bytes:
        raise ValueError(f"Tagged ndarray contains {len(data)} bytes; expected {expected_bytes}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _read_json(data: bytes, member: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=lambda value: _raise_json_constant(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{member} must contain valid UTF-8 JSON") from error


def _raise_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is unsupported")


def _restore_rng(generator: np.random.Generator, state: Any, name: str) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"metadata.training.{name} must be a mapping")
    try:
        generator.bit_generator.state = dict(state)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata.training.{name} is invalid") from error


def _info(value: Any, phase: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Environment {phase} info must be a mapping, got {type(value).__name__}")
    return value


def _reward(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TypeError(f"Environment rewards must be finite real numbers; got {value!r}")
    return float(value)


def _probability(name: str, value: Any) -> float:
    number = _real(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return number


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be a finite real number, got {value!r}")
    return float(value)


def _seed(value: int | None) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, "seed")


def _log_interval(value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, "log_interval")


def _timestep_target(value: int) -> int:
    return _nonnegative_integer(value, "total_timesteps")


def _positive_integer(value: Any, name: str) -> int:
    integer = _nonnegative_integer(value, name)
    if integer == 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return integer


def _nonnegative_integer(value: Any, name: str) -> int:
    integer = _integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return integer


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


__all__ = ["TabularQLearning", "TrainingCallback", "TrainingStep"]
