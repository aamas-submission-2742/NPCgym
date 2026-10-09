"""Typed, import-safe specifications for the supported baseline experiments.

The registry in this module is deliberately an explicit allowlist.  It describes
paper configurations without importing an execution backend or constructing an
environment at import time.  Training seeds remain a separate run axis and are
therefore never part of an experiment specification.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from math import isfinite
from typing import Any, Literal, SupportsFloat, cast

import gymnasium as gym
from gymnasium.wrappers import TimeLimit, TransformReward

from experiments import gardener_bolts, merchant_bolts, pacman_bolts, taxi_bolts
from npc_gym.envs import GardenerEnv, MerchantEnv, PacmanEnv, StormTaxiEnv
from npc_gym.monitors import ComplexMonitor, MonitorFactory, MultiMonitor
from npc_gym.monitors.builtins import make_builtin_monitor
from npc_gym.monitors.gardener_monitors import (
    COLLECT_ONE_NORM_ID,
    DRAIN_NORM_ID,
    NO_COLLECT_NORM_ID,
    PERMISSION_AWARE_NORM_ID,
    RESCUE_NORM_ID,
    RESCUE_PER_FROG_NORM_ID,
    make_gardener_monitor,
)
from npc_gym.monitors.merchant_monitors import (
    DELIVERY_NORM_ID,
    ENV_FRIENDLY_NORM_ID,
    PACIFIST_NORM_ID,
)
from npc_gym.monitors.pacman_monitors import (
    CONDITIONAL_VEGAN_NORM_ID,
    HUNGRY_NORM_ID,
    HUNGRY_VEGAN_NORM_ID,
    HUNGRY_VEGAN_PENALTY_NORM_ID,
    HUNGRY_VEGETARIAN_NORM_ID,
    PENALTY_NORM_ID,
    TRAPPED_NORM_ID,
    VEGAN_NORM_ID,
    VEGAN_PREFERENCE_NORM_ID,
    VEGETARIAN_BLUE_NORM_ID,
    VEGETARIAN_ORANGE_NORM_ID,
)
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID
from npc_gym.wrappers.gardener_wrappers import IllegalActionPenaltyWrapper, StateFeatureObsWrapper
from npc_gym.wrappers.merchant_wrappers import IgnoreTimeObservation
from npc_gym.wrappers.pacman_wrappers import TrappedObservation
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant

ExecutionPath = Literal["tabular", "sb3"]
WrapperKind = Literal["observation", "action", "reward"]
EnvironmentFactory = Callable[..., gym.Env[Any, Any]]
WrapperFactory = Callable[[gym.Env[Any, Any], bool], gym.Env[Any, Any]]

_VERSIONED_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*-v[0-9]+$")
_SAFE_EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*-v[0-9]+$")


class SpecificationError(ValueError):
    """Base class for invalid experiment specifications."""


class DuplicateSpecificationIDError(SpecificationError):
    """Raised when a component or experiment identifier is declared twice."""


class UnknownSpecificationIDError(SpecificationError):
    """Raised when an experiment refers to an undeclared component."""


class UnsupportedCombinationError(SpecificationError):
    """Raised when known components are not compatible with one another."""


@dataclass(frozen=True, slots=True)
class _FrozenMapping:
    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _FrozenSequence:
    items: tuple[object, ...]


@dataclass(frozen=True, slots=True, eq=False)
class KeywordArguments:
    """An immutable, deterministic collection of constructor keyword arguments."""

    _items: tuple[tuple[str, object], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> KeywordArguments:
        """Freeze JSON-like keyword arguments in sorted key order."""
        if not all(isinstance(key, str) for key in values):
            raise TypeError("Keyword argument names must be strings")
        return cls(tuple((key, _freeze(value, path=key)) for key, value in sorted(values.items())))

    def to_dict(self) -> dict[str, object]:
        """Return a detached dictionary suitable for forwarding to a constructor."""
        return {key: _thaw(value) for key, value in self._items}

    def __eq__(self, other: object) -> bool:
        """Compare JSON-like values without equating booleans and numbers."""
        return isinstance(other, KeywordArguments) and _typed_identity(self._items) == _typed_identity(other._items)

    def __hash__(self) -> int:
        return hash(_typed_identity(self._items))


@dataclass(frozen=True, slots=True)
class EnvironmentSpecification:
    """A versioned base environment configuration, separate from wrappers."""

    id: str
    family: str
    factory: EnvironmentFactory
    constructor_kwargs: KeywordArguments
    render_kwargs: KeywordArguments = KeywordArguments(())

    def make(self, *, render: bool = False) -> gym.Env[Any, Any]:
        """Create a fresh base environment, optionally configured for human rendering."""
        kwargs = self.constructor_kwargs.to_dict()
        if render:
            kwargs.update(self.render_kwargs.to_dict())
        return self.factory("human" if render else None, **kwargs)


@dataclass(frozen=True, slots=True)
class WrapperSpecification:
    """A versioned environment wrapper and its declared compatibility."""

    id: str
    kind: WrapperKind
    factory: WrapperFactory
    environment_ids: frozenset[str]
    execution_paths: frozenset[ExecutionPath]

    def apply(self, env: gym.Env[Any, Any], *, training: bool) -> gym.Env[Any, Any]:
        """Apply the wrapper to one environment."""
        return self.factory(env, training)


@dataclass(frozen=True, slots=True)
class ScenarioSpecification:
    """A versioned normative scenario with fresh monitor factories."""

    id: str
    environment_ids: frozenset[str]
    monitor_factories: tuple[tuple[str, MonitorFactory], ...]

    def make_monitors(self) -> dict[str, ComplexMonitor | MultiMonitor]:
        """Create one independent monitor instance for every norm in the scenario."""
        return {monitor_id: factory() for monitor_id, factory in self.monitor_factories}


@dataclass(frozen=True, slots=True)
class AlgorithmSpecification:
    """A learner constructor and its fully resolved seed-independent arguments."""

    id: str
    execution_path: ExecutionPath
    implementation: str
    policy: str | None
    constructor_kwargs: KeywordArguments
    environment_ids: frozenset[str]
    optional_extra: str | None = None


@dataclass(frozen=True, slots=True)
class TechniqueSpecification:
    """A normative technique supported by selected learner execution paths."""

    id: str
    execution_paths: frozenset[ExecutionPath]


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Seed-independent training and evaluation settings for one experiment."""

    training_steps: int
    max_episode_steps: int
    final_evaluation_episodes: int
    intermediate_evaluation_episodes: int
    intermediate_evaluation_frequency: int
    learning_kwargs: KeywordArguments
    evaluation_seed_offset: int
    deterministic_evaluation: bool
    task_return_units: str
    training_reward_transform: str

    def __post_init__(self) -> None:
        for name in (
            "training_steps",
            "max_episode_steps",
            "final_evaluation_episodes",
            "intermediate_evaluation_episodes",
            "intermediate_evaluation_frequency",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SpecificationError(f"{name} must be a positive integer; got {value!r}")
        if (
            isinstance(self.evaluation_seed_offset, bool)
            or not isinstance(self.evaluation_seed_offset, int)
            or self.evaluation_seed_offset < 0
        ):
            raise SpecificationError(
                f"evaluation_seed_offset must be a non-negative integer; got {self.evaluation_seed_offset!r}"
            )
        if not isinstance(self.deterministic_evaluation, bool):
            raise SpecificationError("deterministic_evaluation must be a bool")
        if not self.task_return_units:
            raise SpecificationError("task_return_units must not be empty")
        if not self.training_reward_transform:
            raise SpecificationError("training_reward_transform must not be empty")


@dataclass(frozen=True, slots=True)
class OFTENSettings:
    """Pretraining, local planner and paired evaluation for an OFTEN experiment.

    The experiment's training_steps counts combined teaching transitions. Every
    checkpoint uses consecutive episode seeds starting at evaluation_seed.
    ``norm_id`` selects the teaching recipe independently of the scenario's
    monitored norms, which may include obligations the planner cannot optimize.
    """

    norm_id: str = VEGAN_NORM_ID
    pretraining_steps: int = 5_000_000
    teaching_seed_offset: int = 20_000
    evaluation_seed: int = 50_000
    margin: float = 0.5
    horizon: int = 1
    radius: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.norm_id, str) or not _VERSIONED_ID.fullmatch(self.norm_id):
            raise SpecificationError("norm_id must be a versioned policy-fix recipe ID")
        for name in ("pretraining_steps", "horizon", "radius", "teaching_seed_offset", "evaluation_seed"):
            value = getattr(self, name)
            minimum = 0 if name.endswith(("offset", "seed")) else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SpecificationError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.margin, bool) or not isinstance(self.margin, (int, float)):
            raise SpecificationError("margin must be a finite positive number")
        if not isfinite(self.margin) or self.margin <= 0:
            raise SpecificationError("margin must be a finite positive number")


@dataclass(frozen=True, slots=True)
class PacmanOFTENSettings(OFTENSettings):
    """Pacman phase limits: base training and evaluation can differ from teaching."""

    pretraining_max_episode_steps: int = 300
    evaluation_max_episode_steps: int = 300
    pretraining_evaluation_frequency: int = 250_000
    pretraining_evaluation_seed_offset: int = 10_000

    def __post_init__(self) -> None:
        OFTENSettings.__post_init__(self)
        for name in (
            "pretraining_max_episode_steps",
            "evaluation_max_episode_steps",
            "pretraining_evaluation_frequency",
            "pretraining_evaluation_seed_offset",
        ):
            value = getattr(self, name)
            minimum = 0 if name.endswith("offset") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SpecificationError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True)
class GardenerOFTENSettings(OFTENSettings):
    """Gardener teaching overrides; pretraining retains the base DQN settings."""

    norm_id: str = NO_COLLECT_NORM_ID
    pretraining_steps: int = 250_000
    evaluation_seed: int = 10_000
    margin: float = 5.0
    radius: int = 4
    batch_size: int = 256
    gradient_steps: int = -1
    final_expert_weight: float = 0.5

    def __post_init__(self) -> None:
        OFTENSettings.__post_init__(self)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise SpecificationError("batch_size must be a positive integer")
        if (
            isinstance(self.gradient_steps, bool)
            or not isinstance(self.gradient_steps, int)
            or self.gradient_steps < -1
        ):
            raise SpecificationError("gradient_steps must be an integer >= -1")
        if (
            isinstance(self.final_expert_weight, bool)
            or not isinstance(self.final_expert_weight, (int, float))
            or not isfinite(self.final_expert_weight)
            or self.final_expert_weight < 0
        ):
            raise SpecificationError("final_expert_weight must be a finite nonnegative number")


@dataclass(frozen=True, slots=True)
class ExperimentSpecification:
    """An explicitly supported, stable experiment configuration."""

    id: str
    environment_id: str
    wrapper_ids: tuple[str, ...]
    scenario_id: str
    algorithm_id: str
    technique_id: str
    run: RunSettings
    often: OFTENSettings | None = None


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    """An experiment with all component references resolved deterministically."""

    specification: ExperimentSpecification
    environment: EnvironmentSpecification
    wrappers: tuple[WrapperSpecification, ...]
    scenario: ScenarioSpecification
    algorithm: AlgorithmSpecification
    technique: TechniqueSpecification

    def make_env(self, *, training: bool, render: bool = False) -> gym.Env[Any, Any]:
        """Create a fresh environment with the declared wrappers in order."""
        env = self.environment.make(render=render)
        try:
            for wrapper in self.wrappers:
                env = wrapper.apply(env, training=training)
            limit = self.specification.run.max_episode_steps
            settings = self.specification.often
            if not training and isinstance(settings, PacmanOFTENSettings):
                limit = settings.evaluation_max_episode_steps
            return TimeLimit(env, max_episode_steps=limit)
        except BaseException:
            env.close()
            raise

    def make_monitors(self) -> dict[str, ComplexMonitor | MultiMonitor]:
        """Create fresh monitors for the declared normative scenario."""
        return self.scenario.make_monitors()


class ExperimentRegistry:
    """Validate and resolve an explicit set of supported experiment specifications."""

    def __init__(
        self,
        *,
        environments: Iterable[EnvironmentSpecification] = (),
        wrappers: Iterable[WrapperSpecification] = (),
        scenarios: Iterable[ScenarioSpecification] = (),
        algorithms: Iterable[AlgorithmSpecification] = (),
        techniques: Iterable[TechniqueSpecification] = (),
        experiments: Iterable[ExperimentSpecification] = (),
    ) -> None:
        self._environments = _index("environment", environments)
        self._wrappers = _index("wrapper", wrappers)
        self._scenarios = _index("scenario", scenarios)
        self._algorithms = _index("algorithm", algorithms)
        self._techniques = _index("technique", techniques)
        self._experiments = _index("experiment", experiments)
        self._validate_components()
        for experiment in self._experiments.values():
            self._validate_experiment(experiment)

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        """Return supported experiment identifiers in deterministic order."""
        return tuple(sorted(self._experiments))

    def resolve(self, experiment_id: str) -> ResolvedExperiment:
        """Resolve one supported experiment identifier into typed components."""
        experiment = _lookup("experiment", experiment_id, self._experiments)
        environment = _lookup("environment", experiment.environment_id, self._environments)
        wrappers = tuple(_lookup("wrapper", wrapper_id, self._wrappers) for wrapper_id in experiment.wrapper_ids)
        scenario = _lookup("scenario", experiment.scenario_id, self._scenarios)
        algorithm = _lookup("algorithm", experiment.algorithm_id, self._algorithms)
        technique = _lookup("technique", experiment.technique_id, self._techniques)
        return ResolvedExperiment(experiment, environment, wrappers, scenario, algorithm, technique)

    def _validate_experiment(self, experiment: ExperimentSpecification) -> None:
        resolved = self.resolve(experiment.id)
        environment_id = resolved.environment.id
        path = resolved.algorithm.execution_path
        if len(set(experiment.wrapper_ids)) != len(experiment.wrapper_ids):
            raise DuplicateSpecificationIDError(f"Experiment {experiment.id!r} contains duplicate wrapper IDs")
        if environment_id not in resolved.scenario.environment_ids:
            raise UnsupportedCombinationError(
                f"Scenario {resolved.scenario.id!r} does not support environment {environment_id!r}"
            )
        for wrapper in resolved.wrappers:
            if environment_id not in wrapper.environment_ids:
                raise UnsupportedCombinationError(
                    f"Wrapper {wrapper.id!r} does not support environment {environment_id!r}"
                )
            if path not in wrapper.execution_paths:
                raise UnsupportedCombinationError(f"Wrapper {wrapper.id!r} does not support execution path {path!r}")
        if environment_id not in resolved.algorithm.environment_ids:
            raise UnsupportedCombinationError(
                f"Algorithm {resolved.algorithm.id!r} does not support environment {environment_id!r}"
            )
        if path not in resolved.technique.execution_paths:
            raise UnsupportedCombinationError(
                f"Technique {resolved.technique.id!r} does not support execution path {path!r}"
            )

    def _validate_components(self) -> None:
        valid_paths = {"tabular", "sb3"}
        for environment in self._environments.values():
            if not environment.family:
                raise SpecificationError(f"Environment {environment.id!r} must declare a family")
            if not callable(environment.factory):
                raise TypeError(f"Environment {environment.id!r} factory must be callable")
        for wrapper in self._wrappers.values():
            if not callable(wrapper.factory):
                raise TypeError(f"Wrapper {wrapper.id!r} factory must be callable")
            if not wrapper.environment_ids:
                raise SpecificationError(f"Wrapper {wrapper.id!r} must support at least one environment")
            if not wrapper.execution_paths or not wrapper.execution_paths <= valid_paths:
                raise SpecificationError(f"Wrapper {wrapper.id!r} has invalid execution paths")
            for environment_id in wrapper.environment_ids:
                _lookup("environment", environment_id, self._environments)
        for scenario in self._scenarios.values():
            if not scenario.environment_ids:
                raise SpecificationError(f"Scenario {scenario.id!r} must support at least one environment")
            for environment_id in scenario.environment_ids:
                _lookup("environment", environment_id, self._environments)
            monitor_ids = [monitor_id for monitor_id, _factory in scenario.monitor_factories]
            if len(set(monitor_ids)) != len(monitor_ids):
                raise DuplicateSpecificationIDError(f"Scenario {scenario.id!r} contains duplicate monitor IDs")
            for monitor_id, factory in scenario.monitor_factories:
                if not _VERSIONED_ID.fullmatch(monitor_id):
                    raise SpecificationError(f"Monitor ID {monitor_id!r} must be a lowercase versioned ID")
                if not callable(factory):
                    raise TypeError(f"Monitor {monitor_id!r} factory must be callable")
        for algorithm in self._algorithms.values():
            if algorithm.execution_path not in valid_paths:
                raise SpecificationError(f"Algorithm {algorithm.id!r} has an invalid execution path")
            if not algorithm.environment_ids:
                raise SpecificationError(f"Algorithm {algorithm.id!r} must support at least one environment")
            for environment_id in algorithm.environment_ids:
                _lookup("environment", environment_id, self._environments)
        for technique in self._techniques.values():
            if not technique.execution_paths or not technique.execution_paths <= valid_paths:
                raise SpecificationError(f"Technique {technique.id!r} has invalid execution paths")


def _freeze(value: object, *, path: str) -> object:
    if isinstance(value, float) and not isfinite(value):
        raise TypeError(f"Keyword argument {path!r} must be finite; got {value!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Keyword argument {path!r} contains a non-string mapping key")
        return _FrozenMapping(tuple((key, _freeze(item, path=f"{path}.{key}")) for key, item in sorted(value.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _FrozenSequence(tuple(_freeze(item, path=path) for item in value))
    raise TypeError(f"Keyword argument {path!r} has unsupported value {value!r}")


def _thaw(value: object) -> object:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw(item) for key, item in value.items}
    if isinstance(value, _FrozenSequence):
        return [_thaw(item) for item in value.items]
    return value


def _typed_identity(value: object) -> object:
    if isinstance(value, tuple):
        return (tuple, tuple(_typed_identity(item) for item in value))
    if isinstance(value, _FrozenMapping):
        return (_FrozenMapping, _typed_identity(value.items))
    if isinstance(value, _FrozenSequence):
        return (_FrozenSequence, _typed_identity(value.items))
    return (type(value), value)


def _index(name: str, values: Iterable[Any]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for value in values:
        component_id = value.id
        pattern = _SAFE_EXPERIMENT_ID if name == "experiment" else _VERSIONED_ID
        if not isinstance(component_id, str) or not pattern.fullmatch(component_id):
            qualifier = "filesystem-safe lowercase versioned ID" if name == "experiment" else "lowercase versioned ID"
            raise SpecificationError(f"{name.capitalize()} ID {component_id!r} must be a {qualifier}")
        if component_id in indexed:
            raise DuplicateSpecificationIDError(f"Duplicate {name} ID {component_id!r}")
        indexed[component_id] = value
    return indexed


def _lookup(name: str, component_id: str, values: Mapping[str, Any]) -> Any:
    try:
        return values[component_id]
    except KeyError as error:
        available = ", ".join(sorted(values)) or "none"
        raise UnknownSpecificationIDError(
            f"Unknown {name} ID {component_id!r}. Available {name} IDs: {available}"
        ) from error


def _kwargs(**values: object) -> KeywordArguments:
    return KeywordArguments.from_mapping(values)


def _taxi_environment(render_mode: str | None, **kwargs: object) -> gym.Env[Any, Any]:
    constructor = cast(Callable[..., gym.Env[Any, Any]], StormTaxiEnv)
    return constructor(render_mode=render_mode, **kwargs)


def _merchant_environment(render_mode: str | None, **kwargs: object) -> gym.Env[Any, Any]:
    constructor = cast(Callable[..., gym.Env[Any, Any]], MerchantEnv)
    return constructor(render_mode=render_mode, **kwargs)


def _gardener_environment(render_mode: str | None, **kwargs: object) -> gym.Env[Any, Any]:
    constructor = cast(Callable[..., gym.Env[Any, Any]], GardenerEnv)
    return constructor(render_mode=render_mode, **kwargs)


def _pacman_environment(render_mode: str | None, **kwargs: object) -> gym.Env[Any, Any]:
    constructor = cast(Callable[..., gym.Env[Any, Any]], PacmanEnv)
    return constructor(render_mode=render_mode, **kwargs)


def _ignore_weather(env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
    wrapper = cast(Callable[[gym.Env[Any, Any]], gym.Env[Any, Any]], IgnoreWeatherRelevant)
    return wrapper(env)


def _ignore_time(env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
    wrapper = cast(Callable[[gym.Env[Any, Any]], gym.Env[Any, Any]], IgnoreTimeObservation)
    return wrapper(env)


def _state_features(env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
    wrapper = cast(Callable[[gym.Env[Any, Any]], gym.Env[Any, Any]], StateFeatureObsWrapper)
    return wrapper(env)


def _frog_state_features(env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
    return StateFeatureObsWrapper(env, include_frogs=True)


def _trapped_observation(env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
    return TrappedObservation(env)


def _illegal_to_stay(env: gym.Env[Any, Any], training: bool) -> gym.Env[Any, Any]:
    wrapper = cast(Callable[..., gym.Env[Any, Any]], IllegalActionPenaltyWrapper)
    return wrapper(env, penalty=-1.0 if training else 0.0)


def _divide_reward(reward: SupportsFloat) -> float:
    return float(reward) / 100


def _pacman_training_reward(env: gym.Env[Any, Any], training: bool) -> gym.Env[Any, Any]:
    return TransformReward(env, _divide_reward) if training else env


def _pacman_pixels(env: gym.Env[Any, Any], _training: bool) -> gym.Env[Any, Any]:
    """Two 80-by-210 grayscale frames, preserving optional numeric mode features."""
    from npc_gym.wrappers import PacmanPixelObservation

    return PacmanPixelObservation(env)


TAXI_ENVIRONMENT_ID = "taxi/storm-v1"
MERCHANT_ENVIRONMENT_ID = "merchant/basic-v2"
GARDENER_ENVIRONMENT_ID = "gardener/size-15-v0"
PACMAN_ENVIRONMENT_ID = "pacman/small-v1"
PACMAN_IMAGES_ENVIRONMENT_ID = "pacman/small-images-v1"

ENVIRONMENTS: tuple[EnvironmentSpecification, ...] = (
    EnvironmentSpecification(
        TAXI_ENVIRONMENT_ID,
        "taxi",
        _taxi_environment,
        _kwargs(fickle_passenger=False),
    ),
    EnvironmentSpecification(
        MERCHANT_ENVIRONMENT_ID,
        "merchant",
        _merchant_environment,
        _kwargs(layout="basic", risk_fight=0.75, risk_death=0.25, capacity=5, sunset=28),
        _kwargs(step_delay_ms=500),
    ),
    EnvironmentSpecification(
        GARDENER_ENVIRONMENT_ID,
        "gardener",
        _gardener_environment,
        _kwargs(size=15, grass_respawn=50, puddle_respawn=20, score_limit=300, frog_freeze=5),
    ),
    EnvironmentSpecification(
        PACMAN_ENVIRONMENT_ID,
        "pacman",
        _pacman_environment,
        _kwargs(layout="small", features="complete", dfas=None),
    ),
    EnvironmentSpecification(
        PACMAN_IMAGES_ENVIRONMENT_ID,
        "pacman",
        _pacman_environment,
        _kwargs(layout="small", features="image-full", dfas=None),
    ),
)

TAXI_OBSERVATION_ID = "taxi/ignore-weather-relevant-v0"
MERCHANT_OBSERVATION_ID = "merchant/ignore-time-v0"
GARDENER_OBSERVATION_ID = "gardener/state-features-v0"
GARDENER_FROG_OBSERVATION_ID = "gardener/state-features-frogs-v0"
GARDENER_ACTION_ID = "gardener/illegal-to-stay-v0"
PACMAN_REWARD_ID = "pacman/training-reward-divide-100-v0"
PACMAN_TRAPPED_OBSERVATION_ID = "pacman/trapped-observation-v1"
PACMAN_PIXELS_ID = "pacman/grayscale-two-frames-v0"

WRAPPERS: tuple[WrapperSpecification, ...] = (
    WrapperSpecification(
        TAXI_OBSERVATION_ID,
        "observation",
        _ignore_weather,
        frozenset({TAXI_ENVIRONMENT_ID}),
        frozenset({"tabular"}),
    ),
    WrapperSpecification(
        MERCHANT_OBSERVATION_ID,
        "observation",
        _ignore_time,
        frozenset({MERCHANT_ENVIRONMENT_ID}),
        frozenset({"tabular"}),
    ),
    WrapperSpecification(
        GARDENER_OBSERVATION_ID,
        "observation",
        _state_features,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
    WrapperSpecification(
        GARDENER_FROG_OBSERVATION_ID,
        "observation",
        _frog_state_features,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
    WrapperSpecification(
        GARDENER_ACTION_ID,
        "action",
        _illegal_to_stay,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
    WrapperSpecification(
        PACMAN_REWARD_ID,
        "reward",
        _pacman_training_reward,
        frozenset({PACMAN_ENVIRONMENT_ID, PACMAN_IMAGES_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
    WrapperSpecification(
        PACMAN_TRAPPED_OBSERVATION_ID,
        "observation",
        _trapped_observation,
        frozenset({PACMAN_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
    WrapperSpecification(
        PACMAN_PIXELS_ID,
        "observation",
        _pacman_pixels,
        frozenset({PACMAN_IMAGES_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
)


def _monitor_factories(
    monitor_ids: Sequence[str], factory: Callable[[str], ComplexMonitor | MultiMonitor]
) -> tuple[tuple[str, MonitorFactory], ...]:
    return tuple((monitor_id, lambda monitor_id=monitor_id: factory(monitor_id)) for monitor_id in monitor_ids)


TAXI_SCENARIO_ID = "taxi/baseline-norms-v0"
MERCHANT_SCENARIO_ID = "merchant/baseline-norms-v0"
MERCHANT_ENV_FRIENDLY_SCENARIO_ID = "merchant/env-friendly-with-delivery-v0"
GARDENER_SCENARIO_ID = "gardener/permission-drain-rescue-v0"
GARDENER_PERMISSION_DRAIN_SCENARIO_ID = "gardener/permission-drain-v0"
PACMAN_SCENARIO_ID = "pacman/baseline-norms-v3"
PACMAN_VEGAN_SCENARIO_ID = "pacman/vegan-v0"
PACMAN_VEGETARIAN_SCENARIO_ID = "pacman/vegetarian-orange-v0"

MERCHANT_MONITOR_IDS = (
    ENV_FRIENDLY_NORM_ID,
    DELIVERY_NORM_ID,
    PACIFIST_NORM_ID,
    merchant_bolts.RECIPES["delivery-pacifist"],
)

SCENARIOS = (
    ScenarioSpecification(
        TAXI_SCENARIO_ID,
        frozenset({TAXI_ENVIRONMENT_ID}),
        _monitor_factories((EMERGENCY_NORM_ID,), make_builtin_monitor),
    ),
    ScenarioSpecification(
        MERCHANT_SCENARIO_ID,
        frozenset({MERCHANT_ENVIRONMENT_ID}),
        _monitor_factories(MERCHANT_MONITOR_IDS, make_builtin_monitor),
    ),
    ScenarioSpecification(
        MERCHANT_ENV_FRIENDLY_SCENARIO_ID,
        frozenset({MERCHANT_ENVIRONMENT_ID}),
        _monitor_factories(MERCHANT_MONITOR_IDS, make_builtin_monitor),
    ),
    ScenarioSpecification(
        GARDENER_SCENARIO_ID,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        tuple(
            (norm_id, partial(make_gardener_monitor, norm_id, num_frogs=2, num_puddles=4))
            for norm_id in (PERMISSION_AWARE_NORM_ID, DRAIN_NORM_ID, RESCUE_NORM_ID)
        ),
    ),
    ScenarioSpecification(
        PACMAN_SCENARIO_ID,
        frozenset({PACMAN_ENVIRONMENT_ID, PACMAN_IMAGES_ENVIRONMENT_ID}),
        _monitor_factories(
            (
                VEGAN_NORM_ID,
                VEGETARIAN_BLUE_NORM_ID,
                VEGETARIAN_ORANGE_NORM_ID,
                CONDITIONAL_VEGAN_NORM_ID,
                PENALTY_NORM_ID,
                HUNGRY_NORM_ID,
                HUNGRY_VEGAN_NORM_ID,
                HUNGRY_VEGAN_PENALTY_NORM_ID,
                VEGAN_PREFERENCE_NORM_ID,
                HUNGRY_VEGETARIAN_NORM_ID,
                TRAPPED_NORM_ID,
            ),
            make_builtin_monitor,
        ),
    ),
    ScenarioSpecification(
        PACMAN_VEGAN_SCENARIO_ID,
        frozenset({PACMAN_ENVIRONMENT_ID}),
        _monitor_factories((VEGAN_NORM_ID,), make_builtin_monitor),
    ),
    ScenarioSpecification(
        PACMAN_VEGETARIAN_SCENARIO_ID,
        frozenset({PACMAN_ENVIRONMENT_ID}),
        _monitor_factories((VEGETARIAN_ORANGE_NORM_ID,), make_builtin_monitor),
    ),
    ScenarioSpecification(
        TRAPPED_NORM_ID,
        frozenset({PACMAN_ENVIRONMENT_ID}),
        _monitor_factories((TRAPPED_NORM_ID,), make_builtin_monitor),
    ),
    ScenarioSpecification(
        GARDENER_PERMISSION_DRAIN_SCENARIO_ID,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        tuple(
            (norm_id, partial(make_gardener_monitor, norm_id, num_frogs=2, num_puddles=4))
            for norm_id in (PERMISSION_AWARE_NORM_ID, DRAIN_NORM_ID)
        ),
    ),
    *(
        ScenarioSpecification(
            norm_id,
            frozenset({GARDENER_ENVIRONMENT_ID}),
            # Object counts for the declared size-15 environment.
            ((norm_id, partial(make_gardener_monitor, norm_id, num_frogs=2, num_puddles=4)),),
        )
        for norm_id in (NO_COLLECT_NORM_ID, DRAIN_NORM_ID, PERMISSION_AWARE_NORM_ID)
    ),
)

TABULAR_TAXI_ALGORITHM_ID = "tabular/taxi-baseline-v0"
TABULAR_MERCHANT_ALGORITHM_ID = "tabular/merchant-baseline-v0"
DQN_GARDENER_ALGORITHM_ID = "sb3/dqn-gardener-baseline-v0"
PPO_PACMAN_ALGORITHM_ID = "sb3/ppo-pacman-baseline-v0"
DQN_GARDENER_OFTEN_ALGORITHM_ID = "sb3/dqn-gardener-discount-095-v0"
PPO_GARDENER_ALGORITHM_ID = "sb3/ppo-gardener-default-v0"
DQN_PACMAN_DEFAULT_ALGORITHM_ID = "sb3/dqn-pacman-default-v0"
DQN_PACMAN_IMAGES_ALGORITHM_ID = "sb3/dqn-pacman-images-v0"

ALGORITHMS = (
    AlgorithmSpecification(
        TABULAR_TAXI_ALGORITHM_ID,
        "tabular",
        "npc_gym.algorithms.TabularQLearning",
        None,
        _kwargs(
            learning_rate=0.2,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            use_action_mask=False,
            log_interval=10_000,
        ),
        frozenset({TAXI_ENVIRONMENT_ID}),
    ),
    AlgorithmSpecification(
        TABULAR_MERCHANT_ALGORITHM_ID,
        "tabular",
        "npc_gym.algorithms.TabularQLearning",
        None,
        _kwargs(
            learning_rate=0.5,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.2,
            use_action_mask=True,
            log_interval=10_000,
        ),
        frozenset({MERCHANT_ENVIRONMENT_ID}),
    ),
    AlgorithmSpecification(
        DQN_GARDENER_ALGORITHM_ID,
        "sb3",
        "stable_baselines3.DQN",
        "MlpPolicy",
        _kwargs(
            learning_rate=5e-4,
            buffer_size=50_000,
            learning_starts=5_000,
            batch_size=128,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=2_000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10.0,
            stats_window_size=100,
            policy_kwargs={"net_arch": [32, 32]},
            verbose=1,
            device="cpu",
        ),
        frozenset({GARDENER_ENVIRONMENT_ID}),
        "sb3",
    ),
    AlgorithmSpecification(
        PPO_PACMAN_ALGORITHM_ID,
        "sb3",
        "stable_baselines3.PPO",
        "MlpPolicy",
        _kwargs(
            learning_rate=3e-4,
            n_steps=2_048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            rollout_buffer_class=None,
            rollout_buffer_kwargs=None,
            target_kl=None,
            stats_window_size=100,
            policy_kwargs=None,
            verbose=1,
            device="cpu",
        ),
        frozenset({PACMAN_ENVIRONMENT_ID}),
        "sb3",
    ),
    *(
        AlgorithmSpecification(
            algorithm_id,
            "sb3",
            "stable_baselines3.DQN",
            "MlpPolicy",
            _kwargs(
                learning_rate=1e-4,
                buffer_size=1_000_000,
                learning_starts=100,
                batch_size=32,
                tau=1.0,
                gamma=0.95,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=10_000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10.0,
                policy_kwargs={"net_arch": [64, 64]},
                device="cpu",
                verbose=0,
            ),
            frozenset({environment_id}),
            "sb3",
        )
        for algorithm_id, environment_id in ((DQN_GARDENER_OFTEN_ALGORITHM_ID, GARDENER_ENVIRONMENT_ID),)
    ),
    AlgorithmSpecification(
        PPO_GARDENER_ALGORITHM_ID,
        "sb3",
        "stable_baselines3.PPO",
        "MlpPolicy",
        _kwargs(
            learning_rate=3e-4,
            n_steps=2_048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            rollout_buffer_class=None,
            rollout_buffer_kwargs=None,
            target_kl=None,
            stats_window_size=100,
            policy_kwargs=None,
            verbose=1,
            device="cpu",
        ),
        frozenset({GARDENER_ENVIRONMENT_ID}),
        "sb3",
    ),
    *(
        AlgorithmSpecification(
            algorithm_id,
            "sb3",
            "stable_baselines3.DQN",
            "CnnPolicy" if images else "MlpPolicy",
            _kwargs(
                learning_rate=1e-4,
                buffer_size=100_000 if images else 1_000_000,
                learning_starts=100,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                target_update_interval=10_000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10.0,
                stats_window_size=100,
                policy_kwargs={"features_extractor_kwargs": {"features_dim": 128}, "net_arch": [64, 64]}
                if images
                else None,
                verbose=1,
                device="cpu",
            ),
            frozenset({environment_id}),
            "sb3,render" if images else "sb3",
        )
        for algorithm_id, environment_id, images in (
            (DQN_PACMAN_DEFAULT_ALGORITHM_ID, PACMAN_ENVIRONMENT_ID, False),
            (DQN_PACMAN_IMAGES_ALGORITHM_ID, PACMAN_IMAGES_ENVIRONMENT_ID, True),
        )
    ),
)

UNCONSTRAINED_TECHNIQUE_ID = "technique/unconstrained-v0"
OFTEN_TECHNIQUE_ID = "technique/often-v0"
POLICY_FIX_TECHNIQUE_ID = "technique/policy-fixes-v0"
TECHNIQUES: tuple[TechniqueSpecification, ...] = (
    TechniqueSpecification(UNCONSTRAINED_TECHNIQUE_ID, frozenset({"tabular", "sb3"})),
    TechniqueSpecification(OFTEN_TECHNIQUE_ID, frozenset({"sb3"})),
    TechniqueSpecification(POLICY_FIX_TECHNIQUE_ID, frozenset({"tabular"})),
)


EVALUATION_EPISODES = 1_000


def _run(
    training_steps: int,
    max_episode_steps: int,
    *,
    learning_kwargs: Mapping[str, object] | None = None,
    training_reward_transform: str = "identity",
) -> RunSettings:
    return RunSettings(
        training_steps=training_steps,
        max_episode_steps=max_episode_steps,
        final_evaluation_episodes=EVALUATION_EPISODES,
        intermediate_evaluation_episodes=EVALUATION_EPISODES,
        intermediate_evaluation_frequency=training_steps // 20,
        learning_kwargs=KeywordArguments.from_mapping(learning_kwargs or {}),
        evaluation_seed_offset=10_000,
        deterministic_evaluation=True,
        task_return_units="unscaled task reward",
        training_reward_transform=training_reward_transform,
    )


TAXI_EXPERIMENT_ID = "taxi-tabular-unconstrained-v1"
TAXI_POLICY_FIX_EXPERIMENT_ID = "taxi-tabular-policy-fixes-warn-v1"
MERCHANT_EXPERIMENT_ID = "merchant-tabular-unconstrained-v2"
MERCHANT_POLICY_FIX_EXPERIMENT_ID = "merchant-tabular-policy-fixes-env-friendly-v0"
GARDENER_EXPERIMENT_ID = "gardener-dqn-unconstrained-v0"
PACMAN_EXPERIMENT_ID = "pacman-ppo-unconstrained-v2"
PACMAN_OFTEN_EXPERIMENT_ID = "pacman-dqn-often-v1"
PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID = "pacman-dqn-often-vegetarian-v1"
PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID = "pacman-dqn-often-trapped-v2"
GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID = "gardener-dqn-often-no-collect-v0"
GARDENER_DRAIN_OFTEN_EXPERIMENT_ID = "gardener-dqn-often-drain-v0"
GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID = "gardener-dqn-often-permission-v0"
GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID = "gardener-dqn-often-permission-drain-v0"
GARDENER_PPO_EXPERIMENT_ID = "gardener-ppo-unconstrained-v0"
PACMAN_DQN_EXPERIMENT_ID = "pacman-dqn-unconstrained-v1"
PACMAN_IMAGES_EXPERIMENT_ID = "pacman-images-dqn-unconstrained-v1"


def _pacman_often(experiment_id: str, scenario_id: str) -> ExperimentSpecification:
    return ExperimentSpecification(
        experiment_id,
        PACMAN_ENVIRONMENT_ID,
        (PACMAN_TRAPPED_OBSERVATION_ID, PACMAN_REWARD_ID) if scenario_id == TRAPPED_NORM_ID else (PACMAN_REWARD_ID,),
        scenario_id,
        DQN_PACMAN_DEFAULT_ALGORITHM_ID,
        OFTEN_TECHNIQUE_ID,
        replace(
            _run(10_000_000, 500, training_reward_transform="reward divided by 100"),
            intermediate_evaluation_frequency=1_000_000,
        ),
        often=PacmanOFTENSettings(norm_id=scenario_id),
    )


def _gardener_often(experiment_id: str, norm_id: str) -> ExperimentSpecification:
    return ExperimentSpecification(
        experiment_id,
        GARDENER_ENVIRONMENT_ID,
        (GARDENER_FROG_OBSERVATION_ID, GARDENER_ACTION_ID),
        GARDENER_SCENARIO_ID,
        DQN_GARDENER_OFTEN_ALGORITHM_ID,
        OFTEN_TECHNIQUE_ID,
        replace(
            _run(250_000, 1_000, training_reward_transform="illegal actions become stay with penalty -1"),
            intermediate_evaluation_frequency=250_000,
        ),
        often=GardenerOFTENSettings(norm_id=norm_id),
    )


EXPERIMENTS: tuple[ExperimentSpecification, ...] = (
    ExperimentSpecification(
        TAXI_EXPERIMENT_ID,
        TAXI_ENVIRONMENT_ID,
        (TAXI_OBSERVATION_ID,),
        TAXI_SCENARIO_ID,
        TABULAR_TAXI_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(5_000_000, 50),
    ),
    ExperimentSpecification(
        MERCHANT_EXPERIMENT_ID,
        MERCHANT_ENVIRONMENT_ID,
        (MERCHANT_OBSERVATION_ID,),
        MERCHANT_SCENARIO_ID,
        TABULAR_MERCHANT_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(5_000_000, 150),
    ),
    ExperimentSpecification(
        GARDENER_EXPERIMENT_ID,
        GARDENER_ENVIRONMENT_ID,
        (GARDENER_OBSERVATION_ID, GARDENER_ACTION_ID),
        GARDENER_SCENARIO_ID,
        DQN_GARDENER_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(
            100_000,
            1_000,
            learning_kwargs={"log_interval": 100},
            training_reward_transform="illegal actions become stay with penalty -1",
        ),
    ),
    ExperimentSpecification(
        PACMAN_EXPERIMENT_ID,
        PACMAN_ENVIRONMENT_ID,
        (PACMAN_REWARD_ID,),
        PACMAN_SCENARIO_ID,
        PPO_PACMAN_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(
            5_000_000,
            300,
            learning_kwargs={"log_interval": 1},
            training_reward_transform="reward divided by 100",
        ),
    ),
    ExperimentSpecification(
        TAXI_POLICY_FIX_EXPERIMENT_ID,
        TAXI_ENVIRONMENT_ID,
        (TAXI_OBSERVATION_ID,),
        TAXI_SCENARIO_ID,
        TABULAR_TAXI_ALGORITHM_ID,
        POLICY_FIX_TECHNIQUE_ID,
        replace(
            _run(5_000_000, 50),
            intermediate_evaluation_frequency=5_000_000,
        ),
    ),
    ExperimentSpecification(
        MERCHANT_POLICY_FIX_EXPERIMENT_ID,
        MERCHANT_ENVIRONMENT_ID,
        (MERCHANT_OBSERVATION_ID,),
        MERCHANT_ENV_FRIENDLY_SCENARIO_ID,
        TABULAR_MERCHANT_ALGORITHM_ID,
        POLICY_FIX_TECHNIQUE_ID,
        replace(
            _run(5_000_000, 150),
            intermediate_evaluation_frequency=5_000_000,
        ),
    ),
    _pacman_often(PACMAN_OFTEN_EXPERIMENT_ID, PACMAN_VEGAN_SCENARIO_ID),
    _pacman_often(PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID, PACMAN_VEGETARIAN_SCENARIO_ID),
    _pacman_often(PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID, TRAPPED_NORM_ID),
    _gardener_often(GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID, NO_COLLECT_NORM_ID),
    _gardener_often(GARDENER_DRAIN_OFTEN_EXPERIMENT_ID, DRAIN_NORM_ID),
    _gardener_often(GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID, PERMISSION_AWARE_NORM_ID),
    _gardener_often(GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID, GARDENER_PERMISSION_DRAIN_SCENARIO_ID),
    ExperimentSpecification(
        GARDENER_PPO_EXPERIMENT_ID,
        GARDENER_ENVIRONMENT_ID,
        (GARDENER_OBSERVATION_ID, GARDENER_ACTION_ID),
        GARDENER_SCENARIO_ID,
        PPO_GARDENER_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(100_000, 1_000, training_reward_transform="illegal actions become stay with penalty -1"),
    ),
    ExperimentSpecification(
        PACMAN_DQN_EXPERIMENT_ID,
        PACMAN_ENVIRONMENT_ID,
        (PACMAN_REWARD_ID,),
        PACMAN_SCENARIO_ID,
        DQN_PACMAN_DEFAULT_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(5_000_000, 300, training_reward_transform="reward divided by 100"),
    ),
    ExperimentSpecification(
        PACMAN_IMAGES_EXPERIMENT_ID,
        PACMAN_IMAGES_ENVIRONMENT_ID,
        (PACMAN_PIXELS_ID, PACMAN_REWARD_ID),
        PACMAN_SCENARIO_ID,
        DQN_PACMAN_IMAGES_ALGORITHM_ID,
        UNCONSTRAINED_TECHNIQUE_ID,
        _run(5_000_000, 300, training_reward_transform="reward divided by 100"),
    ),
)

# Keep resolved defaults independent of whichever SB3 version is installed.
# The v1 reward wrappers expose minimized DFA observations.
PACMAN_BOLTS_ENVIRONMENT_ID = "pacman/smallclassic-random-bolts-v0"
RESTRAINING_BOLT_TECHNIQUE_ID = "technique/restraining-bolts-v0"
PACMAN_BOLT_EXPERIMENT_IDS = tuple(
    pacman_bolts.experiment_id(norm, algorithm) for norm in pacman_bolts.BUDGETS for algorithm in ("dqn", "ppo")
)
ENVIRONMENTS += (
    EnvironmentSpecification(
        PACMAN_BOLTS_ENVIRONMENT_ID,
        "pacman",
        pacman_bolts.make_environment,
        _kwargs(layout=pacman_bolts.LAYOUT, features="complete", ghost_behavior="random"),
    ),
)
WRAPPERS += tuple(
    WrapperSpecification(
        f"pacman/kr2026-{norm}-bolts-and-reward-v1",
        "reward",
        partial(pacman_bolts.wrap_bolts, norm=norm),
        frozenset({PACMAN_BOLTS_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    )
    for norm in pacman_bolts.RECIPES
)
SCENARIOS += (
    ScenarioSpecification(
        "pacman/submission-bolt-evaluation-v0",
        frozenset({PACMAN_BOLTS_ENVIRONMENT_ID}),
        _monitor_factories(tuple(pacman_bolts.RECIPES[norm] for norm in pacman_bolts.BUDGETS), make_builtin_monitor),
    ),
)
ALGORITHMS += tuple(
    replace(
        next(item for item in ALGORITHMS if item.id == baseline_id),
        id=f"sb3/{algorithm}-default-bolts-v0",
        environment_ids=frozenset({PACMAN_BOLTS_ENVIRONMENT_ID}),
        constructor_kwargs=_kwargs(
            **(
                next(item for item in ALGORITHMS if item.id == baseline_id).constructor_kwargs.to_dict()
                | {"verbose": 0, "tensorboard_log": None}
                | ({"n_steps": 1} if algorithm == "dqn" else {})
            )
        ),
    )
    for algorithm, baseline_id in (("dqn", DQN_PACMAN_DEFAULT_ALGORITHM_ID), ("ppo", PPO_PACMAN_ALGORITHM_ID))
)
TECHNIQUES += (TechniqueSpecification(RESTRAINING_BOLT_TECHNIQUE_ID, frozenset({"tabular", "sb3"})),)
EXPERIMENTS += tuple(
    ExperimentSpecification(
        pacman_bolts.experiment_id(norm, algorithm),
        PACMAN_BOLTS_ENVIRONMENT_ID,
        (f"pacman/kr2026-{norm}-bolts-and-reward-v1",),
        "pacman/submission-bolt-evaluation-v0",
        f"sb3/{algorithm}-default-bolts-v0",
        RESTRAINING_BOLT_TECHNIQUE_ID,
        _run(
            steps,
            300,
            learning_kwargs={"log_interval": 1} if algorithm == "ppo" else {},
            training_reward_transform="(task reward - configured norm punishments) / 100",
        ),
    )
    for norm, budgets in pacman_bolts.BUDGETS.items()
    for algorithm, steps in budgets.items()
)


# Budget candidates are distinct experiments: DQN's exploration schedule depends
# on the total requested steps. Final runs reuse the selected identity with new seeds.
SCENARIOS += (
    ScenarioSpecification(
        "pacman/submission-bolt-evaluation-v1",
        frozenset({PACMAN_BOLTS_ENVIRONMENT_ID}),
        _monitor_factories(tuple(pacman_bolts.RECIPES.values()), make_builtin_monitor),
    ),
)
PACMAN_BOLT_STUDY_CONFIGURATIONS = tuple(
    (norm, algorithm, steps, pacman_bolts.experiment_id(norm, algorithm, steps))
    for norm in pacman_bolts.PILOT_NORMS
    for algorithm in ("dqn", "ppo")
    for steps in pacman_bolts.PILOT_BUDGETS
) + tuple(
    ("hungry-vegan-penalty", algorithm, 20_000_000, pacman_bolts.experiment_id("hungry-vegan-penalty", algorithm))
    for algorithm in ("dqn", "ppo")
)
PACMAN_BOLT_EXTENDED_PILOT_CONFIGURATIONS = (
    (
        "trapped",
        "ppo",
        pacman_bolts.TRAPPED_PPO_EXTENSION_BUDGET,
        pacman_bolts.experiment_id("trapped", "ppo", pacman_bolts.TRAPPED_PPO_EXTENSION_BUDGET),
    ),
)
EXPERIMENTS += tuple(
    ExperimentSpecification(
        identifier,
        PACMAN_BOLTS_ENVIRONMENT_ID,
        (f"pacman/kr2026-{norm}-bolts-and-reward-v1",),
        "pacman/submission-bolt-evaluation-v1",
        f"sb3/{algorithm}-default-bolts-v0",
        RESTRAINING_BOLT_TECHNIQUE_ID,
        _run(
            steps,
            300,
            learning_kwargs={"log_interval": 1} if algorithm == "ppo" else {},
            training_reward_transform="(task reward - configured norm punishments) / 100",
        ),
    )
    for norm, algorithm, steps, identifier in (
        *PACMAN_BOLT_STUDY_CONFIGURATIONS,
        *PACMAN_BOLT_EXTENDED_PILOT_CONFIGURATIONS,
    )
)

PACMAN_BOLT_EXPERIMENT_IDS += tuple(
    item[3] for item in (*PACMAN_BOLT_STUDY_CONFIGURATIONS, *PACMAN_BOLT_EXTENDED_PILOT_CONFIGURATIONS)
)

MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS = (
    "merchant-tabular-bolts-env-friendly-minimized-v0",
    "merchant-tabular-bolts-delivery-pacifist-minimized-v1",
)
MERCHANT_BOLT_SCENARIO_ID = "merchant/bolt-evaluation-v0"
SCENARIOS += (
    ScenarioSpecification(
        MERCHANT_BOLT_SCENARIO_ID,
        frozenset({MERCHANT_ENVIRONMENT_ID}),
        _monitor_factories(MERCHANT_MONITOR_IDS, make_builtin_monitor),
    ),
)
WRAPPERS += tuple(
    WrapperSpecification(
        merchant_bolts.MINIMIZED_WRAPPER_IDS[norm],
        "reward",
        partial(merchant_bolts.wrap_bolts, norm=norm, minimize=True),
        frozenset({MERCHANT_ENVIRONMENT_ID}),
        frozenset({"tabular"}),
    )
    for norm in merchant_bolts.RECIPES
)
EXPERIMENTS += tuple(
    ExperimentSpecification(
        identifier,
        MERCHANT_ENVIRONMENT_ID,
        ((MERCHANT_OBSERVATION_ID,) if norm == "env-friendly" else ()) + (merchant_bolts.MINIMIZED_WRAPPER_IDS[norm],),
        MERCHANT_BOLT_SCENARIO_ID,
        TABULAR_MERCHANT_ALGORITHM_ID,
        RESTRAINING_BOLT_TECHNIQUE_ID,
        _run(5_000_000, 150, training_reward_transform="task reward - configured norm punishments"),
    )
    for norm, identifier in zip(merchant_bolts.RECIPES, MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS, strict=True)
)

TAXI_BOLT_EXPERIMENT_ID = "taxi-tabular-bolts-emergency-50m-v1"
WRAPPERS += (
    WrapperSpecification(
        taxi_bolts.WRAPPER_ID,
        "reward",
        taxi_bolts.wrap_bolts,
        frozenset({TAXI_ENVIRONMENT_ID}),
        frozenset({"tabular"}),
    ),
    WrapperSpecification(
        taxi_bolts.WARN_WRAPPER_ID,
        "reward",
        partial(taxi_bolts.wrap_bolts, norm="warn"),
        frozenset({TAXI_ENVIRONMENT_ID}),
        frozenset({"tabular"}),
    ),
)
EXPERIMENTS += (
    ExperimentSpecification(
        TAXI_BOLT_EXPERIMENT_ID,
        TAXI_ENVIRONMENT_ID,
        (taxi_bolts.WRAPPER_ID,),
        TAXI_SCENARIO_ID,
        TABULAR_TAXI_ALGORITHM_ID,
        RESTRAINING_BOLT_TECHNIQUE_ID,
        replace(
            _run(50_000_000, 50, training_reward_transform="task reward - configured norm punishments"),
            intermediate_evaluation_frequency=250_000,
        ),
    ),
)

TAXI_WARN_BOLT_EXPERIMENT_ID = "taxi-tabular-bolts-warn-v5"
EXPERIMENTS += (
    replace(
        next(item for item in EXPERIMENTS if item.id == TAXI_BOLT_EXPERIMENT_ID),
        id=TAXI_WARN_BOLT_EXPERIMENT_ID,
        wrapper_ids=(taxi_bolts.WARN_WRAPPER_ID,),
        run=replace(
            next(item for item in EXPERIMENTS if item.id == TAXI_BOLT_EXPERIMENT_ID).run,
            training_steps=10_000_000,
        ),
    ),
)

GARDENER_BOLT_EXPERIMENT_IDS = (
    "gardener-dqn-bolts-collection-rescue-1m-v0",
    "gardener-ppo-bolts-collection-rescue-1m-v0",
)
GARDENER_BOLT_SCENARIO_ID = "gardener/collection-rescue-v0"
SCENARIOS += (
    ScenarioSpecification(
        GARDENER_BOLT_SCENARIO_ID,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        tuple(
            (norm_id, partial(make_gardener_monitor, norm_id, num_frogs=2, num_puddles=4))
            for norm_id in (COLLECT_ONE_NORM_ID, RESCUE_PER_FROG_NORM_ID, PERMISSION_AWARE_NORM_ID, DRAIN_NORM_ID)
        ),
    ),
)
WRAPPERS += (
    WrapperSpecification(
        gardener_bolts.WRAPPER_ID,
        "reward",
        gardener_bolts.wrap_bolts,
        frozenset({GARDENER_ENVIRONMENT_ID}),
        frozenset({"sb3"}),
    ),
)
ALGORITHMS += tuple(
    replace(
        next(item for item in ALGORITHMS if item.id == f"sb3/{algorithm}-default-bolts-v0"),
        id=f"sb3/{algorithm}-gardener-default-bolts-v0",
        environment_ids=frozenset({GARDENER_ENVIRONMENT_ID}),
    )
    for algorithm in ("dqn", "ppo")
)
EXPERIMENTS += tuple(
    ExperimentSpecification(
        identifier,
        GARDENER_ENVIRONMENT_ID,
        (GARDENER_FROG_OBSERVATION_ID, GARDENER_ACTION_ID, gardener_bolts.WRAPPER_ID),
        GARDENER_BOLT_SCENARIO_ID,
        f"sb3/{algorithm}-gardener-default-bolts-v0",
        RESTRAINING_BOLT_TECHNIQUE_ID,
        _run(
            1_000_000,
            1_000,
            training_reward_transform="task reward - illegal-action penalty - configured norm punishments",
        ),
    )
    for algorithm, identifier in zip(("dqn", "ppo"), GARDENER_BOLT_EXPERIMENT_IDS, strict=True)
)

REGISTRY = ExperimentRegistry(
    environments=ENVIRONMENTS,
    wrappers=WRAPPERS,
    scenarios=SCENARIOS,
    algorithms=ALGORITHMS,
    techniques=TECHNIQUES,
    experiments=EXPERIMENTS,
)


def get_experiment(experiment_id: str) -> ResolvedExperiment:
    """Resolve one supported experiment."""
    return REGISTRY.resolve(experiment_id)


__all__ = [
    "ALGORITHMS",
    "ENVIRONMENTS",
    "EXPERIMENTS",
    "GARDENER_BOLT_EXPERIMENT_IDS",
    "GARDENER_BOLT_SCENARIO_ID",
    "GARDENER_DRAIN_OFTEN_EXPERIMENT_ID",
    "GARDENER_EXPERIMENT_ID",
    "GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID",
    "GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID",
    "GARDENER_PERMISSION_DRAIN_SCENARIO_ID",
    "GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID",
    "GARDENER_PPO_EXPERIMENT_ID",
    "MERCHANT_BOLT_SCENARIO_ID",
    "MERCHANT_ENV_FRIENDLY_SCENARIO_ID",
    "MERCHANT_EXPERIMENT_ID",
    "MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS",
    "MERCHANT_POLICY_FIX_EXPERIMENT_ID",
    "PACMAN_BOLTS_ENVIRONMENT_ID",
    "PACMAN_BOLT_EXPERIMENT_IDS",
    "PACMAN_DQN_EXPERIMENT_ID",
    "PACMAN_EXPERIMENT_ID",
    "PACMAN_IMAGES_EXPERIMENT_ID",
    "PACMAN_OFTEN_EXPERIMENT_ID",
    "PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID",
    "PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID",
    "POLICY_FIX_TECHNIQUE_ID",
    "REGISTRY",
    "RESTRAINING_BOLT_TECHNIQUE_ID",
    "SCENARIOS",
    "TAXI_BOLT_EXPERIMENT_ID",
    "TAXI_EXPERIMENT_ID",
    "TAXI_POLICY_FIX_EXPERIMENT_ID",
    "TAXI_WARN_BOLT_EXPERIMENT_ID",
    "TECHNIQUES",
    "AlgorithmSpecification",
    "DuplicateSpecificationIDError",
    "EnvironmentSpecification",
    "ExperimentRegistry",
    "ExperimentSpecification",
    "GardenerOFTENSettings",
    "KeywordArguments",
    "OFTENSettings",
    "PacmanOFTENSettings",
    "ResolvedExperiment",
    "RunSettings",
    "ScenarioSpecification",
    "SpecificationError",
    "TechniqueSpecification",
    "UnknownSpecificationIDError",
    "UnsupportedCombinationError",
    "WrapperSpecification",
    "get_experiment",
]
