"""Stable-Baselines3 policy and evaluation callback adapters."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from npc_gym.evaluation import EpisodeResult, EvaluationSummary, TerminationClass
from npc_gym.evaluation._recording import (
    MonitorRecorder,
    MonitorSource,
    check_monitor_count_schema,
    episode_metrics,
    termination_class,
    validate_monitor_factories,
    validate_monitor_source,
)
from npc_gym.monitors import MonitorFactory

_IntermediateEvaluator = Callable[[BaseAlgorithm], EvaluationSummary]
_EvaluationHook = Callable[[int, EvaluationSummary], None]


@dataclass(frozen=True, slots=True)
class IntermediateEvaluation:
    """One evaluation, all thresholds it covers, and its actual transition count."""

    thresholds: tuple[int, ...]
    timesteps: int
    summary: EvaluationSummary


class SB3Policy:
    """Expose an SB3 model's ``predict`` method as an evaluation policy.

    SB3 policies choose from the whole action space, so the public ``info`` is
    ignored and an ``info["action_mask"]`` does not reach the model.
    """

    def __init__(self, model: BaseAlgorithm, *, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def __call__(self, observation: Any, info: Mapping[str, Any], /) -> Any:
        """Predict one action without changing its scalar or array shape."""
        del info
        action, _state = self.model.predict(observation, deterministic=self.deterministic)
        return action


@dataclass(slots=True)
class _EpisodeAccumulator:
    recorder: MonitorRecorder
    episode_return: float = 0.0
    length: int = 0

    def start(self, reset_info: Mapping[str, Any]) -> None:
        self.episode_return = 0.0
        self.length = 0
        self.recorder.record(reset_info, reset=True)

    def update(self, reward: Any, info: Mapping[str, Any], *, terminated: bool, truncated: bool) -> None:
        try:
            self.episode_return += float(reward)
        except (TypeError, ValueError) as error:
            raise TypeError(f"SB3 rewards must contain real numbers; got {reward!r}") from error
        self.length += 1
        self.recorder.record(info, reset=False, terminated=terminated, truncated=truncated)

    def finish(self, episode: int, info: Mapping[str, Any], termination: TerminationClass) -> EpisodeResult:
        return EpisodeResult(
            episode=episode,
            episode_return=self.episode_return,
            length=self.length,
            termination=termination,
            metrics=episode_metrics(info),
            monitor_counts=self.recorder.counts,
        )


class SB3EvaluationCallback(BaseCallback):
    """Collect training episodes and run isolated, environment-neutral evaluations.

    Training episode state is reconstructed from vector ``rewards``, ``dones``,
    terminal ``infos``, and the reset information retained by SB3's underlying
    vector environment. By default one factory-owned monitor set is maintained
    per vector slot. ``monitor_source="wrapper"`` instead consumes snapshots
    from MonitorWrapper inside the vector boundary; omit ``monitors`` in that
    mode. Reset events and terminal snapshots are counted once. Returns
    total the rewards SB3 reports, so a reward-transforming vector wrapper such
    as ``VecNormalize`` is reflected in ``episode_return``.

    SB3 records a step that both terminates and truncates as termination alone,
    so such an episode is classed :attr:`TerminationClass.TERMINATED` here while
    :func:`~npc_gym.evaluation.evaluate` classes it as both.

    One callback records one ``learn()`` call. Training-episode collection must
    start at timestep zero because a callback attached to a resumed run cannot
    reconstruct an episode already in flight. A fresh callback may observe a
    resumed call only when ``collect_training_episodes=False``.

    ``evaluation_frequency`` counts individual environment transitions. When a
    fresh training call starts at zero, the untrained policy is evaluated once.
    A resumed call starts at the next future threshold. When a vector step crosses
    one or more thresholds, the evaluator and hook run once, and the resulting
    ``IntermediateEvaluation.thresholds`` records every crossed threshold.
    The evaluator owns its evaluation environment
    and returns an :class:`~npc_gym.evaluation.EvaluationSummary`; its episodes
    are never included in ``training_episodes``. Python, NumPy, and CPU Torch
    random state, the model's action-space generator, and every policy-module
    training mode are restored after the evaluator and hook, including when
    either raises. Stochastic DQN prediction samples exploration actions from
    that generator, which training shares. This isolation is intended for seeded
    CPU PPO and DQN runs; arbitrary evaluator mutations to the model are outside
    the contract.
    """

    def __init__(
        self,
        monitors: Mapping[str, MonitorFactory] | None = None,
        *,
        collect_training_episodes: bool = True,
        monitor_source: MonitorSource = "factories",
        evaluator: _IntermediateEvaluator | None = None,
        evaluation_frequency: int | None = None,
        on_evaluation: _EvaluationHook | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        validate_monitor_source(monitor_source, monitors)
        self._monitor_source = monitor_source
        self._monitor_factories = validate_monitor_factories(monitors if monitors is not None else {})
        if not isinstance(collect_training_episodes, bool):
            raise TypeError("collect_training_episodes must be a bool")
        self.collect_training_episodes = collect_training_episodes
        if evaluator is not None and not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.evaluator = evaluator
        self.evaluation_frequency = _validate_frequency(evaluation_frequency)
        self.on_evaluation = on_evaluation
        if (self.evaluator is None) != (self.evaluation_frequency is None):
            raise ValueError("evaluator and evaluation_frequency must be provided together")
        if self.on_evaluation is not None:
            if self.evaluator is None:
                raise ValueError("on_evaluation requires an evaluator")
            if not callable(self.on_evaluation):
                raise TypeError("on_evaluation must be callable")

        self._started = False
        self._episodes: list[EpisodeResult] = []
        self._evaluations: list[IntermediateEvaluation] = []
        self._accumulators: list[_EpisodeAccumulator] = []
        self._next_evaluation: int | None = None

    @property
    def training_episodes(self) -> tuple[EpisodeResult, ...]:
        """Completed training episodes in vector-step and slot order."""
        return tuple(self._episodes)

    @property
    def evaluations(self) -> tuple[IntermediateEvaluation, ...]:
        """Evaluations in actual-timestep order, including zero for fresh training."""
        return tuple(self._evaluations)

    def training_summary(self) -> EvaluationSummary:
        """Aggregate completed training episodes collected by this callback."""
        if not self.collect_training_episodes:
            raise RuntimeError("Training episode aggregation is disabled")
        return EvaluationSummary.from_episodes(self._episodes)

    def _on_training_start(self) -> None:
        if self._started:
            raise RuntimeError(
                "This SB3EvaluationCallback has already recorded a learn() call; callbacks are single-use"
            )
        if self.collect_training_episodes and self.num_timesteps != 0:
            raise RuntimeError(
                "Training episode collection must start at timestep zero; "
                "use collect_training_episodes=False for a resumed learn() call"
            )
        self._started = True
        if self.collect_training_episodes:
            reset_infos = _reset_infos(self.training_env)
            self._accumulators = [
                _EpisodeAccumulator(MonitorRecorder(self._monitor_factories, self._monitor_source))
                for _ in range(self.training_env.num_envs)
            ]
            for accumulator, reset_info in zip(self._accumulators, reset_infos, strict=True):
                accumulator.start(reset_info)

        if self.evaluation_frequency is not None:
            start = int(self.num_timesteps)
            self._next_evaluation = (
                0 if start == 0 else (start // self.evaluation_frequency + 1) * self.evaluation_frequency
            )
            self._run_due_evaluations()
        else:
            self._next_evaluation = None

    def _on_step(self) -> bool:
        if self.collect_training_episodes:
            self._consume_training_step()
        self._run_due_evaluations()
        return True

    def _consume_training_step(self) -> None:
        infos = _vector_values(self.locals.get("infos"), "infos", self.training_env.num_envs)
        rewards = _vector_values(self.locals.get("rewards"), "rewards", self.training_env.num_envs)
        dones = _vector_values(self.locals.get("dones"), "dones", self.training_env.num_envs)
        reset_infos = _reset_infos(self.training_env)

        for index, (raw_info, reward, done) in enumerate(zip(infos, rewards, dones, strict=True)):
            info = _info_mapping(raw_info, "step")
            is_done = bool(done)
            terminated, truncated = _terminal_flags(info, is_done)
            accumulator = self._accumulators[index]
            accumulator.update(reward, info, terminated=terminated, truncated=truncated)
            if is_done:
                result = accumulator.finish(len(self._episodes), info, termination_class(terminated, truncated))
                if self._episodes:
                    check_monitor_count_schema(self._episodes[0].monitor_counts, result.monitor_counts)
                self._episodes.append(result)
                accumulator.start(reset_infos[index])

    def _run_due_evaluations(self) -> None:
        if self._next_evaluation is None:
            return
        assert self.evaluator is not None
        assert self.evaluation_frequency is not None
        if self.num_timesteps < self._next_evaluation:
            return
        timesteps = int(self.num_timesteps)
        thresholds = tuple(range(self._next_evaluation, timesteps + 1, self.evaluation_frequency))
        with _preserve_training_state(self.model):
            summary = self.evaluator(self.model)
            if not isinstance(summary, EvaluationSummary):
                raise TypeError(f"evaluator must return EvaluationSummary, got {type(summary).__name__}")
            self._evaluations.append(
                IntermediateEvaluation(thresholds=thresholds, timesteps=timesteps, summary=summary)
            )
            if self.on_evaluation is not None:
                self.on_evaluation(timesteps, summary)
        self._next_evaluation = thresholds[-1] + self.evaluation_frequency


@contextmanager
def _preserve_training_state(model: BaseAlgorithm) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    action_space = getattr(model, "action_space", None)
    action_generator = action_space.np_random.bit_generator if isinstance(action_space, gym.Space) else None
    action_generator_state = action_generator.state if action_generator is not None else {}
    policy = getattr(model, "policy", None)
    module_modes = (
        tuple((module, module.training) for module in policy.modules()) if isinstance(policy, torch.nn.Module) else ()
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if action_generator is not None:
            action_generator.state = action_generator_state
        for module, training in module_modes:
            module.training = training


def _validate_frequency(frequency: int | None) -> int | None:
    if frequency is None:
        return None
    if isinstance(frequency, bool) or not isinstance(frequency, Integral) or frequency < 1:
        raise ValueError(f"evaluation_frequency must be a positive integer; got {frequency!r}")
    return int(frequency)


def _reset_infos(env: VecEnv) -> list[Mapping[str, Any]]:
    source = env
    while isinstance(source, VecEnvWrapper):
        source = source.venv
    if len(source.reset_infos) != env.num_envs:
        raise RuntimeError("SB3 vector environment has an inconsistent reset_infos length")
    return [_info_mapping(info, "reset") for info in source.reset_infos]


def _vector_values(value: Any, name: str, expected: int) -> Sequence[Any]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"SB3 callback locals must provide vector {name!r}")
    values = list(value)
    if len(values) != expected:
        raise RuntimeError(f"SB3 callback local {name!r} has {len(values)} entries; expected {expected}")
    return values


def _info_mapping(info: Any, phase: str) -> Mapping[str, Any]:
    if not isinstance(info, Mapping):
        raise TypeError(f"SB3 {phase} info must be a mapping, got {type(info).__name__}")
    return info


def _terminal_flags(info: Mapping[str, Any], done: bool) -> tuple[bool, bool]:
    if not done:
        return False, False
    if "terminated" in info or "truncated" in info:
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if terminated or truncated:
            return terminated, truncated
    truncated = bool(info.get("TimeLimit.truncated", False))
    return not truncated, truncated


__all__ = ["IntermediateEvaluation", "SB3EvaluationCallback", "SB3Policy"]
