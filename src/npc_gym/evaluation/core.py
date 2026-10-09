"""Environment-neutral policy evaluation.

The evaluator uses only the ordinary Gymnasium API. Environments may expose
cumulative numeric domain measurements in ``info["episode_metrics"]``; taking
the final transition's snapshot makes those values survive an external
``TimeLimit`` wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from statistics import fmean, pstdev
from typing import Any, Protocol, TypeVar

import gymnasium as gym

from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.evaluation._recording import (
    MonitorRecorder,
    MonitorSource,
    TerminationClass,
    check_monitor_count_schema,
    copy_monitor_counts,
    episode_metrics,
    termination_class,
    validate_monitor_factories,
    validate_monitor_source,
)
from npc_gym.monitors import MonitorFactory

ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)
ActionT_co = TypeVar("ActionT_co", covariant=True)
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class Policy(Protocol[ObservationT_contra, ActionT_co]):
    """A policy callable receiving the current observation and public information."""

    def __call__(self, observation: ObservationT_contra, info: Mapping[str, Any], /) -> ActionT_co:
        """Choose an action without consulting private environment state."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Completed episode measurements, including final named monitor counts.

    Counts are nested by configured monitor and member name. Simple/complex
    roots expose one field, ``count``; collections expose members and derived
    counts. No implicit collection total is calculated.
    """

    episode: int
    episode_return: float
    length: int
    termination: TerminationClass
    metrics: Mapping[str, int | float]
    monitor_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "monitor_counts", copy_monitor_counts(self.monitor_counts))


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Episode results and population aggregates for one evaluation run."""

    episodes: tuple[EpisodeResult, ...]
    mean_return: float
    std_return: float
    mean_length: float
    std_length: float
    mean_metrics: Mapping[str, float]
    std_metrics: Mapping[str, float]
    mean_monitor_counts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    std_monitor_counts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @classmethod
    def from_episodes(cls, episodes: Sequence[EpisodeResult]) -> EvaluationSummary:
        """Build a summary, requiring one stable metric and monitor schema."""
        if not episodes:
            raise ValueError("An evaluation summary requires at least one episode")
        episode_tuple = tuple(episodes)
        metric_keys = _consistent_keys(episode_tuple, "metrics")
        mean_metrics, std_metrics = _mapping_stats(episode_tuple, "metrics", metric_keys)
        first_summaries = episode_tuple[0].monitor_counts
        for result in episode_tuple[1:]:
            check_monitor_count_schema(first_summaries, result.monitor_counts)
        mean_summaries: dict[str, dict[str, float]] = {}
        std_summaries: dict[str, dict[str, float]] = {}
        for monitor_id, fields in sorted(first_summaries.items()):
            mean_summaries[monitor_id] = {}
            std_summaries[monitor_id] = {}
            for key in sorted(fields):
                values = [result.monitor_counts[monitor_id][key] for result in episode_tuple]
                mean_summaries[monitor_id][key] = fmean(values)
                std_summaries[monitor_id][key] = pstdev(values)
        returns = [result.episode_return for result in episode_tuple]
        lengths = [result.length for result in episode_tuple]
        return cls(
            episodes=episode_tuple,
            mean_return=fmean(returns),
            std_return=pstdev(returns),
            mean_length=fmean(lengths),
            std_length=pstdev(lengths),
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            mean_monitor_counts=mean_summaries,
            std_monitor_counts=std_summaries,
        )


def evaluate(
    env: gym.Env[ObservationT, ActionT],
    policy: Policy[ObservationT, ActionT],
    *,
    episodes: int = 1,
    monitors: Mapping[str, MonitorFactory] | None = None,
    seed: int | None = None,
    monitor_source: MonitorSource = "factories",
) -> EvaluationSummary:
    """Evaluate ``policy`` on one ordinary Gymnasium environment.

    ``seed`` is applied to the first reset. Later resets continue that seeded
    environment's random stream. Factories are called once per run. Results
    snapshot final member and derived counts before the next reset.
    Set ``monitor_source="wrapper"`` to record ``info["monitors"]`` instead;
    ``monitors`` must then be omitted. Bolt diagnostics are never counted.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes < 1:
        raise ValueError(f"episodes must be a positive integer; got {episodes!r}")
    validate_monitor_source(monitor_source, monitors)
    recorder = MonitorRecorder(validate_monitor_factories(monitors if monitors is not None else {}), monitor_source)
    results: list[EpisodeResult] = []
    for episode in range(int(episodes)):
        observation, info = env.reset(seed=seed if episode == 0 else None)
        current_info = _validate_info(info, "reset")
        recorder.record(current_info, reset=True)

        episode_return = 0.0
        length = 0
        while True:
            action = policy(observation, current_info)
            observation, reward, terminated, truncated, info = env.step(action)
            current_info = _validate_info(info, "step")
            episode_return += float(reward)
            length += 1
            recorder.record(current_info, reset=False, terminated=bool(terminated), truncated=bool(truncated))
            if terminated or truncated:
                result = EpisodeResult(
                    episode=episode,
                    episode_return=episode_return,
                    length=length,
                    termination=termination_class(bool(terminated), bool(truncated)),
                    metrics=episode_metrics(current_info),
                    monitor_counts=recorder.counts,
                )
                if results:
                    check_monitor_count_schema(results[0].monitor_counts, result.monitor_counts)
                results.append(result)
                break
    return EvaluationSummary.from_episodes(results)


def _validate_info(info: Any, phase: str) -> Mapping[str, Any]:
    if not isinstance(info, Mapping):
        raise TypeError(f"Environment {phase} info must be a mapping, got {type(info).__name__}")
    return info


def _consistent_keys(episodes: tuple[EpisodeResult, ...], attribute: str) -> tuple[str, ...]:
    expected = set(getattr(episodes[0], attribute))
    for result in episodes[1:]:
        actual = set(getattr(result, attribute))
        if actual != expected:
            raise ValueError(f"Episode {result.episode} has inconsistent {attribute} keys")
    return tuple(sorted(expected))


def _mapping_stats(
    episodes: tuple[EpisodeResult, ...], attribute: str, keys: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for key in keys:
        values = [float(getattr(result, attribute)[key]) for result in episodes]
        means[key] = fmean(values)
        standard_deviations[key] = pstdev(values)
    return means, standard_deviations


__all__ = [
    "EPISODE_METRICS_KEY",
    "EpisodeResult",
    "EvaluationSummary",
    "Policy",
    "TerminationClass",
    "evaluate",
]
