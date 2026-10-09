"""Paired Taxi and Merchant policy-fix evaluation for the experiment runner."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from time import perf_counter
from typing import Any, cast

import gymnasium as gym

from experiments.specifications import TAXI_ENVIRONMENT_ID, ResolvedExperiment
from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import MerchantEnv
from npc_gym.evaluation import EvaluationSummary, evaluate
from npc_gym.monitors import MonitorInput
from npc_gym.policy_fixes import ASPPlanner, FixedPolicy, MerchantModel, PlanningProblem, TaxiModel


class _WarningHistory(gym.Wrapper[Any, Any, Any, Any]):
    """Synchronize the Taxi model with real resets and executed transitions."""

    def __init__(self, env: gym.Env[Any, Any], model: TaxiModel) -> None:
        super().__init__(env)
        self.model = model

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.model.reset()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.model.advance(MonitorInput(info["labels"], terminated, truncated))
        return observation, float(reward), terminated, truncated, info


def evaluate_policy_fix(
    env: gym.Env[Any, Any],
    resolved: ResolvedExperiment,
    policy: TabularQLearning[Any],
    *,
    episodes: int,
    seed: int,
    fixed: bool = True,
) -> tuple[EvaluationSummary, dict[str, object]]:
    """Evaluate consecutive episode seeds without changing the learned Q-table.

    Taxi uses no action mask and fixes only Warn; Merchant uses the baseline
    mask and fixes Environment Friendly. Both prioritize violations, then rank.
    Timings include policy selection; evaluation time also includes environment
    steps and monitoring.
    """
    corrected = None
    if fixed:
        allowed_actions: Callable[[Any, Mapping[str, Any]], list[int]] | None = None
        if resolved.environment.id == TAXI_ENVIRONMENT_ID:
            taxi = TaxiModel()
            env = _WarningHistory(env, taxi)

            def problem(observation: Any, info: Mapping[str, Any]) -> PlanningProblem:
                return taxi.problem()
        else:
            merchant = MerchantModel()
            raw = cast(MerchantEnv, env.unwrapped)

            def problem(observation: Any, info: Mapping[str, Any]) -> PlanningProblem:
                return merchant.problem(raw.labeling_state())

            def allowed_actions(observation: Any, info: Mapping[str, Any]) -> list[int]:
                return [a for a, allowed in enumerate(info["action_mask"]) if allowed]

        corrected = FixedPolicy(
            ASPPlanner(),
            lambda observation, info: dict(enumerate(policy.q_values(observation))),
            problem,
            allowed_actions=allowed_actions,
        )
    interventions = fallbacks = decisions = 0
    planning_seconds = 0.0

    def choose(observation: Any, info: Mapping[str, Any]) -> int:
        nonlocal interventions, fallbacks, decisions, planning_seconds
        if corrected is None:
            return policy(observation, info)
        started = perf_counter()
        action = corrected(observation, info)
        planning_seconds += perf_counter() - started
        decision = corrected.last_decision
        assert decision is not None
        interventions += int(decision.changed)
        fallbacks += int(not decision.optimal)
        decisions += 1
        return action

    started = perf_counter()
    results = []
    for index in range(episodes):
        summary = evaluate(
            env, choose, episodes=1, monitors=dict(resolved.scenario.monitor_factories), seed=seed + index
        )
        results.append(replace(summary.episodes[0], episode=index))
    return EvaluationSummary.from_episodes(results), {
        "variant": "fixed" if fixed else "base",
        "interventions": interventions,
        "fallbacks": fallbacks,
        "decisions": decisions,
        "planning_seconds": planning_seconds,
        "evaluation_seconds": perf_counter() - started,
    }
