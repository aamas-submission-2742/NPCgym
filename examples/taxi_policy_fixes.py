"""Compare a learned Storm Taxi policy with and without ASP Warn fixes.

Install ``npc-gym[asp]`` to run this example.
"""

import os
from collections.abc import Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import StormTaxiEnv
from npc_gym.evaluation import EvaluationJSONWriter, EvaluationSummary, Policy, evaluate
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID, make_taxi_monitor
from npc_gym.policy_fixes import ASPPlanner, FixedPolicy, PlanningProblem, TaxiModel
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant

TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "1000000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
MAX_EPISODE_STEPS = int(os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", "50"))
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))


def make_env() -> gym.Env[int, int]:
    """Keep the baseline's compact observations, task rewards and episode limit."""
    return TimeLimit(IgnoreWeatherRelevant(StormTaxiEnv()), max_episode_steps=MAX_EPISODE_STEPS)


class PlanningTaxi(gym.Wrapper[int, int, int, int]):
    """Keep planning history synchronized with actual resets and transitions.

    ``policy`` proposes from the unchanged Q-table, fixes one action, and replans
    next time. The wrapper only tracks history; it never changes actions/rewards.
    ``interventions`` counts changed proposals across all evaluation episodes.
    """

    def __init__(self, env: gym.Env[int, int], learner: TabularQLearning[int], planner: ASPPlanner) -> None:
        super().__init__(env)
        self.model: TaxiModel | None = None
        self.interventions = 0
        self.policy = FixedPolicy(
            planner, lambda observation, info: dict(enumerate(learner.q_values(observation))), self.problem
        )

    def reset(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        if self.model is None:
            self.model = TaxiModel()
        else:
            self.model.reset()
        self.policy.last_decision = None
        return observation, info

    def problem(self, observation: int, info: Mapping[str, Any]) -> PlanningProblem:
        """Plan from the rain history of actual transitions."""
        if self.model is None:
            raise RuntimeError("Reset PlanningTaxi before choosing an action")
        return self.model.problem()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Reset PlanningTaxi before stepping")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.model.advance(MonitorInput(info["labels"], terminated, truncated))
        decision = self.policy.last_decision
        if decision is not None:
            self.interventions += int(decision.changed)
            self.policy.last_decision = None
        return observation, float(reward), terminated, truncated, info


def paired_evaluation(env: gym.Env[int, int], policy: Policy[int, int]) -> EvaluationSummary:
    """Reseed every episode so unequal lengths do not shift later reset samples."""
    results = []
    for episode in range(EVALUATION_EPISODES):
        summary = evaluate(
            env,
            policy,
            episodes=1,
            monitors={EMERGENCY_NORM_ID: lambda: make_taxi_monitor(EMERGENCY_NORM_ID)},
            seed=SEED + 10_000 + episode,
        )
        results.append(replace(summary.episodes[0], episode=episode))
    return EvaluationSummary.from_episodes(results)


def main() -> dict[str, EvaluationSummary]:
    """Train once, reload once, compare paired episodes; optionally retain JSON.

    ``NPC_GYM_OUTPUT_DIR`` must name a new directory. Saved policies are always
    temporary. JSON uses the existing evaluation writer and includes budgets,
    settings, intervention counts and wall times; no baseline artifact is edited.
    """
    for name, value in (
        ("NPC_GYM_TRAINING_STEPS", TRAINING_STEPS),
        ("NPC_GYM_EVALUATION_EPISODES", EVALUATION_EPISODES),
        ("NPC_GYM_MAX_EPISODE_STEPS", MAX_EPISODE_STEPS),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if SEED < 0:
        raise ValueError("NPC_GYM_SEED must be non-negative")
    planner = ASPPlanner()  # Fail before training if the optional asp extra is missing.
    output = Path(os.environ["NPC_GYM_OUTPUT_DIR"]) if os.environ.get("NPC_GYM_OUTPUT_DIR") else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    with closing(make_env()) as training_env, closing(make_env()) as base_env:
        learner = TabularQLearning(training_env, seed=SEED)
        started = perf_counter()
        learner.learn(TRAINING_STEPS)
        training_seconds = perf_counter() - started
        with TemporaryDirectory(prefix="npc-gym-taxi-fixes-") as directory:
            path = Path(directory) / "base.zip"
            learner.save(path)
            policy = TabularQLearning.load(path, env=base_env)
            with closing(PlanningTaxi(make_env(), policy, planner)) as fixed_env:
                for variant, env, action_policy in (
                    ("base", base_env, policy),
                    ("fixed", fixed_env, fixed_env.policy),
                ):
                    started = perf_counter()
                    summary = paired_evaluation(env, action_policy)
                    evaluation_seconds = perf_counter() - started
                    interventions = fixed_env.interventions if variant == "fixed" else 0
                    transitions = sum(episode.length for episode in summary.episodes)
                    metadata = {
                        "example": "taxi_policy_fixes",
                        "variant": variant,
                        "training_seed": SEED,
                        "evaluation_seeds": list(range(SEED + 10_000, SEED + 10_000 + EVALUATION_EPISODES)),
                        "training_steps": policy.num_timesteps,
                        "teaching_steps": 0,
                        "evaluation_steps": transitions,
                        "max_episode_steps": MAX_EPISODE_STEPS,
                        "horizon": 1,
                        "norm_id": EMERGENCY_NORM_ID,
                        "norm_component": "Warn Violations",
                        "interventions": interventions,
                        "training_seconds": training_seconds,
                        "evaluation_seconds": evaluation_seconds,
                    }
                    print(f"{variant}: return={summary.mean_return:.2f}, length={summary.mean_length:.2f}")
                    print(f"  Task outcomes: {dict(summary.mean_metrics)}")
                    print(f"  Norm counts: {dict(summary.mean_monitor_counts)}")
                    print(f"  Interventions: {interventions}/{transitions}")
                    print(f"  Training: {policy.num_timesteps} steps, {training_seconds:.2f}s (shared base)")
                    print(f"  Evaluation: {transitions} steps, {evaluation_seconds:.2f}s; teaching: 0 steps")
                    if output is not None:
                        with EvaluationJSONWriter(output / f"{variant}.json") as writer:
                            writer.write(summary, metadata=metadata)
                    summaries[variant] = summary
    return summaries


if __name__ == "__main__":
    main()
