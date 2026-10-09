"""Compare a learned Merchant policy with Environment Friendly fixes; requires asp."""

import os
from collections.abc import Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, cast

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import MerchantEnv
from npc_gym.envs.merchant.labels import MerchantState
from npc_gym.evaluation import EvaluationJSONWriter, EvaluationSummary, Policy, evaluate
from npc_gym.monitors.merchant_monitors import ENV_FRIENDLY_NORM_ID, make_merchant_monitor
from npc_gym.policy_fixes import ASPPlanner, FixedPolicy, MerchantModel
from npc_gym.wrappers.merchant_wrappers import IgnoreTimeObservation

TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "1000000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
MAX_EPISODE_STEPS = int(os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", "150"))
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))


def make_env() -> gym.Env[MerchantState, int]:
    """Keep the baseline's compact observations, task rewards and episode limit."""
    env = MerchantEnv(layout="basic", risk_fight=0.75, risk_death=0.25, capacity=5, sunset=28)
    return TimeLimit(IgnoreTimeObservation(env), max_episode_steps=MAX_EPISODE_STEPS)


def paired_evaluation(env: gym.Env[MerchantState, int], policy: Policy[MerchantState, int]) -> EvaluationSummary:
    """Reseed every episode so unequal lengths do not shift later reset samples."""
    results = []
    for episode in range(EVALUATION_EPISODES):
        summary = evaluate(
            env,
            policy,
            episodes=1,
            monitors={ENV_FRIENDLY_NORM_ID: lambda: make_merchant_monitor(ENV_FRIENDLY_NORM_ID)},
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
        learner = TabularQLearning(
            training_env,
            learning_rate=0.5,
            exploration_final_eps=0.2,
            seed=SEED,
            use_action_mask=True,
        )
        started = perf_counter()
        learner.learn(TRAINING_STEPS)
        training_seconds = perf_counter() - started
        with TemporaryDirectory(prefix="npc-gym-merchant-fixes-") as directory:
            path = Path(directory) / "base.zip"
            learner.save(path)
            policy = TabularQLearning.load(path, env=base_env)
            with closing(make_env()) as fixed_env:
                base = cast(MerchantEnv, fixed_env.unwrapped)
                model = MerchantModel()
                fixed = FixedPolicy(
                    planner,
                    lambda observation, info: dict(enumerate(policy.q_values(observation))),
                    lambda observation, info: model.problem(base.labeling_state()),
                    allowed_actions=lambda observation, info: [
                        action for action, allowed in enumerate(info["action_mask"]) if allowed
                    ],
                )
                interventions = 0

                def fixed_policy(observation: MerchantState, info: Mapping[str, Any]) -> int:
                    nonlocal interventions
                    action = fixed(observation, info)
                    assert fixed.last_decision is not None
                    interventions += int(fixed.last_decision.changed)
                    return action

                for variant, env, action_policy in (
                    ("base", base_env, policy),
                    ("fixed", fixed_env, fixed_policy),
                ):
                    started = perf_counter()
                    summary = paired_evaluation(env, action_policy)
                    evaluation_seconds = perf_counter() - started
                    changed = interventions if variant == "fixed" else 0
                    transitions = sum(episode.length for episode in summary.episodes)
                    metadata = {
                        "example": "merchant_policy_fixes",
                        "variant": variant,
                        "training_seed": SEED,
                        "evaluation_seeds": list(range(SEED + 10_000, SEED + 10_000 + EVALUATION_EPISODES)),
                        "training_steps": policy.num_timesteps,
                        "teaching_steps": 0,
                        "evaluation_steps": transitions,
                        "max_episode_steps": MAX_EPISODE_STEPS,
                        "horizon": 1,
                        "layout": base.layout,
                        "risk_fight": base.risk_fight,
                        "risk_death": base.risk_death,
                        "use_action_mask": True,
                        "norm_id": ENV_FRIENDLY_NORM_ID,
                        "interventions": changed,
                        "training_seconds": training_seconds,
                        "evaluation_seconds": evaluation_seconds,
                    }
                    print(f"{variant}: return={summary.mean_return:.2f}, length={summary.mean_length:.2f}")
                    print(f"  Task outcomes: {dict(summary.mean_metrics)}")
                    print(f"  Norm counts: {dict(summary.mean_monitor_counts)}")
                    print(f"  Interventions: {changed}/{transitions}")
                    print(f"  Training: {policy.num_timesteps} steps, {training_seconds:.2f}s (shared base)")
                    print(f"  Evaluation: {transitions} steps, {evaluation_seconds:.2f}s; teaching: 0 steps")
                    if output is not None:
                        with EvaluationJSONWriter(output / f"{variant}.json") as writer:
                            writer.write(summary, metadata=metadata)
                    summaries[variant] = summary
    return summaries


if __name__ == "__main__":
    main()
