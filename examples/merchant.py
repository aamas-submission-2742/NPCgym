"""Train, persist, and evaluate masked tabular Q-learning on Merchant."""

import logging
import os
from contextlib import closing
from tempfile import TemporaryDirectory
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import MerchantEnv
from npc_gym.evaluation import EvaluationSummary, evaluate
from npc_gym.monitors.builtins import make_builtin_monitor
from npc_gym.monitors.merchant_monitors import DELIVERY_NORM_ID
from npc_gym.wrappers.merchant_wrappers import IgnoreTimeObservation

TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "5000000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
MAX_EPISODE_STEPS = int(os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", "150"))
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))


def make_env() -> gym.Env[Any, int]:
    """Create one headless Merchant environment with the baseline observation view."""
    env: gym.Env[Any, int] = MerchantEnv(layout="basic", risk_fight=0.75, risk_death=0.25, capacity=5, sunset=28)
    env = IgnoreTimeObservation(env)
    return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


def main() -> tuple[EvaluationSummary, int]:
    """Run the example; return evaluation and actual steps after cleaning up models."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with closing(make_env()) as train_env, closing(make_env()) as eval_env:
        model = TabularQLearning(
            train_env,
            learning_rate=0.5,
            exploration_final_eps=0.2,
            seed=SEED,
            use_action_mask=True,
        )
        model.learn(TRAINING_STEPS)

        with TemporaryDirectory(prefix="npc-gym-merchant-") as directory:
            model_path = f"{directory}/merchant-model.zip"
            model.save(model_path)
            loaded_model = TabularQLearning.load(model_path, env=eval_env)
            summary = evaluate(
                eval_env,
                loaded_model,
                episodes=EVALUATION_EPISODES,
                monitors={DELIVERY_NORM_ID: lambda: make_builtin_monitor(DELIVERY_NORM_ID)},
                seed=SEED + 10_000,
            )

        print(f"Merchant mean task return: {summary.mean_return:.2f}")
        print(f"Mean task outcomes: {dict(summary.mean_metrics)}")
        print(f"Mean monitor measurements: {dict(summary.mean_monitor_counts)}")
        return summary, loaded_model.num_timesteps


if __name__ == "__main__":
    main()
