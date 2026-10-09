"""Train, persist, and evaluate Stable-Baselines3 DQN (default) or PPO on Pacman.

Install the optional dependency with ``pip install 'npc-gym[sb3]'``.
"""

import os
from contextlib import closing
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from gymnasium.wrappers import TimeLimit, TransformReward
from stable_baselines3 import DQN, PPO

from npc_gym.envs import PacmanEnv, PacmanLayout
from npc_gym.evaluation import EvaluationSummary, evaluate
from npc_gym.integrations.sb3 import SB3Policy
from npc_gym.monitors.builtins import make_builtin_monitor
from npc_gym.monitors.pacman_monitors import VEGAN_NORM_ID

ALGORITHM = os.environ.get("NPC_GYM_ALGORITHM", "dqn")
TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "5000000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
LAYOUT = os.environ.get("NPC_GYM_LAYOUT", "small")
MAX_EPISODE_STEPS = int(
    os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", str(PacmanLayout.bundled(LAYOUT).default_episode_steps))
)
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))


def make_env(*, training: bool = False) -> gym.Env[Any, int]:
    """Use rewards divided by 100 for training; report raw rewards in evaluation."""
    env: gym.Env[Any, int] = TimeLimit(
        PacmanEnv(layout=LAYOUT, features="complete"), max_episode_steps=MAX_EPISODE_STEPS
    )
    return TransformReward(env, lambda reward: float(reward) / 100) if training else env


def main() -> tuple[EvaluationSummary, int]:
    """Train, save to a new file, reload, and evaluate on unscaled task rewards."""
    if ALGORITHM not in ("dqn", "ppo"):
        raise ValueError("NPC_GYM_ALGORITHM must be dqn or ppo")
    path = Path(os.environ.get("NPC_GYM_MODEL_PATH", f"pacman-{ALGORITHM}.zip"))
    if path.suffix != ".zip":
        raise ValueError("NPC_GYM_MODEL_PATH must end in .zip")
    if path.exists():
        raise FileExistsError(path)
    # Short budgets also shorten PPO rollouts so smoke runs perform updates.
    ppo_options: dict[str, Any] = (
        {"n_steps": TRAINING_STEPS, "batch_size": TRAINING_STEPS} if TRAINING_STEPS < 2048 else {}
    )
    torch.set_num_threads(1)
    with closing(make_env(training=ALGORITHM == "dqn")) as train_env, closing(make_env()) as eval_env:
        model = (
            DQN("MlpPolicy", train_env, gamma=0.95, seed=SEED, device="cpu", verbose=1)
            if ALGORITHM == "dqn"
            else PPO("MlpPolicy", train_env, seed=SEED, device="cpu", verbose=1, **ppo_options)
        )
        model.learn(total_timesteps=TRAINING_STEPS, log_interval=1)
        model.save(path)
        loaded_model = type(model).load(path, env=eval_env, device="cpu")
        summary = evaluate(
            eval_env,
            SB3Policy(loaded_model, deterministic=True),
            episodes=EVALUATION_EPISODES,
            monitors={VEGAN_NORM_ID: lambda: make_builtin_monitor(VEGAN_NORM_ID)},
            seed=SEED + 10_000,
        )
        print(f"Saved {path}")
        print(f"Pacman mean task return: {summary.mean_return:.2f}")
        print(f"Mean task outcomes: {dict(summary.mean_metrics)}")
        print(f"Mean monitor measurements: {dict(summary.mean_monitor_counts)}")
        return summary, loaded_model.num_timesteps


if __name__ == "__main__":
    main()
