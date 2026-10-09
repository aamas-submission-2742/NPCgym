"""Train, persist, and evaluate Gardener with DQN (default) or PPO.

Install ``npc-gym[sb3]``; select ``NPC_GYM_ALGORITHM=dqn|ppo``.
"""

import os
from contextlib import closing
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO

from npc_gym.envs import GardenerEnv
from npc_gym.evaluation import EvaluationSummary, evaluate
from npc_gym.integrations.sb3 import SB3Policy
from npc_gym.monitors.builtins import make_builtin_monitor
from npc_gym.monitors.gardener_monitors import COLLECT_ONE_NORM_ID
from npc_gym.wrappers.gardener_wrappers import IllegalActionPenaltyWrapper, StateFeatureObsWrapper

ALGORITHM = os.environ.get("NPC_GYM_ALGORITHM", "dqn")
TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "100000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
MAX_EPISODE_STEPS = int(os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", "1000"))
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))


def make_env(*, training: bool) -> gym.Env[Any, int]:
    """Create one headless Gardener environment with task-appropriate wrappers."""
    env: gym.Env[Any, int] = StateFeatureObsWrapper(GardenerEnv(size=15, score_limit=300))
    env = IllegalActionPenaltyWrapper(env, penalty=-1.0 if training else 0.0)
    return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


def main() -> tuple[EvaluationSummary, int]:
    """Train, save to a new file, reload, and evaluate without the training penalty."""
    if ALGORITHM not in ("dqn", "ppo"):
        raise ValueError("NPC_GYM_ALGORITHM must be dqn or ppo")
    path = Path(os.environ.get("NPC_GYM_MODEL_PATH", f"gardener-{ALGORITHM}.zip"))
    if path.suffix != ".zip":
        raise ValueError("NPC_GYM_MODEL_PATH must end in .zip")
    if path.exists():
        raise FileExistsError(path)
    # Short budgets also shorten PPO rollouts so smoke runs perform updates.
    ppo_options: dict[str, Any] = (
        {"n_steps": TRAINING_STEPS, "batch_size": TRAINING_STEPS} if TRAINING_STEPS < 2048 else {}
    )
    with closing(make_env(training=True)) as train_env, closing(make_env(training=False)) as eval_env:
        model = (
            DQN(
                "MlpPolicy",
                train_env,
                seed=SEED,
                device="cpu",
                learning_rate=5e-4,
                learning_starts=min(5_000, TRAINING_STEPS // 2),
                buffer_size=50_000,
                batch_size=128,
                target_update_interval=2_000,
                exploration_fraction=0.3,
                policy_kwargs={"net_arch": [32, 32]},
                verbose=1,
            )
            if ALGORITHM == "dqn"
            else PPO("MlpPolicy", train_env, seed=SEED, device="cpu", verbose=1, **ppo_options)
        )
        model.learn(total_timesteps=TRAINING_STEPS, log_interval=100)

        model.save(path)
        loaded_model = type(model).load(path, env=eval_env, device="cpu")
        summary = evaluate(
            eval_env,
            SB3Policy(loaded_model, deterministic=True),
            episodes=EVALUATION_EPISODES,
            monitors={COLLECT_ONE_NORM_ID: lambda: make_builtin_monitor(COLLECT_ONE_NORM_ID)},
            seed=SEED + 10_000,
        )
        print(f"Saved {path}")
        print(f"Gardener mean task return: {summary.mean_return:.2f}")
        print(f"Mean task outcomes: {dict(summary.mean_metrics)}")
        print(f"Mean monitor measurements: {dict(summary.mean_monitor_counts)}")
        return summary, loaded_model.num_timesteps


if __name__ == "__main__":
    main()
