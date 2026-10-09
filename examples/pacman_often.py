"""Teach a saved Pacman DQN with OFTEN and compare with ordinary continued DQN.

Install ``npc-gym[asp,sb3]`` and first run ``examples/pacman.py`` with DQN.
Training uses rewards divided by 100; evaluation reports raw task rewards.
"""

import os
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import pacman as task
import torch
from stable_baselines3 import DQN

from npc_gym.algorithms import TeachingSession
from npc_gym.envs import PacmanEnv
from npc_gym.evaluation import EvaluationSummary, evaluate
from npc_gym.integrations.sb3 import DQNOFTEN, SB3Policy
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.builtins import make_builtin_monitor
from npc_gym.monitors.pacman_monitors import VEGAN_NORM_ID
from npc_gym.policy_fixes import ASPPlanner, PacmanModel, PlanningProblem

TEACHING_STEPS = int(os.environ.get("NPC_GYM_TEACHING_STEPS", "5000000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "1000"))
HORIZON = int(os.environ.get("NPC_GYM_PLANNING_HORIZON", "1"))


def sessions(env: gym.Env[Any, int]) -> Callable[[Any, Mapping[str, Any]], TeachingSession]:
    """Reuse a planner per stream; reset the episode's model and remaining steps."""
    planner = ASPPlanner()
    raw = cast(PacmanEnv, env.unwrapped)

    def start(observation: Any, info: Mapping[str, Any]) -> TeachingSession:
        model = PacmanModel(raw.labeling_state(), horizon=HORIZON)
        remaining = task.MAX_EPISODE_STEPS

        def problem(observation: Any, info: Mapping[str, Any]) -> PlanningProblem:
            return model.problem(raw.labeling_state(), remaining_steps=remaining)

        def advance(actual: MonitorInput) -> None:
            nonlocal remaining
            remaining -= 1

        return TeachingSession(planner, problem, advance)

    return start


def assess(model: DQN) -> EvaluationSummary:
    """Evaluate each variant with the same episode seeds and no planning."""
    episodes = []
    with closing(task.make_env()) as env:
        for index in range(EVALUATION_EPISODES):
            summary = evaluate(
                env,
                SB3Policy(model, deterministic=True),
                episodes=1,
                seed=50_000 + index,
                monitors={VEGAN_NORM_ID: lambda: make_builtin_monitor(VEGAN_NORM_ID)},
            )
            episodes.append(replace(summary.episodes[0], episode=index))
    return EvaluationSummary.from_episodes(episodes)


def main() -> dict[str, EvaluationSummary]:
    """Load one base policy, then independently teach or continue its task learning."""
    path = Path(os.environ.get("NPC_GYM_MODEL_PATH", "pacman-dqn.zip"))
    smoke = os.environ.get("NPC_GYM_SMOKE", "0")
    if smoke not in ("0", "1"):
        raise ValueError("NPC_GYM_SMOKE must be 0 or 1")
    if not path.is_file() and (smoke != "1" or "NPC_GYM_MODEL_PATH" in os.environ):
        raise FileNotFoundError(f"Missing {path}; first train a DQN with examples/pacman.py")
    if min(TEACHING_STEPS, HORIZON, EVALUATION_EPISODES, task.MAX_EPISODE_STEPS) < 1:
        raise ValueError("Training, evaluation, episode and planning budgets must be positive")
    torch.set_num_threads(1)
    with closing(task.make_env(training=True)) as ordinary, closing(task.make_env(training=True)) as expert:

        def load_model() -> DQN:
            return (
                DQN.load(path, env=ordinary, device="cpu")
                if path.is_file()
                else DQN("MlpPolicy", ordinary, gamma=0.95, seed=task.SEED, device="cpu")
            )

        model = load_model()
        summaries = {"base": assess(model)}
        teacher = DQNOFTEN(
            model,
            ordinary,
            expert,
            ordinary_session=sessions(ordinary),
            expert_session=sessions(expert),
            margin=0.5,
            seed=task.SEED + 20_000,
        ).learn(TEACHING_STEPS, schedule_timesteps=max(5_000_000, TEACHING_STEPS))
        summaries["often"] = assess(model)
        print(f"OFTEN: {teacher.stats.ordinary_steps} ordinary + {teacher.stats.expert_steps} expert steps")
        print(f"Teaching statistics: {teacher.stats}")
        model = load_model()
        model.set_random_seed(task.SEED + 20_000)
        model.learn(TEACHING_STEPS)
        summaries["continued"] = assess(model)
    for name, summary in summaries.items():
        print(
            f"{name}: return={summary.mean_return:.2f}; length={summary.mean_length:.2f}; task={dict(summary.mean_metrics)}"
        )
        print(f"  Episode endings: {dict(Counter(episode.termination.value for episode in summary.episodes))}")
        print(f"  Norm counts: {dict(summary.mean_monitor_counts)}")
    return summaries


if __name__ == "__main__":
    main()
