"""Learn FrozenLake while monitoring a checkpoint-before-goal rule."""

import os
from contextlib import closing

import gymnasium as gym

from npc_gym.algorithms import TabularQLearning
from npc_gym.evaluation import evaluate
from npc_gym.labels import Transition
from npc_gym.monitors.automaton import AutomatonMonitor
from npc_gym.monitors.regex import from_regex
from npc_gym.wrappers import LabelingWrapper

CHECKPOINT_NORM_ID = "example/checkpoint-before-goal-v0"
TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "10000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
MAX_EPISODE_STEPS = int(os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", "100"))
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))


def label_transition(transition: Transition[int, int]) -> frozenset[str]:
    """Use public tile observations on the standard 4-by-4 FrozenLake map."""
    labels = set()
    if transition.state == 6:
        labels.add("checkpoint")
    if transition.state == 15 and transition.terminated:
        labels.add("goal")
    return frozenset(labels)


def make_checkpoint_monitor() -> AutomatonMonitor:
    """Accept traces ending at the goal with no earlier checkpoint visit."""
    return from_regex(
        "[!checkpoint]*[goal]",
        propositions={
            "checkpoint": lambda step: "checkpoint" in step.labels,
            "goal": lambda step: "goal" in step.labels,
        },
        reporting="prefix",
    )


def make_env() -> gym.Env[int, int]:
    """Label an imported environment outside its time limit to retain lifecycle flags."""
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, max_episode_steps=MAX_EPISODE_STEPS)
    return LabelingWrapper(env, label_transition)


def main() -> None:
    """Train on task rewards, then observe norm compliance during evaluation."""
    with closing(make_env()) as train_env, closing(make_env()) as eval_env:
        model = TabularQLearning(train_env, seed=SEED)
        model.learn(TRAINING_STEPS)
        summary = evaluate(
            eval_env,
            model,
            episodes=EVALUATION_EPISODES,
            monitors={CHECKPOINT_NORM_ID: make_checkpoint_monitor},
            seed=SEED + 10_000,
        )
        print(f"FrozenLake success rate (mean task return): {summary.mean_return:.2f}")
        print(f"Mean norm violations: {dict(summary.mean_monitor_counts)}")


if __name__ == "__main__":
    main()
