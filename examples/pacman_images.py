"""DQN with pixels through a CNN and shipped Vegan bolts beside it.

Install ``npc-gym[render,sb3]``. Small runs check execution only.
"""

import os
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from tempfile import TemporaryDirectory
from typing import Any

import gymnasium as gym
import torch
from gymnasium.wrappers import TimeLimit, TransformReward
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from npc_gym.bolts import BoltSpec, make_builtin_bolts
from npc_gym.envs import PacmanEnv, PacmanLayout
from npc_gym.monitors.pacman_monitors import VEGAN_NORM_ID
from npc_gym.wrappers import RestrainingBoltWrapper
from npc_gym.wrappers.pacman_wrappers import PacmanPixelObservation

TRAINING_STEPS = int(os.environ.get("NPC_GYM_TRAINING_STEPS", "5000000"))
EVALUATION_EPISODES = int(os.environ.get("NPC_GYM_EVALUATION_EPISODES", "100"))
LAYOUT = os.environ.get("NPC_GYM_LAYOUT", "small")
MAX_EPISODE_STEPS = int(
    os.environ.get("NPC_GYM_MAX_EPISODE_STEPS", str(PacmanLayout.bundled(LAYOUT).default_episode_steps))
)
LEARNING_STARTS = int(os.environ.get("NPC_GYM_DQN_LEARNING_STARTS", "100"))
SEED = int(os.environ.get("NPC_GYM_SEED", "0"))
DEVICE = os.environ.get("NPC_GYM_DEVICE", "auto")


class PixelBoltFeatures(BaseFeaturesExtractor):
    """SB3 supplies channel-first pixels normalized to [0, 1]; one-hots stay intact."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        numeric = [space for key, space in observation_space.spaces.items() if key != "observation"]
        super().__init__(observation_space, 128 + sum(space.shape[0] for space in numeric if space.shape))
        self.numeric_keys = tuple(key for key in observation_space.spaces if key != "observation")
        self.cnn = NatureCNN(observation_space["observation"], features_dim=128)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (self.cnn(observations["observation"]), *(observations[key] for key in self.numeric_keys)), dim=1
        )


def make_env(bolts: Mapping[str, BoltSpec]) -> gym.Env[Any, int]:
    """Two 80-by-210 grayscale frames with numeric features; scale the combined reward."""
    env: gym.Env[Any, int] = PacmanEnv(layout=LAYOUT, features="image-full")
    env = RestrainingBoltWrapper(TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS), bolts=bolts)
    env = PacmanPixelObservation(env)
    return TransformReward(env, lambda reward: float(reward) / 100)


def evaluate_model(model: DQN, env: gym.Env[Any, int]) -> dict[str, float]:
    """Report mean episode returns in unscaled Pacman units and task outcomes."""
    totals: defaultdict[str, float] = defaultdict(float)
    for episode in range(EVALUATION_EPISODES):
        observation, _ = env.reset(seed=SEED + 10_000 if episode == 0 else None)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            diagnostics = info["restraining_bolts"]
            totals["augmented"] += float(reward) * 100
            totals["steps"] += 1
            totals["wrapped"] += diagnostics["wrapped_reward"]
            for name, adjustment in diagnostics["reward_adjustments"].items():
                totals[name] += adjustment
            if terminated or truncated:
                for name, value in info["episode_metrics"].items():
                    totals[name] += value
                break
    return {name: total / EVALUATION_EPISODES for name, total in totals.items()}


def main() -> dict[str, float]:
    """Compile once, train, reload a temporary model, and evaluate independently."""
    bolts = make_builtin_bolts(VEGAN_NORM_ID, reward=-300)
    with closing(make_env(bolts)) as train_env, closing(make_env(bolts)) as eval_env:
        model = DQN(
            "MultiInputPolicy",
            train_env,
            seed=SEED,
            device=DEVICE,
            learning_starts=LEARNING_STARTS,
            buffer_size=min(TRAINING_STEPS, 100_000),  # 6.72 GB of pixels; default buffer would need 67.2 GB.
            policy_kwargs={"features_extractor_class": PixelBoltFeatures},
            verbose=1,
        )
        model.learn(total_timesteps=TRAINING_STEPS)
        with TemporaryDirectory(prefix="npc-gym-pacman-images-") as directory:
            model.save(f"{directory}/model")
            loaded = DQN.load(f"{directory}/model", env=eval_env, device=DEVICE)
            returns = evaluate_model(loaded, eval_env)
        print(f"Mean augmented return (unscaled): {returns['augmented']:.2f}")
        print(f"Mean wrapped return (unscaled task reward): {returns['wrapped']:.2f}")
        print(f"Mean bolt adjustments: { {name: returns[name] for name in bolts} }")
        print(f"Win rate: {returns['won']:.1%}; ghosts eaten: {returns['blue_eaten'] + returns['orange_eaten']:.2f}")
        return returns


if __name__ == "__main__":
    main()
