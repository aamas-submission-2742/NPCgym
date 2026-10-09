from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit, TransformReward
from pacman_fixtures import TERMINAL_LAYOUT
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from npc_gym.envs.gardener.gardener import GardenerEnv, action_dict
from npc_gym.envs.pacman.pacman_env import PacmanEnv
from npc_gym.evaluation import EpisodeResult, EvaluationSummary, TerminationClass, evaluate
from npc_gym.integrations.sb3 import SB3EvaluationCallback, SB3Policy
from npc_gym.wrappers.gardener_wrappers import IllegalActionPenaltyWrapper


class LearningEnv(gym.Env[np.ndarray, int]):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, trajectory: list[tuple[int, int, float, int]]) -> None:
        self.trajectory = trajectory

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = 0
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        reward = 1.0 if action == self.state else -0.25
        self.step_count += 1
        self.state = (self.state + action + 1) % 2
        self.trajectory.append((self.step_count, action, reward, self.state))
        return self._observation(), reward, self.step_count == 4, False, {}

    def _observation(self) -> np.ndarray:
        return np.array([self.state, self.step_count / 4], dtype=np.float32)


@dataclass(frozen=True)
class TrainingResult:
    parameters: dict[str, torch.Tensor]
    trajectory: tuple[tuple[int, int, float, int], ...]
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: torch.Tensor
    action_space_state: dict[str, Any]
    policy_modes: tuple[bool, ...]
    updates: int
    evaluations: int


def _summary() -> EvaluationSummary:
    return EvaluationSummary.from_episodes((EpisodeResult(0, 0.0, 1, TerminationClass.TERMINATED, {}),))


def _train(algorithm: str, evaluation_frequency: int | None, *, deterministic: bool = True) -> TrainingResult:
    trajectory: list[tuple[int, int, float, int]] = []
    train_env = DummyVecEnv([lambda: LearningEnv(trajectory)])
    eval_env = LearningEnv([])
    seed = 37
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if algorithm == "ppo":
        model: BaseAlgorithm = PPO(
            "MlpPolicy",
            train_env,
            seed=seed,
            device="cpu",
            n_steps=4,
            batch_size=4,
            n_epochs=2,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            verbose=0,
        )
        total_timesteps = 16
    else:
        model = DQN(
            "MlpPolicy",
            train_env,
            seed=seed,
            device="cpu",
            learning_starts=4,
            buffer_size=64,
            batch_size=4,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=4,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
        )
        total_timesteps = 16

    callback = None
    if evaluation_frequency is not None:

        def evaluator(current_model: BaseAlgorithm) -> EvaluationSummary:
            policy = SB3Policy(current_model, deterministic=deterministic)
            summary = evaluate(eval_env, policy, episodes=1, seed=901)
            random.random()
            np.random.random()
            torch.rand(1)
            return summary

        def hook(step: int, summary: EvaluationSummary) -> None:
            assert step >= 0
            assert summary.episodes
            random.random()
            np.random.random()
            torch.rand(1)

        callback = SB3EvaluationCallback(
            collect_training_episodes=False,
            evaluator=evaluator,
            evaluation_frequency=evaluation_frequency,
            on_evaluation=hook,
        )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        return TrainingResult(
            parameters={name: value.detach().clone() for name, value in model.policy.state_dict().items()},
            trajectory=tuple(trajectory),
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            torch_state=torch.random.get_rng_state(),
            action_space_state=deepcopy(model.action_space.np_random.bit_generator.state),
            policy_modes=tuple(module.training for module in model.policy.modules()),
            updates=model._n_updates,
            evaluations=len(callback.evaluations) if callback is not None else 0,
        )
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_seeded_cpu_training_is_invariant_to_intermediate_evaluation_frequency(algorithm, deterministic):
    baseline = _train(algorithm, None)

    assert baseline.updates > 0
    assert baseline.evaluations == 0
    for frequency in (1, 7):
        evaluated = _train(algorithm, frequency, deterministic=deterministic)
        assert evaluated.updates == baseline.updates
        assert evaluated.evaluations == 1 + 16 // frequency
        assert evaluated.trajectory == baseline.trajectory
        assert evaluated.parameters.keys() == baseline.parameters.keys()
        assert all(torch.equal(evaluated.parameters[name], baseline.parameters[name]) for name in baseline.parameters)
        assert evaluated.python_state == baseline.python_state
        _assert_numpy_random_states_equal(evaluated.numpy_state, baseline.numpy_state)
        assert torch.equal(evaluated.torch_state, baseline.torch_state)
        assert evaluated.action_space_state == baseline.action_space_state
        assert evaluated.policy_modes == baseline.policy_modes


class CallbackModel:
    def __init__(self, env: DummyVecEnv) -> None:
        self.env = env
        self.num_timesteps = 0
        self.policy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 2))
        self.action_space = gym.spaces.Discrete(2, seed=14)

    def get_env(self):
        return self.env


@pytest.mark.parametrize("at_start", [True, False])
@pytest.mark.parametrize("failure", ["evaluator", "hook"])
def test_intermediate_evaluation_restores_rng_and_policy_modes_after_failures(failure, at_start):
    env = DummyVecEnv([lambda: LearningEnv([])])
    model = CallbackModel(env)
    model.policy.train()
    model.policy[1].eval()

    def mutate_training_state() -> None:
        random.random()
        np.random.random()
        torch.rand(1)
        model.action_space.sample()
        model.policy.eval()
        model.policy[0].train()

    def evaluator(current_model) -> EvaluationSummary:
        assert current_model is model
        mutate_training_state()
        if failure == "evaluator":
            raise RuntimeError("evaluator failed")
        return _summary()

    def hook(step: int, summary: EvaluationSummary) -> None:
        assert step == (0 if at_start else 2)
        assert summary.episodes
        mutate_training_state()
        raise RuntimeError("hook failed")

    callback = SB3EvaluationCallback(
        collect_training_episodes=False,
        evaluator=evaluator,
        evaluation_frequency=1,
        on_evaluation=hook,
    )
    try:
        env.reset()
        callback.init_callback(model)
        if not at_start:
            # Start as a resumed call so the failure targets the periodic boundary.
            model.num_timesteps = 1
            callback.on_training_start({}, {})
            model.num_timesteps = 2

        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        action_space_state = model.action_space.np_random.bit_generator.state
        policy_modes = tuple(module.training for module in model.policy.modules())

        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            if at_start:
                callback.on_training_start({}, {})
            else:
                callback.on_step()

        assert random.getstate() == python_state
        _assert_numpy_random_states_equal(np.random.get_state(), numpy_state)
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert model.action_space.np_random.bit_generator.state == action_space_state
        assert tuple(module.training for module in model.policy.modules()) == policy_modes
    finally:
        env.close()


def _assert_numpy_random_states_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_gardener_evaluation_remaps_illegal_actions_without_the_training_penalty():
    train_base = GardenerEnv(size=8)
    eval_base = GardenerEnv(size=8)
    train_env = IllegalActionPenaltyWrapper(train_base, penalty=-1.0)
    eval_env = IllegalActionPenaltyWrapper(eval_base, penalty=0.0)
    try:
        train_observation, _ = train_env.reset(seed=12)
        eval_observation, _ = eval_env.reset(seed=12)
        assert train_observation == eval_observation
        illegal = int(np.flatnonzero(train_base.action_mask() == 0)[0])

        train_result = train_env.step(illegal)
        eval_result = eval_env.step(illegal)

        assert train_result[:1] == eval_result[:1]
        assert train_result[2:4] == eval_result[2:4]
        assert train_result[1] == pytest.approx(eval_result[1] - 1.0)
        assert train_base.action == eval_base.action == action_dict["stay"]
        assert train_base.labeling_state() == eval_base.labeling_state()
    finally:
        train_env.close()
        eval_env.close()


def test_pacman_evaluation_reports_unscaled_task_return():
    def rollout(env: gym.Env, policy: Callable[[Any, dict[str, Any]], int]) -> float:
        summary = evaluate(env, policy, seed=0)
        return summary.episodes[0].episode_return

    policy = lambda observation, info: 3
    task_env = TimeLimit(PacmanEnv(layout=TERMINAL_LAYOUT, features="essential"), max_episode_steps=300)
    training_env = TimeLimit(
        TransformReward(PacmanEnv(layout=TERMINAL_LAYOUT, features="essential"), lambda reward: reward / 100),
        max_episode_steps=300,
    )
    try:
        task_return = rollout(task_env, policy)
        transformed_training_return = rollout(training_env, policy)

        assert task_return == pytest.approx(transformed_training_return * 100)
        assert abs(task_return) > 100
    finally:
        task_env.close()
        training_env.close()
