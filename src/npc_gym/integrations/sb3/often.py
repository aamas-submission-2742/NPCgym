"""Optional two-stream OFTEN teaching of a Stable-Baselines3 DQN model."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from npc_gym.algorithms import TeachingSession, TeachingStats, TeachingStep
from npc_gym.algorithms.tabular_often import _Stream
from npc_gym.algorithms.tabular_q_learning import (
    _boolean,
    _info,
    _nonnegative_integer,
    _positive_integer,
    _real,
    _reward,
    _seed,
)
from npc_gym.monitors import MonitorInput


@dataclass(frozen=True)
class _Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    terminated: bool
    next_allowed: tuple[int, ...]


class _Replay:
    """FIFO capacity counts every visit; only eligible entries can be sampled.

    A dense list of eligible entries permits uniform sampling without rejection
    loops. Slot indices let ignored visits evict stale evidence in constant time.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: list[_Transition] = []
        self.slots: list[int] = []
        self.indices: dict[int, int] = {}
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.entries)

    def append(self, entry: _Transition | None) -> None:
        index = self.indices.pop(self.cursor, None)
        if index is not None:
            last_entry, last_slot = self.entries.pop(), self.slots.pop()
            if index < len(self.entries):
                self.entries[index], self.slots[index] = last_entry, last_slot
                self.indices[last_slot] = index
        if entry is not None:
            self.indices[self.cursor] = len(self.entries)
            self.entries.append(entry)
            self.slots.append(self.cursor)
        self.cursor = (self.cursor + 1) % self.capacity


def _expert_loss(online: torch.Tensor, target: torch.Tensor, actions: torch.Tensor, margin: float) -> torch.Tensor:
    """Upstream absolute margin loss; the expert target is always detached."""
    margins = torch.full_like(online, margin)
    margins.scatter_(1, actions, 0.0)
    maximum = (online + margins).max(dim=1, keepdim=True).values
    return F.l1_loss(maximum, target.detach().gather(1, actions))


class DQNOFTEN:
    """Teach an existing SB3 ``DQN`` in place with two ordinary environments.

    ``ordinary_env`` executes proposals and ``expert_env`` executes fixes. Supply
    fresh ``TeachingSession`` factories exactly as for ``TabularOFTEN``. Model,
    ordinary and expert observation/action spaces must match. Collection accepts
    no vector or autoreset environments and no VecNormalize transformations.

    Online/target parameters and optimizer state are retained. DQN settings come
    from ``model``: replay capacity, batch size, warm-up, exploration, learning
    rate, discount, rollout/update frequency, target synchronization and clipping.
    Ordinary and expert rollouts alternate; one combined TD/expert training phase
    follows each pair. Expert weight rises linearly from its initial to final
    value. ``margin`` is in the environment's training reward units.

    Save ``model`` through SB3 for inference without Clingo. Further teaching
    constructs a new trainer with fresh environments and session factories; this
    starts new replay and teaching counters. CPU and CUDA
    devices are supported; tests use CPU. The caller closes the environments.
    """

    def __init__(
        self,
        model: DQN,
        ordinary_env: gym.Env[Any, int],
        expert_env: gym.Env[Any, int],
        *,
        ordinary_session: Callable[[Any, Mapping[str, Any]], TeachingSession],
        expert_session: Callable[[Any, Mapping[str, Any]], TeachingSession],
        margin: float = 50.0,
        initial_expert_weight: float = 0.1,
        final_expert_weight: float = 1.0,
        filter_replay: bool = True,
        use_action_mask: bool = False,
        seed: int | None = None,
    ) -> None:
        if not isinstance(model, DQN):
            raise TypeError("model must be a Stable-Baselines3 DQN")
        if model.n_envs != 1:
            raise ValueError("DQNOFTEN supports a model with one environment only")
        if model.get_vec_normalize_env() is not None:
            raise ValueError(
                "DQNOFTEN does not support VecNormalize; supply explicitly transformed ordinary environments"
            )
        if getattr(model, "n_steps", 1) != 1:
            raise ValueError("DQNOFTEN requires one-step DQN (n_steps=1)")
        for name, env in (("ordinary_env", ordinary_env), ("expert_env", expert_env)):
            if not isinstance(env, gym.Env) or not isinstance(env.action_space, Discrete):
                raise TypeError(
                    f"{name} must be an ordinary Gymnasium Env with Discrete actions, not a vector environment"
                )
            if env.action_space.start != 0:
                raise ValueError("SB3 DQN requires zero-based Discrete actions")
            if env.action_space != model.action_space or env.observation_space != model.observation_space:
                raise ValueError(f"{name} action/observation spaces must match the DQN model")
            current = env
            while isinstance(current, gym.Wrapper):
                if isinstance(current, gym.wrappers.Autoreset):
                    raise TypeError("DQNOFTEN requires environments without Autoreset")
                current = current.env
        if ordinary_env.unwrapped is expert_env.unwrapped:
            raise ValueError("ordinary and expert environments must be independent")
        if not callable(ordinary_session) or not callable(expert_session):
            raise TypeError("session factories must be callable")
        if model.device.type not in ("cpu", "cuda"):
            raise ValueError("DQNOFTEN supports CPU and CUDA devices")
        self.buffer_size = _positive_integer(model.buffer_size, "model.buffer_size")
        self.batch_size = _positive_integer(model.batch_size, "model.batch_size")
        _nonnegative_integer(model.learning_starts, "model.learning_starts")
        _positive_integer(model.target_update_interval, "model.target_update_interval")
        self._train_freq = cast(TrainFreq, model.train_freq)
        if not isinstance(self._train_freq, TrainFreq):
            raise TypeError("model must have an initialized SB3 train_freq")
        _positive_integer(self._train_freq.frequency, "model.train_freq.frequency")
        if model.gradient_steps < -1:
            raise ValueError("model.gradient_steps must be -1 or nonnegative")
        self.margin = _real(margin, "margin")
        self.initial_expert_weight = _real(initial_expert_weight, "initial_expert_weight")
        self.final_expert_weight = _real(final_expert_weight, "final_expert_weight")
        if min(self.margin, self.initial_expert_weight, self.final_expert_weight) < 0:
            raise ValueError("margin and expert weights must be nonnegative")
        self.filter_replay = _boolean(filter_replay, "filter_replay")
        self.use_action_mask = _boolean(use_action_mask, "use_action_mask")
        seeds = np.random.SeedSequence(_seed(seed)).spawn(6)
        self._streams = (
            _Stream(
                ordinary_env, ordinary_session, np.random.default_rng(seeds[0]), int(seeds[2].generate_state(1)[0])
            ),
            _Stream(expert_env, expert_session, np.random.default_rng(seeds[1]), int(seeds[3].generate_state(1)[0])),
        )
        self._replay_rng = np.random.default_rng(seeds[4])
        torch_seed = int(seeds[5].generate_state(1)[0])
        self._torch_state = torch.Generator(device="cpu").manual_seed(torch_seed).get_state()
        self._cuda_devices = (
            [model.device.index if model.device.index is not None else torch.cuda.current_device()]
            if model.device.type == "cuda"
            else []
        )
        self._cuda_state = (
            torch.Generator(device=model.device).manual_seed(torch_seed).get_state() if self._cuda_devices else None
        )
        self.model = model
        assert isinstance(ordinary_env.action_space, Discrete)
        self.n_actions = int(ordinary_env.action_space.n)
        self._buffers = (_Replay(self.buffer_size), _Replay(self.buffer_size))
        self._counts = [0, 0]
        self._episodes = [0, 0]
        self._returns = [0.0, 0.0]
        self._interventions = [0, 0]
        self._fallbacks = [0, 0]
        self._filtered = 0
        self._updates = 0
        self._phase = 0
        self._rollout_steps = 0
        self._rollout_episodes = 0
        self._schedule_steps = 1
        self.progress_remaining = 1.0
        self.exploration_rate = 0.0
        self.expert_weight = self.initial_expert_weight

    def q_values(self, observation: Any) -> NDArray[np.float64]:
        """Return a detached row for one unbatched observation, without RNG use.

        Uses SB3 observation preprocessing, temporarily switches networks to
        evaluation mode and restores all module modes afterwards. Values cover
        all actions; current masks are applied by the collector or its caller.
        """
        with self._network_modes(training=False), torch.no_grad():
            tensor, vectorized = self.model.policy.obs_to_tensor(observation)
            if vectorized:
                raise ValueError("q_values requires one unbatched observation")
            values = np.array(self.model.q_net(tensor).detach().cpu().numpy()[0], dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise FloatingPointError("DQN produced non-finite action values")
        return values

    def _allowed_actions(self, info: Mapping[str, Any], *, phase: str) -> NDArray[np.intp]:
        if not self.use_action_mask:
            return np.arange(self.n_actions, dtype=np.intp)
        if "action_mask" not in info:
            raise ValueError(f"Masked {phase} requires info['action_mask']")
        mask = np.asarray(info["action_mask"])
        if mask.shape != (self.n_actions,) or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"action_mask must contain {self.n_actions} entries, each 0 or 1")
        allowed = np.flatnonzero(mask).astype(np.intp, copy=False)
        if allowed.size == 0:
            raise ValueError("action_mask cannot exclude all available actions")
        return allowed

    @property
    def stats(self) -> TeachingStats:
        """Detached cumulative diagnostics for this trainer."""
        return TeachingStats(
            ordinary_steps=self._counts[0],
            expert_steps=self._counts[1],
            ordinary_episodes=self._episodes[0],
            expert_episodes=self._episodes[1],
            ordinary_return=self._returns[0],
            expert_return=self._returns[1],
            ordinary_interventions=self._interventions[0],
            expert_interventions=self._interventions[1],
            ordinary_fallbacks=self._fallbacks[0],
            expert_fallbacks=self._fallbacks[1],
            filtered_steps=self._filtered,
            ordinary_buffer_size=len(self._buffers[0]),
            expert_buffer_size=len(self._buffers[1]),
            updates=self._updates,
        )

    def learn(
        self,
        total_timesteps: int,
        *,
        schedule_timesteps: int | None = None,
        callback: Callable[[TeachingStep], bool | None] | None = None,
    ) -> Self:
        """Collect up to a cumulative ordinary-plus-expert transition budget.

        ``schedule_timesteps`` defaults to the budget. Set a longer horizon to
        inspect a prefix without compressing learning-rate, exploration or expert
        schedules. Episodes, partial rollouts, replay and RNGs continue across
        calls. Callbacks run after each transition and any completed training
        phase; returning False stops without resetting the streams.
        """
        target = _nonnegative_integer(total_timesteps, "total_timesteps")
        horizon = target if schedule_timesteps is None else _positive_integer(schedule_timesteps, "schedule_timesteps")
        if horizon < target:
            raise ValueError("schedule_timesteps must cover total_timesteps")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        if target <= self.stats.total_steps:
            return self
        self._schedule_steps = (horizon + 1) // 2
        while self.stats.total_steps < target:
            index = self._phase
            stream = self._streams[index]
            if stream.session is None:
                self._reset(stream)
            assert stream.session is not None and stream.info is not None
            session = stream.session
            allowed = self._allowed_actions(stream.info, phase="OFTEN action selection")
            values = self.q_values(stream.observation)
            # The reference counts ordinary transitions, temporarily advancing
            # that counter during the expert rollout, then restoring it.
            schedule_count = self._counts[0] + (self._rollout_steps if index == 1 else 0)
            exploring = schedule_count < self.model.learning_starts or stream.rng.random() < self.exploration_rate
            proposal = int(stream.rng.choice(allowed)) if exploring else int(allowed[np.argmax(values[allowed])])
            decision = session.planner.solve(
                session.problem(stream.observation, stream.info),
                {i: float(value) for i, value in enumerate(values)},
                proposed_action=proposal,
                allowed_actions=[int(i) for i in allowed],
            )
            # Expert exploration executes random actions too; the fixer checks
            # their replay eligibility. Only greedy expert actions are replaced.
            fixed = index == 1 and not exploring
            action = decision.action if fixed else proposal
            disagreement = decision.optimal and decision.changed
            retained = not (self.filter_replay and disagreement and not fixed)
            if index == 1:
                retained = retained and decision.expert_eligible
            state = deepcopy(stream.observation)
            observation, raw_reward, terminated, truncated, raw_info = stream.env.step(action)
            info = _info(raw_info, "step")
            reward = _reward(raw_reward)
            if session.advance is not None:
                if "labels" not in info:
                    raise ValueError("TeachingSession.advance requires step info['labels']")
                session.advance(MonitorInput(info["labels"], bool(terminated), bool(truncated)))
            next_allowed = (
                () if terminated else tuple(int(i) for i in self._allowed_actions(info, phase="OFTEN bootstrapping"))
            )
            self._buffers[index].append(
                _Transition(state, action, reward, deepcopy(observation), bool(terminated), next_allowed)
                if retained
                else None
            )
            self._counts[index] += 1
            self._returns[index] += reward
            self._interventions[index] += int(disagreement)
            self._fallbacks[index] += int(not decision.optimal)
            self._filtered += int(index == 0 and not retained)
            self._episodes[index] += int(terminated or truncated)
            stream.observation, stream.info = observation, info
            if terminated or truncated:
                stream.session = None
            self._rollout_steps += 1
            self._rollout_episodes += int(terminated or truncated)
            self._on_step()
            losses = self._finish_rollout()
            if callback is not None:
                result = callback(
                    TeachingStep(
                        "ordinary" if index == 0 else "expert",
                        action,
                        reward,
                        bool(terminated),
                        bool(truncated),
                        retained,
                        decision,
                        *losses,
                        self.stats,
                    )
                )
                if result is not None and not isinstance(result, bool):
                    raise TypeError("callback must return bool or None")
                if result is False:
                    break
        return self

    def _on_step(self) -> None:
        count = self._counts[0] + (self._rollout_steps if self._phase == 1 else 0)
        self.progress_remaining = max(0.0, 1.0 - count / self._schedule_steps)
        self.exploration_rate = float(self.model.exploration_schedule(self.progress_remaining))
        self.expert_weight = self.initial_expert_weight + (1.0 - self.progress_remaining) * (
            self.final_expert_weight - self.initial_expert_weight
        )
        if self.stats.total_steps % self.model.target_update_interval == 0:
            polyak_update(self.model.q_net.parameters(), self.model.q_net_target.parameters(), self.model.tau)
            # Match SB3: running statistics are copied even for a soft target update.
            polyak_update(self.model.batch_norm_stats, self.model.batch_norm_stats_target, 1.0)

    def _finish_rollout(self) -> tuple[float, float, float]:
        frequency = self._train_freq
        collected = self._rollout_steps if frequency.unit == TrainFrequencyUnit.STEP else self._rollout_episodes
        losses = (0.0, 0.0, 0.0)
        if collected < frequency.frequency:
            return losses
        if self._phase == 1 and self._counts[0] > self.model.learning_starts:
            steps = self._rollout_steps if self.model.gradient_steps == -1 else self.model.gradient_steps
            rate = float(self.model.lr_schedule(self.progress_remaining))
            if not np.isfinite(rate) or rate < 0:
                raise ValueError("model learning-rate schedule must return a finite nonnegative value")
            for group in self.model.policy.optimizer.param_groups:
                group["lr"] = rate
            for _ in range(steps):
                losses = self._update()
        self._phase = 1 - self._phase
        self._rollout_steps = self._rollout_episodes = 0
        return losses

    def _reset(self, stream: _Stream) -> None:
        observation, raw_info = stream.env.reset(seed=stream.seed if not stream.seeded else None)
        stream.seeded = True
        info = _info(raw_info, "reset")
        session = stream.factory(observation, info)
        if not isinstance(session, TeachingSession):
            raise TypeError("session factory must return TeachingSession")
        if any(other is not stream and other.session is session for other in self._streams):
            raise ValueError("streams must have independent TeachingSession instances")
        stream.observation, stream.info, stream.session = observation, info, session

    def _batch(self, observations: list[Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        batch = (
            {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
            if isinstance(observations[0], dict)
            else np.stack(observations)
        )
        return self.model.policy.obs_to_tensor(batch)[0]

    @contextmanager
    def _network_modes(self, *, training: bool) -> Iterator[None]:
        modes = [(module, module.training) for module in self.model.policy.modules()]
        self.model.policy.set_training_mode(training)
        self.model.q_net_target.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode

    @contextmanager
    def _training_randomness(self) -> Iterator[None]:
        # Network stochasticity (for example custom dropout) gets a private
        # stream; restore process RNG even when a forward pass raises.
        with torch.random.fork_rng(devices=self._cuda_devices):
            torch.random.set_rng_state(self._torch_state)
            if self._cuda_state is not None:
                torch.cuda.set_rng_state(self._cuda_state, self._cuda_devices[0])
            try:
                yield
            finally:
                self._torch_state = torch.random.get_rng_state()
                if self._cuda_state is not None:
                    self._cuda_state = torch.cuda.get_rng_state(self._cuda_devices[0])

    def _update(self) -> tuple[float, float, float]:
        if not all(self._buffers):
            return 0.0, 0.0, 0.0
        losses = [0.0, 0.0, 0.0]
        with self._network_modes(training=True), self._training_randomness():
            terms = []
            for index, buffer in enumerate(self._buffers):
                samples = [buffer.entries[int(i)] for i in self._replay_rng.integers(len(buffer), size=self.batch_size)]
                states = self._batch([item.state for item in samples])
                actions = torch.tensor([[item.action] for item in samples], device=self.model.device, dtype=torch.long)
                rewards = torch.tensor(
                    [[item.reward] for item in samples], device=self.model.device, dtype=torch.float32
                )
                with torch.no_grad():
                    # Terminated samples need no next-state evaluation; avoid
                    # zero times infinity when a final mask has no actions.
                    targets = rewards.clone()
                    continuing = [i for i, item in enumerate(samples) if not item.terminated]
                    if continuing:
                        next_states = self._batch([samples[i].next_state for i in continuing])
                        next_values = self.model.q_net_target(next_states)
                        mask = torch.zeros_like(next_values, dtype=torch.bool)
                        for row, i in enumerate(continuing):
                            mask[row, list(samples[i].next_allowed)] = True
                        bootstrap = next_values.masked_fill(~mask, -torch.inf).max(dim=1, keepdim=True).values
                        targets[continuing] += self.model.gamma * bootstrap
                    expert_targets = self.model.q_net_target(states) if index == 1 else None
                online = self.model.q_net(states)
                td = F.smooth_l1_loss(online.gather(1, actions), targets)
                terms.append(td)
                losses[index] = float(td.detach().cpu())
                if expert_targets is not None:
                    expert = _expert_loss(online, expert_targets, actions, self.margin)
                    losses[2] = float(expert.detach().cpu())
                    terms.append(self.expert_weight * expert)
            loss = torch.stack(terms).sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("OFTEN loss is non-finite")
            optimizer = self.model.policy.optimizer
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.model.q_net.parameters(), self.model.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            self._updates += 1
        return losses[0], losses[1], losses[2]


__all__ = ["DQNOFTEN"]
