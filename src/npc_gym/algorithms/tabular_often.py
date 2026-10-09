"""Two-stream OFTEN teaching for an existing tabular task policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

from npc_gym.algorithms.tabular_q_learning import (
    TabularQLearning,
    _boolean,
    _info,
    _nonnegative_integer,
    _positive_integer,
    _probability,
    _real,
    _reward,
    _seed,
    _state_key,
)
from npc_gym.monitors import MonitorInput
from npc_gym.policy_fixes import ASPPlanner, PlanningDecision, PlanningProblem


@dataclass(frozen=True)
class TeachingSession:
    """One episode's planner and public-state problem builder.

    A factory creates a fresh session after each reset. ``problem(observation,
    info)`` reads the current state; optional ``advance(input)`` consumes every
    actual transition, including ordinary proposals rejected by the fixer.
    The factory initializes domain history from the reset snapshot. Callbacks
    must not step/reset the environment or consume its random generator.
    """

    planner: ASPPlanner
    problem: Callable[[Any, Mapping[str, Any]], PlanningProblem]
    advance: Callable[[MonitorInput], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.planner, ASPPlanner):
            raise TypeError("planner must be an ASPPlanner")
        if not callable(self.problem):
            raise TypeError("problem must be callable")
        if self.advance is not None and not callable(self.advance):
            raise TypeError("advance must be callable or None")


@dataclass(frozen=True)
class TeachingStats:
    """Cumulative teaching budgets and diagnostics, excluding base training.

    Returns sum unchanged environment rewards, including partial episodes.
    Interventions count certified fixer disagreements, even on ordinary steps
    where the proposal was executed. Buffer sizes count retained transitions.
    """

    ordinary_steps: int = 0
    expert_steps: int = 0
    ordinary_episodes: int = 0
    expert_episodes: int = 0
    ordinary_return: float = 0.0
    expert_return: float = 0.0
    ordinary_interventions: int = 0
    expert_interventions: int = 0
    ordinary_fallbacks: int = 0
    expert_fallbacks: int = 0
    filtered_steps: int = 0
    ordinary_buffer_size: int = 0
    expert_buffer_size: int = 0
    updates: int = 0

    @property
    def total_steps(self) -> int:
        """Actual ordinary plus demonstration environment transitions."""
        return self.ordinary_steps + self.expert_steps


@dataclass(frozen=True)
class TeachingStep:
    """Callback result after collection and replay, with unweighted mean losses."""

    stream: Literal["ordinary", "expert"]
    action: int
    reward: float
    terminated: bool
    truncated: bool
    retained: bool
    decision: PlanningDecision
    ordinary_td_loss: float
    expert_td_loss: float
    expert_loss: float
    stats: TeachingStats


@dataclass(frozen=True)
class _Transition:
    state: str
    action: int  # Q-row index, not the potentially offset environment action ID.
    reward: float
    next_state: str
    terminated: bool
    next_allowed: tuple[int, ...]


@dataclass
class _Stream:
    env: gym.Env[Any, int]
    factory: Callable[[Any, Mapping[str, Any]], TeachingSession]
    rng: np.random.Generator
    seed: int
    observation: Any = None
    info: Mapping[str, Any] | None = None
    session: TeachingSession | None = None
    seeded: bool = False


class TabularOFTEN:
    """Teach ``policy`` in place using ordinary and fixed demonstration streams.

    ``policy.env`` and ``expert_env`` must be independent ordinary Gymnasium
    environments with equal Discrete action spaces and compatible observations.
    Factories receive reset observations/info and return fresh TeachingSession
    instances. The caller owns environments and closes them.

    ``learn(total_timesteps)`` is a cumulative combined teaching budget, separate
    from ``policy.num_timesteps``. Streams alternate, ordinary first. Every call
    starts fresh episodes while retaining replay, target values, RNGs and counts.
    Save ``policy`` with its existing safe format; constructing a new trainer
    starts a fresh teaching run (not a replay/optimizer checkpoint).
    """

    def __init__(
        self,
        policy: TabularQLearning[Any],
        expert_env: gym.Env[Any, int],
        *,
        ordinary_session: Callable[[Any, Mapping[str, Any]], TeachingSession],
        expert_session: Callable[[Any, Mapping[str, Any]], TeachingSession],
        buffer_size: int = 10_000,
        batch_size: int = 32,
        target_update_interval: int = 100,
        epsilon: float = 0.1,
        expert_epsilon: float = 0.0,
        margin: float = 0.8,
        expert_weight: float = 1.0,
        filter_replay: bool = True,
        seed: int | None = None,
    ) -> None:
        if not isinstance(policy, TabularQLearning):
            raise TypeError("policy must be a TabularQLearning instance")
        if not isinstance(expert_env, gym.Env) or not isinstance(expert_env.action_space, Discrete):
            raise TypeError("expert_env must be an ordinary Gymnasium Env with Discrete actions")
        if expert_env.unwrapped is policy.env.unwrapped:
            raise ValueError("ordinary and expert environments must be independent")
        if expert_env.action_space != policy.env.action_space:
            raise ValueError("ordinary and expert action spaces must match")
        if expert_env.observation_space != policy.env.observation_space:
            raise ValueError("ordinary and expert observation spaces must match")
        if not callable(ordinary_session) or not callable(expert_session):
            raise TypeError("session factories must be callable")
        self.buffer_size = _positive_integer(buffer_size, "buffer_size")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.target_update_interval = _positive_integer(target_update_interval, "target_update_interval")
        self.epsilon = _probability("epsilon", epsilon)
        self.expert_epsilon = _probability("expert_epsilon", expert_epsilon)
        self.margin = _real(margin, "margin")
        self.expert_weight = _real(expert_weight, "expert_weight")
        if self.margin < 0 or self.expert_weight < 0:
            raise ValueError("margin and expert_weight must be nonnegative")
        self.filter_replay = _boolean(filter_replay, "filter_replay")
        seeds = np.random.SeedSequence(_seed(seed)).spawn(5)
        self._streams = (
            _Stream(policy.env, ordinary_session, np.random.default_rng(seeds[0]), int(seeds[2].generate_state(1)[0])),
            _Stream(expert_env, expert_session, np.random.default_rng(seeds[1]), int(seeds[3].generate_state(1)[0])),
        )
        self._replay_rng = np.random.default_rng(seeds[4])
        self.policy = policy
        self._buffers: tuple[list[_Transition], list[_Transition]] = ([], [])
        self._buffer_positions = [0, 0]
        self._target = {key: row.copy() for key, row in policy.q_table.items()}
        self._counts = [0, 0]
        self._episodes = [0, 0]
        self._returns = [0.0, 0.0]
        self._interventions = [0, 0]
        self._fallbacks = [0, 0]
        self._filtered = 0
        self._updates = 0

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

    def learn(self, total_timesteps: int, *, callback: Callable[[TeachingStep], bool | None] | None = None) -> Self:
        """Collect up to the cumulative budget; callback False stops after a step.

        Ordinary steps execute the epsilon-greedy proposal and discard replay
        only on certified disagreement when filtering is enabled. Demonstrations
        execute the fixed action; fallback steps never enter the expert buffer.
        One replay update follows each transition if either buffer is nonempty.
        """
        target = _nonnegative_integer(total_timesteps, "total_timesteps")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        if target <= self.stats.total_steps:
            return self
        for stream in self._streams:
            stream.session = None
        while self.stats.total_steps < target:
            index = self.stats.total_steps % 2
            stream = self._streams[index]
            if stream.session is None:
                self._reset(stream)
            assert stream.session is not None and stream.info is not None
            session = stream.session
            allowed = self.policy._allowed_actions(stream.info, phase="OFTEN action selection")
            values = self.policy.q_values(stream.observation)
            proposal_index = self.policy._epsilon_greedy(
                values, allowed, stream.rng, self.epsilon if index == 0 else self.expert_epsilon
            )
            proposal = self.policy._action_value(proposal_index)
            decision = session.planner.solve(
                session.problem(stream.observation, stream.info),
                {self.policy._action_value(i): float(value) for i, value in enumerate(values)},
                proposed_action=proposal,
                allowed_actions=[self.policy._action_value(int(i)) for i in allowed],
            )
            action = proposal if index == 0 else decision.action
            # Encode before stepping: some environments reuse mutable observations.
            state = _state_key(stream.observation)
            observation, raw_reward, terminated, truncated, raw_info = stream.env.step(action)
            info = _info(raw_info, "step")
            reward = _reward(raw_reward)
            if session.advance is not None:
                if "labels" not in info:
                    raise ValueError("TeachingSession.advance requires step info['labels']")
                session.advance(MonitorInput(info["labels"], bool(terminated), bool(truncated)))
            next_allowed = (
                ()
                if terminated
                else tuple(int(i) for i in self.policy._allowed_actions(info, phase="OFTEN bootstrapping"))
            )
            disagreement = decision.optimal and decision.changed
            retained = not (self.filter_replay and disagreement) if index == 0 else decision.expert_eligible
            if retained:
                transition = _Transition(
                    state,
                    action - self.policy.action_start,
                    reward,
                    _state_key(observation),
                    bool(terminated),
                    next_allowed,
                )
                self._append(index, transition)
            self._counts[index] += 1
            self._returns[index] += reward
            self._interventions[index] += int(disagreement)
            self._fallbacks[index] += int(not decision.optimal)
            self._filtered += int(index == 0 and not retained)
            self._episodes[index] += int(terminated or truncated)
            stream.observation, stream.info = observation, info
            if terminated or truncated:
                stream.session = None
            losses = self._update()
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

    def _append(self, index: int, transition: _Transition) -> None:
        buffer = self._buffers[index]
        if len(buffer) < self.buffer_size:
            buffer.append(transition)
        else:
            buffer[self._buffer_positions[index]] = transition
        self._buffer_positions[index] = (self._buffer_positions[index] + 1) % self.buffer_size

    def _update(self) -> tuple[float, float, float]:
        gradients: dict[str, NDArray[np.float64]] = {}
        losses = [0.0, 0.0, 0.0]
        for index, buffer in enumerate(self._buffers):
            if not buffer:
                continue
            for sample in self._replay_rng.integers(len(buffer), size=self.batch_size):
                transition = buffer[int(sample)]
                row = self.policy._q.get(transition.state, np.zeros(self.policy.n_actions))
                frozen = self._target.get(transition.next_state, np.zeros(self.policy.n_actions))
                bootstrap = 0.0 if transition.terminated else float(np.max(frozen[list(transition.next_allowed)]))
                difference = float(row[transition.action]) - transition.reward - self.policy.gamma * bootstrap
                gradient = gradients.setdefault(transition.state, np.zeros(self.policy.n_actions))
                gradient[transition.action] += difference / self.batch_size
                losses[index] += 0.5 * difference**2 / self.batch_size
                if index == 1:
                    margins = np.full(self.policy.n_actions, self.margin)
                    margins[transition.action] = 0.0
                    maximum = int(np.argmax(row + margins))
                    expert_target = self._target.get(transition.state, np.zeros(self.policy.n_actions))
                    error = float(row[maximum] + margins[maximum] - expert_target[transition.action])
                    losses[2] += abs(error) / self.batch_size
                    gradient[maximum] += self.expert_weight * float(np.sign(error)) / self.batch_size
        if gradients:
            # All terms use pre-update online rows; target rows are detached.
            updated = {
                state: self.policy._q.get(state, np.zeros(self.policy.n_actions)) - self.policy.learning_rate * gradient
                for state, gradient in gradients.items()
            }
            if any(not np.isfinite(row).all() for row in updated.values()):
                raise FloatingPointError("OFTEN update produced non-finite Q-values")
            self.policy._q.update(updated)
            self._updates += 1
            if self._updates % self.target_update_interval == 0:
                self._target = {key: row.copy() for key, row in self.policy._q.items()}
        return losses[0], losses[1], losses[2]


__all__ = ["TabularOFTEN", "TeachingSession", "TeachingStats", "TeachingStep"]
