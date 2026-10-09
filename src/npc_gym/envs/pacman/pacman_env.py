"""Gymnasium adapter for the original NPC Gym maze-chase environment."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded
from numpy.typing import NDArray

from npc_gym.envs.metrics import EPISODE_METRICS_KEY

from .labels import PacmanAuthorityState, PacmanGhostState, PacmanLayoutState, PacmanPlayerState, authority_labels
from .layout import BUNDLED_LAYOUTS, PacmanLayout
from .observations import (
    SUPPORTED_FEATURES,
    SUPPORTED_VECTOR_FEATURES,
    Observation,
    PacmanObservation,
    validate_feature_mode,
)
from .rendering import PacmanRenderer
from .simulation import Simulation
from .state import GhostBehavior, GhostConfig, PacmanSnapshot

PACMAN_ACTIONS = (0, 1, 2, 3, 4)


class PacmanEnv(gym.Env[Observation, int]):
    """Maze chase with independent gameplay, features and rendering.

    Actions are stop/north/south/east/west (0–4). Blocked moves become Stop.
    Random ghosts preserve NPC Gym's historical transition semantics. Scheduled
    ghosts expose [chase flag, remaining turns, paused flag] in observations.
    Truncation belongs to an external TimeLimit; small/medium/large registrations
    use 300/500/800 turns. ``ghost_config`` affects only scheduled behaviors;
    random mode retains the resolved configuration in snapshots but ignores it.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}  # noqa: RUF012
    layouts: ClassVar = BUNDLED_LAYOUTS
    action_space: spaces.Discrete[np.int64]  # type: ignore[assignment]

    def __init__(
        self,
        layout: str | PacmanLayout = "small",
        features: str = "complete",
        dfas: Mapping[str, float] | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
        *,
        ghost_behavior: GhostBehavior = "random",
        ghost_config: GhostConfig | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        validate_feature_mode(features)
        if dfas is not None and not isinstance(dfas, Mapping):
            raise TypeError("dfas must be a mapping")
        self.layout = layout if isinstance(layout, PacmanLayout) else PacmanLayout.bundled(layout)
        # Validate configuration and resolve map defaults before defining spaces.
        initial = Simulation(self.layout, np.random.default_rng(0), behavior=ghost_behavior, config=ghost_config)
        self.ghost_behavior, self.ghost_config = ghost_behavior, initial.config
        self.render_mode, self.feature_mode = render_mode, features
        self._renderer = PacmanRenderer(
            self.layout,
            render_mode,
            enabled=render_mode is not None or "image" in features,
            render_fps=int(self.metadata["render_fps"]),
        )
        max_phase = max(initial.config.scatter_turns or 0, initial.config.chase_turns or 0) if initial.phase else 0
        self._observations = PacmanObservation(
            features, self.layout, dict(dfas or {}), self._renderer, max_phase_turns=max_phase
        )
        self.action_space = spaces.Discrete(5)
        self.observation_space = cast(spaces.Space[Observation], self._observations.space)
        self._sim: Simulation | None = None
        self.step_counter = 0
        self.cum_reward = 0.0
        self.illegal_move_counter = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a seeded episode; nonempty reset options are unsupported."""
        if options:
            raise ValueError("PacmanEnv does not support reset options")
        super().reset(seed=seed)
        self._sim = Simulation(self.layout, self.np_random, behavior=self.ghost_behavior, config=self.ghost_config)
        self.step_counter = self.illegal_move_counter = 0
        self.cum_reward = 0.0
        try:
            self._renderer.reset(self._sim)
            observation = self._observations.observe(self._sim, None)
            if self.render_mode == "human":
                self.render()
        except BaseException:
            self.close()
            raise
        return observation, {
            "step_counter": [[0]],
            "labels": authority_labels(self._sim.snapshot(), None),
            EPISODE_METRICS_KEY: self._metrics(),
        }

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Advance one complete turn and report raw score change and current labels."""
        if self._sim is None:
            raise ResetNeeded("PacmanEnv must be reset before step()")
        if self._sim.terminated:
            raise ResetNeeded("PacmanEnv must be reset after the episode terminates")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Pacman action {action!r}; expected an integer from 0 through 4")
        action = int(action)
        reward, blocked = self._sim.step(action)
        self.step_counter += 1
        self.illegal_move_counter += int(blocked)
        self.cum_reward += reward
        try:
            self._renderer.update(self._sim)
            observation = self._observations.observe(self._sim, action)
            if self.render_mode == "human":
                self.render()
        except BaseException:
            self.close()
            raise
        terminated = self._sim.terminated
        info: dict[str, Any] = {
            "step_counter": [[self.step_counter]],
            "episode": None,
            "agent_eaten": [0, *(g.eaten for g in self._sim.ghosts)],
            "labels": authority_labels(self._sim.snapshot(), action),
            EPISODE_METRICS_KEY: self._metrics(),
        }
        if terminated:
            info["episode"] = [{"r": self.cum_reward, "l": self.step_counter, "w": self._sim.won}]
        return observation, reward, terminated, False, info

    def _metrics(self) -> dict[str, int | float]:
        assert self._sim is not None
        ghosts = self._sim.ghosts
        return {
            "score": self._sim.score,
            "blue_eaten": ghosts[0].eaten if ghosts else 0,
            "orange_eaten": ghosts[1].eaten if len(ghosts) > 1 else 0,
            "food_remaining": len(self._sim.food),
            "won": int(self._sim.won),
            "lost": int(self._sim.lost),
        }

    def labeling_state(self) -> PacmanAuthorityState:
        """Return the detached authority snapshot used by monitors and policy fixes."""
        if self._sim is None:
            raise ResetNeeded("PacmanEnv must be reset before labeling_state()")
        return self._sim.snapshot()

    def state(self) -> PacmanSnapshot:
        """Return detached gameplay state for custom observations, including schedules."""
        if self._sim is None:
            raise ResetNeeded("PacmanEnv must be reset before state()")
        return self._sim.full_snapshot()

    def get_action_meanings(self) -> list[int]:
        """Return the stable integer IDs for stop, north, south, east and west."""
        return list(PACMAN_ACTIONS)

    def render(self) -> NDArray[np.uint8] | None:
        """Produce a detached RGB frame or present the configured human window."""
        if self.render_mode is None:
            gym.logger.warn("Specify render_mode at construction to produce an image.")
            return None
        try:
            return self._renderer.render()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release this renderer without invalidating other live environments."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            renderer.close()


__all__ = [
    "SUPPORTED_FEATURES",
    "SUPPORTED_VECTOR_FEATURES",
    "GhostConfig",
    "PacmanAuthorityState",
    "PacmanEnv",
    "PacmanGhostState",
    "PacmanLayout",
    "PacmanLayoutState",
    "PacmanPlayerState",
    "PacmanSnapshot",
]
