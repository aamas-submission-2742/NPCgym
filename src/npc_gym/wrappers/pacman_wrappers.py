"""Optional norm-history observations for Pacman's feature vectors."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from npc_gym.envs.pacman.labels import PacmanLabel


class TrappedObservation(gym.Wrapper[Any, Any, Any, Any]):
    """Append a 0/1 Trapped-active feature without changing rewards or labels.

    Requires a one-dimensional floating-point Box observation and Pacman labels
    in ``info["labels"]``. Reset labels are consumed: score zero activates the
    restriction, score greater than 400 clears it, and other inputs retain it.
    A high-score label takes precedence if both are present. Final observations
    include the final transition's updated flag. Reset clears previous history.
    The output keeps the original observation's dtype and features.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 1
            or not np.issubdtype(space.dtype, np.floating)
        ):
            raise TypeError("TrappedObservation requires a one-dimensional floating-point Box observation")
        self.observation_space = gym.spaces.Box(
            low=np.append(space.low, 0).astype(space.dtype),
            high=np.append(space.high, 1).astype(space.dtype),
            dtype=space.low.dtype.type,
        )
        self._active = False

    def reset(self, **kwargs: Any) -> tuple[NDArray[Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._active = False
        return self._observation(observation, info), info

    def step(self, action: Any) -> tuple[NDArray[Any], Any, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(observation, info), reward, terminated, truncated, info

    def _observation(self, observation: NDArray[Any], info: dict[str, Any]) -> NDArray[Any]:
        labels = info["labels"]
        if PacmanLabel.SCORE_GREATER_400 in labels:
            self._active = False
        elif PacmanLabel.SCORE_0 in labels:
            self._active = True
        return np.append(observation, self._active).astype(self.observation_space.dtype)


class PacmanPixelObservation(gym.Wrapper[Any, Any, Any, Any]):
    """Grayscale/stack Pacman pixels while retaining numeric side channels.

    Accept raw image observations, scheduled image dictionaries, or either
    wrapped by ``RestrainingBoltWrapper``. Output has ``observation`` pixels,
    ``ghost_mode`` when present, and ``automata`` when present, in one flat Dict.
    With no numeric fields output is an image Box. Only pixels are transformed.
    Frames are oldest first on the last axis; reset repeats the initial frame.
    """

    def __init__(self, env: gym.Env[Any, Any], *, height: int = 80, width: int = 210, frames: int = 2) -> None:
        super().__init__(env)
        for name, value in (("height", height), ("width", width), ("frames", frames)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._height, self._width, self._frames = height, width, frames
        space = env.observation_space
        side: dict[str, gym.Space[Any]] = {}
        self._bolts = isinstance(space, gym.spaces.Dict) and "automata" in space.spaces
        if self._bolts:
            assert isinstance(space, gym.spaces.Dict)
            side["automata"] = space["automata"]
            space = space["observation"]
        self._scheduled = isinstance(space, gym.spaces.Dict)
        if self._scheduled:
            assert isinstance(space, gym.spaces.Dict)
            if set(space.spaces) != {"observation", "ghost_mode"}:
                raise TypeError("expected Pacman image and ghost_mode observation fields")
            side["ghost_mode"] = space["ghost_mode"]
            space = space["observation"]
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 3
            or space.shape[-1] != 3
            or space.dtype != np.uint8
        ):
            raise TypeError("PacmanPixelObservation requires RGB images; use bolts for numeric automaton features")
        image_space = gym.spaces.Box(0, 255, (height, width, frames), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({"observation": image_space, **side}) if side else image_space
        self._history: list[NDArray[np.uint8]] = []

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._history.clear()
        return self._transform(observation), info

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(observation), reward, terminated, truncated, info

    def _transform(self, observation: Any) -> Any:
        from PIL import Image

        side: dict[str, Any] = {}
        if self._bolts:
            side["automata"] = observation["automata"].copy()
            observation = observation["observation"]
        if self._scheduled:
            side["ghost_mode"] = observation["ghost_mode"].copy()
            observation = observation["observation"]
        frame = np.asarray(
            Image.fromarray(observation).convert("L").resize((self._width, self._height), Image.Resampling.BOX)
        )
        if not self._history:
            self._history = [frame] * self._frames
        else:
            self._history = [*self._history[1:], frame]
        pixels = np.stack(self._history, axis=-1)
        return {"observation": pixels, **side} if side else pixels
