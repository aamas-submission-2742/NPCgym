"""The small boundary to Gymnasium 1.3's dry Taxi implementation.

NPC Gym owns storm state and randomness. The backend supplies deterministic base
outcomes and masks without calling upstream reset/step or consuming its RNG.
"""

from typing import Protocol, cast

import numpy as np
from gymnasium.envs.toy_text.taxi import TaxiEnv as GymnasiumTaxi
from numpy.typing import NDArray

BaseOutcome = tuple[float, int, int, bool]


class _TaxiDrawing(Protocol):
    # The pinned upstream implementation leaves these two methods unannotated.
    def render(self) -> object: ...

    def get_surf_loc(self, map_loc: tuple[int, int]) -> tuple[float, float]: ...


class TaxiBackend:
    """Own one upstream environment; never expose its mutable state to callers."""

    def __init__(self, render_mode: str | None = None) -> None:
        self._env = GymnasiumTaxi(render_mode=render_mode)
        self._drawing = cast(_TaxiDrawing, self._env)
        self._transitions = cast(dict[int, dict[int, list[BaseOutcome]]], self._env.P)
        self.locations = tuple(self._env.locs)
        self.initial_distribution = cast(NDArray[np.float64], self._env.initial_state_distrib)
        # Upstream has only 500 states. Store detached seven-action masks once;
        # each caller receives a copy, and Warn is always permitted.
        self.masks: NDArray[np.int8] = np.ones((500, 7), dtype=np.int8)
        for state in range(500):
            self.masks[state, :6] = self._env.action_mask(state)
        self.masks.flags.writeable = False

    def outcome(self, state: int, action: int) -> BaseOutcome:
        """Read a deterministic upstream outcome, or the storm-only Warn no-op."""
        if action == 6:
            return 1.0, state, -1, False
        return self._transitions[state][action][0]

    def reset_orientation(self) -> None:
        """Reset the direction retained by upstream between rendered movements."""
        self._env.taxi_orientation = 0

    def render(self, state: int, action: int | None) -> str | NDArray[np.uint8]:
        """Synchronize only rendering state; Warn has no upstream action number."""
        self._env.s = state
        self._env.lastaction = action if action != 6 else None
        frame = self._drawing.render()
        if isinstance(frame, str):
            return frame.removesuffix("\n") + "  (Warn)\n" if action == 6 else frame
        if isinstance(frame, np.ndarray) and frame.dtype == np.uint8:
            return cast(NDArray[np.uint8], frame)
        raise TypeError("Gymnasium Taxi renderer did not return text or a uint8 RGB frame")

    @property
    def cell_size(self) -> tuple[float, float]:
        return float(self._env.cell_size[0]), float(self._env.cell_size[1])

    def surface_location(self, location: tuple[int, int]) -> tuple[float, float]:
        x, y = self._drawing.get_surf_loc(location)
        return float(x), float(y)

    def release_rendering(self) -> None:
        """Drop upstream surfaces without its process-wide pygame.quit().

        Gymnasium 1.3 stores its rendering resources in these attributes. NPC
        Gym's display owner releases shared Pygame state after the last renderer.
        Dynamics and the instance itself remain reusable after close/reset.
        """
        self._env.window = None
        self._env.clock = None
        self._env.taxi_imgs = None
        self._env.passenger_img = None
        self._env.destination_img = None
        self._env.median_horiz = None
        self._env.median_vert = None
        self._env.background_img = None
