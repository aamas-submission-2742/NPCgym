"""NPC Gym storm overlays on frames produced by the installed Gymnasium renderer."""

from typing import TYPE_CHECKING

import numpy as np
from gymnasium.error import DependencyNotInstalled
from numpy.typing import NDArray

from npc_gym.envs import pygame_display
from npc_gym.envs.taxi._taxi_backend import TaxiBackend
from npc_gym.envs.taxi.labels import TaxiAuthorityState

if TYPE_CHECKING:
    from pygame import Surface
    from pygame.font import Font
    from pygame.time import Clock


class StormRenderer:
    """Own one share of Pygame and present only the completed, composed frame."""

    def __init__(self, backend: TaxiBackend) -> None:
        self._backend = backend
        self._active = False
        self._font: Font | None = None
        self._small_font: Font | None = None
        self._clock: Clock | None = None

    def render(
        self,
        state: int,
        action: int | None,
        snapshot: TaxiAuthorityState,
        *,
        human: bool,
        fps: int,
    ) -> NDArray[np.uint8] | None:
        try:
            import pygame
        except ImportError as error:
            raise DependencyNotInstalled('pygame is not installed, run `pip install "npc-gym[render]"`') from error
        if not self._active:
            pygame_display.acquire()
            self._active = True
        # Upstream is always in rgb_array mode: it neither presents nor paces.
        pixels = self._backend.render(state // 704, action)
        if not isinstance(pixels, np.ndarray):
            raise TypeError("Graphical Storm Taxi rendering requires an upstream RGB array")
        surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
        if self._font is None:
            self._font = pygame.font.SysFont("Arial", 16)
            self._small_font = pygame.font.SysFont("Arial", 14)
        self._overlays(surface, snapshot)
        if human:
            pygame_display.present(surface, caption="Storm Taxi")
            if self._clock is None:
                self._clock = pygame.time.Clock()
            self._clock.tick(fps)
            return None
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def _overlays(self, surface: "Surface", state: TaxiAuthorityState) -> None:
        import pygame

        assert self._font is not None and self._small_font is not None
        width, height = surface.get_size()
        for enabled, tint_color in (
            (state.raining, (70, 110, 170, 45)),
            (state.hurricane > 0, (65, 65, 95, min(120, 25 + state.hurricane * 8))),
            (state.flood_risk, (50, 120, 180, 45)),
        ):
            if enabled:
                tint = pygame.Surface((width, height), pygame.SRCALPHA)
                tint.fill(tint_color)
                surface.blit(tint, (0, 0))
        weather = f"Hurricane L{state.hurricane}" if state.hurricane else ("Rain" if state.raining else "Clear")
        for rect, text, color in (
            ((8, 8, 180, 28), f"Weather: {weather}", (255, 255, 255)),
            (
                (width - 158, 8, 150, 28),
                f"Flood: {'ON' if state.flood_risk else 'OFF'}",
                (160, 220, 255) if state.flood_risk else (255, 255, 255),
            ),
        ):
            badge = pygame.Surface(rect[2:], pygame.SRCALPHA)
            badge.fill((20, 20, 20, 180))
            surface.blit(badge, rect[:2])
            surface.blit(self._font.render(text, True, color), (rect[0] + 6, rect[1] + 5))
        pygame.draw.rect(surface, (20, 20, 20), (8, 42, 180, 16))
        pygame.draw.rect(surface, (80, 80, 80), (10, 44, 176, 12))
        if state.hurricane:
            pygame.draw.rect(surface, (220, 80, 80), (10, 44, int(176 * state.hurricane / 10), 12))
        cell_width, cell_height = self._backend.cell_size
        for location, text, color in ((state.home, "H", (80, 255, 255)), (state.shelter, "S", (255, 170, 80))):
            if location is not None:
                x, y = self._backend.surface_location(self._backend.locations[location])
                center = (int(x + cell_width // 2), int(y + cell_height // 2))
                pygame.draw.circle(surface, color, center, int(min(cell_width, cell_height) * 0.22), 3)
                surface.blit(self._small_font.render(text, True, color), (center[0] - 5, center[1] - 8))

    def close(self) -> None:
        """Release this instance's resources; keep other live environments usable."""
        self._backend.release_rendering()
        self._font = None
        self._small_font = None
        self._clock = None
        if self._active:
            self._active = False
            pygame_display.release()
