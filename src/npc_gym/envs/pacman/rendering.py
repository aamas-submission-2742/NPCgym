"""Original minimalist Pygame graphics with cached geometry and lazy RGB frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from gymnasium.error import DependencyNotInstalled, ResetNeeded
from numpy.typing import NDArray

from npc_gym.envs import pygame_display

from .layout import PacmanLayout, Position
from .simulation import Simulation

if TYPE_CHECKING:
    import pygame

RenderMode = Literal["human", "rgb_array"]
BACKGROUND = (15, 21, 32)
# Blue/orange identities remain stable; luminance also separates all four ghosts.
COLORS = ((45, 95, 190), (255, 175, 65), (55, 180, 95), (245, 225, 255))
CELL = 30


class PacmanRenderer:
    """Own off-screen surfaces; share only Pygame's process-wide display lease."""

    def __init__(
        self, layout: PacmanLayout, render_mode: RenderMode | None, *, enabled: bool, render_fps: int = 10
    ) -> None:
        self.layout, self.render_mode, self.enabled = layout, render_mode, enabled
        self._fps = render_fps
        self._active = False
        self._sim: Simulation | None = None
        self._base: pygame.Surface | None = None
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._sprites: dict[tuple[int, bool, int], pygame.Surface] = {}
        self._pixels: NDArray[np.uint8] | None = None
        self._dirty = True
        if enabled:
            try:
                import pygame  # noqa: F401
            except ImportError as error:
                raise DependencyNotInstalled("Pacman graphics require `pip install 'npc-gym[render]'`") from error

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.layout.height + 1) * CELL, (self.layout.width + 1) * CELL

    def _point(self, position: Position) -> tuple[int, int]:
        return round((position[0] + 1) * CELL), round((self.layout.height - position[1]) * CELL)

    def reset(self, sim: Simulation) -> None:
        self._sim = sim
        self._base = None
        self._pixels = None
        self._dirty = True
        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        import pygame

        if not self._active:
            pygame_display.acquire()
            self._active = True
        pygame.font.init()
        if self._font is None:
            self._font = pygame.font.Font(None, 20)
        height, width = self.frame_shape
        self._base = pygame.Surface((width, height))
        self._base.fill(BACKGROUND)
        for wall in self.layout.walls:
            x, y = self._point(wall)
            pygame.draw.rect(self._base, (38, 58, 78), (x - 14, y - 14, 28, 28), border_radius=6)
            pygame.draw.rect(self._base, (53, 79, 101), (x - 14, y - 14, 28, 28), width=1, border_radius=6)
        assert self._sim is not None
        self._food = set(self._sim.food)
        self._capsules = set(self._sim.capsules)
        for cell in self._food:
            pygame.draw.circle(self._base, (211, 222, 222), self._point(cell), 3)
        for cell in self._capsules:
            pygame.draw.circle(self._base, (241, 214, 134), self._point(cell), 7, width=2)
        self._surface = self._base.copy()

    def update(self, sim: Simulation) -> None:
        self._sim = sim
        self._dirty = True
        self._pixels = None

    def _sprite(self, identity: int, frightened: bool, direction: int) -> pygame.Surface:
        import pygame

        key = identity, frightened, direction
        if key not in self._sprites:
            sprite = pygame.Surface((28, 28), pygame.SRCALPHA)
            if identity == 0:
                pygame.draw.circle(sprite, (252, 210, 86), (14, 14), 11)
                dx, dy = ((1, 0), (0, -1), (0, 1), (1, 0), (-1, 0))[direction]
                pygame.draw.polygon(
                    sprite,
                    BACKGROUND,
                    (
                        (14, 14),
                        (14 + 14 * dx - 6 * dy, 14 + 14 * dy + 6 * dx),
                        (14 + 14 * dx + 6 * dy, 14 + 14 * dy - 6 * dx),
                    ),
                )
            else:
                color = COLORS[(identity - 1) % len(COLORS)]
                pygame.draw.rect(sprite, color, (3, 3, 22, 22), border_radius=8)
                for x in (9, 19):
                    pygame.draw.circle(sprite, BACKGROUND, (x, 12), 3)
                if frightened:
                    pygame.draw.lines(sprite, BACKGROUND, False, ((7, 20), (10, 18), (13, 20), (16, 18), (20, 20)), 3)
            self._sprites[key] = sprite
        return self._sprites[key]

    def _draw(self) -> None:
        import pygame

        if self._sim is None:
            raise ResetNeeded("PacmanEnv must be reset before render()")
        if self._base is None:
            self._initialize()
        if not self._dirty:
            return
        assert self._base is not None and self._surface is not None and self._font is not None
        sim = self._sim
        for cell in (self._food - sim.food) | (self._capsules - sim.capsules):
            x, y = self._point(cell)
            pygame.draw.rect(self._base, BACKGROUND, (x - 9, y - 9, 18, 18))
        self._food, self._capsules = set(sim.food), set(sim.capsules)
        self._surface.blit(self._base, (0, 0))
        for index, ghost in enumerate(sim.ghosts, 1):
            x, y = self._point(ghost.position)
            self._surface.blit(self._sprite(index, bool(ghost.timer), ghost.direction), (x - 14, y - 14))
        x, y = self._point(sim.player)
        self._surface.blit(self._sprite(0, False, sim.direction), (x - 14, y - 14))
        status = "WIN" if sim.won else "GAME OVER" if sim.lost else ""
        phase = f"{sim.phase} {sim.phase_remaining}" if sim.phase else "random"
        frightened = max((ghost.timer for ghost in sim.ghosts), default=0)
        if frightened:
            phase = f"frightened {frightened} / {phase}"
        caption = f"{int(sim.score)}   {phase}   {status}"
        self._surface.blit(self._font.render(caption, True, (210, 219, 227)), (18, self.frame_shape[0] - 23))
        self._dirty = False

    def frame(self, *, crop: bool = False) -> NDArray[np.uint8]:
        if not self.enabled:
            raise RuntimeError("Pacman renderer is not enabled")
        self._draw()
        if self._pixels is None:
            import pygame

            assert self._surface is not None
            self._pixels = np.frombuffer(pygame.image.tobytes(self._surface, "RGB"), dtype=np.uint8).reshape(
                (*self.frame_shape, 3)
            )
        if not crop:
            return self._pixels.copy()
        from PIL import Image

        assert self._sim is not None
        height, width = self.frame_shape
        sx, sy = width / self.layout.width, height / self.layout.height
        x, y = self._sim.player
        extent = (
            int(sx * (x - 1)),
            int(sy * (self.layout.height - (y + 2.2))),
            int(sx * (x + 2)),
            int(sy * (self.layout.height - (y - 1.2))),
        )
        return np.array(Image.fromarray(self._pixels).crop(extent).resize((84, 84)), dtype=np.uint8)

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode == "rgb_array":
            return self.frame()
        if self.render_mode == "human":
            import pygame

            self._draw()
            assert self._surface is not None
            pygame_display.present(self._surface, caption="NPC Gym — maze chase")
            if self._clock is None:
                self._clock = pygame.time.Clock()
            self._clock.tick(self._fps)
        return None

    def close(self) -> None:
        self._base = self._surface = self._font = self._clock = None
        self._pixels = None
        self._sprites.clear()
        self._dirty = True
        if self._active:
            self._active = False
            pygame_display.release()
