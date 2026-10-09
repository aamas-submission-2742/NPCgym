from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

import pygame

from npc_gym.envs import pygame_display

if TYPE_CHECKING:
    from npc_gym.envs.gardener.gardener import GardenerEnv


class GardenerRenderer:
    def __init__(self) -> None:
        self._display_active = False
        self.clock: pygame.time.Clock | None = None

    def close(self) -> None:
        self.clock = None
        if hasattr(self, "font"):
            del self.font
        if hasattr(self, "screen"):
            del self.screen
        if self._display_active:
            self._display_active = False
            pygame_display.release()

    def draw(self, state: GardenerEnv) -> None:
        """Draw the current world, present it, and pace the human frame."""
        if not hasattr(self, "screen"):
            pygame_display.acquire()
            self._display_active = True
            pygame.init()
            self.window_size = 600
            self.cell_size = self.window_size // state.size
            self.screen = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw grid lines
        for i in range(state.size + 1):
            pygame.draw.line(
                self.screen, (220, 220, 220), (i * self.cell_size, 0), (i * self.cell_size, self.window_size)
            )
            pygame.draw.line(
                self.screen, (220, 220, 220), (0, i * self.cell_size), (self.window_size, i * self.cell_size)
            )

        # Draw grass patches
        for idx, (gx, gy) in enumerate(state.grass):
            self._draw_grass(int(gx), int(gy), bool(state.grass_active[idx]))

        # Draw puddles
        for idx, (px, py) in enumerate(state.puddles):
            self._draw_puddle(int(px), int(py), bool(state.puddles_full[idx]))

        # Draw walls
        for wx, wy in state.walls:
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                pygame.Rect(wx * self.cell_size, wy * self.cell_size, self.cell_size, self.cell_size),
            )

        # Draw frogs
        for i, (fx, fy) in enumerate(state.frogs):
            status = 0
            if state.captured_frogs[i] or state.collected_frogs[i]:
                status = 1
            self._draw_frog(int(fx), int(fy), status)

        # Draw agent
        ax, ay = state.agent
        self._draw_human(int(ax), int(ay))

        # Draw score
        if not hasattr(self, "font"):
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)
        score = self.font.render(f"{state.score}", True, (255, 0, 0))
        self.screen.blit(score, (5, 5))

        pygame_display.present(self.screen, caption="GardenerEnv")
        assert self.clock is not None
        self.clock.tick(state.metadata["render_fps"])

    def save_screenshot(self, filename: str | PathLike[str]) -> None:
        """Save the current frame if one exists; do nothing before drawing or after close."""
        if hasattr(self, "screen"):
            pygame.image.save(self.screen, filename)

    def _draw_grass(self, x: int, y: int, active: bool) -> None:
        base_x = x * self.cell_size
        base_y = y * self.cell_size
        if active:
            # Draw blades of grass
            for i in range(5):
                offset_x = base_x + (i + 0.5) * (self.cell_size / 5)
                pygame.draw.line(
                    self.screen,
                    (50, 50, 50),
                    (offset_x, base_y + self.cell_size),
                    (offset_x, base_y + self.cell_size * 0.4),
                    2,
                )
        else:
            # Draw stubble
            for i in range(5):
                offset_x = base_x + (i + 0.5) * (self.cell_size / 5)
                pygame.draw.line(
                    self.screen,
                    (150, 150, 150),
                    (offset_x, base_y + self.cell_size),
                    (offset_x, base_y + self.cell_size * 0.8),
                    1,
                )

    def _draw_puddle(self, x: int, y: int, full: bool) -> None:
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        color = (180, 180, 180) if full else (230, 230, 230)
        pygame.draw.rect(self.screen, color, rect)
        # Draw waves
        start_y = y * self.cell_size + self.cell_size // 2
        start_x = x * self.cell_size + 5
        end_x = (x + 1) * self.cell_size - 5
        pygame.draw.line(self.screen, (100, 100, 100), (start_x, start_y), (end_x, start_y), 2)
        pygame.draw.line(self.screen, (100, 100, 100), (start_x + 5, start_y + 5), (end_x - 5, start_y + 5), 2)

    def _draw_frog(self, x: int, y: int, status: int) -> None:
        center_x = int((x + 0.5) * self.cell_size)
        center_y = int((y + 0.5) * self.cell_size)
        radius = int(self.cell_size * 0.3)

        # Body
        color = (80, 80, 80) if status == 0 else (200, 200, 200)
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), radius, 1)

        # Eyes
        eye_radius = radius // 3
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x - radius // 2, center_y - radius // 2), eye_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x + radius // 2, center_y - radius // 2), eye_radius)
        pygame.draw.circle(self.screen, (0, 0, 0), (center_x - radius // 2, center_y - radius // 2), eye_radius, 1)
        pygame.draw.circle(self.screen, (0, 0, 0), (center_x + radius // 2, center_y - radius // 2), eye_radius, 1)

    def _draw_human(self, x: int, y: int) -> None:
        center_x = int((x + 0.5) * self.cell_size)
        center_y = int((y + 0.5) * self.cell_size)

        # Head
        head_radius = int(self.cell_size * 0.15)
        head_center = (center_x, center_y - int(self.cell_size * 0.2))
        pygame.draw.circle(self.screen, (0, 0, 0), head_center, head_radius)

        # Body
        body_top = (center_x, head_center[1] + head_radius)
        body_bottom = (center_x, center_y + int(self.cell_size * 0.2))
        pygame.draw.line(self.screen, (0, 0, 0), body_top, body_bottom, 2)

        # Arms
        arm_y = body_top[1] + int(self.cell_size * 0.1)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (center_x - int(self.cell_size * 0.15), arm_y),
            (center_x + int(self.cell_size * 0.15), arm_y),
            2,
        )

        # Legs
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            body_bottom,
            (center_x - int(self.cell_size * 0.1), body_bottom[1] + int(self.cell_size * 0.2)),
            2,
        )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            body_bottom,
            (center_x + int(self.cell_size * 0.1), body_bottom[1] + int(self.cell_size * 0.2)),
            2,
        )
