from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pygame
from numpy.typing import NDArray

from npc_gym.envs import pygame_display

if TYPE_CHECKING:
    from npc_gym.envs.merchant.merchant import MerchantEnv


class MerchantGraphicsPyGame:
    def __init__(
        self,
        cell_size: int = 48,
        hud_top: int = 20,
        hud_bottom: int = 0,
        show_window: bool = True,
        step_delay_ms: int = 0,
    ) -> None:
        self.cell_size = cell_size
        self.hud_top = hud_top
        self.hud_bottom = hud_bottom
        self.show_window = show_window
        self.step_delay_ms = step_delay_ms
        self.window: pygame.Surface | None = None
        self.image: NDArray[np.uint8] | None = None
        self._display_active = False

    def _dims(self, grid_w: int, grid_h: int) -> tuple[int, int]:
        self.panel_width = 112
        width = grid_w * self.cell_size + self.panel_width
        height = self.hud_top + grid_h * self.cell_size + self.hud_bottom
        return width, height

    def initialize(self, grid_w: int, grid_h: int) -> None:
        if not self._display_active:
            pygame_display.acquire()
            self._display_active = True
        pygame.init()
        width, height = self._dims(grid_w, grid_h)
        self.window = pygame.Surface((width, height))

    def close(self) -> None:
        self.window = None
        self.image = None
        if self._display_active:
            self._display_active = False
            pygame_display.release()

    def _cell_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(x * self.cell_size, self.hud_top + y * self.cell_size, self.cell_size, self.cell_size)

    def _draw_static_cell(self, surface: pygame.Surface, rect: pygame.Rect, tile: str | np.str_) -> None:
        if tile == "X":
            pygame.draw.rect(surface, (115, 115, 125), rect)
            pygame.draw.line(surface, (15, 15, 20), rect.topleft, rect.bottomright, 2)
            pygame.draw.line(surface, (15, 15, 20), rect.topright, rect.bottomleft, 2)
            return

        pygame.draw.rect(surface, (28, 28, 34), rect, 1)
        cx, cy = rect.center
        s = self.cell_size

        if tile == "H":
            # stylized house: base square + roof triangle
            base = pygame.Rect(rect.left + int(0.22 * s), rect.top + int(0.40 * s), int(0.56 * s), int(0.42 * s))
            roof = [
                (rect.left + int(0.14 * s), rect.top + int(0.40 * s)),
                (rect.right - int(0.14 * s), rect.top + int(0.40 * s)),
                (cx, rect.top + int(0.12 * s)),
            ]
            pygame.draw.rect(surface, (220, 195, 130), base)
            pygame.draw.polygon(surface, (210, 65, 65), roof)
        elif tile == "M":
            # market as a flag on a pole
            pole_x = rect.left + int(0.32 * s)
            pole_top = rect.top + int(0.2 * s)
            pole_bot = rect.bottom - int(0.18 * s)
            pygame.draw.line(surface, (190, 190, 205), (pole_x, pole_top), (pole_x, pole_bot), 3)
            flag = [
                (pole_x + 2, pole_top + int(0.04 * s)),
                (rect.right - int(0.18 * s), rect.top + int(0.30 * s)),
                (pole_x + 2, rect.top + int(0.50 * s)),
            ]
            pygame.draw.polygon(surface, (70, 150, 230), flag)
            pygame.draw.polygon(surface, (25, 60, 105), flag, 2)
        elif tile == "T":
            pts = [
                (cx, rect.top + int(0.2 * s)),
                (rect.left + int(0.25 * s), rect.bottom - int(0.2 * s)),
                (rect.right - int(0.25 * s), rect.bottom - int(0.2 * s)),
            ]
            pygame.draw.polygon(surface, (60, 180, 95), pts)
            pygame.draw.line(
                surface, (110, 85, 55), (cx, rect.bottom - int(0.2 * s)), (cx, rect.bottom - int(0.05 * s)), 4
            )
        elif tile == "R":
            # rock-like irregular boulder
            rock = [
                (rect.left + int(0.22 * s), rect.top + int(0.68 * s)),
                (rect.left + int(0.18 * s), rect.top + int(0.45 * s)),
                (rect.left + int(0.30 * s), rect.top + int(0.25 * s)),
                (rect.left + int(0.52 * s), rect.top + int(0.18 * s)),
                (rect.left + int(0.74 * s), rect.top + int(0.28 * s)),
                (rect.left + int(0.80 * s), rect.top + int(0.52 * s)),
                (rect.left + int(0.66 * s), rect.top + int(0.74 * s)),
                (rect.left + int(0.40 * s), rect.top + int(0.80 * s)),
            ]
            pygame.draw.polygon(surface, (108, 116, 136), rock)
            pygame.draw.polygon(surface, (62, 68, 84), rock, 2)
        elif tile == "D":
            pts = [
                (cx, rect.top + int(0.18 * s)),
                (rect.right - int(0.18 * s), cy),
                (cx, rect.bottom - int(0.18 * s)),
                (rect.left + int(0.18 * s), cy),
            ]
            pygame.draw.polygon(surface, (230, 80, 80), pts)
            pygame.draw.rect(surface, (45, 20, 20), pygame.Rect(cx - 2, cy - int(0.15 * s), 4, int(0.22 * s)))
            pygame.draw.circle(surface, (45, 20, 20), (cx, cy + int(0.16 * s)), 2)

    def _draw_depleted_marker(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        s = self.cell_size
        p1 = (rect.left + int(0.12 * s), rect.bottom - int(0.12 * s))
        p2 = (rect.left + int(0.32 * s), rect.bottom - int(0.12 * s))
        p3 = (rect.left + int(0.12 * s), rect.bottom - int(0.32 * s))
        pygame.draw.line(surface, (170, 170, 170), p1, p2, 2)
        pygame.draw.line(surface, (170, 170, 170), p1, p3, 2)

    def _draw_agent(self, surface: pygame.Surface, x: int, y: int, last_action: str | None, attacked: bool) -> None:
        rect = self._cell_rect(x, y)
        cx, cy = rect.center
        s = self.cell_size
        radius = int(0.22 * s)
        pygame.draw.circle(surface, (255, 245, 220), (cx, cy), radius)

        if attacked:
            # fight/attack state: highlight full agent cell in red
            pygame.draw.rect(surface, (205, 60, 60), rect, 4)
            inner = rect.inflate(-6, -6)
            pygame.draw.rect(surface, (140, 30, 30), inner, 2)

    def _draw_time(self, surface: pygame.Surface, clock: int, sunset: int, width: int) -> None:
        y = 4
        bar_rect = pygame.Rect(8, y, width - 16, 10)
        pygame.draw.rect(surface, (35, 35, 45), bar_rect)
        frac = 1.0 if sunset <= 0 else min(clock, sunset) / sunset
        fill_w = int(frac * (bar_rect.width - 2))
        if fill_w > 0:
            pygame.draw.rect(
                surface,
                (255, 195, 90),
                pygame.Rect(bar_rect.left + 1, bar_rect.top + 1, fill_w, bar_rect.height - 2),
            )
        pygame.draw.rect(surface, (120, 120, 135), bar_rect, 1)

    def _draw_inventory(
        self, surface: pygame.Surface, carried_wood: int, carried_ore: int, capacity: int, map_width: int, height: int
    ) -> None:
        panel_x = map_width
        panel_rect = pygame.Rect(panel_x, self.hud_top, self.panel_width, height - self.hud_top)
        # game-like side panel styling: dark blue slate with framed border
        pygame.draw.rect(surface, (26, 32, 46), panel_rect)
        # subtle vertical stripes to separate from maze background
        for x in range(panel_rect.left + 6, panel_rect.right, 10):
            pygame.draw.line(surface, (31, 38, 55), (x, panel_rect.top + 2), (x, panel_rect.bottom - 2), 1)

        # panel frame
        pygame.draw.rect(surface, (96, 110, 138), panel_rect, 2)
        inner_frame = panel_rect.inflate(-6, -6)
        pygame.draw.rect(surface, (56, 68, 90), inner_frame, 1)

        # separator from maze
        pygame.draw.line(surface, (130, 140, 165), (panel_x, self.hud_top), (panel_x, height), 3)

        slot_size = 40
        gap = 8
        col_gap = 16
        content_w = slot_size * 2 + col_gap
        x_wood = panel_x + (self.panel_width - content_w) // 2
        x_ore = x_wood + slot_size + col_gap
        content_h = capacity * slot_size + (capacity - 1) * gap
        y_start = self.hud_top + max(6, ((height - self.hud_top) - content_h) // 2)

        for i in range(capacity):
            y = y_start + i * (slot_size + gap)

            rect_w = pygame.Rect(x_wood, y, slot_size, slot_size)
            pygame.draw.rect(surface, (120, 132, 160), rect_w, 1)
            pygame.draw.rect(surface, (36, 42, 58), rect_w.inflate(-2, -2))
            if i < carried_wood:
                pts = [
                    (rect_w.centerx, rect_w.top + 5),
                    (rect_w.left + 7, rect_w.bottom - 8),
                    (rect_w.right - 7, rect_w.bottom - 8),
                ]
                pygame.draw.polygon(surface, (60, 180, 95), pts)
                pygame.draw.line(
                    surface, (110, 85, 55), (rect_w.centerx, rect_w.bottom - 8), (rect_w.centerx, rect_w.bottom - 2), 4
                )

            rect_o = pygame.Rect(x_ore, y, slot_size, slot_size)
            pygame.draw.rect(surface, (120, 132, 160), rect_o, 1)
            pygame.draw.rect(surface, (36, 42, 58), rect_o.inflate(-2, -2))
            if i < carried_ore:
                s = slot_size
                rock = [
                    (rect_o.left + int(0.24 * s), rect_o.top + int(0.76 * s)),
                    (rect_o.left + int(0.14 * s), rect_o.top + int(0.50 * s)),
                    (rect_o.left + int(0.30 * s), rect_o.top + int(0.26 * s)),
                    (rect_o.left + int(0.56 * s), rect_o.top + int(0.16 * s)),
                    (rect_o.left + int(0.80 * s), rect_o.top + int(0.30 * s)),
                    (rect_o.left + int(0.86 * s), rect_o.top + int(0.58 * s)),
                    (rect_o.left + int(0.68 * s), rect_o.top + int(0.80 * s)),
                    (rect_o.left + int(0.36 * s), rect_o.top + int(0.86 * s)),
                ]
                pygame.draw.polygon(surface, (108, 116, 136), rock)
                pygame.draw.polygon(surface, (62, 68, 84), rock, 2)

    def render(self, env: MerchantEnv) -> NDArray[np.uint8]:
        """Return an independent RGB frame, presenting and delaying it only in human mode."""
        grid_h, grid_w = env.map.shape
        if self.window is None:
            self.initialize(grid_w, grid_h)
        assert self.window is not None

        surface = self.window
        width, height = self._dims(grid_w, grid_h)
        surface.fill((0, 0, 0))

        map_width = grid_w * self.cell_size
        self._draw_time(surface, env.clock, env.sunset, width)

        for y in range(grid_h):
            for x in range(grid_w):
                rect = self._cell_rect(x, y)
                if (x + y) % 2 == 0:
                    pygame.draw.rect(surface, (14, 20, 16), rect)
                else:
                    pygame.draw.rect(surface, (18, 16, 24), rect)
                tile = env.map[y, x]
                if tile == "T" and env.wood[env.wood_positions[(y, x)]] == 0:
                    tile = "."
                if tile == "R" and env.ore[env.ore_positions[(y, x)]] == 0:
                    tile = "."
                self._draw_static_cell(surface, rect, tile)

        self._draw_agent(surface, int(env.pos[0]), int(env.pos[1]), env.action, env.label == "D")
        self._draw_inventory(surface, env.carried_wood, env.carried_ore, env.capacity, map_width, height)

        if self.show_window:
            pygame_display.present(surface, caption="Merchant")
            if self.step_delay_ms > 0:
                time.sleep(self.step_delay_ms / 1000.0)

        self.image = np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)).copy()
        return self.image
