"""Compact turn-based maze simulation, independent of observations and Pygame."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .labels import PacmanAuthorityState, PacmanGhostState, PacmanLayoutState, PacmanPlayerState
from .layout import REVERSE, VECTORS, Cell, PacmanLayout, Position
from .state import GhostBehavior, GhostConfig, PacmanSnapshot

FRIGHTENED_TURNS = 40
COLLISION_DISTANCE = 0.7


@dataclass(slots=True)
class Ghost:
    """Private mutable actor; positions are exact integer or half-cell values."""

    position: Position
    spawn: Cell
    direction: int = 0
    timer: int = 0
    eaten: int = 0
    reverse_pending: bool = False


def nearest_cell(position: Position) -> Cell:
    return int(position[0] + 0.5), int(position[1] + 0.5)


def manhattan(first: Position, second: Position) -> float:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _separated(first: Position, second: Position, opposing: bool) -> bool:
    distance = manhattan(first, second)
    return distance >= 1 or (opposing and distance > 0)


class Simulation:
    """Advance the player, then ghosts, with mode-specific movement ordering.

    Random behavior preserves the characterized NPC Gym transition contract,
    including random pauses, sorted cumulative sampling and timer expiry
    before collision. Scheduled ghosts reserve distinct destinations together,
    then resolve player collisions in identity order. Mutable state never escapes.
    """

    def __init__(
        self,
        layout: PacmanLayout,
        rng: np.random.Generator,
        *,
        behavior: GhostBehavior = "random",
        config: GhostConfig | None = None,
    ) -> None:
        self.layout = layout
        if behavior not in ("random", "deterministic", "partly-deterministic"):
            raise ValueError(f"Unknown ghost_behavior {behavior!r}")
        if config is not None and not isinstance(config, GhostConfig):
            raise TypeError("ghost_config must be a GhostConfig or None")
        self.behavior = behavior
        requested = config or GhostConfig()
        # Approximately 7:20 scatter/chase, with cycle lengths scaled to maze size.
        defaults = {"small": (14, 40), "medium": (18, 50), "large": (28, 80)}.get(layout.name, (20, 70))
        self.config = GhostConfig(
            requested.scatter_turns or defaults[0],
            requested.chase_turns or defaults[1],
            requested.random_turn_probability,
        )
        self.phase: Literal["scatter", "chase"] | None = "scatter" if behavior != "random" else None
        self.phase_remaining = int(self.config.scatter_turns or 0) if self.phase else 0
        self.rng = rng
        self.geometry = PacmanLayoutState(layout.name, layout.width, layout.height, layout.walls)
        self.player: Position = layout.player
        self.direction = 0
        self.ghosts = [Ghost(cell, cell) for cell in layout.ghosts]
        self.food = set(layout.food)
        self.capsules = set(layout.capsules)
        self.score = 0.0
        self.won = False
        self.lost = False
        self.killed_blue = False
        self.killed_orange = False
        self.ate_capsule = False
        self.last_action: int | None = None
        self.steps = 0
        self.exits = {
            cell: tuple(
                action
                for action, (dx, dy) in enumerate(VECTORS[1:], 1)
                if (cell[0] + dx, cell[1] + dy) not in layout.walls
            )
            for cell in ((x, y) for x in range(1, layout.width - 1) for y in range(1, layout.height - 1))
            if cell not in layout.walls
        }

    @property
    def terminated(self) -> bool:
        return self.won or self.lost

    def legal_ghost_actions(self, ghost: Ghost, *, scheduled: bool = False) -> tuple[int, ...]:
        if ghost.position != nearest_cell(ghost.position):
            return (ghost.direction,)
        legal = self.exits[nearest_cell(ghost.position)]
        if not scheduled:
            legal = (*legal, 0)
        reverse = REVERSE[ghost.direction]
        if len(legal) > (1 if scheduled else 2):
            legal = tuple(action for action in legal if action != reverse)
        return tuple(sorted(legal))

    def ghost_action(self, index: int) -> int:
        """Choose the configured controller action; random singleton choices consume a draw."""
        if self.behavior != "random":
            return self._scheduled_action(index)
        legal = self.legal_ghost_actions(self.ghosts[index])
        if not legal:
            return 0
        threshold = float(self.rng.random())
        probability = 1.0 / len(legal)
        cumulative = 0.0
        for action in legal:
            cumulative += probability
            if threshold <= cumulative:
                return action
        return legal[-1]

    def step(self, action: int) -> tuple[float, bool]:
        """Advance one turn; return score delta and whether movement was blocked."""
        if self.terminated:
            raise RuntimeError("cannot advance a terminated simulation")
        before = self.score
        was_frightened = any(ghost.timer for ghost in self.ghosts)
        blocked = action != 0 and action not in self.exits[nearest_cell(self.player)]
        if blocked:
            action = 0
        self.last_action = action
        self.killed_blue = self.killed_orange = self.ate_capsule = False
        dx, dy = VECTORS[action]
        self.player = self.player[0] + dx, self.player[1] + dy
        if action:
            self.direction = action
        cell = nearest_cell(self.player)
        if cell in self.food:
            self.food.remove(cell)
            self.score += 10
            if not self.food:
                self.won = True
                self.score += 500
        if cell in self.capsules:
            self.capsules.remove(cell)
            self.ate_capsule = True
            for ghost in self.ghosts:
                ghost.timer = FRIGHTENED_TURNS
                if self.behavior != "random":
                    ghost.reverse_pending = True
        self.score -= 1
        for index in range(len(self.ghosts)):
            self._collide(index)
        if self.behavior == "random":
            for index, ghost in enumerate(self.ghosts):
                if self.terminated:
                    break
                choice = self.ghost_action(index)
                dx, dy = VECTORS[choice]
                speed = 0.5 if ghost.timer else 1.0
                ghost.position = ghost.position[0] + speed * dx, ghost.position[1] + speed * dy
                if choice:
                    ghost.direction = choice
                if ghost.timer == 1:
                    ghost.position = nearest_cell(ghost.position)
                ghost.timer = max(0, ghost.timer - 1)
                self._collide(index)
        else:
            self._move_scheduled_ghosts()
        self.steps += 1
        if self.phase is not None and not was_frightened and not self.ate_capsule and not self.terminated:
            self.phase_remaining -= 1
            if self.phase_remaining == 0:
                self.phase = "chase" if self.phase == "scatter" else "scatter"
                duration = self.config.chase_turns if self.phase == "chase" else self.config.scatter_turns
                self.phase_remaining = int(duration or 1)
                for ghost in self.ghosts:
                    ghost.reverse_pending = True
        return self.score - before, blocked

    def _move_scheduled_ghosts(self) -> None:
        """Keep followers a cell apart, allowing opposing traffic to pass.

        Prefer feasible moves in ID order, including required departures of
        higher-ID occupants. Waits preserve headings and pending reversals.
        Frightened expiry never snaps a scheduled ghost backwards to a center.
        """
        if self.terminated:
            return
        origins = [ghost.position for ghost in self.ghosts]
        reversals = [ghost.reverse_pending for ghost in self.ghosts]
        actions = [self.ghost_action(index) for index in range(len(self.ghosts))]
        destinations: list[Position] = []
        for ghost, action in zip(self.ghosts, actions, strict=True):
            # A queued ghost may still be between cells when frightened expires.
            # Finish that half-cell before resuming ordinary full-cell strides.
            speed = 0.5 if ghost.timer or ghost.position != nearest_cell(ghost.position) else 1.0
            dx, dy = VECTORS[action]
            target = ghost.position[0] + speed * dx, ghost.position[1] + speed * dy
            destinations.append(target)
        # A move may require other ghosts to leave, or conflict with their moves.
        # Resolve whole dependency chains before granting priority: a blocked
        # contender must not reserve a destination against a feasible crossing.
        count = len(origins)
        requires: list[set[int]] = [set() for _ in origins]
        conflicts: list[set[int]] = [set() for _ in origins]
        stationary = {i for i in range(count) if destinations[i] == origins[i]}
        for index in range(count):
            for other in range(index):
                _, dy = VECTORS[actions[index]]
                opposing = (
                    actions[index] != 0
                    and actions[other] == REVERSE[actions[index]]
                    and (origins[index][0] == origins[other][0] if dy else origins[index][1] == origins[other][1])
                )

                if not _separated(destinations[index], origins[other], opposing):
                    requires[index].add(other)
                if not _separated(origins[index], destinations[other], opposing):
                    requires[other].add(index)
                if not _separated(destinations[index], destinations[other], opposing):
                    conflicts[index].add(other)
                    conflicts[other].add(index)
        moving: set[int] = set()
        for index in range(count):
            candidate = moving.copy()
            pending = [index]
            while pending:
                member = pending.pop()
                if member not in candidate:
                    candidate.add(member)
                    pending.extend(requires[member] - candidate)
            if not candidate & stationary and not any(conflicts[i] & candidate for i in candidate):
                moving = candidate
        for index, ghost in enumerate(self.ghosts):
            if index in moving:
                ghost.position = destinations[index]
                ghost.direction = actions[index]
            else:
                ghost.reverse_pending = reversals[index]
            ghost.timer = max(0, ghost.timer - 1)
        for index in range(len(self.ghosts)):
            if self.terminated:
                break
            self._collide(index)

    def _scheduled_respawn(self, index: int) -> Cell:
        """Find the nearest floor cell with a full-cell gap, or a distinct fallback."""
        occupied = {self.player, *(ghost.position for other, ghost in enumerate(self.ghosts) if other != index)}
        spawn = self.ghosts[index].spawn
        pending = deque([spawn])
        seen = {spawn}
        fallback = None
        while pending:
            cell = pending.popleft()
            if cell not in occupied:
                if all(manhattan(cell, position) >= 1 for position in occupied):
                    return cell
                if fallback is None:
                    fallback = cell
            for neighbor in self.layout.neighbors(cell):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        # Dense custom layouts can lack a full-cell gap around half-cell ghosts.
        # Distinct initial spawns guarantee an unoccupied integer cell exists.
        assert fallback is not None
        return fallback

    def _collide(self, index: int) -> None:
        ghost = self.ghosts[index]
        if manhattan(self.player, ghost.position) > COLLISION_DISTANCE:
            return
        if ghost.timer:
            self.score += 200
            ghost.position = ghost.spawn if self.behavior == "random" else self._scheduled_respawn(index)
            ghost.direction = 0
            ghost.timer = 0
            ghost.eaten += 1
            ghost.reverse_pending = False
            if index == 0:
                self.killed_blue = True
            if index == 1:
                self.killed_orange = True
        elif not self.won:
            self.score -= 500
            self.lost = True

    def snapshot(self) -> PacmanAuthorityState:
        """Build a detached authority value, sharing only immutable geometry."""
        return PacmanAuthorityState(
            self.geometry,
            frozenset(self.food),
            frozenset(self.capsules),
            PacmanPlayerState(self.player, self.direction, not self.lost),
            tuple(
                PacmanGhostState(i + 1, g.position, g.direction, bool(g.timer), g.timer, g.eaten)
                for i, g in enumerate(self.ghosts)
            ),
            self.score,
            self.terminated,
            self.won,
            self.lost,
            self.killed_blue,
            self.killed_orange,
            self.ate_capsule,
            self.last_action,
        )

    def _scheduled_action(self, index: int) -> int:
        ghost = self.ghosts[index]
        cell = nearest_cell(ghost.position)
        if ghost.position != cell:
            return ghost.direction
        reverse = REVERSE[ghost.direction]
        if ghost.reverse_pending:
            ghost.reverse_pending = False
            if reverse in self.exits[cell]:
                return reverse
        legal = self.legal_ghost_actions(ghost, scheduled=True)
        if not legal:
            return 0
        if (
            self.behavior == "partly-deterministic"
            and len(self.exits[cell]) >= 3
            and len(legal) > 1
            and self.rng.random() < self.config.random_turn_probability
        ):
            return legal[min(int(self.rng.random() * len(legal)), len(legal) - 1)]
        corners = (
            (self.layout.width - 2, self.layout.height - 2),
            (1, self.layout.height - 2),
            (self.layout.width - 2, 1),
            (1, 1),
        )
        target: Position = corners[index % 4]
        if ghost.timer:
            target = self.player
        elif self.phase == "chase":
            dx, dy = VECTORS[self.direction]
            if index % 4 == 0:
                target = self.player
            elif index % 4 == 1:
                target = self.player[0] + 4 * dx, self.player[1] + 4 * dy
            elif index % 4 == 2:
                anchor = self.ghosts[0].position
                target = 2 * (self.player[0] + 2 * dx) - anchor[0], 2 * (self.player[1] + 2 * dy) - anchor[1]
            elif manhattan(ghost.position, self.player) > 8:
                target = self.player
        tie_order = (1, 4, 2, 3)

        def priority(action: int) -> tuple[float, int]:
            dx, dy = VECTORS[action]
            distance = (cell[0] + dx - target[0]) ** 2 + (cell[1] + dy - target[1]) ** 2
            return (-distance if ghost.timer else distance), tie_order.index(action)

        return min(legal, key=priority)

    def full_snapshot(self) -> PacmanSnapshot:
        """Return all gameplay fields needed to construct custom observations."""
        return PacmanSnapshot(
            self.snapshot(),
            self.behavior,
            self.config,
            self.phase,
            self.phase_remaining,
            bool(self.phase and any(g.timer for g in self.ghosts)),
            self.layout.ghosts,
            tuple(g.reverse_pending for g in self.ghosts),
            self.steps,
        )
