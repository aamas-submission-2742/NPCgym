from __future__ import annotations

from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, SupportsInt

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.error import DependencyNotInstalled, ResetNeeded
from numpy.typing import NDArray

from npc_gym.envs.gardener.labels import GardenerLabelingFunction, GardenerState
from npc_gym.envs.metrics import EPISODE_METRICS_KEY
from npc_gym.labels import LabelSet, Transition

if TYPE_CHECKING:
    from npc_gym.envs.gardener.gardener_rendering import GardenerRenderer

actions = ["right", "up", "left", "down", "stay"]
action_dict = {name: i for i, name in enumerate(actions)}

directions: dict[str, NDArray[np.int_]] = {
    "right": np.array([1, 0]),
    "up": np.array([0, 1]),
    "left": np.array([-1, 0]),
    "down": np.array([0, -1]),
    "stay": np.array([0, 0]),
}

UNREACHABLE = np.iinfo(np.int32).max

_ACTION_BY_STEP = {(int(d[0]), int(d[1])): action_dict[name] for name, d in directions.items() if name != "stay"}


def target_routes(
    size: int,
    targets: Sequence[tuple[int, int]] | NDArray[np.int_],
    impassable: Collection[tuple[int, int]],
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """Map every passable cell to its `(target index, distance, action toward that target)` triples.

    A breadth-first search runs backwards from each target over cells outside `impassable`; a target
    may itself sit on an impassable cell, which is how puddles stay reachable without becoming
    thoroughfares. Triples are ordered by distance, ties by target index, so the first entry whose
    target is active is the nearest one. Unreachable targets carry `UNREACHABLE` and the stay action.
    """
    routes: dict[tuple[int, int], list[tuple[int, int, int]]] = {
        (x, y): [] for x in range(size) for y in range(size) if (x, y) not in impassable
    }
    for index, target in enumerate(targets):
        source = (int(target[0]), int(target[1]))
        distance = {source: 0}
        toward = {}
        queue = deque([source])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                    continue
                if neighbor in impassable or neighbor in distance:
                    continue
                distance[neighbor] = distance[(x, y)] + 1
                toward[neighbor] = _ACTION_BY_STEP[(-dx, -dy)]
                queue.append(neighbor)
        for cell, entries in routes.items():
            entries.append((index, distance.get(cell, UNREACHABLE), toward.get(cell, action_dict["stay"])))
    for entries in routes.values():
        entries.sort(key=lambda entry: entry[1])
    return routes


@dataclass(frozen=True, slots=True)
class GardenerObservation:
    size: int
    agent: tuple[int, int]
    grass: tuple[tuple[int, int], ...]
    grass_active: tuple[bool, ...]
    grass_timer: tuple[int, ...]
    puddles: tuple[tuple[int, int], ...]
    puddles_full: tuple[bool, ...]
    puddle_timer: tuple[int, ...]
    frogs: tuple[tuple[int, int], ...]
    collected_frogs: tuple[bool, ...]
    captured_frogs: tuple[bool, ...]
    frog_timer: tuple[int, ...]
    walls: tuple[tuple[int, int], ...]
    frog_collected: tuple[bool, ...]
    puddle_drained: tuple[bool, ...]


class GardenerObservationCodec:
    """Decode the public flat observation without consulting environment state."""

    def __init__(self, size: int, num_grass: int, num_puddles: int, num_frogs: int, num_walls: int) -> None:
        self.size = size
        self.counts = (num_grass, num_puddles, num_frogs, num_walls)
        self.length = 2 + 4 * num_grass + 5 * num_puddles + 6 * num_frogs + 2 * num_walls

    def decode(self, observation: Sequence[SupportsInt] | NDArray[np.int_]) -> GardenerObservation:
        """Decode a flat integer sequence of the configured length into an immutable value."""
        if len(observation) != self.length:
            raise ValueError(f"Gardener observation has length {len(observation)}; expected {self.length}")
        values = iter(observation)

        def scalar() -> int:
            return int(next(values))

        def positions(count: int) -> tuple[tuple[int, int], ...]:
            return tuple((scalar(), scalar()) for _ in range(count))

        def scalars(count: int) -> tuple[int, ...]:
            return tuple(scalar() for _ in range(count))

        num_grass, num_puddles, num_frogs, num_walls = self.counts
        return GardenerObservation(
            size=self.size,
            agent=(scalar(), scalar()),
            grass=positions(num_grass),
            grass_active=tuple(map(bool, scalars(num_grass))),
            grass_timer=scalars(num_grass),
            puddles=positions(num_puddles),
            puddles_full=tuple(map(bool, scalars(num_puddles))),
            puddle_timer=scalars(num_puddles),
            frogs=positions(num_frogs),
            collected_frogs=tuple(map(bool, scalars(num_frogs))),
            captured_frogs=tuple(map(bool, scalars(num_frogs))),
            frog_timer=scalars(num_frogs),
            walls=positions(num_walls),
            frog_collected=tuple(map(bool, scalars(num_frogs))),
            puddle_drained=tuple(map(bool, scalars(num_puddles))),
        )


class GardenerEnv(Env[GardenerState, int]):
    # Gymnasium declares metadata as an instance attribute.
    metadata = {"render_modes": ["human"], "render_fps": 10}  # noqa: RUF012

    # Discrete accepts Python ints but samples NumPy ints; Gymnasium's invariant
    # Space type cannot express both while preserving the integer action API.
    action_space: gym.spaces.Discrete[np.int64]  # type: ignore[assignment]
    observation_space: gym.spaces.Tuple

    def __init__(
        self,
        size: int = 15,
        grass_respawn: int = 50,
        puddle_respawn: int = 20,
        score_limit: int = 300,
        frog_freeze: int = 5,
        render_mode: str | None = None,
    ) -> None:
        self._validate_parameters(size, grass_respawn, puddle_respawn, score_limit, frog_freeze)
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected one of {self.metadata['render_modes']} or None"
            )
        # set environment parameters
        self.size = size
        self.grass_respawn = grass_respawn
        self.puddle_respawn = puddle_respawn
        self.score_limit = score_limit
        self.frog_freeze = frog_freeze
        self.num_frogs = max(1, int(size * size * 0.01))
        self.num_puddles = max(1, int(size * size * 0.02))
        self.num_grass = max(1, int(size * size * 0.04))
        self.num_walls = int(size * size * 0.30)
        self.observation_codec = GardenerObservationCodec(
            size, self.num_grass, self.num_puddles, self.num_frogs, self.num_walls
        )
        self.render_mode = render_mode
        self.display: GardenerRenderer | None = None
        self.labeling_function = GardenerLabelingFunction(
            num_grass=self.num_grass,
            num_puddles=self.num_puddles,
            num_frogs=self.num_frogs,
            num_walls=self.num_walls,
        )
        self.action: int | None = None
        self.reward = 0.0
        self.score_delta = 0
        # the state reported by the previous transition, passed on as the next transition's origin
        self._last_state: GardenerState | None = None
        # set action space
        self.action_space = gym.spaces.Discrete(len(action_dict))
        # observation = the whole world state as a flat Tuple of Discretes
        # (same style as MerchantEnv). Wrappers filter this down to whatever
        # feature shape a particular learner wants.
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(size),  # agent x
                gym.spaces.Discrete(size),  # agent y
            )
            # grass: positions, active flags, regrowth timers
            + (gym.spaces.Discrete(size),) * (2 * self.num_grass)
            + (gym.spaces.Discrete(2),) * self.num_grass
            + (gym.spaces.Discrete(grass_respawn + 1),) * self.num_grass
            # puddles: positions, full flags, refill timers
            + (gym.spaces.Discrete(size),) * (2 * self.num_puddles)
            + (gym.spaces.Discrete(2),) * self.num_puddles
            + (gym.spaces.Discrete(puddle_respawn + 1),) * self.num_puddles
            # frogs: positions, collected flags, captured flags, freeze timers
            + (gym.spaces.Discrete(size),) * (2 * self.num_frogs)
            + (gym.spaces.Discrete(2),) * self.num_frogs
            + (gym.spaces.Discrete(2),) * self.num_frogs
            + (gym.spaces.Discrete(frog_freeze + 1),) * self.num_frogs
            # walls: positions
            + (gym.spaces.Discrete(size),) * (2 * self.num_walls)
            # step events: frog collected, puddle drained
            + (gym.spaces.Discrete(2),) * self.num_frogs
            + (gym.spaces.Discrete(2),) * self.num_puddles
        )
        # setup all state variables
        self.reset()
        self._labeling_state_ready = False

    @staticmethod
    def _validate_parameters(
        size: int, grass_respawn: int, puddle_respawn: int, score_limit: int, frog_freeze: int
    ) -> None:
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 3:
            raise ValueError(f"size must be an integer of at least 3; got {size!r}")
        for name, value in (("grass_respawn", grass_respawn), ("puddle_respawn", puddle_respawn)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if isinstance(score_limit, bool) or not isinstance(score_limit, Integral) or score_limit < 1:
            raise ValueError(f"score_limit must be a positive integer; got {score_limit!r}")
        if isinstance(frog_freeze, bool) or not isinstance(frog_freeze, Integral) or frog_freeze < 0:
            raise ValueError(f"frog_freeze must be a non-negative integer; got {frog_freeze!r}")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GardenerState, dict[str, Any]]:
        super().reset(seed=seed)
        self.score = 0
        # place the agent at a random cell
        self.agent: NDArray[np.int_] = self.np_random.integers(0, self.size, size=2, dtype=int)
        # candidate cells: every grid cell except the agent's
        all_positions = {(x, y) for x in range(self.size) for y in range(self.size)}
        all_positions.discard(tuple(self.agent))
        # place frogs
        frog_positions = self.np_random.choice(list(all_positions), size=self.num_frogs, replace=False)
        for fp in frog_positions:
            all_positions.discard(tuple(fp))
        # place puddles; also exclude their 4-neighborhood so that they remain accessible for the agent
        puddle_positions = self.np_random.choice(list(all_positions), size=self.num_puddles, replace=False)
        for pp in puddle_positions:
            all_positions.discard(tuple(pp))
            px, py = pp
            for nx, ny in [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    all_positions.discard((nx, ny))
        # place grass
        grass_positions = self.np_random.choice(list(all_positions), size=self.num_grass, replace=False)
        for gp in grass_positions:
            all_positions.discard(tuple(gp))

        # place walls greedily, ensuring all remaining free cells stay reachable
        def is_accessible(excluded: Collection[tuple[int, int]]) -> bool:
            free = {(x, y) for x in range(self.size) for y in range(self.size)}
            free -= set(map(tuple, puddle_positions))
            free -= set(excluded)
            if not free:
                return True
            start = next(iter(free))
            stack = [start]
            visited = {start}
            while stack:
                cx, cy = stack.pop()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in free and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            return visited == free

        wall_positions: list[tuple[int, int]] = []
        remaining = list(all_positions)
        self.np_random.shuffle(remaining)
        for pos in remaining:
            if len(wall_positions) == self.num_walls:
                break
            trial = wall_positions + [pos]
            if is_accessible(trial):
                wall_positions.append(pos)
        # if the random draw could not fit enough walls, redraw
        if len(wall_positions) < self.num_walls:
            return self.reset()

        self.frogs: NDArray[np.int_] = np.array(frog_positions, dtype=int)
        self.puddles: NDArray[np.int_] = np.array(puddle_positions, dtype=int)
        self.grass: NDArray[np.int_] = np.array(grass_positions, dtype=int)
        self.walls: NDArray[np.int_] = np.array(wall_positions, dtype=int)
        # cached for BFS over walkable cells
        self._walls_set = {(int(x), int(y)) for x, y in self.walls}
        self._puddles_set = {(int(x), int(y)) for x, y in self.puddles}
        # initialize per-entity state and timers
        self.collected_frogs: NDArray[np.bool_] = np.zeros(self.num_frogs, dtype=bool)
        self.captured_frogs: NDArray[np.bool_] = np.zeros(self.num_frogs, dtype=bool)
        self.frog_timer: NDArray[np.int_] = np.zeros(self.num_frogs, dtype=int)
        self.puddles_full: NDArray[np.bool_] = np.ones(self.num_puddles, dtype=bool)
        self.puddle_timer: NDArray[np.int_] = np.ones(self.num_puddles, dtype=int)
        self.grass_active: NDArray[np.bool_] = np.ones(self.num_grass, dtype=bool)
        self.grass_timer: NDArray[np.int_] = np.zeros(self.num_grass, dtype=int)
        self.frog_collected: NDArray[np.bool_] = np.zeros(self.num_frogs, dtype=bool)
        self.puddle_drained: NDArray[np.bool_] = np.zeros(self.num_puddles, dtype=bool)
        # precompute static helpers
        self._compute_pos_actions()
        # frogs steer toward the nearest full puddle; puddles block the routes between them
        self.puddle_dict = target_routes(self.size, self.puddles, self._walls_set | self._puddles_set)
        self.action = None
        self.reward = 0.0
        if self.render_mode == "human":
            self.render()
        state = self.get_state()
        self._last_state = state
        self._labeling_state_ready = True
        transition_labels = self.labeling_function(
            Transition(previous_state=None, action=None, state=state, terminated=False, truncated=False)
        )
        return state, self._get_info(transition_labels)

    def _compute_pos_actions(self) -> None:
        size = self.size
        walls_set = {tuple(w) for w in self.walls}
        puddles_set = {tuple(p) for p in self.puddles}
        self.pos_actions: NDArray[np.int_] = np.zeros((size, size, len(actions)), dtype=int)
        for x in range(size):
            for y in range(size):
                if (x, y) in walls_set or (x, y) in puddles_set:
                    continue
                for ai, name in enumerate(actions):
                    d = directions[name]
                    nx, ny = x + int(d[0]), y + int(d[1])
                    if not (0 <= nx < size and 0 <= ny < size):
                        continue
                    if (nx, ny) in walls_set or (nx, ny) in puddles_set:
                        continue
                    self.pos_actions[x, y, ai] = 1

    def _get_obs(self) -> GardenerState:
        return (
            int(self.agent[0]),
            int(self.agent[1]),
            *(int(v) for p in self.grass for v in p),
            *self.grass_active.astype(int).tolist(),
            *self.grass_timer.astype(int).tolist(),
            *(int(v) for p in self.puddles for v in p),
            *self.puddles_full.astype(int).tolist(),
            *self.puddle_timer.astype(int).tolist(),
            *(int(v) for p in self.frogs for v in p),
            *self.collected_frogs.astype(int).tolist(),
            *self.captured_frogs.astype(int).tolist(),
            *self.frog_timer.astype(int).tolist(),
            *(int(v) for p in self.walls for v in p),
            *self.frog_collected.astype(int).tolist(),
            *self.puddle_drained.astype(int).tolist(),
        )

    def _get_info(self, transition_labels: LabelSet) -> dict[str, Any]:
        return {
            "action_mask": self.action_mask(),
            "labels": transition_labels,
            EPISODE_METRICS_KEY: {"score": self.score},
        }

    def action_mask(self) -> NDArray[np.int8]:
        """Return the actions available in the current state."""
        x, y = int(self.agent[0]), int(self.agent[1])
        return self.pos_actions[x, y, :].astype(np.int8, copy=True)

    def _proceed(self, reward: float, terminal: bool) -> tuple[GardenerState, float, bool, bool, dict[str, Any]]:
        # remember the resulting (possibly terminal) reward
        self.reward = reward
        # refresh visualization on every state transition
        if self.render_mode in ["human"]:
            self.render()
        state = self.get_state()
        previous_state, self._last_state = self._last_state, state
        transition_labels = self.labeling_function(
            Transition(
                previous_state=previous_state,
                action=self.action,
                state=state,
                terminated=terminal,
                truncated=False,
            )
        )
        return (
            state,
            self.reward,
            terminal,
            False,
            self._get_info(transition_labels),
        )

    def step(self, action: int | np.int64) -> tuple[GardenerState, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        action = int(action)
        ax, ay = int(self.agent[0]), int(self.agent[1])
        if self.pos_actions[ax, ay, action] == 0:
            action = action_dict["stay"]
        self.action = action
        self.frog_collected.fill(False)
        self.puddle_drained.fill(False)
        # move the agent
        self.agent = self.agent + directions[actions[action]]
        # Frogs move next, biased toward the nearest full puddle.
        self._move_frogs()
        # If the agent ends up sharing a cell with an uncollected frog, the frog is collected.
        if self.num_frogs > 0 and np.any(np.all(self.agent == self.frogs, axis=1)):
            for i, (fx, fy) in enumerate(self.frogs):
                if self.collected_frogs[i]:
                    continue
                if self.agent[0] == fx and self.agent[1] == fy:
                    self.collected_frogs[i] = True
                    self.captured_frogs[i] = True
                    self.frog_collected[i] = True
        reward = 0
        # tick frog freeze timers
        for f in range(self.num_frogs):
            if self.frog_timer[f] > 0:
                self.frog_timer[f] -= 1
        # grass: stepping on active grass yields +10 and starts the regrowth timer
        for i, (gx, gy) in enumerate(self.grass):
            if self.agent[0] == gx and self.agent[1] == gy:
                if self.grass_active[i]:
                    self.grass_active[i] = False
                    reward += 10
                    self.grass_timer[i] = self.grass_respawn
            else:
                if not self.grass_active[i] and self.grass_timer[i] > 0:
                    self.grass_timer[i] -= 1
                    if self.grass_timer[i] == 0:
                        self.grass_active[i] = True
        # Puddles: standing 4-adjacent to a full puddle drains it for +5 and freezes nearby frogs.
        for i, (px, py) in enumerate(self.puddles):
            if self.puddle_timer[i] > 0:
                self.puddle_timer[i] -= 1
                if self.puddle_timer[i] == 0:
                    self.puddles_full[i] = True
            if self.puddles_full[i] and abs(self.agent[0] - px) + abs(self.agent[1] - py) == 1:
                reward += 5
                self.puddles_full[i] = False
                self.puddle_timer[i] = self.puddle_respawn
                self.puddle_drained[i] = True
                for f, (c, r) in enumerate(self.frogs):
                    if self.collected_frogs[f]:
                        continue
                    if self._is_frog_near_puddle(px, py, c, r):
                        self.frog_timer[f] = self.frog_freeze
        # cap reported score at score_limit; agent reward stays uncapped
        self.score_delta = min(reward, self.score_limit - self.score)
        self.score += self.score_delta
        terminal = self.score >= self.score_limit
        # training signal only: punish unnecessary steps
        return self._proceed(reward - 0.1, terminal)

    @staticmethod
    def _is_frog_near_puddle(px: SupportsInt, py: SupportsInt, fx: SupportsInt, fy: SupportsInt) -> bool:
        dx, dy = abs(int(px) - int(fx)), abs(int(py) - int(fy))
        return (dx + dy == 1) or (dx + dy == 2 and (dx == 1 or dy == 1))

    def _move_frogs(self) -> None:
        if self.num_frogs == 0:
            return
        new_positions = []
        for i in range(self.num_frogs):
            fx, fy = int(self.frogs[i][0]), int(self.frogs[i][1])
            if self.collected_frogs[i] or self.frog_timer[i] > 0:
                new_positions.append([fx, fy])
                continue
            # frogs only consider the four cardinal directions, never stay
            valid = [a for a in range(4) if self.pos_actions[fx, fy, a] == 1]
            if not valid:
                new_positions.append([fx, fy])
                continue
            preferred = None
            if (fx, fy) in self.puddle_dict:
                for puddle_idx, _, a in self.puddle_dict[(fx, fy)]:
                    if self.puddles_full[puddle_idx]:
                        preferred = a
                        break
            chosen = None
            if preferred is not None and preferred in valid and self.np_random.random() < 0.7:
                chosen = preferred
            if chosen is None:
                if preferred is not None:
                    other = [a for a in valid if a != preferred]
                    if other:
                        chosen = int(self.np_random.choice(other))
                    elif preferred in valid:
                        chosen = preferred
                else:
                    chosen = int(self.np_random.choice(valid))
            if chosen is not None:
                d = directions[actions[chosen]]
                new_positions.append([fx + int(d[0]), fy + int(d[1])])
            else:
                new_positions.append([fx, fy])
        self.frogs = np.array(new_positions, dtype=int)

    def render(self) -> None:
        """Present and pace a human frame using the constructor-selected mode."""
        if self.render_mode not in ["human"]:
            return
        if self.display is None:
            try:
                from npc_gym.envs.gardener.gardener_rendering import GardenerRenderer
            except ModuleNotFoundError as error:
                if error.name != "pygame":
                    raise
                raise DependencyNotInstalled(
                    'Gardener rendering requires pygame; install it with `pip install "npc-gym[render]"`'
                ) from error

            self.display = GardenerRenderer()
        try:
            self.display.draw(self)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        display = self.display
        self.display = None
        if display is not None:
            display.close()

    def get_state(self) -> GardenerState:
        """Return the flat integer observation decoded by ``observation_codec``."""
        return self._get_obs()

    def labeling_state(self) -> GardenerObservation:
        """Return the detached immutable decoded observation for external labelers."""
        if not self._labeling_state_ready:
            raise ResetNeeded("GardenerEnv must be reset before labeling_state()")
        return self.observation_codec.decode(self._get_obs())
