from collections.abc import Sequence
from numbers import Real
from typing import Any, Protocol, SupportsInt, TypeAlias, TypeVar, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from npc_gym.envs.gardener.gardener import (
    UNREACHABLE,
    GardenerObservation,
    GardenerObservationCodec,
    action_dict,
    actions,
    target_routes,
)

EncodedObservationT = TypeVar("EncodedObservationT", bound=Sequence[SupportsInt] | NDArray[np.int_])
ObservationT = TypeVar("ObservationT")


class _ActionMaskProvider(Protocol):
    def action_mask(self) -> NDArray[np.int8]: ...


FloatObservation: TypeAlias = NDArray[np.float32]
RouteTable: TypeAlias = dict[tuple[int, int], list[tuple[int, int, int]]]


class StateFeatureObsWrapper(gym.ObservationWrapper[FloatObservation, int, EncodedObservationT]):
    """Compact task features, optionally extended with one-step frog risks.

    The default 12 features describe proximity and routes to grass and puddles.
    ``include_frogs=True`` appends five possible-collection counts divided by
    the total frog count, five possible harmful-drainage counts divided by the
    total puddle count, then five action-legality bits. Each group is ordered
    Right, Up, Left, Down, Stay. Zero denominators produce zero counts.

    Risks include every legal frog move, freeze status and puddle refill, with
    collection before drainage. Collection counts are attainable maxima;
    drainage counts can combine mutually exclusive outcomes. They are not
    probabilities. Blocked proposals have Stay's risks and legality zero.
    Features are computed from the supplied observation without a solver.
    """

    # grass/puddle proximity: 1 / (distance + 1), or 0 when no active target is reachable
    # dir_lawn one-hot   (5 entries: which action heads toward nearest active grass)
    # dir_puddle one-hot (5 entries)
    n_features = 2 + 2 * len(actions)

    def __init__(self, env: gym.Env[EncodedObservationT, int], *, include_frogs: bool = False) -> None:
        super().__init__(env)
        if not isinstance(include_frogs, bool):
            raise TypeError("include_frogs must be Boolean")
        self.include_frogs = include_frogs
        if include_frogs:
            self.n_features += 3 * len(actions)
        self._codec = _observation_codec(env)
        self._route_layout: (
            tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], frozenset[tuple[int, int]]] | None
        ) = None
        self._route_tables: tuple[RouteTable, RouteTable] | None = None
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)

    def observation(self, obs: EncodedObservationT) -> FloatObservation:
        state = self._codec.decode(obs)
        layout = (state.grass, state.puddles, frozenset(state.walls))
        if layout != self._route_layout:
            impassable = layout[2] | frozenset(state.puddles)
            self._route_tables = (
                target_routes(self._codec.size, state.grass, impassable),
                target_routes(self._codec.size, state.puddles, impassable),
            )
            self._route_layout = layout
        assert self._route_tables is not None
        grass_routes, puddle_routes = self._route_tables
        dist_lawn, dir_lawn = self._best_target(grass_routes, state.grass_active, state.agent)
        dist_puddle, dir_puddle = self._best_target(puddle_routes, state.puddles_full, state.agent)

        feat = np.zeros(self.n_features, dtype=np.float32)
        feat[0] = 1.0 / (dist_lawn + 1.0) if dist_lawn < UNREACHABLE else 0.0
        feat[1] = 1.0 / (dist_puddle + 1.0) if dist_puddle < UNREACHABLE else 0.0
        if dir_lawn is not None:
            feat[2 + dir_lawn] = 1.0
        if dir_puddle is not None:
            feat[2 + len(actions) + dir_puddle] = 1.0
        if self.include_frogs:
            feat[-3 * len(actions) :] = _frog_features(state)
        return feat

    @staticmethod
    def _best_target(routes: RouteTable, active: tuple[bool, ...], origin: tuple[int, int]) -> tuple[int, int | None]:
        """Return the distance to the nearest active target and the action heading toward it."""
        for index, distance, action in routes.get(origin, ()):
            if active[index] and distance < UNREACHABLE:
                return distance, action
        return UNREACHABLE, None


def _frog_features(state: GardenerObservation) -> FloatObservation:
    """Count possible events separately; do not enumerate joint frog outcomes."""
    vectors = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
    blocked = frozenset((*state.walls, *state.puddles))

    def legal(position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < state.size and 0 <= y < state.size and position not in blocked

    frog_positions = []
    for (x, y), collected, timer in zip(state.frogs, state.collected_frogs, state.frog_timer, strict=True):
        if collected:
            continue
        moves = {(x + dx, y + dy) for dx, dy in vectors[:4] if legal((x + dx, y + dy))} if timer == 0 else set()
        frog_positions.append(moves or {(x, y)})
    possible_positions = {position for positions in frog_positions for position in positions}
    full_puddles = [
        position
        for position, full, timer in zip(state.puddles, state.puddles_full, state.puddle_timer, strict=True)
        if full or timer == 1
    ]
    features = np.zeros(3 * len(actions), dtype=np.float32)
    for action, (dx, dy) in enumerate(vectors):
        destination = (state.agent[0] + dx, state.agent[1] + dy)
        allowed = legal(destination)
        features[2 * len(actions) + action] = allowed
        if not allowed:
            destination = state.agent
        features[action] = sum(destination in positions for positions in frog_positions) / max(1, len(state.frogs))
        survivors = possible_positions - {destination}
        features[len(actions) + action] = sum(
            abs(px - destination[0]) + abs(py - destination[1]) == 1
            and any(max(abs(px - fx), abs(py - fy)) == 1 for fx, fy in survivors)
            for px, py in full_puddles
        ) / max(1, len(state.puddles))
    return features


class LocalGridObsWrapper(gym.ObservationWrapper[FloatObservation, int, EncodedObservationT]):
    """Local terrain channels followed by directions to grass and puddles.

    ``include_frogs=True`` adds one channel for each remaining freeze timer,
    from zero (mobile) through the environment's ``frog_freeze``. Each cell
    holds the number of uncollected frogs with that timer divided by the total
    number of frogs. Collected and out-of-window frogs are omitted. Channels
    precede the eight direction bits; the default encoding omits frogs.
    """

    # Flat Box observation:
    #   - (2*radius+1) x (2*radius+1) local view centered on the agent,
    #     one binary channel per entity type, flattened.
    #   - 4-bit Manhattan direction (right, up, left, down) toward the
    #     nearest active grass, then the same toward the nearest full puddle.
    # Channel layout: wall (also marks out-of-bounds), puddle_full,
    # puddle_empty, grass_active, grass_inactive
    N_CHANNELS = 5
    CH_WALL = 0
    CH_PUDDLE_FULL = 1
    CH_PUDDLE_EMPTY = 2
    CH_GRASS_ACTIVE = 3
    CH_GRASS_INACTIVE = 4

    N_DIR_BITS = 4
    n_dir_features = 2 * N_DIR_BITS

    def __init__(self, env: gym.Env[EncodedObservationT, int], radius: int = 2, *, include_frogs: bool = False) -> None:
        super().__init__(env)
        if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
            raise ValueError(f"radius must be a non-negative integer; got {radius!r}")
        self.radius = int(radius)
        self._codec = _observation_codec(env)
        if not isinstance(include_frogs, bool):
            raise TypeError("include_frogs must be Boolean")
        self.include_frogs = include_frogs
        self.n_channels = self.N_CHANNELS
        if include_frogs:
            self.n_channels += int(env.get_wrapper_attr("frog_freeze")) + 1
        self.window = 2 * self.radius + 1
        self.n_grid_features = self.n_channels * self.window * self.window
        self.n_features = self.n_grid_features + self.n_dir_features
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)

    def observation(self, obs: EncodedObservationT) -> FloatObservation:
        state = self._codec.decode(obs)
        ax, ay = state.agent
        size = self._codec.size
        radius = self.radius
        window = self.window

        grid = np.zeros((self.n_channels, window, window), dtype=np.float32)

        # Out-of-bounds cells are treated as walls (same effect on movement)
        for i in range(window):
            for j in range(window):
                gx = ax + (i - radius)
                gy = ay + (j - radius)
                if not (0 <= gx < size and 0 <= gy < size):
                    grid[self.CH_WALL, i, j] = 1.0

        def to_local(gx: int, gy: int) -> tuple[int, int] | None:
            i = int(gx) - ax + radius
            j = int(gy) - ay + radius
            if 0 <= i < window and 0 <= j < window:
                return i, j
            return None

        for wx, wy in state.walls:
            loc = to_local(wx, wy)
            if loc is not None:
                grid[self.CH_WALL, loc[0], loc[1]] = 1.0

        for idx, (px, py) in enumerate(state.puddles):
            loc = to_local(px, py)
            if loc is None:
                continue
            ch = self.CH_PUDDLE_FULL if state.puddles_full[idx] else self.CH_PUDDLE_EMPTY
            grid[ch, loc[0], loc[1]] = 1.0

        for idx, (gx, gy) in enumerate(state.grass):
            loc = to_local(gx, gy)
            if loc is None:
                continue
            ch = self.CH_GRASS_ACTIVE if state.grass_active[idx] else self.CH_GRASS_INACTIVE
            grid[ch, loc[0], loc[1]] = 1.0

        if self.include_frogs:
            for position, collected, timer in zip(state.frogs, state.collected_frogs, state.frog_timer, strict=True):
                loc = to_local(*position)
                if not collected and loc is not None:
                    grid[self.N_CHANNELS + timer, loc[0], loc[1]] += 1
            grid[self.N_CHANNELS :] /= max(1, len(state.frogs))

        feat = np.zeros(self.n_features, dtype=np.float32)
        feat[: self.n_grid_features] = grid.ravel()
        self._fill_target(feat, self.n_grid_features, state.grass, state.grass_active, ax, ay)
        self._fill_target(feat, self.n_grid_features + self.N_DIR_BITS, state.puddles, state.puddles_full, ax, ay)
        return feat

    @staticmethod
    def _fill_target(
        dirs: FloatObservation,
        offset: int,
        positions: tuple[tuple[int, int], ...],
        active_mask: tuple[bool, ...],
        ax: int,
        ay: int,
    ) -> None:
        best_d = UNREACHABLE
        best_dx = 0
        best_dy = 0
        for idx, (tx, ty) in enumerate(positions):
            if not bool(active_mask[idx]):
                continue
            dx = int(tx) - ax
            dy = int(ty) - ay
            d = abs(dx) + abs(dy)
            if d < best_d:
                best_d = d
                best_dx = dx
                best_dy = dy
        if best_d == UNREACHABLE:
            return
        # Bit order matches `actions`: 0=right(+x), 1=up(+y), 2=left(-x), 3=down(-y)
        if best_dx > 0:
            dirs[offset + 0] = 1.0
        elif best_dx < 0:
            dirs[offset + 2] = 1.0
        if best_dy > 0:
            dirs[offset + 1] = 1.0
        elif best_dy < 0:
            dirs[offset + 3] = 1.0


class IllegalActionPenaltyWrapper(gym.Wrapper[ObservationT, int, ObservationT, int]):
    """Re-map illegal actions to 'stay' and apply a penalty."""

    def __init__(self, env: gym.Env[ObservationT, int], penalty: float = -1.0) -> None:
        super().__init__(env)
        if isinstance(penalty, bool) or not isinstance(penalty, Real) or not np.isfinite(penalty):
            raise ValueError(f"penalty must be a finite number; got {penalty!r}")
        self._stay: int = action_dict["stay"]
        self._penalty: float = float(penalty)

    def step(self, action: int) -> tuple[ObservationT, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        mask = cast(_ActionMaskProvider, self.env.unwrapped).action_mask()
        extra = 0.0
        if not mask[int(action)]:
            action = self._stay
            extra = self._penalty
        obs, r, term, trunc, info = self.env.step(int(action))
        return obs, float(r) + extra, term, trunc, info


def _observation_codec(env: gym.Env[ObservationT, int]) -> GardenerObservationCodec:
    codec = getattr(env.unwrapped, "observation_codec", None)
    if not isinstance(codec, GardenerObservationCodec):
        raise TypeError("Gardener observation wrappers require an environment with a GardenerObservationCodec")
    return codec
