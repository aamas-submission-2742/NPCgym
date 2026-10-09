"""State-based Pacman observations with explicit, characterized feature contracts."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from inspect import isabstract
from numbers import Real
from typing import Any, Protocol, cast

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from npc_gym.automata import pacman_dfas
from npc_gym.automata.automaton import AutomatonDefinition

from .labels import PacmanLabel, authority_labels
from .layout import VECTORS, Cell, PacmanLayout, Position
from .simulation import Simulation, nearest_cell

SUPPORTED_VECTOR_FEATURES = (
    "complete",
    "complete-distinguish",
    "deep-rl",
    "dfa",
    "dfa-distinguish",
    "essential",
    "essential-na",
    "hungry",
    "labelled",
    "next-action",
)
SUPPORTED_FEATURES = frozenset(
    (
        *SUPPORTED_VECTOR_FEATURES,
        "image-crop",
        "image-full",
        *(f"image-full+{mode}" for mode in SUPPORTED_VECTOR_FEATURES),
    )
)
DFA_FEATURES = {"complete", "complete-distinguish", "dfa", "dfa-distinguish"}
STATE_LABELS = tuple(PacmanLabel)[5:]
FloatArray = NDArray[np.float64]
Observation = FloatArray | NDArray[np.uint8] | dict[str, Any]


class FrameSource(Protocol):
    @property
    def frame_shape(self) -> tuple[int, int]: ...
    def frame(self, *, crop: bool = False) -> NDArray[np.uint8]: ...


def validate_feature_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("features must be a string")
    if mode not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported Pacman features {mode!r}")
    return mode


class Routes:
    """Lazily cache breadth-first route order for immutable geometry.

    Wall origins support one-step feature probes. Search ties use north, south,
    east, west; ghost distances include half-cell offsets and collision tolerance.
    """

    def __init__(self, layout: PacmanLayout) -> None:
        self.layout = layout
        self._cache: dict[Cell, dict[Cell, tuple[int, int, int]]] = {}

    def from_cell(self, origin: Cell) -> dict[Cell, tuple[int, int, int]]:
        if origin not in self._cache:
            routes = {origin: (0, 0, 0)}
            pending = deque([origin])
            while pending:
                cell = pending.popleft()
                distance, first, _ = routes[cell]
                for target in self.layout.neighbors(cell):
                    if target in routes:
                        continue
                    vector = target[0] - cell[0], target[1] - cell[1]
                    action = VECTORS.index(vector)
                    routes[target] = (distance + 1, first or action, len(routes))
                    pending.append(target)
            self._cache[origin] = routes
        return self._cache[origin]

    def nearest(self, origin: Cell, targets: set[Cell]) -> tuple[float, int] | None:
        routes = self.from_cell(origin)
        route = min((routes[cell] for cell in targets if cell in routes), key=lambda value: value[2], default=None)
        return None if route is None else (float(route[0]), route[1])

    def ghost(self, origin: Cell, target: Position) -> tuple[float, int]:
        routes = self.from_cell(origin)
        candidates = [
            (x, y)
            for x in {math.floor(target[0]), math.ceil(target[0])}
            for y in {math.floor(target[1]), math.ceil(target[1])}
            if abs(x - target[0]) <= 0.7 and abs(y - target[1]) <= 0.7 and (x, y) in routes
        ]
        if not candidates:
            raise RuntimeError(f"No traversable path from Pacman at {origin} to ghost at {target}")
        cell = min(candidates, key=lambda point: routes[point][2])
        distance, direction, _ = routes[cell]
        return distance + abs(cell[0] - target[0]) + abs(cell[1] - target[1]), direction


def one_hot(features: dict[str, float], name: str, value: int, *, stop: bool = True, size: int = 5) -> None:
    for index in range(0 if stop else 1, size):
        features[f"{name}-{index}"] = float(index == value)


def feature_array(features: dict[str, float]) -> FloatArray:
    # Directional all-ghost/scared counts follow every other scalar feature.
    order = sorted(
        features, key=lambda key: (key.startswith(("#-of-ghosts-1-step-away-", "#-of-scared-ghosts-1-step-away-")), key)
    )
    return np.array([features[key] for key in order], dtype=np.float64)


class VectorFeatures:
    """Compose existing named vector contracts from reusable state queries."""

    def __init__(self, mode: str, layout: PacmanLayout, dfas: Mapping[str, float]) -> None:
        self.mode, self.layout = mode, layout
        self.routes = Routes(layout)
        self.dfas: list[AutomatonDefinition] = []
        for name, reward in dfas.items():
            definition = getattr(pacman_dfas, name, None)
            if (
                not isinstance(definition, type)
                or not issubclass(definition, AutomatonDefinition)
                or isabstract(definition)
                or definition.__module__ != pacman_dfas.__name__
            ):
                raise ValueError(f"Unknown Pacman DFA {name!r}")
            if isinstance(reward, bool) or not isinstance(reward, Real):
                raise TypeError("DFA rewards must be finite real numbers")
            if not math.isfinite(reward):
                raise ValueError("DFA rewards must be finite real numbers")
            self.dfas.append(definition(frozenset(PacmanLabel), reward=reward))
        if dfas and mode not in DFA_FEATURES:
            raise ValueError(f"Pacman features {mode!r} ignore dfas")

    def space(self) -> spaces.Box:
        ghosts = len(self.layout.ghosts)
        states = sum(len(tuple(dfa.states)) for dfa in self.dfas)
        sizes = {
            "hungry": 6,
            "next-action": 20,
            "essential": 19 + 13 * ghosts,
            "essential-na": 37 + 13 * ghosts,
            "deep-rl": 21 + 24 * ghosts,
            "labelled": 40 + len(STATE_LABELS) + 11 * ghosts,
            "complete": 41 + states + 11 * ghosts,
            "dfa": 41 + states + 11 * ghosts,
            "complete-distinguish": 21 + states + 31 * ghosts,
            "dfa-distinguish": 21 + states + 31 * ghosts,
        }
        size = sizes[self.mode]
        if self.mode == "next-action":
            high = np.ones(size)
            high[-8:] = ghosts
            return spaces.Box(np.zeros(size), high, dtype=np.float64)
        if self.mode == "essential":
            return spaces.Box(-1.0, float(self.layout.width * self.layout.height), (size,), dtype=np.float64)
        return spaces.Box(-np.inf, np.inf, (size,), dtype=np.float64)

    def _capsule(self, sim: Simulation, origin: Cell) -> tuple[float, int]:
        if not sim.capsules:
            return -1.0, -1
        result = self.routes.nearest(origin, sim.capsules)
        if result is None:
            raise TypeError("No traversable path to a remaining Pacman capsule")
        return result

    def _near_counts(self, sim: Simulation, origin: Cell) -> tuple[int, int]:
        adjacent = [
            ghost
            for ghost in sim.ghosts
            if origin in self.layout.neighbors(nearest_cell(ghost.position), include_stop=True)
        ]
        return len(adjacent), sum(bool(ghost.timer) for ghost in adjacent)

    def values(self, sim: Simulation, action: int | None) -> dict[str, float]:
        mode = self.mode
        origin = nearest_cell(sim.player)
        if mode == "hungry":
            dx, dy = VECTORS[action or 0]
            probe = origin[0] + dx, origin[1] + dy
            all_count, scared = self._near_counts(sim, probe)
            food = self.routes.nearest(probe, sim.food)
            capsule, _ = self._capsule(sim, probe)
            return {
                key: value / 10
                for key, value in {
                    "bias": 1.0,
                    "#-of-ghosts-1-step-away": all_count,
                    "#-of-scared-ghosts-1-step-away": scared,
                    "closest-capsule-dist": capsule,
                    "eats-food": float(all_count == 0 and probe in sim.food),
                    "closest-food": food[0] / (self.layout.width * self.layout.height) if food else 0.0,
                }.items()
            }
        features: dict[str, float] = {}
        complete = mode in {*DFA_FEATURES, "labelled"}
        distinguish = mode.endswith("distinguish")
        scale = self.layout.width + self.layout.height
        stop = not complete
        if mode != "next-action":
            food = self.routes.nearest(origin, sim.food)
            default = scale if mode == "essential-na" or complete else max(self.layout.width, self.layout.height)
            features["closest-food"] = (food[0] if food else default) / (scale if complete else 1)
            one_hot(features, "closest-food-dir", food[1] if food else 0, stop=stop)
            # A reached food has no first direction, including no Stop bit.
            if food and food[0] == 0 and stop:
                features["closest-food-dir-0"] = 0.0
            capsule, direction = self._capsule(sim, origin)
            features["closest-capsule-dist"] = capsule
            one_hot(features, "closest-capsule-dir", direction if capsule != 0 else -1, stop=stop)
        for choice, (dx, dy) in enumerate(VECTORS):
            if choice == 0 and (complete or mode == "next-action"):
                continue
            features[f"poss-dir-{choice}"] = float((origin[0] + dx, origin[1] + dy) not in self.layout.walls)
        if mode not in ("essential", "next-action"):
            features["x"] = origin[0] / (self.layout.width if complete else 1)
            features["y"] = origin[1] / (self.layout.height if complete else 1)
        distances: dict[int, list[tuple[float, bool]]] = {choice: [] for choice in range(5)}
        if mode != "next-action":
            for index, ghost in enumerate(sim.ghosts):
                distance, direction = self.routes.ghost(origin, ghost.position)
                distances[0].append((distance, bool(ghost.timer)))
                prefix = f"ghost-{index}"
                features[prefix + "-scared"] = float(bool(ghost.timer))
                features[prefix + "-scaredtime"] = ghost.timer / 40
                features[prefix + "-dist"] = distance / (scale if complete else 1)
                one_hot(features, prefix + "-dir", direction, stop=stop)
                one_hot(features, prefix + "-heading", ghost.direction, stop=stop)
                if mode == "deep-rl":
                    gx, gy = ghost.position
                    features[prefix + "-x"], features[prefix + "-y"] = gx, gy
                    angle_dx, angle_dy = gx - origin[0], gy - origin[1]
                    # Preserve the asymmetric historical collision-angle test.
                    angle = 0
                    if not (abs(angle_dx) <= 0.7 + abs(angle_dy) <= 0.7):
                        radians = math.atan2(angle_dy, angle_dx)
                        for code, center in (
                            (1, math.pi / 2),
                            (2, math.pi / 4),
                            (3, 0),
                            (4, -math.pi / 4),
                            (5, -math.pi / 2),
                            (6, -3 * math.pi / 4),
                            (8, 3 * math.pi / 4),
                        ):
                            if center - math.pi / 8 <= radians <= center + math.pi / 8:
                                angle = code
                                break
                        else:
                            angle = 7
                    one_hot(features, prefix + "-angle", angle, size=9)
                if complete:
                    for choice in range(1, 5):
                        dx, dy = VECTORS[choice]
                        value = self.routes.ghost((origin[0] + dx, origin[1] + dy), ghost.position)[0]
                        distances[choice].append((value, bool(ghost.timer)))
                    if distinguish:
                        for choice in range(5):
                            ghost_distance, scared = distances[choice][-1]
                            suffix = f"-{choice}" if choice else ""
                            for kind, flag in (("scared", scared), ("non-scared", not scared)):
                                for label, limit in (("1", 1), ("le3", 3)):
                                    # Current frightened le3 intentionally retains the <=1 quirk.
                                    bound = 1 if choice == 0 and kind == "scared" else limit
                                    features[f"{prefix}-{kind}-{label}-step-away{suffix}"] = float(
                                        flag and ghost_distance <= bound
                                    )
        if complete and not distinguish:
            divisor = max(1, len(sim.ghosts))
            for kind, scared in (("non-scared", False), ("scared", True)):
                for label, limit in (("1", 1), ("le3", 3)):
                    name = f"#-of-{kind}-ghosts-{label}-step-away"
                    current = sum(d <= limit and flag == scared for d, flag in distances[0])
                    features[name] = current / divisor
                    for choice in range(1, 5):
                        count = sum(d <= limit and flag == scared for d, flag in distances[choice])
                        features[f"{name}-{choice}"] = (count - current) / divisor
        elif mode not in ("next-action",) and not distinguish:
            ghosts = sim.ghosts[-1:] if mode == "deep-rl" else sim.ghosts
            adjacent = [g for g in ghosts if abs(g.position[0] - origin[0]) + abs(g.position[1] - origin[1]) <= 1]
            features["#-of-ghosts-1-step-away"] = sum(not g.timer for g in adjacent)
            features["#-of-scared-ghosts-1-step-away"] = sum(bool(g.timer) for g in adjacent)
        if complete or mode in ("essential-na", "next-action"):
            for choice in range(1, 5):
                dx, dy = VECTORS[4 if distinguish and sim.ghosts else choice]
                probe = origin[0] + dx, origin[1] + dy
                food = self.routes.nearest(probe, sim.food)
                if complete:
                    features[f"closest-food-{choice}"] = (food[0] - features["closest-food"]) / scale if food else 1.0
                elif mode == "essential-na":
                    features[f"closest-food-{choice}"] = food[0] - features["closest-food"] if food else float(scale)
                else:
                    features[f"closest-food-{choice}"] = food[0] / scale if food else 1.0
                if not complete:
                    count, scared_count = self._near_counts(sim, probe)
                    for name, value in (
                        ("#-of-ghosts-1-step-away", count),
                        ("#-of-scared-ghosts-1-step-away", scared_count),
                    ):
                        features[f"{name}-{choice}"] = value - (features[name] if mode == "essential-na" else 0)
                    features[f"eats-food-{choice}"] = float(
                        features[f"#-of-ghosts-1-step-away-{choice}"] == 0 and probe in sim.food
                    )
        if complete:
            labels = authority_labels(sim.snapshot(), 0)
            if mode == "labelled":
                features.update({label.value: float(label in labels) for label in STATE_LABELS})
            else:
                features["score"] = sim.score / 100
                for index, dfa in enumerate(self.dfas, 1):
                    reached = dfa.transition(labels)
                    for state in dfa.states:
                        features[f"DFA-{index}-state-{state}"] = float(state == reached)
        return features


class PacmanObservation:
    """Build vector/image observations independently of environment transitions."""

    def __init__(
        self,
        mode: str,
        layout: PacmanLayout,
        dfas: Mapping[str, float],
        renderer: FrameSource,
        *,
        max_phase_turns: int = 0,
    ) -> None:
        validate_feature_mode(mode)
        self.mode, self.renderer = mode, renderer
        self.max_phase_turns = max_phase_turns
        vector = mode.removeprefix("image-full+")
        self.vector = VectorFeatures(vector, layout, dfas) if vector in SUPPORTED_VECTOR_FEATURES else None
        if self.vector is None and dfas:
            raise ValueError(f"Pacman features {mode!r} ignore dfas")
        self.space: spaces.Space[Any]
        if "image" in mode:
            shape = (84, 84) if mode == "image-crop" else renderer.frame_shape
            self.space = spaces.Box(0, 255, (*shape, 4 if "+" in mode else 3), dtype=np.uint8)
        else:
            assert self.vector is not None
            self.space = self.vector.space()

        if max_phase_turns:
            mode_space = spaces.Box(
                np.array([0.0, 0.0, 0.0]), np.array([1.0, float(max_phase_turns), 1.0]), dtype=np.float64
            )
            if "image" in mode:
                self.space = spaces.Dict({"observation": self.space, "ghost_mode": mode_space})
            else:
                assert isinstance(self.space, spaces.Box)
                self.space = spaces.Box(
                    np.concatenate((self.space.low, mode_space.low)),
                    np.concatenate((self.space.high, mode_space.high)),
                    dtype=np.float64,
                )

    def observe(self, sim: Simulation, action: int | None) -> Observation:
        observation = self._base_observation(sim, action)
        if not self.max_phase_turns:
            return observation
        mode = np.array(
            [float(sim.phase == "chase"), float(sim.phase_remaining), float(any(g.timer for g in sim.ghosts))],
            dtype=np.float64,
        )
        if "image" in self.mode:
            return {"observation": observation, "ghost_mode": mode}
        return np.concatenate((observation, mode)).astype(np.float64, copy=False)

    def _base_observation(self, sim: Simulation, action: int | None) -> FloatArray | NDArray[np.uint8]:
        if "image" not in self.mode:
            assert self.vector is not None
            return feature_array(self.vector.values(sim, action))
        pixels = self.renderer.frame(crop=self.mode == "image-crop")
        if "+" not in self.mode:
            return pixels
        assert self.vector is not None
        values = self.vector.values(sim, action)
        bits = [value for key, value in values.items() if key.startswith("DFA")]
        if len(bits) > pixels.shape[1]:
            raise ValueError("Configured DFA state vector is wider than the Pacman image")
        channel = np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)
        channel[0, : len(bits), 0] = bits
        return cast(NDArray[np.uint8], np.concatenate((pixels, channel), axis=-1))
