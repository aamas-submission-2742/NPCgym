"""Typed transition labels for the Gardener environment."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from npc_gym.labels import LabelSet, Transition

GardenerState: TypeAlias = tuple[int, ...]


class GardenerLabel(StrEnum):
    """Gardener action and permission propositions."""

    RIGHT = "right"
    UP = "up"
    LEFT = "left"
    DOWN = "down"
    STAY = "stay"
    PERMITTED_COLLECT = "permittedCollect"


@dataclass(frozen=True, slots=True)
class FrogCollected:
    """Collection of the frog identified by ``frog_id``."""

    frog_id: int


@dataclass(frozen=True, slots=True)
class PuddleDrained:
    """Drainage of a puddle while the identified frogs were nearby."""

    puddle_id: int
    nearby_frog_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _DecodedState:
    puddles: tuple[tuple[int, int], ...]
    frogs: tuple[tuple[int, int], ...]
    collected_frogs: tuple[bool, ...]
    frog_collected: tuple[bool, ...]
    puddle_drained: tuple[bool, ...]


_ACTION_LABELS = (
    GardenerLabel.RIGHT,
    GardenerLabel.UP,
    GardenerLabel.LEFT,
    GardenerLabel.DOWN,
    GardenerLabel.STAY,
)


class GardenerLabelingFunction:
    """Produce collection, drainage, permission, and action labels."""

    def __init__(self, *, num_grass: int, num_puddles: int, num_frogs: int, num_walls: int) -> None:
        self._num_grass = num_grass
        self._num_puddles = num_puddles
        self._num_frogs = num_frogs
        self._num_walls = num_walls

    def __call__(self, transition: Transition[GardenerState, int]) -> LabelSet:
        if transition.action is None:
            return frozenset()

        decoded = self._decode(transition.state)
        collected = tuple(FrogCollected(i) for i, occurred in enumerate(decoded.frog_collected) if occurred)
        result: set[GardenerLabel | FrogCollected | PuddleDrained] = {
            _ACTION_LABELS[transition.action],
            *collected,
        }
        if collected and transition.action == 0:
            result.add(GardenerLabel.PERMITTED_COLLECT)

        for puddle_id, drained in enumerate(decoded.puddle_drained):
            if not drained:
                continue
            px, py = decoded.puddles[puddle_id]
            nearby = tuple(
                frog_id
                for frog_id, (fx, fy) in enumerate(decoded.frogs)
                if not decoded.collected_frogs[frog_id] and self._is_frog_near_puddle(px, py, fx, fy)
            )
            if nearby:
                result.add(PuddleDrained(puddle_id, nearby))
        return frozenset(result)

    def _decode(self, state: GardenerState) -> _DecodedState:
        index = 2 + 4 * self._num_grass
        puddle_positions = state[index : index + 2 * self._num_puddles]
        index += 4 * self._num_puddles
        frog_positions = state[index : index + 2 * self._num_frogs]
        index += 2 * self._num_frogs
        collected_frogs = state[index : index + self._num_frogs]
        index += 3 * self._num_frogs + 2 * self._num_walls
        frog_collected = state[index : index + self._num_frogs]
        index += self._num_frogs
        puddle_drained = state[index : index + self._num_puddles]
        return _DecodedState(
            puddles=self._pairs(puddle_positions),
            frogs=self._pairs(frog_positions),
            collected_frogs=tuple(bool(value) for value in collected_frogs),
            frog_collected=tuple(bool(value) for value in frog_collected),
            puddle_drained=tuple(bool(value) for value in puddle_drained),
        )

    @staticmethod
    def _pairs(flat: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2))

    @staticmethod
    def _is_frog_near_puddle(px: int, py: int, fx: int, fy: int) -> bool:
        dx, dy = abs(px - fx), abs(py - fy)
        return (dx + dy == 1) or (dx + dy == 2 and (dx == 1 or dy == 1))


__all__ = ["FrogCollected", "GardenerLabel", "GardenerLabelingFunction", "GardenerState", "PuddleDrained"]
