"""Typed transition labels for Storm Taxi."""

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import TypeAlias

from npc_gym.labels import LabelSet, Transition

TaxiState: TypeAlias = int


@dataclass(frozen=True, slots=True)
class TaxiAuthorityState:
    """Detached Taxi state available to external labeling functions.

    Positions use ``(x, y)`` coordinates. Passenger, destination, home, and
    shelter values index ``locations``; passenger ``4`` means inside the taxi.
    """

    taxi_position: tuple[int, int]
    passenger: int
    destination: int
    raining: bool
    hurricane: int
    home: int
    flood_risk: bool
    shelter: int
    locations: tuple[tuple[int, int], ...]


_LOCATIONS = ((0, 0), (4, 0), (0, 4), (3, 4))
_MAP = (
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
)


class TaxiWeatherLabel(StrEnum):
    """Weather propositions visible to the Taxi authority."""

    RAIN = "rain"
    HURRICANE = "hurricane"
    NEW_HURRICANE = "newHurricane"
    FLOOD_RISK = "floodrisk"


class TaxiLocationLabel(StrEnum):
    """Taxi location and direction-of-travel propositions."""

    AT_HOME = "atHome"
    AT_SHELTER = "atShelter"
    AT_DESTINATION = "atDestination"
    TOWARD_HOME = "toward(atHome)"
    TOWARD_SHELTER = "toward(atShelter)"
    TOWARD_DESTINATION = "toward(atDestination)"


class TaxiPassengerLabel(StrEnum):
    """Passenger propositions visible to the Taxi authority."""

    HAS_PASSENGER = "hasPassenger"


class TaxiActionLabel(StrEnum):
    """Taxi action propositions."""

    SOUTH = "south"
    NORTH = "north"
    EAST = "east"
    WEST = "west"
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    WARN = "warn"


_ACTION_LABELS = tuple(TaxiActionLabel)


class TaxiLabelingFunction:
    """Label actual locations and progress along the completed transition.

    ``at...`` describes the resulting position, including on reset. ``toward``
    requires an action and a previous state, and means that the actual movement
    shortened the shortest route to the resulting state's target. Blocked moves
    and stationary actions make no progress. Weather and passenger labels also
    describe the resulting state.
    """

    def __call__(self, transition: Transition[TaxiState, int]) -> LabelSet:
        row, col, passenger, destination, rain, hurricane, home, flood, shelter = self._decode(transition.state)
        position = (col, row)
        result: set[TaxiWeatherLabel | TaxiLocationLabel | TaxiPassengerLabel | TaxiActionLabel] = set()

        if transition.action is not None:
            result.add(_ACTION_LABELS[transition.action])
        if transition.action is not None and transition.previous_state is not None:
            previous_row, previous_col, *_ = self._decode(transition.previous_state)
            previous_position = (previous_col, previous_row)
            if _toward(previous_position, position, _LOCATIONS[home]):
                result.add(TaxiLocationLabel.TOWARD_HOME)
            if _toward(previous_position, position, _LOCATIONS[destination]):
                result.add(TaxiLocationLabel.TOWARD_DESTINATION)
            if _toward(previous_position, position, _LOCATIONS[shelter]):
                result.add(TaxiLocationLabel.TOWARD_SHELTER)

        if rain:
            result.add(TaxiWeatherLabel.RAIN)
        if hurricane > 0:
            result.add(TaxiWeatherLabel.HURRICANE)
        if hurricane == 1:
            result.add(TaxiWeatherLabel.NEW_HURRICANE)
        if flood:
            result.add(TaxiWeatherLabel.FLOOD_RISK)
        if position == _LOCATIONS[home]:
            result.add(TaxiLocationLabel.AT_HOME)
        if position == _LOCATIONS[shelter]:
            result.add(TaxiLocationLabel.AT_SHELTER)
        if position == _LOCATIONS[destination]:
            result.add(TaxiLocationLabel.AT_DESTINATION)
        if passenger == 4:
            result.add(TaxiPassengerLabel.HAS_PASSENGER)
        return frozenset(result)

    def _decode(self, state: int) -> tuple[int, int, int, int, bool, int, int, bool, int]:
        shelter = state % 4
        state //= 4
        flood = bool(state % 2)
        state //= 2
        home = state % 4
        state //= 4
        hurricane = state % 11
        state //= 11
        rain = bool(state % 2)
        state //= 2
        destination = state % 4
        state //= 4
        passenger = state % 5
        state //= 5
        col = state % 5
        row = state // 5
        return row, col, passenger, destination, rain, hurricane, home, flood, shelter


# Memoize actual progress and route distances on the static map.
@cache
def _toward(previous: tuple[int, int], position: tuple[int, int], target: tuple[int, int]) -> bool:
    return _shortest_route(previous, target) > _shortest_route(position, target)


@cache
def _shortest_route(start: tuple[int, int], target: tuple[int, int]) -> int:
    visited = [[False for _ in range(5)] for _ in range(5)]
    queue = deque([(start, [start])])
    while queue:
        (x, y), path = queue.popleft()
        if not (0 <= x < 5 and 0 <= y < 5) or _MAP[y + 1][2 * x + 1] in "+-|" or visited[x][y]:
            continue
        visited[x][y] = True
        if (x, y) == target:
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < 5
                and 0 <= ny < 5
                and not visited[nx][ny]
                and (
                    (dx > 0 and _MAP[ny + 1][2 * nx] != "|") or (dx < 0 and _MAP[ny + 1][2 * nx + 2] != "|") or dx == 0
                )
            ):
                queue.append(((nx, ny), path + [(nx, ny)]))
    return len(path) - 1


__all__ = [
    "TaxiActionLabel",
    "TaxiAuthorityState",
    "TaxiLabelingFunction",
    "TaxiLocationLabel",
    "TaxiPassengerLabel",
    "TaxiState",
    "TaxiWeatherLabel",
]
