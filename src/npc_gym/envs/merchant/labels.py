"""Typed transition labels for the Merchant environment."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from npc_gym.labels import LabelSet, Transition

MerchantState: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MerchantAuthorityState:
    """Detached Merchant state available to external labeling functions.

    ``position`` uses ``(x, y)`` coordinates and ``cell`` is the label of the
    current cell. Resource availability follows the row-major order of the
    ``"T"`` and ``"R"`` cells in ``layout``.
    """

    position: tuple[int, int]
    cell: str
    carried_wood: int
    carried_ore: int
    time: int
    wood_available: tuple[bool, ...]
    ore_available: tuple[bool, ...]
    layout_name: str
    layout: tuple[str, ...]


class MerchantLabel(StrEnum):
    """Authority-visible Merchant state and action propositions."""

    AT_TREE = "atTree"
    AT_ROCK = "atRock"
    AT_HOME = "atHome"
    AT_MARKET = "atMarket"
    AT_DANGER = "atDanger"
    ATTACK = "attack"
    SUNDOWN = "sundown"
    HAS_WOOD = "hasWood"
    HAS_ORE = "hasOre"
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    EXTRACT = "extract"
    UNLOAD = "unload"
    FIGHT = "fight"


_ACTION_LABELS = (
    MerchantLabel.NORTH,
    MerchantLabel.SOUTH,
    MerchantLabel.EAST,
    MerchantLabel.WEST,
    MerchantLabel.EXTRACT,
    MerchantLabel.UNLOAD,
    MerchantLabel.FIGHT,
)


class MerchantLabelingFunction:
    """Produce the propositions used by the existing Merchant norms."""

    def __init__(self, sunset: int) -> None:
        self._sunset = sunset

    def __call__(self, transition: Transition[MerchantState, int]) -> LabelSet:
        state = transition.state
        result: set[MerchantLabel] = set()

        cell_label = state[2]
        if cell_label == 4:
            result.add(MerchantLabel.AT_TREE)
        elif cell_label == 3:
            result.add(MerchantLabel.AT_ROCK)
        elif cell_label == 0:
            result.add(MerchantLabel.AT_HOME)
        elif cell_label == 2:
            result.add(MerchantLabel.AT_MARKET)
        elif cell_label == 1:
            result.update((MerchantLabel.AT_DANGER, MerchantLabel.ATTACK))

        if state[5] == self._sunset:
            result.add(MerchantLabel.SUNDOWN)
        if state[3] > 0:
            result.add(MerchantLabel.HAS_WOOD)
        if state[4] > 0:
            result.add(MerchantLabel.HAS_ORE)
        if transition.action is not None:
            result.add(_ACTION_LABELS[transition.action])

        return frozenset(result)


__all__ = ["MerchantAuthorityState", "MerchantLabel", "MerchantLabelingFunction", "MerchantState"]
