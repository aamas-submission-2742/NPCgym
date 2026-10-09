"""Stateless projections of authority labels into the built-in regex vocabulary."""

from collections.abc import Mapping
from types import MappingProxyType

from npc_gym.envs.gardener.labels import FrogCollected
from npc_gym.envs.merchant.labels import MerchantLabel as M
from npc_gym.envs.pacman.labels import PacmanLabel as P
from npc_gym.envs.taxi.labels import TaxiActionLabel as A
from npc_gym.envs.taxi.labels import TaxiLocationLabel as L
from npc_gym.envs.taxi.labels import TaxiWeatherLabel as W
from npc_gym.labels import Label
from npc_gym.monitors.automaton import Proposition


def _label(label: Label) -> Proposition:
    return lambda input: label in input.labels


def _vocabulary(labels: Mapping[str, Label], **predicates: Proposition) -> Mapping[str, Proposition]:
    return MappingProxyType({**{atom: _label(label) for atom, label in labels.items()}, **predicates})


BUILTIN_PROPOSITIONS: Mapping[str, Mapping[str, Proposition]] = MappingProxyType(
    {
        "merchant": _vocabulary(
            {
                "danger": M.AT_DANGER,
                "fight": M.FIGHT,
                "home": M.AT_HOME,
                "market": M.AT_MARKET,
                "sundown": M.SUNDOWN,
                "tree": M.AT_TREE,
                "wood": M.HAS_WOOD,
                "extract": M.EXTRACT,
            }
        ),
        "gardener": _vocabulary(
            {},
            collected=lambda input: any(isinstance(label, FrogCollected) for label in input.labels),
            end=lambda input: input.terminated or input.truncated,
        ),
        "pacman": _vocabulary(
            {
                "blue": P.EAT_BLUE_GHOST,
                "orange": P.EAT_ORANGE_GHOST,
                "adjacent": P.ADJACENT_BLUE_GHOST,
                "west": P.WEST_SIDE,
                "pellet": P.EAT_POWER_PELLET,
                "still": P.STAYED_STILL,
                "zero": P.SCORE_0,
                "high": P.SCORE_GREATER_400,
                "corner": P.IN_CORNER,
                "southeast": P.IN_SOUTH_EAST,
            },
            eat=lambda input: bool({P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST} & input.labels),
            deadline=lambda input: bool({P.LOSE, P.SCORE_GREATER_100} & input.labels),
        ),
        "taxi": _vocabulary(
            {"rain": W.RAIN, "warn": A.WARN, "new_hurricane": W.NEW_HURRICANE, "hurricane": W.HURRICANE},
            safe=lambda input: (
                L.AT_SHELTER in input.labels or (L.AT_HOME in input.labels and W.FLOOD_RISK not in input.labels)
            ),
        ),
    }
)
