"""Typed transition labels for the Pacman environment."""

from dataclasses import dataclass
from enum import StrEnum

from npc_gym.labels import LabelSet, Transition


@dataclass(frozen=True, slots=True)
class PacmanLayoutState:
    """Immutable geometry of one bundled Pacman layout."""

    name: str
    width: int
    height: int
    walls: frozenset[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class PacmanPlayerState:
    """Authority-visible player position, direction, and status."""

    position: tuple[float, float]
    direction: int
    alive: bool


@dataclass(frozen=True, slots=True)
class PacmanGhostState:
    """Authority-visible state of one identified ghost."""

    ghost_id: int
    position: tuple[float, float]
    direction: int
    scared: bool
    scared_timer: int
    eaten_count: int


@dataclass(frozen=True, slots=True)
class PacmanAuthorityState:
    """Detached Pacman state available to external labeling functions."""

    layout: PacmanLayoutState
    food: frozenset[tuple[int, int]]
    capsules: frozenset[tuple[int, int]]
    player: PacmanPlayerState
    ghosts: tuple[PacmanGhostState, ...]
    score: float
    terminated: bool
    won: bool
    lost: bool
    killed_blue: bool
    killed_orange: bool
    ate_capsule: bool
    last_action: int | None


class PacmanLabel(StrEnum):
    """Authority-visible Pacman events, state propositions, and actions."""

    STOP = "Stop"
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    EAT_BLUE_GHOST = "eatBlueGhost"
    EAT_ORANGE_GHOST = "eatOrangeGhost"
    ADJACENT_BLUE_GHOST = "adjacentBlueGhost"
    ADJACENT_ORANGE_GHOST = "adjacentOrangeGhost"
    STAYED_STILL = "stayedStill"
    LOSE = "lose"
    SCORE_0 = "score0"
    SCORE_GREATER_70 = "scoreGreater70"
    SCORE_GREATER_80 = "scoreGreater80"
    SCORE_GREATER_90 = "scoreGreater90"
    SCORE_GREATER_100 = "scoreGreater100"
    SCORE_GREATER_150 = "scoreGreater150"
    SCORE_GREATER_400 = "scoreGreater400"
    SCORE_GREATER_500 = "scoreGreater500"
    WEST_SIDE = "westSide"
    EAT_POWER_PELLET = "eatPowerPellet"
    IN_CORNER = "inCorner"
    IN_SOUTH_EAST = "inSouthEast"


_ACTION_LABELS = (
    PacmanLabel.STOP,
    PacmanLabel.NORTH,
    PacmanLabel.SOUTH,
    PacmanLabel.EAST,
    PacmanLabel.WEST,
)


class PacmanLabelingFunction:
    """Produce Pacman propositions from public immutable authority states."""

    def __call__(self, transition: Transition[PacmanAuthorityState, int]) -> LabelSet:
        return authority_labels(transition.state, transition.action)


def authority_labels(state: PacmanAuthorityState, action: int | None) -> LabelSet:
    """Label a detached state with the existing blue/orange event semantics."""
    result: set[PacmanLabel] = set()
    if action is not None:
        result.add(_ACTION_LABELS[action])
    for active, label in (
        (state.killed_blue, PacmanLabel.EAT_BLUE_GHOST),
        (state.killed_orange, PacmanLabel.EAT_ORANGE_GHOST),
        (state.ate_capsule, PacmanLabel.EAT_POWER_PELLET),
        (state.last_action == 0, PacmanLabel.STAYED_STILL),
        (state.lost, PacmanLabel.LOSE),
        (state.score == 0, PacmanLabel.SCORE_0),
        (state.player.position[0] < state.layout.width / 2, PacmanLabel.WEST_SIDE),
    ):
        if active:
            result.add(label)
    for ghost, label in zip(state.ghosts[:2], (PacmanLabel.ADJACENT_BLUE_GHOST, PacmanLabel.ADJACENT_ORANGE_GHOST)):
        dx = abs(ghost.position[0] - state.player.position[0])
        dy = abs(ghost.position[1] - state.player.position[1])
        if ghost.scared and dx <= 1 and dy <= 1 and dx + dy > 0:
            result.add(label)
    for threshold in (70, 80, 90, 100, 150, 400, 500):
        if state.score > threshold:
            result.add(PacmanLabel(f"scoreGreater{threshold}"))
    x, y = state.player.position
    if x in (1, state.layout.width - 2) and y in (1, state.layout.height - 2):
        result.add(PacmanLabel.IN_CORNER)
    if (x, y) == (state.layout.width - 2, state.layout.height - 2):
        result.add(PacmanLabel.IN_SOUTH_EAST)
    return frozenset(result)
