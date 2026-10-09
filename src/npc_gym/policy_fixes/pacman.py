"""Pacman collision and Trapped policy-fix abstractions."""

from __future__ import annotations

from importlib.resources import files
from math import isfinite
from numbers import Real

from npc_gym.envs.pacman.labels import PacmanAuthorityState, PacmanGhostState, PacmanLabel, PacmanPlayerState
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.pacman_monitors import VEGAN_NORM_ID, VEGETARIAN_BLUE_NORM_ID, VEGETARIAN_ORANGE_NORM_ID
from npc_gym.policy_fixes.core import PlanningProblem, integer

_PROTECTED = {VEGAN_NORM_ID: (1, 2), VEGETARIAN_BLUE_NORM_ID: (1,), VEGETARIAN_ORANGE_NORM_ID: (2,)}
_Point = tuple[int, int]


class PacmanModel:
    """Build policy fixes using the OFTEN reference's local collision abstraction.

    Construct from any nonterminal ``env.unwrapped.labeling_state()`` snapshot.
    Reuse the model while the layout and ghost identities match. Only the
    layout, player position and protected ghost positions affect predictions.
    Ghosts move one cardinal cell per turn, regardless of direction or scared
    status. Food, capsules, respawns and Stop are not modeled.

    ``norm_id`` selects protected ghosts: Vegan protects blue and orange;
    Vegetarian Blue or Vegetarian Orange protects only the named color. All
    variants share the same ASP rules and solver reuse.

    ``horizon`` counts future actions (upstream uses this value plus one).
    Defaults of one action and square-window ``radius=5`` match its experiments.
    Ghosts outside the window are ignored; fractional relative positions are
    truncated toward zero. Use radius >= horizon to observe walls at every
    possible planned player position. See the catalogue for the exact costs.
    """

    def __init__(
        self,
        initial_state: PacmanAuthorityState,
        *,
        norm_id: str = VEGAN_NORM_ID,
        horizon: int = 1,
        radius: int = 5,
    ) -> None:
        if not isinstance(norm_id, str) or norm_id not in _PROTECTED:
            raise ValueError(f"Unsupported Pacman policy-fix norm {norm_id!r}; expected one of {tuple(_PROTECTED)}")
        self.norm_id = norm_id
        self.horizon = integer("horizon", horizon, minimum=1)
        integer("upstream horizon", self.horizon + 1, minimum=2)
        self.radius = integer("radius", radius)
        integer("planning extent", self.radius + self.horizon)
        _validate_state(initial_state)
        if initial_state.terminated:
            raise ValueError("Cannot construct a Pacman model from a terminated snapshot")
        self._layout = initial_state.layout
        self._ghost_ids = tuple(ghost.ghost_id for ghost in initial_state.ghosts)
        rules = files("npc_gym.policy_fixes").joinpath("pacman.lp").read_text(encoding="utf-8")
        self._program = f"#const radius={self.radius}. #const ghosts={len(initial_state.ghosts)}.\n" + rules

    def problem(self, state: PacmanAuthorityState, *, remaining_steps: int | None = None) -> PlanningProblem:
        """Replace the local window and ghost inputs without advancing history.

        Candidates are North 1, South 2, East 3 and West 4. Values may also
        include Stop 0, but optimal plans never select it. Moves into known walls
        are infeasible. Costs predict weighted abstract encounters, not monitor counts.
        A terminated snapshot or nonpositive remaining budget is rejected.
        """
        _validate_state(state)
        if state.layout != self._layout or tuple(g.ghost_id for g in state.ghosts) != self._ghost_ids:
            raise ValueError("state must have the model's layout and ghost identities; construct a new model")
        if state.terminated:
            raise ValueError("Cannot plan from a terminated Pacman snapshot; reset the environment")
        horizon = self.horizon
        if remaining_steps is not None:
            horizon = min(horizon, integer("remaining_steps", remaining_steps, minimum=1))
        x, y = map(int, state.player.position)
        layout = state.layout
        inputs = [
            f"wall({dx},{dy})"
            for dx in range(-self.radius, self.radius + 1)
            for dy in range(-self.radius, self.radius + 1)
            if not (0 <= x + dx < layout.width and 0 <= y + dy < layout.height) or (x + dx, y + dy) in layout.walls
        ]
        for index, ghost in enumerate(state.ghosts):
            dx, dy = ghost.position[0] - x, ghost.position[1] - y
            if ghost.ghost_id not in _PROTECTED[self.norm_id] or abs(dx) > self.radius or abs(dy) > self.radius:
                inputs.append(f"goutside({index})")
            else:
                inputs.extend((f"gcol({index},{int(dx)},0)", f"grow({index},{int(dy)},0)"))
        return PlanningProblem.from_externals(
            static=f"#const horizon={horizon + 1}.\n" + self._program,
            true_atoms=inputs,
            horizon=horizon,
        )


class PacmanTrappedModel:
    """Keep Pacman west while the observed Trapped restriction is active.

    Pass reset labels as a ``MonitorInput`` at construction or ``reset``, then
    call ``advance`` once per real transition, including episode endings.
    Score zero activates the restriction; score greater than 400 clears it.
    Use one model per environment. Layouts may change across resets.

    One-step costs hold this flag fixed: an action costs one if its destination
    is outside the west while active, otherwise zero. Score changes, ghosts,
    food and task termination are not predicted. In particular, a move that
    would release the restriction may still cost one; a future activation is
    recognized only after observing it. This is not a compliance guarantee.
    """

    def __init__(self, initial_input: MonitorInput) -> None:
        self._rules = files("npc_gym.policy_fixes").joinpath("pacman_trapped.lp").read_text(encoding="utf-8")
        self.reset(initial_input)

    def reset(self, initial_input: MonitorInput) -> None:
        """Clear episode history and consume the new reset labels, as the monitor does."""
        if not isinstance(initial_input, MonitorInput):
            raise TypeError("reset requires a MonitorInput containing the reset labels")
        self._active = False
        self._ended = False
        self.advance(initial_input)

    def advance(self, input: MonitorInput) -> None:
        """Consume one actual transition; a high score takes precedence over zero.

        Returning west does not clear the restriction. Once released, falling
        below the threshold does not reactivate it unless score zero is observed.
        Planning or advancing after an observed ending requires a reset.
        """
        if not isinstance(input, MonitorInput):
            raise TypeError("advance requires a MonitorInput for one real transition")
        if self._ended:
            raise ValueError("Episode has ended; reset the model before advancing")
        if PacmanLabel.SCORE_GREATER_400 in input.labels:
            self._active = False
        elif PacmanLabel.SCORE_0 in input.labels:
            self._active = True
        self._ended = input.terminated or input.truncated

    def problem(self, state: PacmanAuthorityState) -> PlanningProblem:
        """Build a one-step problem without changing history or the snapshot.

        Supply values for Stop 0, North 1, South 2, East 3 and West 4. Blocked
        directions stay in place, matching the environment. West means
        ``x < layout.width / 2``. Only layout and player position affect the
        prediction; reset and transition labels are the source of norm history.
        Use ``ASPPlanner()`` to minimize violations before policy rank.
        """
        if self._ended:
            raise ValueError("Episode has ended; reset the model before planning")
        _validate_state(state)
        if state.terminated:
            raise ValueError("Cannot plan from a terminated Pacman snapshot; reset the environment")
        x, y = map(int, state.player.position)
        layout = state.layout
        atoms = ["active"] if self._active else []
        for action, (dx, dy) in enumerate(((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < layout.width and 0 <= ny < layout.height) or (nx, ny) in layout.walls:
                nx = x
            if 2 * nx >= layout.width:
                atoms.append(f"outside({action})")
        return PlanningProblem.from_externals(static=self._rules, true_atoms=atoms)


def _point(position: tuple[float, float], *, half: bool) -> _Point:
    if not isinstance(position, tuple) or len(position) != 2:
        raise ValueError("Pacman positions must be coordinate pairs")
    scale = 2 if half else 1
    for value in position:
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
            raise ValueError("Pacman coordinates must be finite nonnegative numbers")
        if float(value * scale) != int(value * scale):
            raise ValueError("Pacman positions must lie on the player grid or ghost half-grid")
    return int(2 * position[0]), int(2 * position[1])


def _validate_state(state: PacmanAuthorityState) -> None:
    if not isinstance(state, PacmanAuthorityState):
        raise TypeError("state must be a PacmanAuthorityState from labeling_state()")
    layout = state.layout
    integer("layout width", layout.width, minimum=1)
    integer("layout height", layout.height, minimum=1)
    # Bound coordinate arithmetic before validating cells and half-grid edges.
    integer("doubled layout extent", 2 * max(layout.width, layout.height) + 2)
    for x, y in layout.walls | state.food | state.capsules:
        _point((x, y), half=False)
        if x >= layout.width or y >= layout.height:
            raise ValueError("Map cells must lie within the layout")
    if (state.food | state.capsules) & layout.walls:
        raise ValueError("Food and capsules cannot occupy walls")
    identities = tuple(g.ghost_id for g in state.ghosts)
    if identities != tuple(range(1, len(identities) + 1)):
        raise ValueError("Ghost identities must be consecutive from 1 in engine movement order")
    agents: tuple[PacmanPlayerState | PacmanGhostState, ...] = (state.player, *state.ghosts)
    for agent in agents:
        x, y = _point(agent.position, half=agent is not state.player)
        if integer("direction", agent.direction) > 4:
            raise ValueError("Pacman directions must be integers from 0 through 4")
        # A fractional coordinate must be on a traversable edge, not in a wall.
        cells = {(x // 2, y // 2), ((x + 1) // 2, (y + 1) // 2)}
        if (
            x % 2
            and y % 2
            or any(cx >= layout.width or cy >= layout.height or (cx, cy) in layout.walls for cx, cy in cells)
        ):
            raise ValueError("Agent positions must lie on traversable cells or half-grid edges")
        if (x % 2 and agent.direction not in (3, 4)) or (y % 2 and agent.direction not in (1, 2)):
            raise ValueError("Half-grid ghosts must face along their edge")
    for ghost in state.ghosts:
        if integer("scared_timer", ghost.scared_timer) > 40 or ghost.scared != (ghost.scared_timer > 0):
            raise ValueError("Ghost scared status must agree with its timer in [0, 40]")
        if not ghost.scared:
            _point(ghost.position, half=False)
        integer("eaten_count", ghost.eaten_count)
    if state.terminated != (state.won or state.lost) or state.player.alive == state.lost:
        raise ValueError("Pacman terminal and player status flags are inconsistent")


__all__ = ["PacmanModel", "PacmanTrappedModel"]
