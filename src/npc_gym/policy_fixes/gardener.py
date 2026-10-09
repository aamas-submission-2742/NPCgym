"""Local Gardener planning for maintenance and punctual norms."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from types import MappingProxyType

from npc_gym.envs.gardener.gardener import GardenerObservation
from npc_gym.monitors.gardener_monitors import DRAIN_NORM_ID, NO_COLLECT_NORM_ID, PERMISSION_AWARE_NORM_ID
from npc_gym.policy_fixes.core import Objective, PlanningProblem, integer

_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
_RECIPES = {
    NO_COLLECT_NORM_ID: ("no_collect",),
    DRAIN_NORM_ID: ("drain",),
    PERMISSION_AWARE_NORM_ID: ("permission_aware",),
    "gardener/permission-drain-v0": ("permission_aware", "drain"),
}


class GardenerModel:
    """Plan from public snapshots without retaining normative history.

    Supports No Collect, Drain and permission-aware collection. The recipe
    ``gardener/permission-drain-v0`` sums unpermitted collections and harmful
    drainages, with the same cost per event. Construct from
    any snapshot; reuse while the layout and frog identities match. Pass the
    environment's actual ``puddle_respawn`` and ``frog_freeze`` parameters.

    ``horizon`` counts future actions; ``radius`` bounds a square around the
    current agent. Frogs and puddles outside it are ignored, as are frog paths
    after leaving it. Grass, score and task termination are not predicted.
    Collection costs count each frog at most once across all possible paths;
    drainage costs count possible harmful events per puddle and step.

    Use ``ASPPlanner(objectives=model.objectives)`` for equal-priority costs:
    norm weight 5 and first-action preference-rank weight 1. Predictions are
    local adverse possibilities, not expected counts or global guarantees.
    """

    def __init__(
        self,
        initial_state: GardenerObservation,
        *,
        norm_id: str = NO_COLLECT_NORM_ID,
        horizon: int = 1,
        radius: int = 4,
        puddle_respawn: int = 20,
        frog_freeze: int = 5,
    ) -> None:
        if not isinstance(norm_id, str) or norm_id not in _RECIPES:
            raise ValueError(f"Unsupported Gardener policy-fix norm {norm_id!r}; expected one of {tuple(_RECIPES)}")
        self.norm_id = norm_id
        self.horizon = integer("horizon", horizon, minimum=1)
        self.radius = integer("radius", radius)
        integer("planning extent", self.radius + self.horizon)
        self.puddle_respawn = integer("puddle_respawn", puddle_respawn, minimum=1)
        self.frog_freeze = integer("frog_freeze", frog_freeze)
        self._validate(initial_state)
        self._initial = initial_state
        self._rules = files("npc_gym.policy_fixes").joinpath("gardener.lp").read_text(encoding="utf-8")
        self.objectives: Mapping[str, Objective] = MappingProxyType(
            {"violations": Objective(weight=5, priority=1), "policy": Objective(priority=1)}
        )
        self._blocked = frozenset((*initial_state.walls, *initial_state.puddles))

    def problem(self, state: GardenerObservation, *, remaining_steps: int | None = None) -> PlanningProblem:
        """Build a local problem without modifying the environment or RNG.

        Supply all five action values. Blocked proposals execute as Stay and
        do not grant collection permission. ``remaining_steps`` optionally
        shortens the plan at a known time limit. Call only before episode end;
        snapshots do not contain a terminal flag.
        """
        self._validate(state)
        initial = self._initial
        if (state.size, state.grass, state.puddles, state.walls, len(state.frogs)) != (
            initial.size,
            initial.grass,
            initial.puddles,
            initial.walls,
            len(initial.frogs),
        ):
            raise ValueError("Snapshot layout and object identities must match; construct a new model")
        horizon = self.horizon
        if remaining_steps is not None:
            horizon = min(horizon, integer("remaining_steps", remaining_steps, minimum=1))
        x, y = state.agent
        window = {
            (u, v)
            for u in range(max(0, x - self.radius), min(state.size, x + self.radius + 1))
            for v in range(max(0, y - self.radius), min(state.size, y + self.radius + 1))
        }
        facts = [f"player(0,{x},{y}).", f"puddle_respawn({self.puddle_respawn}).", f"freeze({self.frog_freeze})."]
        facts.extend(f"recipe({recipe})." for recipe in _RECIPES[self.norm_id])
        for u, v in sorted(window - self._blocked):
            facts.append(f"window({u},{v}).")
            neighbors = []
            for action, (dx, dy) in enumerate(_VECTORS):
                target = (u + dx, v + dy)
                legal = 0 <= target[0] < state.size and 0 <= target[1] < state.size and target not in self._blocked
                a, b = target if legal else (u, v)
                effective = action if legal else 4
                facts.append(f"move({u},{v},{action},{a},{b},{effective}).")
                if action < 4 and legal:
                    neighbors.append((a, b))
            # Leaving the observation window drops a path; it is not a wall.
            for a, b in neighbors or [(u, v)]:
                if (a, b) in window:
                    facts.append(f"hop({u},{v},{a},{b}).")
        for i, ((u, v), full, timer) in enumerate(
            zip(state.puddles, state.puddles_full, state.puddle_timer, strict=True)
        ):
            if (u, v) in window:
                facts.extend((f"puddle({i},{u},{v}).", f"water(0,{i},{int(full)},{timer})."))
        for i, ((u, v), collected, timer) in enumerate(
            zip(state.frogs, state.collected_frogs, state.frog_timer, strict=True)
        ):
            if not collected and (u, v) in window:
                facts.append(f"frog(0,{i},{u},{v},{timer}).")
        return PlanningProblem.from_parts(static=self._rules, dynamic="\n".join(facts), horizon=horizon)

    def _validate(self, state: GardenerObservation) -> None:
        if not isinstance(state, GardenerObservation):
            raise TypeError("state must be a GardenerObservation from labeling_state()")
        integer("size", state.size, minimum=3)
        groups = (
            (state.grass, (state.grass_active,), state.grass_timer, None),
            (state.puddles, (state.puddles_full, state.puddle_drained), state.puddle_timer, self.puddle_respawn),
            (
                state.frogs,
                (state.collected_frogs, state.captured_frogs, state.frog_collected),
                state.frog_timer,
                self.frog_freeze,
            ),
        )
        for positions, flags, timers, maximum in groups:
            if any(len(values) != len(positions) for values in (*flags, timers)):
                raise ValueError("Object positions, flags and timers must have matching lengths")
            if any(type(value) is not bool for values in flags for value in values):
                raise ValueError("Snapshot flags must be Boolean")
            for timer in timers:
                value = integer("timer", timer)
                if maximum is not None and value > maximum:
                    raise ValueError("Snapshot timer exceeds its configured environment parameter")
        for position in (state.agent, *state.grass, *state.puddles, *state.frogs, *state.walls):
            if not isinstance(position, tuple) or len(position) != 2:
                raise ValueError("Positions must be coordinate pairs")
            if any(integer("coordinate", c) >= state.size for c in position):
                raise ValueError("Positions must be inside the grid")
        static = (*state.grass, *state.puddles, *state.walls)
        if len(set(static)) != len(static):
            raise ValueError("Grass, puddles and walls must occupy distinct cells")
        if {state.agent, *state.frogs} & {*state.walls, *state.puddles}:
            raise ValueError("Agent and frogs cannot occupy walls or puddles")
        if state.captured_frogs != state.collected_frogs or any(
            event and not collected
            for event, collected in zip(state.frog_collected, state.collected_frogs, strict=True)
        ):
            raise ValueError("Captured and collection event flags must agree with collected frogs")


__all__ = ["GardenerModel"]
