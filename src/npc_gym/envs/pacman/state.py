"""Detached gameplay snapshots for user-defined observations."""

from dataclasses import dataclass
from typing import Literal

from .labels import PacmanAuthorityState
from .layout import Cell

GhostBehavior = Literal["random", "deterministic", "partly-deterministic"]


@dataclass(frozen=True, slots=True)
class GhostConfig:
    """Fixed turn durations and junction noise; no difficulty progression.

    None durations select bundled-map defaults (custom maps use 20/70).
    ``random_turn_probability`` applies only to partly deterministic ghosts.
    Random behavior ignores all fields; snapshots retain the resolved values
    for inspection, with no active phase or countdown.
    """

    scatter_turns: int | None = None
    chase_turns: int | None = None
    random_turn_probability: float = 0.2

    def __post_init__(self) -> None:
        for name in ("scatter_turns", "chase_turns"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        value = self.random_turn_probability
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("random_turn_probability must be a finite number between 0 and 1")


@dataclass(frozen=True, slots=True)
class PacmanSnapshot:
    """Full observable gameplay state, without mutable engine objects.

    ``game`` preserves the authority/labeling contract. Ghost spawns and pending
    reversals use its ghost order. Phase is the underlying schedule; frightened
    status remains per ghost. Random mode has no phase or countdown. This is an
    observation value, not a checkpoint of the random-number generator.
    """

    game: PacmanAuthorityState
    ghost_behavior: GhostBehavior
    config: GhostConfig
    phase: Literal["scatter", "chase"] | None
    phase_turns_remaining: int
    schedule_paused: bool
    ghost_spawns: tuple[Cell, ...]
    pending_reversals: tuple[bool, ...]
    step_count: int
