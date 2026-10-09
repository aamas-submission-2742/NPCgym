"""Original maze resources and explicit Berkeley-format text loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

Cell = tuple[int, int]
Position = tuple[float, float]
# Public action IDs: stop, north, south, east, west.
VECTORS: tuple[Cell, ...] = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))
REVERSE = (0, 2, 1, 4, 3)
BUNDLED_LAYOUTS = ("small", "medium", "large")


@dataclass(frozen=True, slots=True)
class PacmanLayout:
    """Immutable maze in bottom-left coordinates, with ordered ghost spawns.

    Use ``from_text`` or ``from_file`` to parse rectangular, wall-enclosed .lay
    files. Symbols are %, space, ., o, P, G and 1–4. Exactly one player and at
    least one food pellet are required. All open cells must be connected.
    Ghosts sort by marker (G equals 1), then x and y, matching .lay conventions.
    There is no implicit filesystem lookup for bundled names.
    """

    name: str
    width: int
    height: int
    walls: frozenset[Cell]
    food: frozenset[Cell]
    capsules: frozenset[Cell]
    player: Cell
    ghosts: tuple[Cell, ...]

    @property
    def default_episode_steps(self) -> int:
        """Recommended external TimeLimit: small/custom 300, medium 500, large 800.

        The layout name selects this default, as it does for ghost schedules.
        The simulation itself has no turn limit.
        """
        return {"small": 300, "medium": 500, "large": 800}.get(self.name, 300)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("layout name must be a nonempty string")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 3 for value in (self.width, self.height)
        ):
            raise ValueError("layout dimensions must be integers at least 3")
        for name in ("walls", "food", "capsules"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        object.__setattr__(self, "ghosts", tuple(self.ghosts))
        coordinates = (*self.walls, *self.food, *self.capsules, self.player, *self.ghosts)
        if any(
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in cell)
            or not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height)
            for cell in coordinates
        ):
            raise ValueError("layout coordinates must be integer (x, y) tuples inside the grid")
        boundary = {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if x in (0, self.width - 1) or y in (0, self.height - 1)
        }
        if not boundary <= self.walls:
            raise ValueError("layout must have a closed wall boundary")
        if not self.food or self.walls & (self.food | self.capsules | {self.player, *self.ghosts}):
            raise ValueError("layout requires food and entities on traversable cells")
        if len(set(self.ghosts)) != len(self.ghosts) or self.player in self.ghosts:
            raise ValueError("player and ghost spawns must occupy distinct cells")
        if self.food & self.capsules:
            raise ValueError("food and power pellets cannot occupy the same cell")
        seen = {self.player}
        pending = [self.player]
        while pending:
            for cell in self.neighbors(pending.pop()):
                if cell not in seen:
                    seen.add(cell)
                    pending.append(cell)
        if len(seen) != self.width * self.height - len(self.walls):
            raise ValueError("all open layout cells must be reachable from the player")

    @classmethod
    def from_text(cls, text: str, *, name: str = "custom") -> PacmanLayout:
        """Parse text without trimming meaningful spaces; accept a final newline."""
        if not isinstance(text, str):
            raise TypeError("layout text must be a string")
        rows = [row.rstrip() for row in text.splitlines()]
        # Some .lay resources append redundant wall characters beyond the first
        # row width. Accept only wall-only tails, never discard playable cells.
        if rows:
            width = len(rows[0])
            rows = [row[:width] if len(row) > width and set(row[width:]) <= {"%"} else row for row in rows]
        if len(rows) < 3 or len(rows[0]) < 3 or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("layout must be a rectangular grid at least 3 by 3")
        width, height = len(rows[0]), len(rows)
        if any(char != "%" for char in rows[0] + rows[-1]) or any(row[0] != "%" or row[-1] != "%" for row in rows):
            raise ValueError("layout must have a closed wall boundary")
        cells: dict[str, set[Cell]] = {char: set() for char in "% .oPG1234"}
        for row_index, row in enumerate(rows):
            for x, char in enumerate(row):
                if char not in cells:
                    raise ValueError(f"unknown layout symbol {char!r} at row {row_index + 1}, column {x + 1}")
                cells[char].add((x, height - row_index - 1))
        if len(cells["P"]) != 1:
            raise ValueError("layout must contain exactly one player P")
        if not cells["."]:
            raise ValueError("layout must contain at least one food pellet")
        player = next(iter(cells["P"]))
        walls = frozenset(cells["%"])
        ghosts = sorted((1 if marker == "G" else int(marker), cell) for marker in "G1234" for cell in cells[marker])
        if not isinstance(name, str) or not name:
            raise ValueError("layout name must be a nonempty string")
        return cls(
            name,
            width,
            height,
            walls,
            frozenset(cells["."]),
            frozenset(cells["o"]),
            player,
            tuple(cell for _, cell in ghosts),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> PacmanLayout:
        """Read a user-supplied UTF-8 .lay file; propagate filesystem errors."""
        path = Path(path)
        return cls.from_text(path.read_text(encoding="utf-8"), name=path.stem)

    @classmethod
    def bundled(cls, name: str) -> PacmanLayout:
        """Load one of small, medium or large from package resources."""
        if not isinstance(name, str):
            raise TypeError("layout must be a string or PacmanLayout")
        name = name.removesuffix(".lay")
        if name not in BUNDLED_LAYOUTS:
            raise ValueError(f"Unknown Pacman layout {name!r}; expected one of {BUNDLED_LAYOUTS}")
        return cls.from_text(
            files(__package__).joinpath("layouts", name + ".lay").read_text(encoding="utf-8"), name=name
        )

    def neighbors(self, cell: Cell, *, include_stop: bool = False) -> tuple[Cell, ...]:
        """Return traversable destinations in north/south/east/west order.

        A wall origin is allowed for directional observation probes. Stop, if
        requested, is last and only present when the origin is traversable.
        """
        x, y = cell
        offsets = (*VECTORS[1:], VECTORS[0]) if include_stop else VECTORS[1:]
        return tuple(
            (x + dx, y + dy)
            for dx, dy in offsets
            if 0 <= x + dx < self.width and 0 <= y + dy < self.height and (x + dx, y + dy) not in self.walls
        )
