"""Capture/compare Pacman contracts in an explicitly selected source checkout.

Run with PYTHONPATH pointing to the desired checkout's src directory. Reference
outputs and layouts belong in ignored working directories, never package data.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from npc_gym.envs.pacman.pacman_env import SUPPORTED_VECTOR_FEATURES, PacmanEnv


def canonical(value: Any) -> Any:
    """Convert snapshots and NumPy values to portable comparison data."""
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def food_action(state: Any) -> int:
    """Deterministic food/capsule pursuit, with cardinal-order search ties."""
    start = tuple(map(int, state.player.position))
    queue = deque([(start, 0)])
    seen = {start}
    targets = state.food | state.capsules
    while queue:
        (x, y), first = queue.popleft()
        if (x, y) in targets:
            return first
        for action, (dx, dy) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0)), 1):
            cell = x + dx, y + dy
            if cell in seen or cell in state.layout.walls:
                continue
            if not (0 <= cell[0] < state.layout.width and 0 <= cell[1] < state.layout.height):
                continue
            seen.add(cell)
            queue.append((cell, first or action))
    return 0


def capture(layouts: Path, seeds: int, steps: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path in sorted(layouts.glob("*.lay")):
        for mode in SUPPORTED_VECTOR_FEATURES:
            # The replacement accepts explicit external resources; the frozen
            # implementation resolves the same named resources in its checkout.
            try:
                from npc_gym.envs.pacman.layout import PacmanLayout
            except ModuleNotFoundError:
                layout: Any = path.stem
            else:
                layout = PacmanLayout.from_file(path)
            dfas = {"VegBlueDFA": 1.0} if mode in ("dfa", "dfa-distinguish") else None
            with PacmanEnv(layout=layout, features=mode, dfas=dfas) as env:
                for seed in range(seeds):
                    obs, info = env.reset(seed=seed)
                    rows = []
                    actions = np.random.default_rng(seed + 400)
                    for turn in range(steps + 1):
                        state = env.labeling_state()
                        snapshot = asdict(state)
                        # Compare the authority contract, excluding new additive fields.
                        for ghost in snapshot["ghosts"]:
                            for key in list(ghost):
                                if key not in (
                                    "ghost_id",
                                    "position",
                                    "direction",
                                    "scared",
                                    "scared_timer",
                                    "eaten_count",
                                ):
                                    del ghost[key]
                        rows.append(
                            canonical(
                                {
                                    "state": snapshot,
                                    "observation": obs,
                                    "info": info,
                                    "rng": env.np_random.bit_generator.state,
                                }
                            )
                        )
                        if state.terminated or turn == steps:
                            break
                        action = int(actions.integers(5)) if turn % 4 == 0 else food_action(state)
                        obs, reward, terminated, truncated, info = env.step(action)
                        rows[-1]["transition"] = [action, reward, terminated, truncated]
                    records[f"{path.stem}/{mode}/{seed}"] = rows
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layouts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    data = capture(args.layouts, args.seeds, args.steps)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    if args.compare:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        for key in sorted(expected.keys() | data.keys()):
            if expected.get(key) != data.get(key):
                old, new = expected.get(key, []), data.get(key, [])
                turn = next((i for i, pair in enumerate(zip(old, new)) if pair[0] != pair[1]), min(len(old), len(new)))
                raise AssertionError(f"Pacman parity differs at {key}, turn {turn}; inspect the two output files")
        print(f"Exact parity: {len(data)} trajectories, {sum(map(len, data.values()))} states")
    else:
        print(f"Captured {len(data)} trajectories, {sum(map(len, data.values()))} states")


if __name__ == "__main__":
    main()
