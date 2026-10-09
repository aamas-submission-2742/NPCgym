"""Compare the new simulation against a locally captured reference JSON."""

import argparse
import json
from dataclasses import asdict

import numpy as np
from pacman_reference import canonical

from npc_gym.envs.pacman.labels import authority_labels
from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.observations import VectorFeatures, feature_array
from npc_gym.envs.pacman.simulation import Simulation


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layouts", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--vectors", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.reference.read_text())
    count = 0
    for key, rows in data.items():
        layout, mode, seed = key.split("/")
        if mode != "essential" and not args.vectors:
            continue
        sim = Simulation(PacmanLayout.from_file(args.layouts / f"{layout}.lay"), np.random.default_rng(int(seed)))
        features = VectorFeatures(mode, sim.layout, {"VegBlueDFA": 1.0} if mode in ("dfa", "dfa-distinguish") else {})
        action = None
        for turn, row in enumerate(rows):
            actual = canonical(asdict(sim.snapshot()))
            assert sorted(authority_labels(sim.snapshot(), action)) == row["info"]["labels"], (key, turn, "labels")
            if args.vectors:
                observed = feature_array(features.values(sim, action))
                expected = np.array(row["observation"])
                assert np.array_equal(observed, expected), (key, turn, observed, expected)
            assert actual == row["state"], (key, turn, actual, row["state"])
            assert sim.rng.bit_generator.state == row["rng"], (key, turn, "rng")
            if "transition" in row:
                action, reward, terminated, _ = row["transition"]
                result, _ = sim.step(action)
                assert (result, sim.terminated) == (reward, terminated), (key, turn, "transition")
            count += 1
    print(f"Exact simulation parity: {count} states")


if __name__ == "__main__":
    main()
