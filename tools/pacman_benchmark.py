"""Measure warmed Pacman operations in an explicitly selected source checkout."""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from npc_gym.envs import PacmanEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        from npc_gym.envs.pacman.layout import PacmanLayout
    except ModuleNotFoundError:
        layout: Any = args.layout.stem
    else:
        layout = PacmanLayout.from_file(args.layout)
    results: dict[str, Any] = {"python": platform.python_version(), "layout": args.layout.stem}
    for name, features in [("transitions", "essential"), ("vector", "complete"), ("rgb", "image-full")]:
        durations = []
        with PacmanEnv(layout=layout, features=features) as env:
            env.reset(seed=5)
            for index in range(1500):
                start = time.perf_counter_ns()
                if name == "transitions":
                    if hasattr(env, "_sim"):
                        env._sim.step(0)
                        terminated = env._sim.terminated
                    else:
                        terminated = env._gameplay.step(0).terminated
                else:
                    terminated = env.step(0)[2]
                elapsed = time.perf_counter_ns() - start
                if index >= 500:
                    durations.append(elapsed / 1000)
                if terminated:
                    env.reset(seed=5)
        results[name] = {"median_us": float(np.median(durations)), "p95_us": float(np.percentile(durations, 95))}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
