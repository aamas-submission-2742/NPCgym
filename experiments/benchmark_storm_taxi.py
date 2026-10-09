"""Measure short Storm Taxi workloads; emit measurements to stdout, never research files."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence

import gymnasium as gym
import numpy as np

from npc_gym.envs import StormTaxiEnv
from npc_gym.envs.taxi._storm_weather import weather_outcomes

Factory = Callable[[], StormTaxiEnv]


def measure(factory: Factory, *, steps: int, resets: int, constructions: int) -> dict[str, float]:
    """Measure construction, reset, repeated states, varied states and short episodes."""
    started = time.perf_counter()
    for _ in range(constructions):
        factory().close()
    result = {"construction_us": (time.perf_counter() - started) * 1e6 / constructions}
    env = factory()
    try:
        env.reset(seed=17)
        for index in range(1000):
            env.step(index % 7)
        started = time.perf_counter()
        for _ in range(resets):
            env.reset()
        result["reset_us"] = (time.perf_counter() - started) * 1e6 / resets
        for workload in ("hot", "varied", "rollout"):
            env.reset(seed=17)
            started = time.perf_counter()
            for index in range(steps):
                if workload == "varied":
                    env.s = index * 104729 % 352000
                elif workload == "rollout" and index % 50 == 0:
                    env.reset()
                _, _, terminated, _, _ = env.step(6 if workload == "hot" else index % 7)
                if terminated:
                    env.reset()
            result[f"{workload}_step_us"] = (time.perf_counter() - started) * 1e6 / steps
    finally:
        env.close()
    return result


def memory(factory: Factory) -> dict[str, int]:
    """Measure Python/NumPy allocations separately so tracing does not affect timing."""
    weather_outcomes.cache_clear()
    gc.collect()
    tracemalloc.start()
    env = factory()
    try:
        env.reset(seed=17)
        construction_live, construction_peak = tracemalloc.get_traced_memory()
        for index in range(5000):
            env.s = index * 104729 % 352000
            env.step(index % 7)
        live, peak = tracemalloc.get_traced_memory()
        return {
            "construction_live_bytes": construction_live,
            "construction_peak_bytes": construction_peak,
            "after_varied_steps_live_bytes": live,
            "after_varied_steps_peak_bytes": peak,
        }
    finally:
        env.close()
        tracemalloc.stop()


def main(argv: Sequence[str] | None = None) -> None:
    """Run repeated measurements; small CLI budgets also exercise the utility in CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--constructions", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.steps, args.resets, args.constructions, args.repeats) < 1:
        parser.error("all measurement budgets must be positive")
    factories: dict[str, Factory] = {"storm": StormTaxiEnv}
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in factories}
    for _ in range(args.repeats):
        for name, factory in factories.items():
            samples[name].append(
                measure(factory, steps=args.steps, resets=args.resets, constructions=args.constructions)
            )
    medians = {
        name: {key: statistics.median(sample[key] for sample in values) for key in values[0]}
        for name, values in samples.items()
    }
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gymnasium": gym.__version__,
                "numpy": np.__version__,
                "budgets": vars(args),
                "samples": samples,
                "medians": medians,
                "memory": {name: memory(factory) for name, factory in factories.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
