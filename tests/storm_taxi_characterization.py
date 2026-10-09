"""Fingerprints of Storm Taxi dynamics and labels unaffected by the location correction."""

import hashlib
import json
import struct
from dataclasses import asdict

from npc_gym.envs.taxi.labels import TaxiLocationLabel
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID, make_taxi_monitor
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant

_LOCATION_LABELS = frozenset(TaxiLocationLabel)


def dynamics_fingerprints(env):
    """Hash ordered outcomes and masks, without rounding or merging duplicates."""
    transitions, masks = hashlib.sha256(), hashlib.sha256()
    for state in range(env.observation_space.n):
        assert env.encode(*env.decode(state)) == state
        masks.update(env.action_mask(state).tobytes())
        for action in range(env.action_space.n):
            outcomes = env.transition_distribution(state, action)
            transitions.update(struct.pack("<B", len(outcomes)))
            for outcome in outcomes:
                transitions.update(struct.pack("<dIi?", *outcome))
    return {
        "transitions": transitions.hexdigest(),
        "masks": masks.hexdigest(),
        "initial": hashlib.sha256(env.initial_state_distrib.astype("<f8").tobytes()).hexdigest(),
    }


def trajectory_fingerprint(env):
    """Include resets, snapshots, unchanged labels, RNG state and warning counts.

    Location labels and safety counts are covered by semantic regression tests,
    allowing this fingerprint to guard the unchanged behavior independently.
    """
    records = []
    wrapper = IgnoreWeatherRelevant(env)
    monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
    for seed in range(16):
        for episode in range(3):
            state, info = env.reset(seed=seed if episode == 0 else None)
            monitor.reset(MonitorInput(info["labels"]))
            for step in range(65):
                records.append(
                    {
                        "state": state,
                        "compact": wrapper.observation(state),
                        "info": {
                            **info,
                            "labels": sorted(info["labels"] - _LOCATION_LABELS),
                            "action_mask": info["action_mask"].tolist(),
                        },
                        "snapshot": asdict(env.labeling_state()),
                        "counts": {"Warn Violations": monitor.counts["Warn Violations"]},
                        "fickle": bool(env.fickle_step),
                        "rng": env.np_random.bit_generator.state,
                    }
                )
                state, reward, terminated, truncated, info = env.step((step * 3 + seed + episode) % 7)
                records.append([reward, terminated, truncated])
                monitor.update(MonitorInput(info["labels"], terminated, truncated))
                if terminated or truncated:
                    break
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def delivery_trajectory_fingerprint(env):
    """Seek passengers/destinations with seeded noise, retaining terminal records too."""
    from collections import Counter, deque

    import numpy as np

    noise = np.random.default_rng(103)
    wrapper = IgnoreWeatherRelevant(env)
    monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
    records, coverage = [], Counter()
    # Derive a small routing graph from advisory masks, without stepping the env.
    graph = {}
    for row in range(5):
        for col in range(5):
            mask = env.action_mask(env.encode(row, col, 0, 1, False, 0, 0, False, 2))
            graph[row, col] = [
                (action, (row + dr, col + dc))
                for action, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1)))
                if mask[action]
            ]

    def action_toward(position, goal):
        queue = deque([(position, None)])
        seen = {position}
        while queue:
            cell, first = queue.popleft()
            for action, neighbor in graph[cell]:
                if neighbor in seen:
                    continue
                next_action = action if first is None else first
                if neighbor == goal:
                    return next_action
                seen.add(neighbor)
                queue.append((neighbor, next_action))
        raise AssertionError(f"No route from {position} to {goal}")

    def record(state, info):
        records.append(
            {
                "state": state,
                "compact": wrapper.observation(state),
                "info": {
                    **info,
                    "labels": sorted(info["labels"] - _LOCATION_LABELS),
                    "action_mask": info["action_mask"].tolist(),
                },
                "snapshot": asdict(env.labeling_state()),
                "counts": {"Warn Violations": monitor.counts["Warn Violations"]},
                "fickle": bool(env.fickle_step),
                "rng": env.np_random.bit_generator.state,
            }
        )

    for seed in range(16):
        for episode in range(3):
            state, info = env.reset(seed=seed if episode == 0 else None)
            monitor.reset(MonitorInput(info["labels"]))
            record(state, info)
            for _ in range(80):
                row, col, passenger, destination, *_ = env.decode(state)
                goal = tuple(env.locs[destination if passenger == 4 else passenger])
                action = (5 if passenger == 4 else 4) if (row, col) == goal else action_toward((row, col), goal)
                if noise.random() < 0.15:
                    action = int(noise.integers(7))
                state, reward, terminated, truncated, info = env.step(action)
                after = tuple(env.decode(state))
                coverage["destination_changes"] += int(destination != after[3])
                coverage["deliveries"] += int(terminated)
                coverage["rain_deliveries"] += int(terminated and reward == 20)
                coverage["pickups"] += int(passenger != 4 and after[2] == 4)
                monitor.update(MonitorInput(info["labels"], terminated, truncated))
                records.append([action, reward, terminated, truncated])
                record(state, info)
                if terminated or truncated:
                    break
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, dict(coverage)
