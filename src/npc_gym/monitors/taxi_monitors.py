"""Versioned Taxi regex monitor factories."""

from functools import partial

from npc_gym.monitors.builtins import BUILTIN_IDS, make_builtin_monitor
from npc_gym.monitors.events import Monitor
from npc_gym.monitors.registry import MonitorRegistry

EMERGENCY_NORM_ID = "taxi/emergency-v0"

taxi_monitor_registry = MonitorRegistry()
for identifier in BUILTIN_IDS:
    if identifier.startswith("taxi/"):
        taxi_monitor_registry.register(identifier, partial(make_builtin_monitor, identifier))


def make_taxi_monitor(monitor_id: str) -> Monitor:
    """Create a fresh graph for a registered Taxi monitor."""
    return taxi_monitor_registry.make(monitor_id)


__all__ = ["EMERGENCY_NORM_ID", "make_taxi_monitor", "taxi_monitor_registry"]
