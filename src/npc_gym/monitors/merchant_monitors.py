"""Versioned Merchant regex monitor factories."""

from functools import partial

from npc_gym.monitors.builtins import BUILTIN_IDS, make_builtin_monitor
from npc_gym.monitors.events import Monitor
from npc_gym.monitors.registry import MonitorRegistry

DANGER_NORM_ID = "merchant/danger-v0"
DELIVERY_NORM_ID = "merchant/delivery-v0"
DELIVERY_PACIFIST_NORM_ID = "merchant/delivery-pacifist-v0"
ENV_FRIENDLY_NORM_ID = "merchant/env-friendly-v0"
EVOLVING_NORM_ID = "merchant/evolving-v0"
PACIFIST_NORM_ID = "merchant/pacifist-v0"

merchant_monitor_registry = MonitorRegistry()
for identifier in BUILTIN_IDS:
    if identifier.startswith("merchant/"):
        merchant_monitor_registry.register(identifier, partial(make_builtin_monitor, identifier))


def make_merchant_monitor(monitor_id: str) -> Monitor:
    """Create a fresh graph for a registered Merchant monitor."""
    return merchant_monitor_registry.make(monitor_id)


__all__ = [
    "DANGER_NORM_ID",
    "DELIVERY_NORM_ID",
    "DELIVERY_PACIFIST_NORM_ID",
    "ENV_FRIENDLY_NORM_ID",
    "EVOLVING_NORM_ID",
    "PACIFIST_NORM_ID",
    "make_merchant_monitor",
    "merchant_monitor_registry",
]
