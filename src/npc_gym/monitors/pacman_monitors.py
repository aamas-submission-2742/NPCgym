"""Versioned Pacman regex monitor factories."""

from functools import partial

from npc_gym.monitors.builtins import BUILTIN_IDS, make_builtin_monitor
from npc_gym.monitors.events import Monitor
from npc_gym.monitors.registry import MonitorRegistry

VEGAN_NORM_ID = "pacman/vegan-v0"
VEGETARIAN_BLUE_NORM_ID = "pacman/vegetarian-blue-v0"
VEGETARIAN_ORANGE_NORM_ID = "pacman/vegetarian-orange-v0"
CONDITIONAL_VEGAN_NORM_ID = "pacman/conditional-vegan-v0"
OBL_BLUE_NORM_ID = "pacman/obligation-blue-v1"
VEGAN_PREFERENCE_NORM_ID = "pacman/vegan-preference-v0"
VEGAN_CONFLICT_NORM_ID = "pacman/vegan-conflict-v1"
CAUTIOUS_NORM_ID = "pacman/cautious-v0"
ALL_OR_NOTHING_NORM_ID = "pacman/all-or-nothing-v0"
ONE_TASTE_NORM_ID = "pacman/one-taste-v1"
SWITCH_NORM_ID = "pacman/switch-v0"
PENALTY_NORM_ID = "pacman/penalty-v0"
PENALTY_1_NORM_ID = "pacman/penalty-1-v0"
PENALTY_3_NORM_ID = "pacman/penalty-3-v0"
TRAPPED_NORM_ID = "pacman/trapped-v1"
HUNGRY_NORM_ID = "pacman/hungry-v0"
HUNGRY_VEGAN_NORM_ID = "pacman/hungry-vegan-v0"
HUNGRY_VEGETARIAN_NORM_ID = "pacman/hungry-vegetarian-v0"
HUNGRY_VEGAN_PENALTY_NORM_ID = "pacman/hungry-vegan-penalty-v1"
MAXIMUM_NORM_ID = "pacman/maximum-v2"
ERRAND_NORM_ID = "pacman/errand-v0"
VISIT_NORM_ID = "pacman/visit-v0"
SOLUTION_GUILT_MAXIMUM_NORM_ID = "pacman/solution-guilt-maximum-v2"

pacman_monitor_registry = MonitorRegistry()
for identifier in BUILTIN_IDS:
    if identifier.startswith("pacman/"):
        pacman_monitor_registry.register(identifier, partial(make_builtin_monitor, identifier))


def make_pacman_monitor(monitor_id: str) -> Monitor:
    """Create a fresh graph for a registered Pacman monitor."""
    return pacman_monitor_registry.make(monitor_id)


__all__ = [
    "ALL_OR_NOTHING_NORM_ID",
    "CAUTIOUS_NORM_ID",
    "CONDITIONAL_VEGAN_NORM_ID",
    "ERRAND_NORM_ID",
    "HUNGRY_NORM_ID",
    "HUNGRY_VEGAN_NORM_ID",
    "HUNGRY_VEGAN_PENALTY_NORM_ID",
    "HUNGRY_VEGETARIAN_NORM_ID",
    "MAXIMUM_NORM_ID",
    "OBL_BLUE_NORM_ID",
    "ONE_TASTE_NORM_ID",
    "PENALTY_1_NORM_ID",
    "PENALTY_3_NORM_ID",
    "PENALTY_NORM_ID",
    "SOLUTION_GUILT_MAXIMUM_NORM_ID",
    "SWITCH_NORM_ID",
    "TRAPPED_NORM_ID",
    "VEGAN_CONFLICT_NORM_ID",
    "VEGAN_NORM_ID",
    "VEGAN_PREFERENCE_NORM_ID",
    "VEGETARIAN_BLUE_NORM_ID",
    "VEGETARIAN_ORANGE_NORM_ID",
    "VISIT_NORM_ID",
    "make_pacman_monitor",
    "pacman_monitor_registry",
]
