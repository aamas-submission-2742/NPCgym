"""Input shared by finite-trace monitors."""

from dataclasses import dataclass

from npc_gym.labels import LabelSet


@dataclass(frozen=True, slots=True)
class MonitorInput:
    """Labels and lifecycle flags for one reset or environment transition."""

    labels: LabelSet
    terminated: bool = False
    truncated: bool = False


def validate_monitor_name(name: str) -> None:
    """Require a nonempty string for a monitor, member, or bolt identifier."""
    if not isinstance(name, str):
        raise TypeError("monitor names must be strings")
    if not name:
        raise ValueError("monitor names must not be empty")
