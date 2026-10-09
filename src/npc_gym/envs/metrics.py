"""Public information keys shared by NPC Gym environments."""

from typing import Final

EPISODE_METRICS_KEY: Final = "episode_metrics"
"""Key for cumulative, numeric episode metrics in reset and transition information."""

__all__ = ["EPISODE_METRICS_KEY"]
