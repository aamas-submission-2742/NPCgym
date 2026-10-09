"""Registry for versioned monitor definitions."""

from collections.abc import Callable

from npc_gym.monitors.contracts import validate_monitor_name
from npc_gym.monitors.events import Monitor

MonitorFactory = Callable[[], Monitor]


class MonitorRegistryError(ValueError):
    """Base class for monitor registry errors."""


class DuplicateMonitorIDError(MonitorRegistryError):
    """Raised when registering an identifier that is already in use."""


class UnknownMonitorIDError(MonitorRegistryError):
    """Raised when requesting an identifier that is not registered."""


class MonitorRegistry:
    """Map stable monitor identifiers to factories that create fresh monitors."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Monitor]] = {}

    @property
    def ids(self) -> tuple[str, ...]:
        """Return registered identifiers in deterministic order."""
        return tuple(sorted(self._factories))

    def register(self, monitor_id: str, factory: Callable[[], Monitor]) -> None:
        """Register ``factory`` under ``monitor_id``.

        Identifiers are expected to be stable, versioned names such as
        ``"merchant/danger-v0"``.
        """
        validate_monitor_name(monitor_id)
        if not callable(factory):
            raise TypeError("factory must be callable")
        if monitor_id in self._factories:
            raise DuplicateMonitorIDError(f"Monitor ID {monitor_id!r} is already registered")
        self._factories[monitor_id] = factory

    def make(self, monitor_id: str) -> Monitor:
        """Create a fresh monitor for ``monitor_id``."""
        try:
            factory = self._factories[monitor_id]
        except KeyError as error:
            available = ", ".join(self.ids) or "none"
            raise UnknownMonitorIDError(
                f"Unknown monitor ID {monitor_id!r}. Available monitor IDs: {available}"
            ) from error
        monitor = factory()
        if not isinstance(monitor, Monitor):
            raise TypeError(f"Factory for {monitor_id!r} must return a monitor")
        return monitor


__all__ = [
    "DuplicateMonitorIDError",
    "MonitorFactory",
    "MonitorRegistry",
    "MonitorRegistryError",
    "UnknownMonitorIDError",
]
