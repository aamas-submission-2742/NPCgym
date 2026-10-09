"""Versioned Gardener monitors, including identified object events."""

from functools import partial
from numbers import Integral

from npc_gym.envs.gardener.labels import FrogCollected, GardenerLabel, PuddleDrained
from npc_gym.monitors.automaton import AutomatonMonitor, Proposition
from npc_gym.monitors.builtins import RegexRecipe, make_builtin_monitor
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.events import ComplexMonitor, Monitor, MultiMonitor, SimpleMonitor
from npc_gym.monitors.registry import MonitorRegistry, UnknownMonitorIDError

COLLECT_ONE_NORM_ID = "gardener/collect-one-v0"
COLLECT_PERMISSION_NORM_ID = "gardener/collect-permission-v0"
DRAIN_NORM_ID = "gardener/drain-v0"
NO_COLLECT_NORM_ID = "gardener/no-collect-v0"
RESCUE_NORM_ID = "gardener/rescue-v1"
RESCUE_PER_FROG_NORM_ID = "gardener/rescue-v2"
PERMISSION_AWARE_NORM_ID = "gardener/permission-aware-v0"
OBJECT_NORM_IDS = (COLLECT_PERMISSION_NORM_ID, DRAIN_NORM_ID, NO_COLLECT_NORM_ID, PERMISSION_AWARE_NORM_ID)
GARDENER_NORM_IDS = (COLLECT_ONE_NORM_ID, *OBJECT_NORM_IDS, RESCUE_NORM_ID, RESCUE_PER_FROG_NORM_ID)


def rescue_recipe(frog_id: int) -> tuple[RegexRecipe, dict[str, Proposition]]:
    """Return the per-frog failure language and stateless label bindings.

    Each drainage activates a five-subsequent-step collection obligation.
    Expiry on step six wins over collection; episode end fails pending duties.
    Acceptance combines simultaneous failures for this frog into one event.
    Separate activations can still fail on successive inputs. Reset is ignored.
    """
    if isinstance(frog_id, bool) or not isinstance(frog_id, Integral) or frog_id < 0:
        raise ValueError("frog_id must be a nonnegative integer")
    trigger = "[drained & !collected & !end]"
    pending = "[!collected & !end]"
    expression = (
        ".*("
        + trigger
        + pending * 5
        + "."
        + "|[drained & !collected & end]|"
        + trigger
        + (pending + "?") * 4
        + "[!collected & end])"
    )
    return RegexRecipe(expression, consume_initial=False), {
        "collected": lambda input: FrogCollected(frog_id) in input.labels,
        "drained": lambda input: any(
            isinstance(label, PuddleDrained) and frog_id in label.nearby_frog_ids for label in input.labels
        ),
        "end": lambda input: input.terminated or input.truncated,
    }


class RescueMonitor(SimpleMonitor):
    """Count steps with failed rescue obligations, not the number of obligations.

    Every drainage/frog pair creates an obligation. Collection through the fifth
    subsequent input fulfills it; on the sixth, expiry takes precedence. Ending
    an episode fails all still-pending obligations on that input. Reset labels
    are ignored. Multiple failures on one input count once.
    """

    def __init__(self) -> None:
        super().__init__(consume_initial=False)
        self.reset_history()

    def reset_history(self) -> None:
        self._pending: list[tuple[int, int]] = []
        self._step = 0

    def detect(self, input: MonitorInput) -> bool:
        self._step += 1
        collected = {label.frog_id for label in input.labels if isinstance(label, FrogCollected)}
        failed = any(deadline < self._step for _, deadline in self._pending)
        pending = [
            (frog, deadline) for frog, deadline in self._pending if deadline >= self._step and frog not in collected
        ]
        for label in input.labels:
            if isinstance(label, PuddleDrained):
                pending.extend((frog, self._step + 5) for frog in label.nearby_frog_ids if frog not in collected)
        if input.terminated or input.truncated:
            failed = failed or bool(pending)
            pending = []
        self._pending = pending
        return failed


class _ObjectEvent(SimpleMonitor):
    def __init__(self, object_id: int, *, drainage: bool = False, permission: bool = False) -> None:
        super().__init__(consume_initial=False)
        self.object_id = object_id
        self.drainage = drainage
        self.permission = permission

    def detect(self, input: MonitorInput) -> bool:
        if self.drainage:
            return any(isinstance(x, PuddleDrained) and x.puddle_id == self.object_id for x in input.labels)
        return FrogCollected(self.object_id) in input.labels and (
            not self.permission or GardenerLabel.PERMITTED_COLLECT in input.labels
        )


gardener_monitor_registry = MonitorRegistry()
gardener_monitor_registry.register(COLLECT_ONE_NORM_ID, partial(make_builtin_monitor, COLLECT_ONE_NORM_ID))
gardener_monitor_registry.register(RESCUE_NORM_ID, RescueMonitor)


def make_gardener_monitor(
    monitor_id: str, *, num_frogs: int | None = None, num_puddles: int | None = None, minimize: bool = True
) -> Monitor:
    """Create a fresh monitor with fixed count keys for the configured objects.

    Object recipes require the matching environment's ``num_frogs`` (collection
    recipes) or ``num_puddles`` (Drain). IDs are zero-based. Collect One and
    Rescue v1 need neither size. Rescue v2 requires ``num_frogs`` and counts
    failures separately per frog using minimized DFAs; ``minimize=False``
    retains compiler states. Permission-aware reports raw, permitted and
    unpermitted collections independently, with explicit aggregate counts.
    """
    if monitor_id not in GARDENER_NORM_IDS:
        raise UnknownMonitorIDError(f"Unknown Gardener monitor {monitor_id!r}; available: {GARDENER_NORM_IDS}")
    if not isinstance(minimize, bool):
        raise TypeError("minimize must be a bool")
    if monitor_id == COLLECT_ONE_NORM_ID:
        return make_builtin_monitor(monitor_id, minimize=minimize)
    if monitor_id not in (*OBJECT_NORM_IDS, RESCUE_PER_FROG_NORM_ID):
        return gardener_monitor_registry.make(monitor_id)
    drainage = monitor_id == DRAIN_NORM_ID
    size = num_puddles if drainage else num_frogs
    parameter = "num_puddles" if drainage else "num_frogs"
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
        raise ValueError(f"{monitor_id} requires {parameter} as a nonnegative integer")
    members: dict[str, ComplexMonitor] = {}
    if monitor_id == RESCUE_PER_FROG_NORM_ID:
        for frog_id in range(int(size)):
            recipe, propositions = rescue_recipe(frog_id)
            members[f"Rescue/{frog_id}"] = AutomatonMonitor(
                recipe.compile(minimize=minimize), propositions=propositions, consume_initial=False
            )
        return MultiMonitor(members, derived={"Rescue": lambda c: sum(c.values())})
    for object_id in range(int(size)):
        event = _ObjectEvent(object_id, drainage=drainage)
        if monitor_id == PERMISSION_AWARE_NORM_ID:
            permission = _ObjectEvent(object_id, permission=True)
            members[f"Collected/{object_id}"] = event
            members[f"Permitted/{object_id}"] = permission
            members[f"Unpermitted/{object_id}"] = event & ~permission
        else:
            name = "Drain" if drainage else "CollectPerm" if monitor_id == COLLECT_PERMISSION_NORM_ID else "NoCollect"
            members[f"{name}/{object_id}"] = (
                _ObjectEvent(object_id, permission=True) if monitor_id == COLLECT_PERMISSION_NORM_ID else event
            )
    if monitor_id == PERMISSION_AWARE_NORM_ID:
        return MultiMonitor(
            members,
            derived={
                prefix: lambda c, prefix=prefix: sum(v for k, v in c.items() if k.startswith(prefix + "/"))
                for prefix in ("Collected", "Permitted", "Unpermitted")
            },
        )
    total = "Drain" if drainage else "CollectPerm" if monitor_id == COLLECT_PERMISSION_NORM_ID else "NoCollect"
    return MultiMonitor(members, derived={total: lambda c: sum(c.values())})


__all__ = [
    "COLLECT_ONE_NORM_ID",
    "COLLECT_PERMISSION_NORM_ID",
    "DRAIN_NORM_ID",
    "GARDENER_NORM_IDS",
    "NO_COLLECT_NORM_ID",
    "PERMISSION_AWARE_NORM_ID",
    "RESCUE_NORM_ID",
    "RESCUE_PER_FROG_NORM_ID",
    "RescueMonitor",
    "gardener_monitor_registry",
    "make_gardener_monitor",
    "rescue_recipe",
]
