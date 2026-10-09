"""Hand-specified traces shared by reference and compiled-monitor regression tests.

Expected occurrence identities and counts are literal research behavior, recorded
before the regex migration, except Rescue v1's explicit step-count semantics and
Environment Friendly's overlapping extraction attempts and DeliveryPacifist's
composition of the existing Delivery, Danger and CTD events.
Multiple identities on one input preserve multiplicity.
The first valuation is reset; the last receives the requested episode-end flag.
"""

from npc_gym.envs.gardener.labels import FrogCollected, PuddleDrained
from npc_gym.envs.merchant.labels import MerchantLabel as M
from npc_gym.envs.pacman.labels import PacmanLabel as P
from npc_gym.envs.taxi.labels import TaxiWeatherLabel as T
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.gardener_monitors import gardener_monitor_registry
from npc_gym.monitors.merchant_monitors import merchant_monitor_registry
from npc_gym.monitors.pacman_monitors import pacman_monitor_registry
from npc_gym.monitors.taxi_monitors import taxi_monitor_registry

REGISTRIES = (gardener_monitor_registry, merchant_monitor_registry, pacman_monitor_registry, taxi_monitor_registry)
FACTORIES = {identifier: registry.make for registry in REGISTRIES for identifier in registry.ids}

# The obligation expires before two consecutive simultaneous ghost-eating steps.
PACMAN_TRACE = (
    (P.SCORE_0, P.ADJACENT_BLUE_GHOST),
    (),
    (),
    (P.SCORE_GREATER_100, P.SCORE_GREATER_400),
    (P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST, P.EAT_POWER_PELLET),
    (P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST, P.EAT_POWER_PELLET),
)
GHOSTS = (P.EAT_BLUE_GHOST, P.EAT_ORANGE_GHOST)
BLUE = "pacman/vegetarian-blue-v0"
ORANGE = "pacman/vegetarian-orange-v0"
HUNGRY = "pacman/hungry-v0"
GUILT = "pacman/hungry-vegan-penalty-v1"
SOLUTION_KINDS = ("pacman/hungry-vegetarian-v0", GUILT, "pacman/maximum-v2")

# Values: input valuations, ordered emitted kinds at each input, final measurements.
CASES = {
    "pacman/vegan-v0": (PACMAN_TRACE, ((), (), (), (), GHOSTS, GHOSTS), {"Vegan": 4}),
    BLUE: (PACMAN_TRACE, ((), (), (), (), (None,), (None,)), {"VegetarianBlue": 2}),
    ORANGE: (PACMAN_TRACE, ((), (), (), (), (None,), (None,)), {"VegetarianOrange": 2}),
    "pacman/conditional-vegan-v0": (PACMAN_TRACE, ((), (), (), (), GHOSTS, GHOSTS), {"ConditionalVegan": 4}),
    "pacman/obligation-blue-v1": (PACMAN_TRACE, ((), (), (), (None,), (), ()), {"OblBlue": 1}),
    "pacman/vegan-preference-v0": (
        PACMAN_TRACE,
        ((), (), (), (), (BLUE, ORANGE), (BLUE, ORANGE)),
        {"VegetarianBlue": 2, "VegetarianOrange": 2},
    ),
    "pacman/vegan-conflict-v1": (
        PACMAN_TRACE,
        ((), (), (), ("pacman/obligation-blue-v1",), (BLUE, ORANGE), (BLUE, ORANGE)),
        {"VegetarianBlue": 2, "VegetarianOrange": 2, "OblBlue": 1},
    ),
    "pacman/cautious-v0": (PACMAN_TRACE, ((), (), (), (), (None,), (None,)), {"Cautious": 2}),
    "pacman/all-or-nothing-v0": (PACMAN_TRACE, ((), (), (), (), GHOSTS, ()), {"AllOrNothing": 2}),
    "pacman/one-taste-v1": (PACMAN_TRACE, ((), (), (), (), (), GHOSTS), {"OneTaste": 2}),
    "pacman/switch-v0": (
        PACMAN_TRACE,
        ((), (), (), (), (P.EAT_BLUE_GHOST,), (P.EAT_BLUE_GHOST,)),
        {"Switch": 2},
    ),
    "pacman/penalty-v0": (
        PACMAN_TRACE,
        ((), (), (), (), GHOSTS, ("contrary-to-duty", *GHOSTS)),
        {"Penalty(total)": 5, "CTD": 1},
    ),
    "pacman/penalty-1-v0": (
        PACMAN_TRACE,
        ((), (), (), (), GHOSTS, ("contrary-to-duty", *GHOSTS)),
        {"Penalty1(total)": 5, "CTD": 1},
    ),
    "pacman/penalty-3-v0": (
        PACMAN_TRACE,
        ((), (), (), (), GHOSTS, ("contrary-to-duty", *GHOSTS)),
        {"Penalty3(total)": 5, "CTD": 1},
    ),
    "pacman/trapped-v1": (PACMAN_TRACE, ((None,), (None,), (None,), (), (), ()), {"Trapped": 3}),
    HUNGRY: (PACMAN_TRACE, ((), (), (), (None,), (), ()), {"Hungr": 1}),
    "pacman/hungry-vegan-v0": (
        PACMAN_TRACE,
        ((), (), (), (HUNGRY,), (BLUE, ORANGE), (BLUE, ORANGE)),
        {"VegetarianBlue": 2, "VegetarianOrange": 2, "Hungr": 1},
    ),
    "pacman/hungry-vegetarian-v0": (
        PACMAN_TRACE,
        ((), (), (), (HUNGRY,), (ORANGE,), (ORANGE,)),
        {"VegetarianOrange": 2, "Hungr": 1},
    ),
    GUILT: (
        PACMAN_TRACE,
        ((), (), (), (HUNGRY,), (BLUE, ORANGE), (BLUE, ORANGE, "contrary-to-duty")),
        {"VegetarianBlue": 2, "VegetarianOrange": 2, "Hungr": 1, "CTD": 1},
    ),
    "pacman/maximum-v2": (
        PACMAN_TRACE,
        (("pacman/trapped-v1",), ("pacman/trapped-v1",), ("pacman/trapped-v1",), (GUILT,), (GUILT,) * 2, (GUILT,) * 3),
        {"VegetarianBlue": 2, "VegetarianOrange": 2, "Hungr": 1, "CTD": 1, "Trapped": 3},
    ),
    "pacman/errand-v0": (PACMAN_TRACE, ((), (), (), (None,), (), ()), {"Errand": 1}),
    "pacman/visit-v0": (PACMAN_TRACE, ((), (), (), (None,), (), ()), {"Visit": 1}),
    "pacman/solution-guilt-maximum-v2": (
        PACMAN_TRACE,
        (
            ("pacman/maximum-v2",),
            ("pacman/maximum-v2",),
            ("pacman/maximum-v2",),
            SOLUTION_KINDS,
            SOLUTION_KINDS,
            SOLUTION_KINDS,
        ),
        {"SolutionGuiltMaximum": 12},
    ),
    "merchant/danger-v0": (
        ((M.AT_DANGER,), (M.AT_DANGER,), ()),
        ((None,), (None,), ()),
        {"Danger": 2},
    ),
    "merchant/pacifist-v0": (
        ((M.AT_DANGER, M.FIGHT), (M.AT_DANGER, M.FIGHT), (M.FIGHT,)),
        (("danger",), ("contrary-to-duty", "danger"), ("contrary-to-duty",)),
        {"Pacifist(total)": 4, "Danger": 2, "CTD": 2},
    ),
    "merchant/delivery-pacifist-v0": (
        (
            (M.AT_HOME,),
            (M.AT_DANGER,),
            (M.AT_DANGER, M.FIGHT, M.SUNDOWN),
            (M.UNLOAD,),
            (M.AT_HOME,),
            (M.AT_MARKET,),
            (M.SUNDOWN,),
        ),
        ((), ("Danger",), ("Delivery", "Danger", "CTD"), (), (), (), ()),
        {"DeliveryPacifist": 4, "Delivery": 1, "Danger": 2, "CTD": 1},
    ),
    "merchant/delivery-v0": (
        ((M.AT_HOME,), (M.AT_MARKET, M.SUNDOWN), (M.SUNDOWN,), (M.AT_HOME, M.SUNDOWN)),
        ((), (None,), (), (None,)),
        {"Delivery": 2},
    ),
    "merchant/env-friendly-v0": (
        ((M.AT_TREE, M.HAS_WOOD), (M.EXTRACT, M.AT_TREE, M.HAS_WOOD), (M.EXTRACT,)),
        ((), (None,), (None,)),
        {"EnvFriendly": 2},
    ),
    "merchant/evolving-v0": (
        ((M.AT_TREE,), (M.EXTRACT, M.AT_TREE), (M.EXTRACT,)),
        ((), (None,), ()),
        {"Evolving": 1},
    ),
    "gardener/collect-one-v0": (((FrogCollected(7),), (), ()), ((), (), (None,)), {"CollectOne": 1}),
    "gardener/rescue-v1": (
        ((), (PuddleDrained(0, (0, 1)),), (), (), (), (), (), (), (PuddleDrained(1, (2,)),)),
        ((), (), (), (), (), (), (), (None,), (None,)),
        {"Rescue": 2},
    ),
    "taxi/emergency-v0": (
        ((T.RAIN, T.HURRICANE), (T.RAIN, T.HURRICANE), (T.HURRICANE,)),
        ((), ("stay",), ("warn", "stay")),
        {"Emergency Violations": 3, "Safety Violations": 0, "Stay Violations": 2, "Warn Violations": 1},
    ),
}


def counts(identifier, monitor):
    from npc_gym.monitors import ComplexMonitor

    if isinstance(monitor, ComplexMonitor):
        return {next(iter(CASES[identifier][2])): monitor.count}
    return dict(monitor.counts)


def assert_characterization(identifier, factory, ending):
    valuations, _occurrences, expected = CASES[identifier]
    first, second = factory(identifier), factory(identifier)
    for _ in range(2):
        for monitor in (first, second):
            for index, labels in enumerate(valuations):
                input = MonitorInput(frozenset(labels), **{ending: index == len(valuations) - 1})
                (monitor.reset if index == 0 else monitor.update)(input)
            actual = counts(identifier, monitor)
            assert {key: actual[key] for key in expected} == expected
        first.reset(MonitorInput(frozenset()))
        assert all(value == 0 for value in counts(identifier, first).values())
        assert {key: counts(identifier, second)[key] for key in expected} == expected
