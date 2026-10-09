"""Counts characterized before removal of the imperative implementations.

Episode counts cover ten deterministic 100-input traces per recipe, retaining
final snapshots from 80caf51's fingerprints. Those traces were parity-checked
against the pre-refactor runtime in 818d4ec. Before replacing the fingerprints,
all 1000 per-input snapshots per recipe were checked against those fingerprints
at 39e0017. episode_counts.json retains the final counts of each episode;
hand-written and boundary tests retain per-input assertions. CTD Violations was
renamed CTD without changing values.
Environment Friendly counts were updated for overlapping extraction attempts,
using the adjacent-input predicate independently of the compiled monitor.
Taxi warning counts were updated by checking for a missing warning immediately
after each rain onset, independently of the compiled monitor; other components
retain their characterized values. Trapped traces use the 400-point release label;
separate boundary tests cover its changed threshold.
DeliveryPacifist's expected episode counts sum the recorded Delivery and
Pacifist components from the same traces.
"""

import json
import random
from pathlib import Path

import pytest

from npc_gym.envs.gardener.labels import FrogCollected, GardenerLabel
from npc_gym.envs.merchant.labels import MerchantLabel
from npc_gym.envs.pacman.labels import PacmanLabel
from npc_gym.envs.taxi.labels import TaxiActionLabel, TaxiLocationLabel, TaxiWeatherLabel
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.builtins import BUILTIN_IDS, make_builtin_monitor
from npc_gym.monitors.events import MultiMonitor
from tests.monitor_characterization import CASES


def counts(identifier, monitor):
    if isinstance(monitor, MultiMonitor):
        return monitor.counts
    return {next(iter(CASES[identifier][2])): monitor.count}


@pytest.mark.parametrize("identifier", BUILTIN_IDS)
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_hand_characterized_counts(identifier, ending):
    monitor = make_builtin_monitor(identifier)
    valuations, _, expected = CASES[identifier]
    for _ in range(2):
        for index, labels in enumerate(valuations):
            input = MonitorInput(frozenset(labels), **{ending: index == len(valuations) - 1})
            (monitor.reset if index == 0 else monitor.update)(input)
        actual = counts(identifier, monitor)
        assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("identifier", BUILTIN_IDS)
def test_deterministic_trace_parity(identifier):
    vocabulary = {
        "merchant": tuple(MerchantLabel),
        "pacman": tuple(label for label in PacmanLabel if label != PacmanLabel.SCORE_GREATER_400),
        "taxi": (*TaxiActionLabel, *TaxiLocationLabel, *TaxiWeatherLabel),
        "gardener": (*GardenerLabel, FrogCollected(0)),
    }[identifier.split("/")[0]]
    new = make_builtin_monitor(identifier)
    expected = json.loads(Path(__file__).with_name("episode_counts.json").read_text())[identifier]
    expected_keys = CASES[identifier][2]
    rng = random.Random(6821)
    for episode in range(10):
        for index in range(100):
            input = MonitorInput(
                frozenset(label for label in vocabulary if rng.random() < (0.1 if episode % 2 else 0.6)),
                truncated=index == 99,
            )
            # Preserve the recorded traces while expressing Trapped's release at 400.
            # Dedicated boundary tests distinguish scores 100, 400 and 401.
            if (
                identifier in {"pacman/trapped-v1", "pacman/maximum-v2", "pacman/solution-guilt-maximum-v2"}
                and PacmanLabel.SCORE_GREATER_100 in input.labels
            ):
                input = MonitorInput(input.labels | {PacmanLabel.SCORE_GREATER_400}, truncated=input.truncated)
            (new.reset if index == 0 else new.update)(input)
        actual = counts(identifier, new)
        assert {key: actual[key] for key in expected_keys} == expected[episode], (
            f"{identifier}, episode {episode}, seed 6821, final input 99"
        )


@pytest.mark.parametrize("identifier", BUILTIN_IDS)
def test_explicit_totals_preserve_each_characterized_input(identifier):
    monitor = make_builtin_monitor(identifier)
    valuations, occurrences, _ = CASES[identifier]
    total_names = {
        "VeganPreference",
        "VeganConflict",
        "HungryVegan",
        "HungryVegetarian",
        "HungryVeganPenalty",
        "Maximum",
        "Vegan",
        "ConditionalVegan",
        "AllOrNothing",
        "OneTaste",
        "Penalty(total)",
        "Penalty1(total)",
        "Penalty3(total)",
        "Pacifist(total)",
        "DeliveryPacifist",
        "Emergency Violations",
        "SolutionGuiltMaximum",
    }
    for _ in range(2):
        expected = 0
        for index, labels in enumerate(valuations):
            input = MonitorInput(frozenset(labels), truncated=index == len(valuations) - 1)
            (monitor.reset if index == 0 else monitor.update)(input)
            expected += len(occurrences[index])
            if isinstance(monitor, MultiMonitor):
                keys = (set(monitor.counts) - monitor.members.keys()) & total_names
                assert len(keys) == 1
                assert monitor.counts[next(iter(keys))] == expected
            else:
                assert monitor.count == expected


@pytest.mark.parametrize(
    "identifier,members,total",
    [
        (
            "taxi/emergency-v0",
            ("Warn Violations", "Stay Violations", "Seven-Step Safety Violations", "Three-Step Safety Violations"),
            "Emergency Violations",
        ),
        (
            "pacman/conditional-vegan-v0",
            ("ConditionalVegetarianBlue", "ConditionalVegetarianOrange"),
            "ConditionalVegan",
        ),
        ("pacman/all-or-nothing-v0", ("FirstTasteBlue", "FirstTasteOrange"), "AllOrNothing"),
        ("pacman/one-taste-v1", ("LaterTasteBlue", "LaterTasteOrange"), "OneTaste"),
        ("pacman/vegan-preference-v0", ("VegetarianBlue", "VegetarianOrange"), "VeganPreference"),
        ("pacman/vegan-conflict-v1", ("VegetarianBlue", "VegetarianOrange", "OblBlue"), "VeganConflict"),
        ("pacman/hungry-vegan-v0", ("VegetarianBlue", "VegetarianOrange", "Hungr"), "HungryVegan"),
        ("pacman/hungry-vegetarian-v0", ("VegetarianOrange", "Hungr"), "HungryVegetarian"),
        (
            "pacman/hungry-vegan-penalty-v1",
            ("VegetarianBlue", "VegetarianOrange", "Hungr", "CTD"),
            "HungryVeganPenalty",
        ),
        ("pacman/maximum-v2", ("VegetarianBlue", "VegetarianOrange", "Hungr", "CTD", "Trapped"), "Maximum"),
        (
            "pacman/solution-guilt-maximum-v2",
            ("HungryVegetarian", "HungryVeganPenalty", "Maximum"),
            "SolutionGuiltMaximum",
        ),
    ],
)
def test_research_facing_collection_names_and_totals(identifier, members, total):
    monitor = make_builtin_monitor(identifier)
    assert tuple(monitor.members) == members
    for index, labels in enumerate(CASES[identifier][0]):
        input = MonitorInput(frozenset(labels))
        (monitor.reset if index == 0 else monitor.update)(input)
        assert monitor.counts[total] == sum(monitor.counts[name] for name in members)
    assert "total" not in monitor.counts


@pytest.mark.parametrize("identifier", BUILTIN_IDS)
def test_domain_convenience_factories_are_fresh_and_reject_foreign_ids(identifier):
    from importlib import import_module

    from npc_gym.monitors import UnknownMonitorIDError

    domain = identifier.split("/")[0]
    module = import_module(f"npc_gym.monitors.{domain}_monitors")
    factory_name = f"make_{domain}_monitor"
    assert factory_name in module.__all__
    factory = getattr(module, factory_name)
    first, second = factory(identifier), factory(identifier)
    assert first is not second
    for index, labels in enumerate(CASES[identifier][0]):
        input = MonitorInput(frozenset(labels), truncated=index == len(CASES[identifier][0]) - 1)
        (first.reset if index == 0 else first.update)(input)
    actual = counts(identifier, first)
    assert {key: actual[key] for key in CASES[identifier][2]} == CASES[identifier][2]
    assert all(value == 0 for value in counts(identifier, second).values())
    foreign = "merchant/danger-v0" if domain != "merchant" else "pacman/vegan-v0"
    with pytest.raises(UnknownMonitorIDError):
        factory(foreign)


def test_merchant_late_compilation_size_stays_within_explicit_budget():
    from npc_gym.monitors.builtins import RECIPES
    from npc_gym.monitors.regex import RegexLimits

    # Input 14 expansion is near the default DFA limit. Review budgets if this grows.
    definition = RECIPES["merchant"]["late"].compile(minimize=False)
    assert len(definition.states) == 237
    assert len(RECIPES["merchant"]["late"].compile().states) == 30
    assert len(definition.states) <= RegexLimits().max_dfa_states
    assert sum(map(len, definition.transitions.values())) <= RegexLimits().max_transitions
