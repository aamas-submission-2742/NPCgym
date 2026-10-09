import random

import pytest

from npc_gym.envs.gardener.labels import FrogCollected, GardenerLabel, PuddleDrained
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.gardener_monitors import (
    COLLECT_PERMISSION_NORM_ID,
    DRAIN_NORM_ID,
    NO_COLLECT_NORM_ID,
    PERMISSION_AWARE_NORM_ID,
    RESCUE_PER_FROG_NORM_ID,
    RescueMonitor,
    make_gardener_monitor,
    rescue_recipe,
)


def step(*labels, **flags):
    return MonitorInput(frozenset(labels), **flags)


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_rescue_deadline_and_episode_boundary(ending):
    monitor = RescueMonitor()
    monitor.reset(step(PuddleDrained(0, (0, 1))))
    assert not monitor.update(step(**{ending: True}))  # reset labels ignored
    monitor.reset(step())
    monitor.update(step(PuddleDrained(0, (0, 1))))
    for _ in range(4):
        assert not monitor.update(step())
    assert not monitor.update(step(FrogCollected(0), FrogCollected(1)))  # fifth subsequent input
    assert not monitor.update(step(**{ending: True}))
    monitor.reset(step())
    monitor.update(step(PuddleDrained(0, (0, 1)), PuddleDrained(1, (0,))))
    for _ in range(5):
        assert not monitor.update(step())
    assert monitor.update(step(FrogCollected(0), FrogCollected(1), **{ending: True}))
    assert monitor.count == 1  # three failed obligations, one failure occurrence


def test_rescue_staggered_obligations_and_final_activation():
    monitor = RescueMonitor()
    monitor.reset(step())
    monitor.update(step(PuddleDrained(0, (0,))))
    monitor.update(step(PuddleDrained(1, (0,))))
    for _ in range(4):
        monitor.update(step())
    assert monitor.update(step(FrogCollected(0)))  # first expired, second fulfilled
    assert not monitor.update(step())
    assert monitor.update(step(PuddleDrained(0, (1,)), truncated=True))
    assert monitor.count == 2
    monitor.reset(step())
    assert monitor.count == 0
    assert not monitor.update(step(PuddleDrained(0, (0,)), FrogCollected(0), terminated=True))


@pytest.mark.parametrize(
    "identifier,kwargs,total",
    [
        (NO_COLLECT_NORM_ID, {"num_frogs": 2}, "NoCollect"),
        (COLLECT_PERMISSION_NORM_ID, {"num_frogs": 2}, "CollectPerm"),
        (DRAIN_NORM_ID, {"num_puddles": 2}, "Drain"),
    ],
)
def test_object_counts_preserve_multiplicity(identifier, kwargs, total):
    monitor = make_gardener_monitor(identifier, **kwargs)
    monitor.reset(step())
    monitor.update(
        step(
            FrogCollected(0),
            FrogCollected(1),
            GardenerLabel.PERMITTED_COLLECT,
            PuddleDrained(0, (0,)),
            PuddleDrained(1, (0, 1)),
        )
    )
    assert monitor.counts[total] == 2
    with pytest.raises(ValueError, match="requires"):
        make_gardener_monitor(identifier)


def test_permission_composition_counts_raw_and_exempted_collections():
    monitor = make_gardener_monitor(PERMISSION_AWARE_NORM_ID, num_frogs=2)
    monitor.reset(step())
    monitor.update(step(FrogCollected(0), FrogCollected(1), GardenerLabel.PERMITTED_COLLECT))
    monitor.update(step(FrogCollected(0)))
    assert monitor.counts["Collected"] == 3
    assert monitor.counts["Permitted"] == 2
    assert monitor.counts["Unpermitted"] == 1


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
@pytest.mark.parametrize("minimize", [False, True])
def test_per_frog_rescue_counts_and_boundaries(ending, minimize):
    monitor = make_gardener_monitor(RESCUE_PER_FROG_NORM_ID, num_frogs=2, minimize=minimize)
    monitor.reset(step(PuddleDrained(0, (0, 1))))
    monitor.update(step(**{ending: True}))
    assert monitor.counts == {"Rescue/0": 0, "Rescue/1": 0, "Rescue": 0}
    for delay in (0, 4, 5, 6):
        monitor.reset(step())
        monitor.update(step(PuddleDrained(0, (0, 1)), PuddleDrained(1, (0,))))
        for _ in range(delay):
            monitor.update(step())
        monitor.update(step(FrogCollected(0), **{ending: True}))
        # Collection through the fifth subsequent step fulfills frog 0;
        # frog 1 always fails. Duplicate obligations for frog 0 count once.
        assert monitor.counts == {"Rescue/0": int(delay >= 5), "Rescue/1": 1, "Rescue": 1 + int(delay >= 5)}
    monitor.reset(step())
    monitor.update(step(PuddleDrained(0, (0, 1)), FrogCollected(0), FrogCollected(1), truncated=True))
    assert monitor.counts["Rescue"] == 0


def test_per_frog_rescue_staggered_expiry_and_reset():
    monitor = make_gardener_monitor(RESCUE_PER_FROG_NORM_ID, num_frogs=2)
    monitor.reset(step())
    monitor.update(step(PuddleDrained(0, (0, 1))))
    monitor.update(step(PuddleDrained(1, (0,))))
    for _ in range(4):
        monitor.update(step())
    monitor.update(step())
    assert monitor.counts["Rescue"] == 2
    monitor.update(step())
    assert monitor.counts == {"Rescue/0": 2, "Rescue/1": 1, "Rescue": 3}
    monitor.reset(step())
    monitor.update(step(truncated=True))
    assert monitor.counts["Rescue"] == 0


@pytest.mark.parametrize("minimize", [False, True])
def test_rescue_automata_match_independent_obligation_tracking(minimize):
    # The existing imperative monitor, restricted to one frog at a time, is an
    # independent oracle for overlapping deadlines and terminal handling.
    rng = random.Random(37)
    monitor = make_gardener_monitor(RESCUE_PER_FROG_NORM_ID, num_frogs=2, minimize=minimize)
    references = [RescueMonitor(), RescueMonitor()]
    for _ in range(80):
        monitor.reset(step())
        for reference in references:
            reference.reset(step())
        for index in range(30):
            labels = {FrogCollected(f) for f in range(2) if rng.random() < 0.12}
            labels.update(
                PuddleDrained(p, tuple(f for f in range(2) if rng.random() < 0.7))
                for p in range(2)
                if rng.random() < 0.35
            )
            end = index == 29
            monitor.update(step(*labels, truncated=end))
            for frog, reference in enumerate(references):
                filtered = {label for label in labels if label == FrogCollected(frog)}
                filtered.update(
                    PuddleDrained(label.puddle_id, (frog,))
                    for label in labels
                    if isinstance(label, PuddleDrained) and frog in label.nearby_frog_ids
                )
                reference.update(step(*filtered, truncated=end))
                assert monitor.counts[f"Rescue/{frog}"] == reference.count
            assert monitor.counts["Rescue"] == sum(r.count for r in references)


@pytest.mark.parametrize("size", [None, -1, True, 1.5])
def test_per_frog_rescue_requires_object_count(size):
    with pytest.raises(ValueError, match="num_frogs"):
        make_gardener_monitor(RESCUE_PER_FROG_NORM_ID, num_frogs=size)


def test_empty_rescue_and_minimization_option():
    empty = make_gardener_monitor(RESCUE_PER_FROG_NORM_ID, num_frogs=0)
    empty.reset(step())
    empty.update(step(truncated=True))
    assert empty.counts == {"Rescue": 0}
    with pytest.raises(TypeError, match="minimize"):
        make_gardener_monitor(RESCUE_PER_FROG_NORM_ID, num_frogs=2, minimize=1)
    for value in (-1, True, 0.5):
        with pytest.raises(ValueError, match="frog_id"):
            rescue_recipe(value)
