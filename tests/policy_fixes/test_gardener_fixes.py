"""Gardener planning checked against live dynamics and independent monitors."""

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from npc_gym.envs import GardenerEnv
from npc_gym.envs.gardener.gardener import GardenerObservation, GardenerObservationCodec
from npc_gym.envs.gardener.labels import GardenerLabel, GardenerLabelingFunction, PuddleDrained
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.gardener_monitors import (
    COLLECT_ONE_NORM_ID,
    COLLECT_PERMISSION_NORM_ID,
    DRAIN_NORM_ID,
    NO_COLLECT_NORM_ID,
    PERMISSION_AWARE_NORM_ID,
    RESCUE_NORM_ID,
    make_gardener_monitor,
)
from npc_gym.policy_fixes import ASPPlanner, GardenerModel, Objective, PlanningProblem

pytest.importorskip("clingo")

VALUES = dict.fromkeys(range(5), 0.0)
NORMS = (NO_COLLECT_NORM_ID, DRAIN_NORM_ID, PERMISSION_AWARE_NORM_ID)


def snapshot(**changes):
    state = GardenerObservation(
        size=5,
        agent=(1, 1),
        grass=((4, 4),),
        grass_active=(True,),
        grass_timer=(0,),
        puddles=((3, 3),),
        puddles_full=(True,),
        puddle_timer=(1,),
        frogs=((2, 1),),
        collected_frogs=(False,),
        captured_frogs=(False,),
        frog_timer=(5,),
        walls=(),
        frog_collected=(False,),
        puddle_drained=(False,),
    )
    return replace(state, **changes)


def predict(model, state, actions, *, remaining_steps=None, planner=None):
    problem = model.problem(state, remaining_steps=remaining_steps)
    constraints = "\n".join(f":- not action({t},{a})." for t, a in enumerate(actions))
    fixed = PlanningProblem(problem.program + "\n" + constraints, problem.horizon)
    return (planner or ASPPlanner(objectives=model.objectives)).solve(fixed, VALUES)


def engine(state, **params):
    """Install a small test layout; production models use public snapshots only."""
    env = GardenerEnv(size=state.size, **params)
    env.reset(seed=11)
    env.num_grass, env.num_puddles, env.num_frogs, env.num_walls = map(
        len, (state.grass, state.puddles, state.frogs, state.walls)
    )
    env.observation_codec = GardenerObservationCodec(
        state.size, env.num_grass, env.num_puddles, env.num_frogs, env.num_walls
    )
    env.labeling_function = GardenerLabelingFunction(
        num_grass=env.num_grass, num_puddles=env.num_puddles, num_frogs=env.num_frogs, num_walls=env.num_walls
    )
    for field in state.__dataclass_fields__:
        if field != "size":
            setattr(env, field, np.array(getattr(state, field)))
    env._walls_set = set(state.walls)
    env._puddles_set = set(state.puddles)
    from npc_gym.envs.gardener.gardener import target_routes

    env.puddle_dict = target_routes(state.size, state.puddles, env._walls_set | env._puddles_set)
    env._compute_pos_actions()
    env._last_state = env.get_state()
    return env


def successors(env, action):
    """All positive-probability frog outcomes; engine executes the rest of each turn."""
    options = []
    for i, (x, y) in enumerate(env.frogs):
        choices = [(int(x), int(y))]
        if not env.collected_frogs[i] and env.frog_timer[i] == 0:
            choices = [
                (int(x + dx), int(y + dy))
                for a, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1)))
                if env.pos_actions[x, y, a]
            ] or choices
        options.append(choices)
    for positions in product(*options):
        child = deepcopy(env)

        def move(positions=positions, child=child):
            child.frogs = np.array(positions, dtype=int).reshape((-1, 2))

        child._move_frogs = move
        result = child.step(action)
        yield child, result


def monitor_total(monitor, norm):
    counts = monitor.counts if hasattr(monitor, "counts") else {"count": monitor.count}
    key = {
        NO_COLLECT_NORM_ID: "NoCollect",
        COLLECT_PERMISSION_NORM_ID: "CollectPerm",
        PERMISSION_AWARE_NORM_ID: "Unpermitted",
        DRAIN_NORM_ID: "Drain",
    }.get(norm, "count")
    return counts[key]


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("action", range(5))
def test_simultaneous_collections_and_permission_match_monitor(norm, action):
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))[action]
    state = snapshot(
        frogs=((1 + dx, 1 + dy),) * 2,
        frog_timer=(5, 5),
        collected_frogs=(False, False),
        captured_frogs=(False, False),
        frog_collected=(False, False),
    )
    model = GardenerModel(state, norm_id=norm, horizon=1)
    env = engine(state)
    monitor = make_gardener_monitor(norm, num_frogs=2, num_puddles=1)
    monitor.reset(MonitorInput(frozenset()))
    _, _, term, _, info = env.step(action)
    monitor.update(MonitorInput(info["labels"], terminated=term, truncated=True))
    result = predict(model, state, [action], remaining_steps=1)
    assert result.costs["violations"] == monitor_total(monitor, norm)
    assert sum(env.frog_collected) == 2
    env.close()


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("timer", [0, 1, 2])
def test_two_turn_cost_bounds_every_engine_future(norm, timer):
    state = snapshot(agent=(2, 1), frogs=((2, 2),), frog_timer=(timer,), puddles=((3, 2),))
    for actions in ((4, 4), (1, 0), (2, 1), (3, 0)):
        model = GardenerModel(state, norm_id=norm, horizon=2)
        predicted = predict(model, state, actions, remaining_steps=2).costs["violations"]
        monitor = make_gardener_monitor(norm, num_frogs=1, num_puddles=1)
        monitor.reset(MonitorInput(frozenset()))
        branches = [(engine(state), [])]
        final_counts = []
        for t, action in enumerate(actions):
            following = []
            for parent, history in branches:
                for child, (_, _, term, trunc, info) in successors(parent, action):
                    current = make_gardener_monitor(norm, num_frogs=1, num_puddles=1)
                    current.reset(MonitorInput(frozenset()))
                    trace = [*history, MonitorInput(info["labels"], terminated=term, truncated=trunc or t == 1)]
                    for input in trace:
                        current.update(input)
                    assert monitor_total(current, norm) <= predicted
                    if t == len(actions) - 1:
                        final_counts.append(monitor_total(current, norm))
                    if not term:
                        following.append((child, trace))
                parent.close()
            branches = following
        for env, _ in branches:
            env.close()
        if norm != DRAIN_NORM_ID:
            assert predicted == max(final_counts)


@pytest.mark.parametrize("norm,expected", [(NO_COLLECT_NORM_ID, 1), (PERMISSION_AWARE_NORM_ID, 1)])
def test_blocked_right_is_stay_and_has_no_permission(norm, expected):
    state = snapshot(agent=(0, 1), frogs=((0, 1),), walls=((1, 1),))
    assert predict(GardenerModel(state, norm_id=norm, horizon=1), state, [0]).costs["violations"] == expected
    env = engine(state)
    *_, info = env.step(0)
    assert GardenerLabel.STAY in info["labels"]
    assert GardenerLabel.PERMITTED_COLLECT not in info["labels"]
    env.close()


def test_refill_drain_freeze_and_diagonal_labels():
    state = snapshot(
        agent=(1, 2), puddles=((2, 2),), puddles_full=(False,), puddle_timer=(1,), frogs=((3, 3),), frog_timer=(1,)
    )
    model = GardenerModel(state, norm_id=DRAIN_NORM_ID, horizon=2, puddle_respawn=1)
    assert predict(model, state, [4, 4]).costs["violations"] == 2
    env = engine(state, puddle_respawn=1)
    for _ in range(2):
        *_, info = env.step(4)
        assert info["labels"] >= {PuddleDrained(0, (0,))}
        assert env.frog_timer[0] == 5
        assert env.puddle_timer[0] == 1
    env.close()


@pytest.mark.parametrize("position,expected", [((3, 3), 1), ((2, 4), 0), ((1, 2), 0)])
def test_drain_uses_post_collection_eight_neighborhood(position, expected):
    state = snapshot(agent=(1, 2), puddles=((2, 2),), frogs=(position,))
    assert predict(GardenerModel(state, norm_id=DRAIN_NORM_ID, horizon=1), state, [4]).costs["violations"] == expected


def test_default_objectives_and_override_change_the_fix():
    state = snapshot()
    model = GardenerModel(state, horizon=1)
    values = {0: 10.0, 4: 9.0, 1: 8.0, 2: 7.0, 3: 6.0}
    assert model.objectives == {"violations": Objective(5, 1), "policy": Objective(1, 1)}
    assert ASPPlanner(objectives=model.objectives).solve(model.problem(state), values).action == 4
    planner = ASPPlanner(objectives={"violations": Objective(0, 1), "policy": Objective(1, 1)})
    assert planner.solve(model.problem(state), values).action == 0


def test_radius_clips_outside_frogs():
    state = snapshot(frog_timer=(0,))
    model = GardenerModel(state, horizon=1, radius=0)
    decision = ASPPlanner(objectives=model.objectives).solve(model.problem(state), VALUES)
    assert decision.action == 4
    assert decision.costs["violations"] == 0
    boundary = GardenerModel(snapshot(), horizon=1, radius=1)
    assert predict(boundary, snapshot(), [0]).costs["violations"] == 1


def test_frogs_move_before_collection_and_freeze_one_delays_movement():
    moving = snapshot(frog_timer=(0,))
    frozen = snapshot(frog_timer=(1,))
    assert predict(GardenerModel(moving, horizon=1), moving, [0]).costs["violations"] == 0
    assert predict(GardenerModel(frozen, horizon=1), frozen, [0]).costs["violations"] == 1


def test_two_live_models_preserve_seeded_trajectories():
    first, second = GardenerEnv(size=5), GardenerEnv(size=5)
    first.reset(seed=43)
    second.reset(seed=43)
    model = GardenerModel(first.labeling_state(), norm_id=NO_COLLECT_NORM_ID, horizon=2)
    other = GardenerModel(second.labeling_state(), norm_id=NO_COLLECT_NORM_ID, horizon=2)
    planner = ASPPlanner(objectives=model.objectives)
    for t in range(8):
        before = first.labeling_state()
        problem = model.problem(before)
        assert planner.solve(problem, VALUES) == planner.solve(problem, VALUES)
        assert before == first.labeling_state()
        left, right = first.step(t % 5), second.step(t % 5)
        assert left[:4] == right[:4]
        assert left[4]["labels"] == right[4]["labels"]
        assert model.problem(first.labeling_state()) == other.problem(second.labeling_state())
    first.close()
    second.step(4)
    second.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"radius": -1},
        {"frog_freeze": -1},
        {"puddle_respawn": 0},
        {"radius": None},
        {"norm_id": RESCUE_NORM_ID},
        {"norm_id": COLLECT_ONE_NORM_ID},
        {"norm_id": COLLECT_PERMISSION_NORM_ID},
        {"norm_id": "unknown"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GardenerModel(snapshot(), **kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"agent": (-1, 0)},
        {"frog_timer": (6,)},
        {"frog_timer": ()},
        {"collected_frogs": (1,)},
        {"walls": ((1, 1),)},
        {"grass": ((3, 3),)},
        {"captured_frogs": (True,)},
        {"frog_collected": (True,)},
    ],
)
def test_invalid_snapshots(changes):
    with pytest.raises((TypeError, ValueError)):
        GardenerModel(snapshot(**changes))


def test_snapshot_lifecycle_and_invalid_inputs():
    state = snapshot()
    model = GardenerModel(state)
    with pytest.raises(TypeError):
        GardenerModel(())
    with pytest.raises(ValueError, match="layout"):
        model.problem(replace(state, grass=((0, 0),)))
    for budget in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            model.problem(state, remaining_steps=budget)
    collected = replace(state, collected_frogs=(True,), captured_frogs=(True,), frog_collected=(True,))
    assert predict(model, collected, [0]).costs["violations"] == 0
    assert predict(GardenerModel(collected), collected, [0]).costs["violations"] == 0
    # Reuse after reset to the original layout, without retaining old collections.
    assert predict(model, state, [0]).costs["violations"] == 1


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("timer", [0, 1])
def test_actual_rng_support_matches_enumerated_cardinal_moves(full, timer):
    class Choices:
        def __init__(self, draw, index):
            self.draw, self.index = draw, index

        def random(self):
            return self.draw

        def choice(self, options):
            return options[self.index % len(options)]

    state = snapshot(frog_timer=(timer,), puddles_full=(full,))
    env = engine(state)
    expected = {tuple(child.frogs[0]) for child, _ in successors(env, 4)}
    seen = set()
    for draw, index in product((0.0, 0.99), range(4)):
        child = deepcopy(env)
        child.np_random = Choices(draw, index)
        child.step(4)
        seen.add(tuple(child.frogs[0]))
        child.close()
    assert seen == expected
    env.close()


def test_trapped_frog_stays_and_zero_freeze_does_not_delay_movement():
    trapped = snapshot(frogs=((4, 0),), frog_timer=(0,), walls=((3, 0), (4, 1)))
    env = engine(trapped, frog_freeze=0)
    env.step(4)
    assert tuple(env.frogs[0]) == (4, 0)
    assert predict(GardenerModel(trapped, horizon=2, frog_freeze=0), trapped, [4, 4]).costs["violations"] == 0
    env.close()
    state = snapshot(agent=(1, 2), puddles=((2, 2),), frogs=((3, 2),), frog_timer=(0,))
    assert predict(GardenerModel(state, horizon=2, frog_freeze=0), state, [4, 4]).optimal


@pytest.mark.parametrize("norm", [NO_COLLECT_NORM_ID, PERMISSION_AWARE_NORM_ID])
def test_alternative_collection_times_count_each_frog_only_once(norm):
    state = snapshot(frog_timer=(0,))
    model = GardenerModel(state, norm_id=norm, horizon=2)
    # One frog can meet the agent on either turn, never on both.
    assert predict(model, state, [4, 1]).costs["violations"] == 1
    assert predict(model, state, [4, 0]).costs["violations"] == 1


def test_collected_frog_does_not_move_or_cause_later_drainage():
    state = snapshot(agent=(1, 2), frogs=((1, 2),), puddles=((2, 2),))
    for norm, expected in ((NO_COLLECT_NORM_ID, 1), (DRAIN_NORM_ID, 0)):
        model = GardenerModel(state, norm_id=norm, horizon=3)
        assert predict(model, state, [4, 4, 4]).costs["violations"] == expected


def test_permitted_collection_cannot_become_a_later_unpermitted_collection():
    state = snapshot()
    model = GardenerModel(state, norm_id=PERMISSION_AWARE_NORM_ID, horizon=3)
    assert predict(model, state, [0, 4, 4]).costs["violations"] == 0


def test_window_is_square_and_drops_outside_puddles():
    state = snapshot(agent=(1, 1), frogs=((2, 2),), puddles=((3, 2),))
    model = GardenerModel(state, horizon=2, radius=1)
    assert predict(model, state, [0, 1]).costs["violations"] == 1
    model = GardenerModel(state, norm_id=DRAIN_NORM_ID, horizon=2, radius=1)
    assert predict(model, state, [0, 1]).costs["violations"] == 0


def test_leaving_window_does_not_create_an_artificial_stay():
    state = snapshot(agent=(1, 1), frogs=((1, 1),), frog_timer=(0,))
    assert predict(GardenerModel(state, radius=0), state, [4]).costs["violations"] == 0


def test_known_time_limit_shortens_plan():
    state = snapshot()
    model = GardenerModel(state, horizon=3)
    assert len(predict(model, state, [4], remaining_steps=1).plan) == 1


@pytest.mark.parametrize("timer", [0, 1, 5])
@pytest.mark.parametrize("water", [(True, 0), (False, 1), (False, 2)])
def test_compact_frog_features_match_planner_and_engine_events(timer, water):
    from npc_gym.envs.gardener.labels import FrogCollected
    from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper

    state = snapshot(
        agent=(2, 1),
        frogs=((2, 2),),
        frog_timer=(timer,),
        puddles=((3, 2),),
        puddles_full=(water[0],),
        puddle_timer=(water[1],),
    )
    env = engine(state)
    try:
        features = StateFeatureObsWrapper(env, include_frogs=True).observation(env.get_state())
        for action in range(5):
            counts, drainages = [], set()
            for child, (_, _, _, _, info) in successors(env, action):
                counts.append(sum(isinstance(label, FrogCollected) for label in info["labels"]))
                drainages.update(label.puddle_id for label in info["labels"] if isinstance(label, PuddleDrained))
                child.close()
            assert features[12 + action] == max(counts) / env.num_frogs
            assert features[17 + action] == len(drainages) / env.num_puddles
            for norm in NORMS:
                expected = (
                    features[17 + action] * env.num_puddles
                    if norm == DRAIN_NORM_ID
                    else features[12 + action] * env.num_frogs
                )
                if norm == PERMISSION_AWARE_NORM_ID and action == 0 and features[22]:
                    expected = 0
                assert predict(GardenerModel(state, norm_id=norm), state, [action]).costs[
                    "violations"
                ] == pytest.approx(expected)
    finally:
        env.close()


def test_compact_drainage_counts_union_of_mutually_exclusive_events():
    from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper

    state = snapshot(
        agent=(2, 2),
        frogs=((2, 1),),
        frog_timer=(0,),
        puddles=((1, 2), (3, 2)),
        puddles_full=(True, True),
        puddle_timer=(0, 0),
        puddle_drained=(False, False),
    )
    env = engine(state)
    try:
        features = StateFeatureObsWrapper(env, include_frogs=True).observation(env.get_state())
        actual = []
        for child, (_, _, _, _, info) in successors(env, 4):
            actual.append(sum(isinstance(label, PuddleDrained) for label in info["labels"]))
            child.close()
        assert max(actual) == 1
        assert features[21] * env.num_puddles == 2  # Either puddle can be harmed, never both together.
        assert predict(GardenerModel(state, norm_id=DRAIN_NORM_ID), state, [4]).costs["violations"] == 2
    finally:
        env.close()


def test_compact_features_count_distinct_frogs_and_blocked_right_is_not_permission():
    from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper

    state = snapshot(
        agent=(1, 1),
        frogs=((1, 1),) * 2,
        frog_timer=(5, 5),
        walls=((2, 1),),
        collected_frogs=(False, False),
        captured_frogs=(False, False),
        frog_collected=(False, False),
    )
    env = engine(state)
    try:
        features = StateFeatureObsWrapper(env, include_frogs=True).observation(env.get_state())
        assert features[12] == features[16] == 1
        assert features[22] == 0 and features[26] == 1
        assert predict(GardenerModel(state, norm_id=PERMISSION_AWARE_NORM_ID), state, [0]).costs["violations"] == 2
    finally:
        env.close()


def test_compact_features_include_trapped_frogs_staying_in_place():
    from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper

    state = snapshot(agent=(4, 0), frogs=((4, 0),), frog_timer=(0,), walls=((3, 0), (4, 1)))
    env = engine(state)
    try:
        features = StateFeatureObsWrapper(env, include_frogs=True).observation(env.get_state())
        np.testing.assert_array_equal(features[12:17], np.ones(5))
        np.testing.assert_array_equal(features[22:], [0, 0, 0, 0, 1])
        *_, info = env.step(4)
        assert info["labels"]
        assert env.collected_frogs[0]
    finally:
        env.close()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "action,frogs,walls,unpermitted,drain",
    [
        (0, ((2, 1), (3, 2)), (), 0, 1),
        (1, ((1, 2), (3, 2)), (), 1, 1),
        (0, ((2, 1), (4, 0)), (), 0, 0),
        (0, ((1, 1), (3, 2)), ((2, 1),), 1, 0),
    ],
)
def test_combined_cost_matches_permission_exception_and_drain(action, frogs, walls, unpermitted, drain, reuse):
    state = snapshot(
        puddles=((2, 2),),
        frogs=frogs,
        walls=walls,
        frog_timer=(5, 5),
        collected_frogs=(False, False),
        captured_frogs=(False, False),
        frog_collected=(False, False),
    )
    model = GardenerModel(state, norm_id="gardener/permission-drain-v0")
    planner = ASPPlanner(objectives=model.objectives, reuse_solver=reuse)
    with engine(state) as env:
        monitors = [
            make_gardener_monitor(norm, num_frogs=2, num_puddles=1)
            for norm in (PERMISSION_AWARE_NORM_ID, DRAIN_NORM_ID)
        ]
        for monitor in monitors:
            monitor.reset(MonitorInput(frozenset()))
        _, _, terminated, truncated, info = env.step(action)
        for monitor in monitors:
            monitor.update(MonitorInput(info["labels"], terminated, truncated))
        assert monitors[0].counts["Unpermitted"] == unpermitted
        assert monitors[1].counts["Drain"] == drain
    # Alternating recipes on the same solver must not retain enabled norms.
    for norm, expected in (
        (model.norm_id, unpermitted + drain),
        (NO_COLLECT_NORM_ID, 1),
        (PERMISSION_AWARE_NORM_ID, unpermitted),
        (DRAIN_NORM_ID, drain),
        (model.norm_id, unpermitted + drain),
    ):
        current = GardenerModel(state, norm_id=norm)
        assert predict(current, state, [action], planner=planner).costs["violations"] == expected


def test_combined_fix_avoids_both_kinds_of_violation():
    state = snapshot(
        puddles=((2, 2),),
        frogs=((1, 0), (3, 2)),
        frog_timer=(5, 5),
        collected_frogs=(False, False),
        captured_frogs=(False, False),
        frog_collected=(False, False),
    )
    values = {0: 10.0, 3: 9.0, 4: 8.0, 1: 7.0, 2: 6.0}
    for norm, expected in ((PERMISSION_AWARE_NORM_ID, 0), (DRAIN_NORM_ID, 3), ("gardener/permission-drain-v0", 4)):
        model = GardenerModel(state, norm_id=norm)
        result = ASPPlanner(objectives=model.objectives).solve(model.problem(state), values)
        assert result.action == expected
        assert result.costs["violations"] == 0
