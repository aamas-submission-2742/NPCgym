"""OFTEN abstraction parity, literal costs, and independent environment behavior."""

import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("NPC_GYM_REQUIRE_ASP") == "1":
    import clingo
else:
    clingo = pytest.importorskip("clingo")

from npc_gym.envs import PacmanEnv
from npc_gym.envs.pacman.labels import (
    PacmanLabelingFunction,
)
from npc_gym.envs.pacman.layout import PacmanLayout
from npc_gym.envs.pacman.simulation import Simulation
from npc_gym.labels import Transition
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.pacman_monitors import (
    VEGAN_NORM_ID,
    VEGETARIAN_BLUE_NORM_ID,
    VEGETARIAN_ORANGE_NORM_ID,
    make_pacman_monitor,
)
from npc_gym.policy_fixes import ASPPlanner, NoPlanError, Objective, PacmanModel, PlanningProblem

VALUES = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 3.0}
ROOM = ["%%%%%%%", "% .   %", "%  G  %", "% P G %", "%     %", "%%%%%%%"]


def snapshot(engine):
    return engine.snapshot()


def fixture(*, player=(2, 2), ghosts=(((3, 2), 4, 8),), food=None, capsules=(), lines=ROOM):
    engine = Simulation(PacmanLayout.from_text("\n".join(lines), name="fixture"), np.random.default_rng(0))
    initial = snapshot(engine)
    engine.player = player
    for index, ghost in enumerate(engine.ghosts):
        position, direction, timer = ghosts[index] if index < len(ghosts) else ((5, 4), 0, 0)
        ghost.position, ghost.direction, ghost.timer = position, direction, timer
    if food is not None:
        engine.food = set(food)
        initial = replace(initial, food=frozenset(food))
    engine.capsules = set(capsules)
    initial = replace(initial, capsules=frozenset(capsules))
    return initial, engine


def fixed_plan(model, state, actions):
    problem = model.problem(state)
    assert problem.horizon == len(actions)
    constraints = "\n".join(f":- not action({t},{a})." for t, a in enumerate(actions))
    return ASPPlanner().solve(PlanningProblem(problem.program + constraints, problem.horizon), VALUES)


@pytest.mark.parametrize("behavior", ["random", "deterministic", "partly-deterministic"])
def test_replanning_does_not_mutate_state_rng_observations_rewards_or_labels(behavior):
    first, second = (PacmanEnv(features="essential", ghost_behavior=behavior) for _ in range(2))
    try:
        first.reset(seed=37)
        second.reset(seed=37)
        model = PacmanModel(first.labeling_state(), horizon=2)
        other_model = PacmanModel(second.labeling_state(), horizon=1)
        planner = ASPPlanner()
        for _ in range(8):
            before = first.labeling_state()
            decision = planner.solve(model.problem(before), VALUES)
            planner.solve(other_model.problem(second.labeling_state()), VALUES)
            assert planner.solve(model.problem(before), VALUES) == decision
            assert first.labeling_state() == before
            left, right = first.step(decision.action), second.step(decision.action)
            assert np.array_equal(left[0], right[0]) and left[1:] == right[1:]
            assert first.labeling_state() == second.labeling_state()
            if left[2] or left[3]:
                break
        first.reset(seed=18)
        reset_model = PacmanModel(first.labeling_state(), horizon=2)
        assert model.problem(first.labeling_state()) == reset_model.problem(first.labeling_state())
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"radius": -1}, {"norm_id": "pacman/cautious-v0"}])
def test_invalid_configuration(kwargs):
    initial, _ = fixture()
    with pytest.raises(ValueError):
        PacmanModel(initial, **kwargs)


def test_snapshot_validation_episode_boundaries_and_time_limit():
    initial, engine = fixture()
    model = PacmanModel(initial)
    state = snapshot(engine)
    assert model.problem(state, remaining_steps=1).horizon == 1
    with pytest.raises(ValueError, match="remaining_steps"):
        model.problem(state, remaining_steps=0)
    with pytest.raises(TypeError, match="PacmanAuthorityState"):
        model.problem({})
    # Model construction does not depend on episode history or scared timers.
    current = replace(state, last_action=3, ghosts=(replace(state.ghosts[0], eaten_count=2), *state.ghosts[1:]))
    assert PacmanModel(current).problem(current) == model.problem(current)
    with pytest.raises(ValueError, match="terminated"):
        PacmanModel(replace(state, won=True, terminated=True))
    with pytest.raises(ValueError, match="terminated"):
        model.problem(replace(state, won=True, terminated=True))
    with pytest.raises(ValueError, match="layout"):
        model.problem(replace(state, layout=replace(state.layout, name="other")))
    with pytest.raises(ValueError, match="ghost identities"):
        model.problem(replace(state, ghosts=state.ghosts[:1]))
    with pytest.raises(ValueError, match="half-grid"):
        model.problem(replace(state, player=replace(state.player, position=(2.1, 2))))
    with pytest.raises(ValueError, match="timer"):
        model.problem(replace(state, ghosts=(replace(state.ghosts[0], scared_timer=41), *state.ghosts[1:])))
    # Prior event flags and accumulated counts are not predicted future events.
    normal = replace(initial, killed_blue=True, killed_orange=True)
    assert ASPPlanner().solve(model.problem(normal), VALUES) == ASPPlanner().solve(model.problem(initial), VALUES)


def upstream(state, *, norm=VEGAN_NORM_ID, horizon=1, radius=5, values=VALUES):
    """Independent oracle: unchanged upstream ASP with independently encoded inputs."""
    source = Path(__file__).with_name("fixtures") / "often_pacman.lp"
    assert sha256(source.read_bytes()).hexdigest() == "87ce23e3e7aae45e78a7cba69af6662c4e1c6b1cdbb914c134b69bd883f67bff"
    ctl = clingo.Control(
        [
            "--opt-mode=optN",
            "--models=0",
            "--warn=none",
            "-c",
            f"horizon={horizon + 1}",
            "-c",
            f"radius={radius}",
            "-c",
            f"ghosts={max(1, len(state.ghosts))}",
        ]
    )
    ctl.load(str(source))
    ctl.ground([("base", [])])

    def enable(name, *args):
        ctl.assign_external(clingo.Function(name, list(map(clingo.Number, args))), True)

    # The upstream zero-ghost instance grounds an unbounded ok(T) recursion.
    # An always-outside dummy ghost gives an equivalent, finite empty domain.
    if not state.ghosts:
        enable("goutside", 0)
    x, y = map(int, state.player.position)
    for dx, dy in product(range(-radius, radius + 1), repeat=2):
        ax, ay = x + dx, y + dy
        if ax < 0 or ay < 0 or ax >= state.layout.width or ay >= state.layout.height or (ax, ay) in state.layout.walls:
            enable("wall", dx, dy)
    protected = {VEGAN_NORM_ID: {1, 2}, VEGETARIAN_BLUE_NORM_ID: {1}, VEGETARIAN_ORANGE_NORM_ID: {2}}[norm]
    for index, ghost in enumerate(state.ghosts):
        gx, gy = ghost.position[0] - x, ghost.position[1] - y
        if ghost.ghost_id not in protected or abs(gx) > radius or abs(gy) > radius:
            enable("goutside", index)
        else:
            enable("gcol", index, int(gx), 0)
            enable("grow", index, int(gy), 0)
    action_ids = {3: 0, 4: 1, 1: 2, 2: 3}
    for rank, action in enumerate(sorted(action_ids, key=lambda a: (-values[a], a))):
        enable("action", action_ids[action], rank)
    plans, costs = set(), set()
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            if not model.optimality_proven:
                continue
            cols, rows = {}, {}
            for atom in model.symbols(shown=True):
                (cols if atom.name == "pcol" else rows)[atom.arguments[1].number] = atom.arguments[0].number
            directions = {(0, 1): 1, (0, -1): 2, (1, 0): 3, (-1, 0): 4}
            plans.add(tuple(directions[cols[t + 1] - cols[t], rows[t + 1] - rows[t]] for t in range(horizon)))
            costs.add(tuple(model.cost))
        assert handle.get().exhausted
    return plans, costs


@pytest.mark.parametrize("norm", [VEGAN_NORM_ID, VEGETARIAN_BLUE_NORM_ID, VEGETARIAN_ORANGE_NORM_ID])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize(
    "ghosts", [(((3, 2), 4, 0),), (((3.5, 2), 3, 8),), (((1.5, 2), 4, 1),), (((3, 2), 1, 0), ((3, 2), 2, 8))]
)
def test_plans_and_costs_match_unmodified_upstream(norm, horizon, ghosts):
    initial, engine = fixture(ghosts=ghosts)
    state = snapshot(engine)
    plans, costs = upstream(state, norm=norm, horizon=horizon)
    result = ASPPlanner().solve(PacmanModel(initial, norm_id=norm, horizon=horizon).problem(state), VALUES)
    assert result.plan == min(plans)  # Library tie-break chooses among upstream optima.
    assert (result.costs["violations"], result.costs["policy"]) in costs


@pytest.mark.parametrize("timer", [0, 1, 2, 40])
def test_encounters_ignore_edibility_and_count_once_per_time(timer):
    initial, engine = fixture(ghosts=(((3, 2), 4, timer), ((3, 2), 1, timer)))
    state = snapshot(engine)
    for norm in (VEGAN_NORM_ID, VEGETARIAN_BLUE_NORM_ID, VEGETARIAN_ORANGE_NORM_ID):
        assert fixed_plan(PacmanModel(initial, norm_id=norm), state, (3,)).costs["violations"] == 1
    # A real simultaneous eating event still counts twice in the Vegan monitor;
    # the model's one-time waiver penalty is not that count.
    assert fixed_plan(PacmanModel(initial, horizon=3), state, (3, 4, 3)).costs["violations"] == 3 + 2 + 1


def test_only_a_final_turn_encounter_costs_one():
    initial, engine = fixture(ghosts=(((3, 2), 4, 0),))
    model = PacmanModel(initial, horizon=3, norm_id=VEGETARIAN_BLUE_NORM_ID)
    assert fixed_plan(model, snapshot(engine), (1, 1, 3)).costs["violations"] == 1


@pytest.mark.parametrize("timer,actual_count", [(0, 0), (8, 2)])
def test_abstract_penalties_are_distinct_from_real_monitor_counts(timer, actual_count):
    initial, engine = fixture(ghosts=(((3, 2), 4, timer), ((3, 2), 1, timer)))
    assert fixed_plan(PacmanModel(initial), snapshot(engine), (3,)).costs["violations"] == 1
    previous = engine.snapshot()
    engine.step(3)
    terminated = engine.terminated
    labels = PacmanLabelingFunction()(Transition(previous, 3, engine.snapshot(), terminated, False))
    monitor = make_pacman_monitor(VEGAN_NORM_ID)
    monitor.reset(MonitorInput(frozenset()))
    monitor.update(MonitorInput(labels, terminated, False))
    assert monitor.counts["Vegan"] == actual_count


def test_trapped_player_fails_instead_of_adding_stop():
    # Isolate the zero-ghost grounding regression: the upstream rule recurses
    # without a time bound here, so a regression must time out rather than hang.
    subprocess.run(
        [sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1])['_empty_ghost_layouts']()", __file__],
        check=True,
        timeout=10,
        capture_output=True,
    )


def _empty_ghost_layouts():
    initial, _ = fixture(player=(2, 2), ghosts=())
    layout = replace(initial.layout, walls=frozenset((x, y) for x in range(7) for y in range(6) if (x, y) != (2, 2)))
    state = replace(initial, layout=layout, ghosts=(), food=frozenset(), capsules=frozenset())
    with pytest.raises(NoPlanError):
        ASPPlanner().solve(PacmanModel(state).problem(state), VALUES)
    assert upstream(state)[0] == set()
    open_state = replace(initial, ghosts=())
    decision = ASPPlanner().solve(PacmanModel(open_state).problem(open_state), VALUES)
    assert decision.costs["violations"] == 0
    assert decision.plan == min(upstream(open_state)[0])


@pytest.mark.parametrize(
    "kwargs", [{"radius": None}, {"radius": True}, {"horizon": True}, {"horizon": 2**31 - 1}, {"radius": 2**31 - 1}]
)
def test_invalid_window_and_horizon_bounds(kwargs):
    initial, _ = fixture()
    with pytest.raises((ValueError, TypeError)):
        PacmanModel(initial, **kwargs)


def test_food_capsules_timers_directions_and_past_events_do_not_change_predictions():
    initial, engine = fixture(ghosts=(((3, 2), 4, 0), ((4, 2), 1, 0)), capsules=((3, 2),))
    state = snapshot(engine)
    model, planner = PacmanModel(initial, horizon=3), ASPPlanner()
    expected = planner.solve(model.problem(state), VALUES)
    changed = replace(
        state,
        food=frozenset(),
        capsules=frozenset(),
        killed_blue=True,
        killed_orange=True,
        ghosts=tuple(replace(g, direction=2, scared=True, scared_timer=40, eaten_count=99) for g in state.ghosts),
    )
    assert planner.solve(model.problem(changed), VALUES) == expected


def test_no_stop_or_blocked_move_and_masks_only_restrict_the_current_action():
    initial, engine = fixture(player=(1, 1), ghosts=(((5, 4), 0, 0),))
    model = PacmanModel(initial, horizon=2)
    state = snapshot(engine)
    for action in (0, 2, 4):
        with pytest.raises(NoPlanError):
            fixed_plan(model, state, (action, 3))
    decision = ASPPlanner().solve(model.problem(state), {0: 100.0, 1: 1.0, 2: 0.0, 3: 2.0, 4: 0.0})
    assert decision.proposed_action == 0 and decision.action in (1, 3)
    decision = ASPPlanner().solve(model.problem(state), VALUES, allowed_actions=[3])
    assert decision.plan[0] == 3 and decision.plan[1] in range(1, 5)


@pytest.mark.parametrize("radius", [0, 1, 2, 5])
def test_square_window_fractional_rounding_and_recentring_match_reference(radius):
    initial, engine = fixture(ghosts=(((3.5, 2), 3, 8), ((1.5, 2), 4, 8)))
    model, planner = PacmanModel(initial, radius=radius, horizon=2), ASPPlanner()
    for position in ((2, 2), (4, 2), (2, 2)):
        engine.player, engine.direction = position, 0
        state = snapshot(engine)
        plans, _ = upstream(state, radius=radius, horizon=2)
        assert planner.solve(model.problem(state), VALUES).plan == min(plans)


def test_outside_ghosts_are_ignored_before_rounding_and_diagonal_window_cells_are_included():
    initial, engine = fixture(ghosts=(((3.5, 2), 3, 8), ((3, 3), 1, 0)))
    state = snapshot(engine)
    problem = PacmanModel(initial, radius=1).problem(state)
    assert "goutside(0)." in problem.program  # 1.5 is outside even though int(1.5) is 1.
    assert "gcol(1,1,0)." in problem.program and "grow(1,1,0)." in problem.program
    plans, _ = upstream(state, radius=1)
    assert ASPPlanner().solve(problem, VALUES).plan == min(plans)


@pytest.mark.parametrize("reuse_solver", [True, False])
def test_horizon_changes_rebuild_and_restore_the_same_model(reuse_solver):
    initial, engine = fixture()
    state = snapshot(engine)
    model, planner = PacmanModel(initial, horizon=3), ASPPlanner(reuse_solver=reuse_solver)
    for remaining in (3, 1, 2, 3):
        problem = model.problem(state, remaining_steps=remaining)
        result = planner.solve(problem, VALUES)
        plans, _ = upstream(state, horizon=remaining)
        assert result.plan == min(plans)
        assert result == ASPPlanner().solve(PlanningProblem(problem.program, remaining), VALUES)


@pytest.mark.parametrize("layout", PacmanEnv.layouts)
@pytest.mark.parametrize("behavior", ["random", "deterministic", "partly-deterministic"])
def test_bundled_layouts_missing_colors_and_physical_moves(layout, behavior):
    with PacmanEnv(layout=layout, features="essential", ghost_behavior=behavior) as env:
        env.reset(seed=13)
        state = env.labeling_state()
        model = PacmanModel(state, norm_id=VEGETARIAN_ORANGE_NORM_ID)
        result = ASPPlanner().solve(model.problem(state), VALUES)
        assert result.optimal and result.action in range(1, 5)
        x, y = state.player.position
        dx, dy = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}[result.action]
        assert (x + dx, y + dy) not in state.layout.walls
        if len(state.ghosts) < 2:
            assert result.costs["violations"] == 0


def test_weighted_tradeoffs_remain_explicit():
    initial, engine = fixture()
    problem = PacmanModel(initial).problem(snapshot(engine))
    result = ASPPlanner().solve(problem, VALUES)
    assert result.action != 3 and result.costs["violations"] == 0
    weighted = ASPPlanner(objectives={"violations": Objective(1, 1), "policy": Objective(2, 1)})
    assert weighted.solve(problem, VALUES).action == 3


@pytest.mark.parametrize("nearby_ghost", [1, 2])
@pytest.mark.parametrize("reuse_solver", [False, True])
def test_vegetarian_ignores_only_the_permitted_color(nearby_ghost, reuse_solver):
    ghosts = [((5, 4), 0, 0), ((5, 4), 0, 0)]
    ghosts[nearby_ghost - 1] = ((3, 2), 4, 8)
    initial, engine = fixture(ghosts=tuple(ghosts))
    state = snapshot(engine)
    previous = engine.snapshot()
    engine.step(3)
    labels = PacmanLabelingFunction()(Transition(previous, 3, engine.snapshot(), engine.terminated, False))
    planner = ASPPlanner(reuse_solver=reuse_solver)
    # Switching norm variants on one planner must also clear the previous inputs.
    for norm, color, protected in (
        (VEGAN_NORM_ID, "Vegan", True),
        (VEGETARIAN_ORANGE_NORM_ID, "VegetarianOrange", nearby_ghost == 2),
        (VEGETARIAN_BLUE_NORM_ID, "VegetarianBlue", nearby_ghost == 1),
        (VEGETARIAN_ORANGE_NORM_ID, "VegetarianOrange", nearby_ghost == 2),
    ):
        model = PacmanModel(initial, norm_id=norm)
        assert fixed_plan(model, state, (3,)).costs["violations"] == int(protected)
        decision = planner.solve(model.problem(state), VALUES)
        assert (decision.action == 3) == (not protected)
        plans, costs = upstream(state, norm=norm)
        assert decision.plan == min(plans)
        assert (decision.costs["violations"], decision.costs["policy"]) in costs
        monitor = make_pacman_monitor(norm)
        monitor.reset(MonitorInput(frozenset()))
        monitor.update(MonitorInput(labels, False, False))
        count = monitor.counts[color] if norm == VEGAN_NORM_ID else monitor.count
        assert count == int(protected)
