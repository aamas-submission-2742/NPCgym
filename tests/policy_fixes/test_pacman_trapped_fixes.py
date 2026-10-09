"""Trapped history, one-step movement, and the intentional score abstraction."""

import os
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

if os.environ.get("NPC_GYM_REQUIRE_ASP") == "1":
    import clingo  # noqa: F401
else:
    pytest.importorskip("clingo")

from npc_gym.envs import PacmanEnv
from npc_gym.envs.pacman.labels import PacmanAuthorityState, PacmanLabel, PacmanLayoutState, PacmanPlayerState
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.pacman_monitors import TRAPPED_NORM_ID, make_pacman_monitor
from npc_gym.policy_fixes import ASPPlanner, Objective, PacmanTrappedModel, PlanningProblem

VALUES = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 3.0}
ZERO, HIGH, WEST = PacmanLabel.SCORE_0, PacmanLabel.SCORE_GREATER_400, PacmanLabel.WEST_SIDE


def letter(*labels, **ending):
    return MonitorInput(frozenset(labels), **ending)


def state(*, x=3, width=8, walls=()):
    return PacmanAuthorityState(
        layout=PacmanLayoutState("room", width, 5, frozenset(walls)),
        food=frozenset(),
        capsules=frozenset(),
        player=PacmanPlayerState((x, 2), 0, True),
        ghosts=(),
        score=0,
        terminated=False,
        won=False,
        lost=False,
        killed_blue=False,
        killed_orange=False,
        ate_capsule=False,
        last_action=None,
    )


@pytest.mark.parametrize("reuse", [True, False])
def test_history_matches_monitor_and_predictions_do_not_advance_it(reuse):
    planner = ASPPlanner(reuse_solver=reuse)
    snapshot = state(x=4)
    for history in product(((), (ZERO,), (PacmanLabel.SCORE_GREATER_100,), (HIGH,), (ZERO, HIGH)), repeat=3):
        model = PacmanTrappedModel(letter(*history[0]))
        monitor = make_pacman_monitor(TRAPPED_NORM_ID)
        event = monitor.reset(letter(*history[0]))
        for index, labels in enumerate(history):
            if index:
                model.advance(letter(*labels))
                event = monitor.update(letter(*labels))
            problem = model.problem(snapshot)
            decision = planner.solve(problem, VALUES, allowed_actions=[0])
            assert decision.costs["violations"] == int(event)
            assert model.problem(snapshot) == problem
            assert decision == ASPPlanner().solve(PlanningProblem(problem.program), VALUES, allowed_actions=[0])


@pytest.mark.parametrize("reuse", [True, False])
@pytest.mark.parametrize(
    "snapshot, costs",
    [
        (state(), (0, 0, 0, 1, 0)),
        (state(x=4), (1, 1, 1, 1, 0)),
        (state(walls=((4, 2),)), (0, 0, 0, 0, 0)),
        (state(x=4, walls=((3, 2),)), (1, 1, 1, 1, 1)),
        (state(x=3, width=7), (0, 0, 0, 1, 0)),
        (state(x=4, width=7), (1, 1, 1, 1, 0)),
        (state(x=0, width=1), (0, 0, 0, 0, 0)),
    ],
)
def test_all_actions_boundaries_and_blocked_moves(snapshot, costs, reuse):
    model = PacmanTrappedModel(letter(ZERO))
    planner = ASPPlanner(reuse_solver=reuse)
    for action, expected in enumerate(costs):
        decision = planner.solve(model.problem(snapshot), VALUES, allowed_actions=[action])
        assert decision.plan == (action,) and decision.costs["violations"] == expected
    model.advance(letter(HIGH))
    assert planner.solve(model.problem(snapshot), VALUES).action == 3


def test_release_reactivation_and_policy_preferences():
    model, planner = PacmanTrappedModel(letter(ZERO)), ASPPlanner()
    snapshot = state()
    assert planner.solve(model.problem(snapshot), VALUES).action == 4
    model.advance(letter(WEST))  # Returning west does not release the restriction.
    assert planner.solve(model.problem(snapshot), VALUES).action == 4
    model.advance(letter(HIGH))
    assert planner.solve(model.problem(snapshot), VALUES).action == 3
    model.advance(letter())  # Falling below 101 does not reactivate it.
    assert planner.solve(model.problem(snapshot), VALUES).action == 3
    model.advance(letter(ZERO))
    assert planner.solve(model.problem(snapshot), VALUES).action == 4
    stop = {**VALUES, 0: 100.0}
    assert planner.solve(model.problem(snapshot), stop).action == 0
    unavoidable = planner.solve(model.problem(state(x=5)), VALUES)
    assert unavoidable.action == 3 and unavoidable.costs["violations"] == 1
    disabled = ASPPlanner(objectives={"violations": Objective(0), "policy": Objective(priority=1)})
    assert disabled.solve(model.problem(snapshot), VALUES).action == 3
    model.reset(letter())
    assert planner.solve(model.problem(snapshot), VALUES).action == 3


def test_score_changing_moves_are_deliberately_not_predicted():
    model = PacmanTrappedModel(letter(ZERO, WEST))
    monitor = make_pacman_monitor(TRAPPED_NORM_ID)
    monitor.reset(letter(ZERO, WEST))
    snapshot = replace(state(), score=400, food=frozenset({(4, 2)}))
    decision = ASPPlanner().solve(model.problem(snapshot), VALUES, allowed_actions=[3])
    assert decision.costs["violations"] == 1
    assert not monitor.update(letter(HIGH))  # The real move can release the norm immediately.
    model.advance(letter(HIGH))
    assert ASPPlanner().solve(model.problem(state(x=4)), VALUES).costs["violations"] == 0
    assert monitor.update(letter(ZERO))  # Conversely, a new activation was not predicted.
    model.advance(letter(ZERO))
    assert ASPPlanner().solve(model.problem(state(x=4)), VALUES, allowed_actions=[0]).costs["violations"] == 1


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_episode_endings_reset_and_invalid_inputs(ending):
    model = PacmanTrappedModel(letter(ZERO))
    with pytest.raises(TypeError, match="MonitorInput"):
        model.advance([])
    with pytest.raises(TypeError, match="MonitorInput"):
        model.reset([])
    with pytest.raises(TypeError, match="MonitorInput"):
        PacmanTrappedModel([])
    with pytest.raises(TypeError, match="PacmanAuthorityState"):
        model.problem({})
    with pytest.raises(ValueError, match="terminated"):
        model.problem(replace(state(), terminated=True, won=True))
    model.advance(letter(ZERO, **{ending: True}))
    with pytest.raises(ValueError, match="ended"):
        model.problem(state())
    with pytest.raises(ValueError, match="ended"):
        model.advance(letter())
    model.reset(letter(HIGH))
    assert ASPPlanner().solve(model.problem(state(width=9)), VALUES).action == 3
    model.reset(letter(ZERO, **{ending: True}))
    with pytest.raises(ValueError, match="ended"):
        model.problem(state())


@pytest.mark.parametrize("reuse", [True, False])
def test_real_movement_monitor_counts_and_environment_independence(reuse):
    with PacmanEnv(features="essential") as env, PacmanEnv(features="essential") as control:
        _, info = env.reset(seed=37)
        control.reset(seed=37)
        model = PacmanTrappedModel(MonitorInput(info["labels"]))
        monitor = make_pacman_monitor(TRAPPED_NORM_ID)
        monitor.reset(MonitorInput(info["labels"]))
        planner = ASPPlanner(reuse_solver=reuse)
        for action in [3, 3, 0, 1, 4, 2] * 3:
            before = env.labeling_state()
            problem = model.problem(before)
            decision = planner.solve(problem, VALUES, allowed_actions=[action])
            assert model.problem(before) == problem and env.labeling_state() == before
            observed = env.step(decision.action)
            expected = control.step(decision.action)
            assert np.array_equal(observed[0], expected[0]) and observed[1:] == expected[1:]
            _, _, terminated, truncated, info = observed
            input = MonitorInput(info["labels"], terminated, truncated)
            actual = monitor.update(input)
            if HIGH not in input.labels:  # This short trace starts active; only release may differ.
                assert decision.costs["violations"] == int(actual)
            model.advance(input)
            if terminated or truncated:
                break
        _, info = env.reset(seed=18)
        model.reset(MonitorInput(info["labels"]))
        fresh = PacmanTrappedModel(MonitorInput(info["labels"]))
        assert model.problem(env.labeling_state()) == fresh.problem(env.labeling_state())
