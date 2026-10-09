"""Warn-only fixing agrees with Taxi monitor events and preserves the engine."""

from contextlib import closing
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from npc_gym.envs import StormTaxiEnv
from npc_gym.envs.taxi.labels import TaxiActionLabel as A
from npc_gym.envs.taxi.labels import TaxiLocationLabel as L
from npc_gym.envs.taxi.labels import TaxiWeatherLabel as W
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.taxi_monitors import EMERGENCY_NORM_ID, make_taxi_monitor
from npc_gym.policy_fixes import ASPPlanner, Objective, PlanningProblem, TaxiModel

pytest.importorskip("clingo")

VALUES = {action: float(7 - action) for action in range(7)}


def letter(*labels, **flags):
    return MonitorInput(frozenset(labels), **flags)


@pytest.mark.parametrize("reuse", [True, False])
def test_every_action_matches_warning_monitor_after_short_histories(reuse):
    model = TaxiModel()
    planner = ASPPlanner(reuse_solver=reuse)
    inputs = (letter(), letter(W.RAIN), letter(A.WARN), letter(W.RAIN, A.WARN))
    for history in product(inputs, repeat=3):
        model.reset()
        for input in history:
            model.advance(input)
        problem = model.problem()
        assert problem.horizon == 1
        for action in range(7):
            decision = planner.solve(problem, VALUES, allowed_actions=[action])
            for rainy, ending in product((False, True), (None, "terminated", "truncated")):
                monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
                monitor.reset(letter(W.RAIN))  # Reset rain is ignored by the monitor.
                for input in history:
                    monitor.update(input)
                labels = ([W.RAIN] if rainy else []) + ([A.WARN] if action == 6 else [])
                events = monitor.update(letter(*labels, **({ending: True} if ending else {})))
                assert decision.costs["violations"] == int(events["Warn Violations"])
                assert decision.optimal and decision.plan == (action,)
        assert model.problem() == problem  # Predictions never consume history.


@pytest.mark.parametrize("reuse", [True, False])
def test_best_action_and_reused_solver_follow_each_rain_spell(reuse):
    model = TaxiModel()
    planner = ASPPlanner(reuse_solver=reuse)
    assert planner.solve(model.problem(), VALUES).action == 0
    for input, expected in (
        (letter(W.RAIN, A.WARN), 6),  # Warning on the onset is too early.
        (letter(W.RAIN), 0),  # Even a missed response expires after one step.
        (letter(W.RAIN), 0),
        (letter(), 0),
        (letter(W.RAIN), 6),
        (letter(A.WARN), 0),  # Warn is still required if rain stops on the response.
        (letter(W.RAIN), 6),
    ):
        model.advance(input)
        problem = model.problem()
        decision = planner.solve(problem, VALUES)
        assert decision.action == expected and decision.costs["violations"] == 0
        assert decision == ASPPlanner().solve(PlanningProblem(problem.program), VALUES)
    model.reset()
    assert planner.solve(model.problem(), VALUES).action == 0


def test_masks_and_objective_override_use_standard_planner_contract():
    model = TaxiModel()
    model.advance(letter(W.RAIN))
    problem = model.problem()
    masked = ASPPlanner().solve(problem, VALUES, allowed_actions=[2, 3])
    assert masked.action == 2 and masked.costs["violations"] == 1
    disabled = ASPPlanner(objectives={"violations": Objective(0), "policy": Objective(priority=1)})
    assert disabled.solve(problem, VALUES).action == 0
    decision = ASPPlanner().solve(problem, VALUES, proposed_action=6)
    assert decision.action == 6 and not decision.changed


def test_safety_deadlines_and_stay_are_monitored_but_not_planned():
    model = TaxiModel()
    monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
    monitor.reset(letter())
    model.advance(letter(W.RAIN))
    monitor.update(letter(W.RAIN))
    history = [letter(W.RAIN, W.NEW_HURRICANE, A.WARN)] + [letter(W.RAIN, W.HURRICANE)] * 3
    history += [letter(W.RAIN, W.HURRICANE, L.AT_SHELTER), letter(W.RAIN, W.HURRICANE)]
    for input in history:
        model.advance(input)
        monitor.update(input)
        assert ASPPlanner().solve(model.problem(), VALUES).action == 0
    assert monitor.counts["Three-Step Safety Violations"] == 1
    assert monitor.counts["Stay Violations"] == 1
    assert monitor.counts["Warn Violations"] == 0
    model.reset()
    monitor.reset(letter())
    for input in [letter(W.NEW_HURRICANE)] + [letter(W.HURRICANE)] * 7:
        model.advance(input)
        monitor.update(input)
        assert ASPPlanner().solve(model.problem(), VALUES).costs["violations"] == 0
    assert monitor.counts["Seven-Step Safety Violations"] == 1


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_ending_requires_reset_and_clears_pending_warning(ending):
    model = TaxiModel()
    model.advance(letter(W.RAIN, **{ending: True}))
    with pytest.raises(ValueError, match="ended"):
        model.problem()
    with pytest.raises(ValueError, match="ended"):
        model.advance(letter())
    model.reset()
    assert ASPPlanner().solve(model.problem(), VALUES).action == 0
    model.advance(letter(W.RAIN))
    assert ASPPlanner().solve(model.problem(), VALUES).action == 6


def test_invalid_input_fails_without_changing_history():
    model = TaxiModel()
    model.advance(letter(W.RAIN))
    problem = model.problem()
    with pytest.raises(TypeError, match="MonitorInput"):
        model.advance({"rain"})
    assert model.problem() == problem


@pytest.mark.parametrize("reuse", [True, False])
def test_fixed_policy_has_no_warning_violations_and_leaves_seeded_dynamics_unchanged(reuse):
    model = TaxiModel()
    planner = ASPPlanner(reuse_solver=reuse)
    interventions = 0
    with closing(StormTaxiEnv()) as env, closing(StormTaxiEnv()) as reference:
        for seed in range(4):
            _, info = env.reset(seed=seed)
            reference.reset(seed=seed)
            model.reset()
            monitor = make_taxi_monitor(EMERGENCY_NORM_ID)
            monitor.reset(letter(*info["labels"]))
            for _ in range(50):
                state = env.labeling_state()
                rng = deepcopy(env.np_random.bit_generator.state)
                decision = planner.solve(model.problem(), VALUES)
                assert env.labeling_state() == state and env.np_random.bit_generator.state == rng
                actual = env.step(decision.action)
                expected = reference.step(decision.action)
                assert actual[:4] == expected[:4]
                assert actual[4]["labels"] == expected[4]["labels"]
                np.testing.assert_array_equal(actual[4]["action_mask"], expected[4]["action_mask"])
                _, _, terminated, truncated, info = actual
                input = MonitorInput(info["labels"], terminated, truncated)
                model.advance(input)
                assert not monitor.update(input)["Warn Violations"]
                interventions += decision.changed
                if terminated or truncated:
                    break
    assert interventions > 0
