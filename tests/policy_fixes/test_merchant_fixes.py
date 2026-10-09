"""Environment Friendly fixing agrees with actual Merchant monitor events."""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest

from npc_gym.envs import MerchantEnv
from npc_gym.envs.merchant.labels import MerchantAuthorityState
from npc_gym.labels import Transition
from npc_gym.monitors import MonitorInput, make_builtin_monitor
from npc_gym.monitors.merchant_monitors import ENV_FRIENDLY_NORM_ID, PACIFIST_NORM_ID
from npc_gym.policy_fixes import ASPPlanner, MerchantModel, Objective, PlanningProblem

pytest.importorskip("clingo")

VALUES = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 10.0, 5: 0.0, 6: -1.0}
LAYOUT = ("XXXXXX", "XHTRMX", "X.D..X", "XXXXXX")


def snapshot(**changes):
    return replace(MerchantAuthorityState((2, 1), "T", 1, 0, 0, (True,), (True,), "fixture", LAYOUT), **changes)


@pytest.mark.parametrize("reuse", [True, False])
def test_costs_match_engine_and_monitor_for_all_cells_inventories_and_actions(reuse):
    model = MerchantModel()
    planner = ASPPlanner(reuse_solver=reuse)
    with closing(MerchantEnv(risk_fight=0.5, risk_death=0.5)) as env:
        env.reset(seed=7)
        for cell in ("H", "D", "M", "R", "T", "C", "."):
            marker = "T" if cell == "C" else cell
            y, x = np.argwhere(env.map == marker)[0]
            for wood, ore in ((0, 0), (1, 0), (2, 1), (1, 4), (0, 5)):
                env.pos = np.array([x, y])
                env.label = cell
                env.carried_wood, env.carried_ore = wood, ore
                env.wood = [int(cell != "C") for _ in env.wood]
                env._last_state = env.get_state()
                state = env.labeling_state()
                labels = env.labeling_function(Transition(None, None, env.get_state(), False, False))
                for action in range(7):
                    monitor = make_builtin_monitor(ENV_FRIENDLY_NORM_ID)
                    monitor.reset(MonitorInput(labels))
                    before_rng = deepcopy(env.np_random.bit_generator.state)
                    decision = planner.solve(model.problem(state), VALUES, allowed_actions=[action])
                    assert env.labeling_state() == state
                    assert env.np_random.bit_generator.state == before_rng
                    with closing(deepcopy(env)) as branch:
                        _, _, terminated, truncated, info = branch.step(action)
                    monitor.update(MonitorInput(info["labels"], terminated, truncated))
                    assert decision.optimal and decision.plan == (action,)
                    assert decision.costs["violations"] == monitor.count
                    assert monitor.count == int(cell == "T" and wood > 0 and action == 4)


@pytest.mark.parametrize("reuse", [True, False])
def test_reused_solver_clears_inputs_and_preserves_best_permitted_action(reuse):
    model = MerchantModel()
    planner = ASPPlanner(reuse_solver=reuse)
    for state, expected in (
        (snapshot(), 3),
        (snapshot(cell="R"), 4),  # Ore remains allowed while carrying wood.
        (snapshot(carried_wood=0), 4),  # The first wood is allowed.
        (snapshot(), 3),
        (snapshot(cell="C"), 4),  # An already depleted tree does not trigger the norm.
        (snapshot(carried_wood=5), 3),  # Full inventory does not excuse attempts.
    ):
        problem = model.problem(state)
        decision = planner.solve(problem, VALUES)
        assert decision.action == expected and decision.costs["violations"] == 0
        assert decision == ASPPlanner().solve(PlanningProblem(problem.program), VALUES)
    masked = planner.solve(model.problem(snapshot()), VALUES, allowed_actions=[0, 2, 4])
    assert masked.action == 2 and masked.proposed_action == 4
    forced = planner.solve(model.problem(snapshot()), VALUES, allowed_actions=[4])
    assert forced.action == 4 and forced.costs["violations"] == 1


def test_objective_override_and_proposal_use_the_standard_planner_contract():
    problem = MerchantModel().problem(snapshot())
    disabled = ASPPlanner(objectives={"violations": Objective(0), "policy": Objective(priority=1)})
    assert disabled.solve(problem, VALUES).action == 4
    decision = ASPPlanner().solve(problem, VALUES, proposed_action=0)
    assert decision.action == 0 and not decision.changed


def test_every_failed_attempt_at_capacity_matches_the_monitor():
    model = MerchantModel()
    planner = ASPPlanner()
    with closing(MerchantEnv(capacity=1, risk_fight=0)) as env:
        _, info = env.reset(seed=7)
        monitor = make_builtin_monitor(ENV_FRIENDLY_NORM_ID)
        monitor.reset(MonitorInput(info["labels"]))
        for action in [0] * 3 + [2] * 5 + [4] + [2] * 4:
            *_, info = env.step(action)
            monitor.update(MonitorInput(info["labels"]))
        assert monitor.count == 0 and env.labeling_state().cell == "T"
        for count in range(1, 5):
            prediction = planner.solve(model.problem(env.labeling_state()), VALUES, allowed_actions=[4])
            _, reward, terminated, truncated, info = env.step(4)
            assert monitor.update(MonitorInput(info["labels"], terminated, truncated))
            assert reward == 0 and prediction.costs["violations"] == 1 and monitor.count == count


def test_irrelevant_snapshot_fields_do_not_change_the_problem():
    model = MerchantModel()
    state = snapshot()
    other = replace(
        state,
        position=(3, 8),
        carried_wood=4,
        carried_ore=1,
        time=28,
        layout_name="another",
        layout=("X" * 10,) * 10,
        wood_available=(),
        ore_available=(),
    )
    assert model.problem(state) == model.problem(other)


def test_one_model_can_be_reused_across_layouts_and_resets():
    model = MerchantModel()
    planner = ASPPlanner()
    layouts = files("npc_gym.envs.merchant").joinpath("merchant_layouts")
    for path in sorted(layouts.iterdir(), key=lambda p: p.name):
        if path.name.endswith(".txt"):
            with closing(MerchantEnv(layout=path.name.removesuffix(".txt"))) as env:
                for seed in (7, 8):
                    env.reset(seed=seed)
                    assert planner.solve(model.problem(env.labeling_state()), VALUES).action == 4
    assert make_builtin_monitor(PACIFIST_NORM_ID) is not None


@pytest.mark.parametrize(
    "changes",
    [{"cell": "X"}, {"cell": ""}, {"cell": []}, {"carried_wood": -1}, {"carried_wood": True}, {"carried_wood": 1.5}],
)
def test_invalid_relevant_fields_fail(changes):
    with pytest.raises((TypeError, ValueError), match="cell|carried_wood"):
        MerchantModel().problem(snapshot(**changes))


def test_invalid_snapshot_type_fails():
    with pytest.raises(TypeError, match="MerchantAuthorityState"):
        MerchantModel().problem(())
