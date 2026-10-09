"""Ground once, replace all inputs, and preserve ordinary planning semantics."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

pytest.importorskip("clingo")

from npc_gym.policy_fixes import ASPPlanner, NoPlanError, Objective, PlanningModelError, PlanningProblem, _solver

RULES = """
    candidate(T,0..1) :- time(T).
    #external harmful(0..1).
    #external blocked.
    :- blocked.
    cost("violations",1,harm) :- action(0,A), harmful(A).
"""
VALUES = {0: 10.0, 1: 0.0}


def problem(*atoms, static=RULES, horizon=3):
    return PlanningProblem.from_externals(static=static, true_atoms=atoms, horizon=horizon)


@pytest.fixture(params=[True, False])
def reuse_solver(request):
    return request.param


def test_changing_state_preferences_proposals_and_masks_match_fresh_solves(monkeypatch, reuse_solver):
    planner = ASPPlanner(reuse_solver=reuse_solver)
    ground = _solver.Control.ground
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return ground(self, *args, **kwargs)

    monkeypatch.setattr(_solver.Control, "ground", counted)
    cases = [
        (("harmful(0)",), VALUES, {}),
        (("harmful(1)",), VALUES, {}),
        ((), {0: 0.0, 1: 10.0}, {}),
        ((), VALUES, {"proposed_action": 1}),
        ((), VALUES, {"allowed_actions": [1]}),
        (("harmful(0)", "harmful(0)"), VALUES, {}),
        ((), VALUES, {}),
    ]
    decisions = [planner.solve(problem(*atoms), values, **kwargs) for atoms, values, kwargs in cases]
    assert len(calls) == (1 if reuse_solver else len(cases))
    assert len({id(ctl) for ctl in calls}) == len(calls)
    if not reuse_solver:
        assert planner._solve.prepared is None  # No solver retained between calls.
    assert [d.action for d in decisions] == [1, 0, 1, 1, 1, 1, 0]
    for decision, (atoms, values, kwargs) in zip(decisions, cases, strict=True):
        current = problem(*atoms)
        assert decision == ASPPlanner().solve(PlanningProblem(current.program, current.horizon), values, **kwargs)


def test_external_custom_model_needs_no_environment_specific_backend(reuse_solver):
    class CrossingModel:
        def __init__(self, exits):
            self.rules = (
                exits
                + """
                #external occupied(north;south).
                candidate(T,A) :- time(T), exit(A,_).
                cost("violations",1,enter(T,S)) :- action(T,A), exit(A,S), occupied(S).
            """
            )

        def problem(self, side):
            return PlanningProblem.from_externals(static=self.rules, true_atoms=[f"occupied({side})"], horizon=2)

    planner = ASPPlanner(reuse_solver=reuse_solver)
    for exits in ("exit(0,north). exit(1,south).", "exit(1,north). exit(0,south)."):
        model = CrossingModel(exits)
        for side in ("north", "south", "north"):
            current = model.problem(side)
            decision = planner.solve(current, VALUES)
            assert decision == ASPPlanner().solve(PlanningProblem(current.program, 2), VALUES)
            assert decision.costs["violations"] == 0
            assert decision.plan[0] == decision.plan[1]


def test_equally_optimal_cost_breakdowns_preserve_plan_and_weighted_totals(reuse_solver):
    source = """
        candidate(0,0..1).
        #external e(0..5).
        k(1..6).
        1 { pick(K,a); pick(K,b) } 1 :- k(K).
        cost("a",1,K) :- pick(K,a).
        cost("b",1,K) :- pick(K,b).
        cost("a",1,x(K)) :- pick(K,b), e(K-1).
        cost("b",1,x(K)) :- pick(K,a), e(K-1).
    """
    objectives = {"a": Objective(), "b": Objective(), "policy": Objective(priority=1)}
    planner = ASPPlanner(objectives=objectives, reuse_solver=reuse_solver)
    for inputs in ((), (0, 2, 4), tuple(range(6)), (1, 3), ()):
        current = PlanningProblem.from_externals(static=source, true_atoms=[f"e({i})" for i in inputs])
        decision = planner.solve(current, {0: 0.0, 1: 1.0})
        assert decision.plan == (1,) and decision.optimal
        assert decision.objective == {2: 6 + len(inputs), 1: 0}
        assert decision.costs["a"] + decision.costs["b"] == 6 + len(inputs)
        assert decision.costs["policy"] == 0
        # Every answer set contributes one cost per k plus one per enabled input.
        # Choosing all a or all b gives opposite, equally optimal breakdowns.
        assert len(inputs) <= decision.costs["a"] <= 6
        assert len(inputs) <= decision.costs["b"] <= 6


def test_configuration_changes_rebuild_and_validate_current_inputs(reuse_solver):
    planner = ASPPlanner(reuse_solver=reuse_solver)
    for horizon in (3, 1, 2, 3):
        assert len(planner.solve(problem(horizon=horizon), VALUES).plan) == horizon
    with pytest.raises(PlanningModelError, match="Invalid candidate"):
        planner.solve(problem(), {0: 1.0})
    assert planner.solve(problem("harmful(0)"), VALUES).action == 1
    planner.objectives = {"policy": Objective()}
    with pytest.raises(PlanningModelError, match="Unknown cost"):
        planner.solve(problem(), VALUES)
    planner.objectives = {"policy": Objective(), "violations": Objective(weight=0)}
    assert planner.solve(problem("harmful(0)"), VALUES).action == 0
    planner.conflict_limit = 0
    planner.on_failure = "base_policy"
    assert planner.solve(problem("harmful(0)"), VALUES).status == "limit"
    planner.conflict_limit = None
    assert planner.solve(problem("harmful(0)"), VALUES).optimal


@pytest.mark.parametrize("fallback", [False, True])
def test_recovers_after_unsatisfiable_and_invalid_inputs(fallback, reuse_solver):
    planner = ASPPlanner(on_failure="base_policy" if fallback else "raise", reuse_solver=reuse_solver)
    assert planner.solve(problem("harmful(0)"), VALUES).action == 1
    if fallback:
        result = planner.solve(problem("blocked"), VALUES)
        assert result.status == "unsatisfiable" and not result.expert_eligible
    else:
        with pytest.raises(NoPlanError):
            planner.solve(problem("blocked"), VALUES)
    with pytest.raises(PlanningModelError, match="declared domain external"):
        planner.solve(problem("harmful(1)", "unknown"), VALUES)
    assert planner.solve(problem("harmful(1)"), VALUES).action == 0
    assert planner.solve(problem(), VALUES).action == 0


@pytest.mark.parametrize("atom", ["unknown", "candidate(0,0)", "npc_allowed(0)", "preference(0,0)", "42", '"x"'])
def test_rejects_unknown_nonexternal_and_framework_inputs(atom, reuse_solver):
    with pytest.raises(PlanningModelError, match="declared domain external"):
        ASPPlanner(on_failure="base_policy", reuse_solver=reuse_solver).solve(problem(atom), VALUES)


@pytest.mark.parametrize("atom", ["harmful(0).", "harmful(X)", "harmful(0). blocked", "harmful(0)% comment"])
def test_inputs_are_atoms_not_arbitrary_source(atom, reuse_solver):
    with pytest.raises(PlanningModelError, match="[Ii]nput"):
        ASPPlanner(reuse_solver=reuse_solver).solve(problem(atom), VALUES)


@pytest.mark.parametrize(
    "source,match",
    [
        ("#external active. [true]", "default false"),
        ("#external active. [free]", "default false"),
        ('cost("unknown",1,k) :- harmful(0).', "Unknown cost"),
        ('cost("violations",oops,k) :- harmful(0).', "integer"),
        ("candidate(99,99) :- harmful(0).", "Invalid candidate"),
        ("#minimize {1}.", "optimization directives"),
        ("preference(0,0).", "framework predicate"),
        ("npc_allowed(0).", "framework predicate"),
    ],
)
def test_prepared_program_validation_covers_inactive_branches(source, match, reuse_solver):
    with pytest.raises(PlanningModelError, match=match):
        ASPPlanner(reuse_solver=reuse_solver).solve(problem(static=RULES + source), VALUES)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_reuse_option_requires_boolean(value):
    with pytest.raises(TypeError, match="reuse_solver must be a Boolean"):
        ASPPlanner(reuse_solver=value)


@pytest.mark.parametrize("parts", [False, True])
def test_ordinary_problems_always_use_fresh_solvers(monkeypatch, reuse_solver, parts):
    planner = ASPPlanner(reuse_solver=reuse_solver)
    current = PlanningProblem.from_parts(static=RULES) if parts else PlanningProblem(RULES)
    controls = []
    control = _solver.Control

    def counted(*args, **kwargs):
        ctl = control(*args, **kwargs)
        controls.append(ctl)
        return ctl

    monkeypatch.setattr(_solver, "Control", counted)
    assert planner.solve(current, VALUES) == planner.solve(current, VALUES)
    assert len(controls) == 2 and controls[0] is not controls[1]


def test_program_sections_and_detached_source_preserve_input_scope():
    current = problem("harmful(0)", static=RULES + "#program unused. candidate(99,99).")
    planner = ASPPlanner()
    assert planner.solve(current, VALUES).action == 1
    assert planner.solve(current, VALUES) == planner.solve(replace(current), VALUES)


def test_independent_planners_and_threads_do_not_share_assignments():
    planners = [ASPPlanner(), ASPPlanner()]
    assert planners[0].solve(problem("harmful(0)"), VALUES).action == 1
    assert planners[1].solve(problem("harmful(1)"), VALUES).action == 0
    assert planners[0].solve(problem(), VALUES).action == 0
    barrier = Barrier(2)
    shared = ASPPlanner()

    def run(harmful):
        for _ in range(5):
            barrier.wait(timeout=10)
            assert shared.solve(problem(f"harmful({harmful})"), VALUES).action == 1 - harmful
        return shared._solve.prepared.ctl

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(run, (0, 1))
    assert first is not second
