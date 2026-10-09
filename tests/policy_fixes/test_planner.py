"""Real Clingo checks for policy fixing contracts, independent of any domain."""

import os
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

if os.environ.get("NPC_GYM_REQUIRE_ASP") == "1":
    import clingo  # noqa: F401 -- the ASP gate must fail rather than skip without Clingo
else:
    pytest.importorskip("clingo")

from npc_gym.policy_fixes import (
    ASPPlanner,
    FixedPolicy,
    NoPlanError,
    Objective,
    PlanningLimitError,
    PlanningModelError,
    PlanningProblem,
)

CANDIDATES = "candidate(T,0..2) :- time(T)."
HARM = CANDIDATES + 'cost("violations",1,harm) :- action(0,0).'


def test_norm_priority_dominates_policy_rank_and_raw_q_magnitudes():
    planner = ASPPlanner()
    for values in ({0: 1000000.0, 1: 1.0, 2: 0.0}, {0: 0.002, 1: 0.001, 2: -3.0}):
        result = planner.solve(PlanningProblem(HARM), values)
        assert result.action == 1
        assert result.proposed_action == 0
        assert result.changed and result.optimal and result.expert_eligible
        assert result.costs == {"violations": 0, "policy": 1}
        assert result.objective == {2: 0, 1: 1}
        with pytest.raises(TypeError):
            result.costs["violations"] = 10
        with pytest.raises(FrozenInstanceError):
            result.action = 2


def test_equal_priority_weighted_tradeoffs_and_disabled_cost():
    problem = PlanningProblem(HARM)
    values = {0: 10.0, 1: 2.0, 2: 1.0}
    planner = ASPPlanner(objectives={"violations": Objective(1, 1), "policy": Objective(2, 1)})
    assert planner.solve(problem, values).action == 0  # cost 1 beats deviation cost 2
    planner = ASPPlanner(objectives={"violations": Objective(3, 1), "policy": Objective(2, 1)})
    assert planner.solve(problem, values).action == 1
    planner = ASPPlanner(objectives={"violations": Objective(0, 1), "policy": Objective(1, 1)})
    assert planner.solve(problem, values).action == 0


def test_equal_cost_events_have_distinct_identities_and_bonuses_are_signed():
    problem = PlanningProblem(
        CANDIDATES
        + 'cost("violations",1,frog(0)) :- action(0,0).'
        + 'cost("violations",1,frog(1)) :- action(0,0).'
        + 'cost("violations",-2,target) :- action(0,0).'
    )
    result = ASPPlanner().solve(problem, {0: 1.0, 1: 0.0, 2: 0.0})
    assert result.action == 0
    assert result.costs["violations"] == 0


def test_horizon_and_lexicographic_action_ties():
    planner = ASPPlanner(objectives={"policy": Objective(0, 1)})
    problem = PlanningProblem(CANDIDATES + ":- action(0,0), action(1,0).", horizon=2)
    for _ in range(3):
        assert planner.solve(problem, {2: 4.0, 1: 4.0, 0: 4.0}).plan == (0, 1)


def test_future_cost_affects_first_action():
    problem = PlanningProblem(CANDIDATES + 'cost("violations",5,late) :- action(0,0), action(1,_).', horizon=2)
    assert ASPPlanner().solve(problem, {0: 5.0, 1: 1.0, 2: 0.0}).plan == (1, 0)


@pytest.mark.parametrize(
    "output",
    [
        "#show.",
        "#show irrelevant/1.",
        '#show action(0,99). #show cost("unknown",1,fake).',
        "#show action/0. #show -action/2. #show -cost/3.",
    ],
)
def test_domain_output_directives_do_not_change_decisions(output):
    # Displayed terms need not be true atoms, and domains can hide all output.
    program = HARM + "irrelevant(1..1000). -action(0,99)." + '-cost("unknown",oops,negative).' + output
    values = {0: 3.0, 1: 2.0, 2: 1.0}
    planner = ASPPlanner()
    assert planner.solve(PlanningProblem(program, horizon=2), values) == planner.solve(
        PlanningProblem(HARM, horizon=2), values
    )


def test_domain_constants_and_program_sections_preserve_scope():
    program = (
        "#const last=2. candidate(T,0..last) :- time(T)."
        "#program unused. candidate(99,99)."
        '#program base. cost("violations",1,harm) :- action(0,0).'
    )
    values = {0: 3.0, 1: 2.0, 2: 1.0}
    planner = ASPPlanner()
    assert planner.solve(PlanningProblem(program, horizon=2), values) == planner.solve(
        PlanningProblem(HARM, horizon=2), values
    )


def test_adverse_outcomes_are_encoded_by_domain_not_chosen_optimistically():
    problem = PlanningProblem(
        CANDIDATES
        + "outcome(0,dry,0). outcome(0,wet,4). outcome(1,dry,1). outcome(1,wet,1). outcome(2,dry,2)."
        + 'cost("violations",V,worst) :- action(0,A), V = #max { C,S : outcome(A,S,C) }.'
    )
    result = ASPPlanner().solve(problem, {0: 10.0, 1: 1.0, 2: 0.0})
    assert result.action == 1 and result.costs["violations"] == 1


def test_proposal_is_promoted_and_current_mask_does_not_constrain_future():
    planner = ASPPlanner()
    values = {0: 10.0, 1: 2.0, 2: 1.0}
    result = planner.solve(PlanningProblem(CANDIDATES, horizon=2), values, proposed_action=2)
    assert result.plan == (2, 0) and not result.changed
    result = planner.solve(PlanningProblem(CANDIDATES, horizon=2), values, allowed_actions=[1, 2])
    assert result.plan == (1, 0) and result.proposed_action == 1


def test_offset_actions_and_numpy_scalar_values():
    result = ASPPlanner().solve(
        PlanningProblem("candidate(0,-2). candidate(0,7)."),
        {np.int64(-2): np.float32(1), np.int64(7): np.float64(1)},
    )
    assert result.plan == (-2,)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("reason", ["unsatisfiable", "limit"])
def test_failure_is_explicit_and_never_expert_evidence(fallback, reason):
    planner = ASPPlanner(
        conflict_limit=0 if reason == "limit" else None,
        on_failure="base_policy" if fallback else "raise",
    )
    problem = PlanningProblem(HARM if reason == "limit" else HARM + ":- action(0,_).")
    if not fallback:
        with pytest.raises(PlanningLimitError if reason == "limit" else NoPlanError):
            planner.solve(problem, {0: 3.0, 1: 2.0, 2: 1.0})
    else:
        decision = planner.solve(problem, {0: 3.0, 1: 2.0, 2: 1.0})
        assert decision.action == decision.proposed_action == 0
        assert decision.status == reason
        assert not decision.optimal and not decision.expert_eligible
        assert decision.plan == () and decision.costs == {} and decision.objective == {}


@pytest.mark.parametrize(
    "program,match",
    [
        ("not valid ASP!", "parse or ground"),
        ('candidate(0,0). cost("unknown",1,x).', "Unknown cost"),
        ('candidate(0,0). cost("violations",oops,x).', "integer"),
        ("candidate(0,99).", "Invalid candidate"),
        ("candidate(3,0).", "Invalid candidate"),
        ("candidate(0,0;1).", "arity"),
        ("candidate(0,0). action(0).", "arity"),
        ("candidate(0,0). cost(0).", "arity"),
        ("candidate(0,0). -candidate(0).", "arity"),
        ("candidate(0,0). -action(0).", "arity"),
        ("candidate(0,0). -cost(0).", "arity"),
        ("candidate(0,0). #minimize {1 : action(0,0)}.", "optimization directives"),
        ("candidate(0,0). #maximize {1 : action(0,0)}.", "optimization directives"),
        ("candidate(0,0). :~ action(0,0). [1@1]", "optimization directives"),
        ("candidate(0,0). #script (python)\nraise AssertionError('must not execute')\n#end.", "scripts"),
        ('candidate(0,0). cost("violations",1,x). cost("violations",2,x).', "multiple values"),
    ],
)
def test_model_errors_never_fall_back(program, match):
    planner = ASPPlanner(on_failure="base_policy")
    with pytest.raises(PlanningModelError, match=match):
        planner.solve(PlanningProblem(program), {0: 1.0})


def test_weighted_cost_overflow_is_rejected():
    planner = ASPPlanner(objectives={"policy": Objective(2), "violations": Objective(2)})
    with pytest.raises(PlanningModelError, match="weighted cost"):
        planner.solve(PlanningProblem('candidate(0,0). cost("violations",2147483647,x).'), {0: 0.0})


@pytest.mark.parametrize("form", ["plain", "static", "dynamic", "external", "external_fresh"])
@pytest.mark.parametrize(
    "definition",
    [
        "npc_allowed(0).",
        "preference(0,0) :- time(0).",
        "available_action(99).",
        "time(99).",
        "action(0,0).",
        "{npc_allowed(0)}.",
        "npc_allowed(0) | other.",
        "1 #sum {1:npc_allowed(0): time(0); 2:other} 1.",
        "#external npc_allowed(0).",
        "npc_custom.",
        "-npc_custom.",
        "npc_allowed(0) :- nonexistent.",
    ],
)
def test_reserved_definitions_are_rejected_before_grounding(form, definition):
    source = CANDIDATES + definition
    if form == "plain":
        problem = PlanningProblem(source)
    elif form == "dynamic":
        problem = PlanningProblem.from_parts(static=CANDIDATES, dynamic=definition)
    elif form == "static":
        problem = PlanningProblem.from_parts(static=source)
    else:
        problem = PlanningProblem.from_externals(static=source)
    planner = ASPPlanner(reuse_solver=form != "external_fresh", on_failure="base_policy")
    with pytest.raises(PlanningModelError, match="framework predicate"):
        planner.solve(problem, {0: 3.0, 1: 2.0, 2: 1.0}, allowed_actions=[1])


def test_framework_references_in_head_conditions_and_constraints_are_allowed():
    source = CANDIDATES + "{chosen(A): action(0,A)}. :- chosen(A), not available_action(A)."
    result = ASPPlanner().solve(PlanningProblem(source), {0: 2.0, 1: 1.0, 2: 0.0}, allowed_actions=[1])
    assert result.action == 1


@pytest.mark.parametrize("parts", [False, True])
def test_dynamic_includes_cannot_define_framework_predicates(tmp_path, parts):
    path = tmp_path / "domain.lp"
    path.write_text("npc_allowed(0).", encoding="utf-8")
    source = f'{CANDIDATES}\n#include "{path.as_posix()}".'
    problem = PlanningProblem.from_parts(static="", dynamic=source) if parts else PlanningProblem(source)
    with pytest.raises(PlanningModelError, match="framework predicate"):
        ASPPlanner().solve(problem, {0: 1.0, 1: 0.0, 2: 0.0}, allowed_actions=[1])


@pytest.mark.parametrize("values", [{}, {True: 1.0}, {0: float("nan")}, {0: float("inf")}, {0: True}, {0: "1"}])
def test_invalid_action_values(values):
    with pytest.raises((TypeError, ValueError)):
        ASPPlanner().solve(PlanningProblem(CANDIDATES), values)


@pytest.mark.parametrize("kwargs", [{"allowed_actions": []}, {"allowed_actions": [7]}, {"proposed_action": 7}])
def test_invalid_current_action_constraints(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ASPPlanner().solve(PlanningProblem(CANDIDATES), {0: 1.0}, **kwargs)


@pytest.mark.parametrize("bad", [True, -1, 1.5, 2**31])
def test_invalid_weights_and_conflict_limits(bad):
    with pytest.raises((TypeError, ValueError)):
        Objective(weight=bad)
    with pytest.raises((TypeError, ValueError)):
        ASPPlanner(conflict_limit=bad)


@pytest.mark.parametrize("bad", [0, True, -1, 1.5])
def test_invalid_horizon_and_priority(bad):
    with pytest.raises((TypeError, ValueError)):
        PlanningProblem("", horizon=bad)
    with pytest.raises((TypeError, ValueError)):
        Objective(priority=bad)


def test_independent_planners_and_repeated_decisions_have_no_stale_state():
    first, second = ASPPlanner(), ASPPlanner()
    first.solve(PlanningProblem(HARM), {0: 9.0, 1: 1.0, 2: 0.0})
    with pytest.raises(NoPlanError):
        first.solve(PlanningProblem(":- ."), {0: 1.0})
    problem = PlanningProblem("candidate(0,0).")
    assert second.solve(problem, {0: 1.0}).action == 0
    assert first.solve(problem, {0: 1.0}).action == 0


def test_fixed_policy_passes_inputs_through_and_clears_failed_diagnostics():
    info = {"labels": frozenset(), "episode": 0}
    values = {0: 2.0, 1: 1.0, 2: 0.0}

    def problem(observation, received):
        assert observation == 8 and received is info
        return PlanningProblem(HARM if received["episode"] == 0 else ":- .")

    policy = FixedPolicy(ASPPlanner(), lambda obs, info: values, problem)
    assert policy(8, info) == 1
    assert policy.last_decision.expert_eligible
    info["episode"] = 1
    with pytest.raises(NoPlanError):
        policy(8, info)
    assert policy.last_decision is None
    assert values == {0: 2.0, 1: 1.0, 2: 0.0}


def test_fixed_policy_applies_current_mask_before_proposing_an_action():
    policy = FixedPolicy(
        ASPPlanner(),
        lambda obs, info: {0: 10.0, 1: 1.0, 2: 0.0},
        lambda obs, info: PlanningProblem(CANDIDATES),
        allowed_actions=lambda obs, info: [a for a, allowed in enumerate(info["action_mask"]) if allowed],
    )
    assert policy(None, {"action_mask": [0, 1, 1]}) == 1
    assert policy.last_decision.proposed_action == 1
    assert not policy.last_decision.changed


def test_solving_does_not_consume_global_randomness():
    import random

    python_before, numpy_before = random.getstate(), np.random.get_state()
    ASPPlanner().solve(PlanningProblem(HARM), {0: 10.0, 1: 2.0, 2: 0.0})
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_out_of_range_real_value_is_actionable():
    with pytest.raises(ValueError, match="finite and real"):
        ASPPlanner().solve(PlanningProblem(CANDIDATES), {0: 10**1000})
