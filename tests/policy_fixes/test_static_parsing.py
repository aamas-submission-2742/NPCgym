"""Static parsing must not retain decisions or bypass current-input validation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier

import pytest

pytest.importorskip("clingo")

from npc_gym.envs import GardenerEnv, MerchantEnv, PacmanEnv, StormTaxiEnv
from npc_gym.policy_fixes import (
    ASPPlanner,
    GardenerModel,
    MerchantModel,
    Objective,
    PacmanModel,
    PlanningModelError,
    PlanningProblem,
    TaxiModel,
    _solver,
)

RULES = 'candidate(T,0..1) :- time(T). cost("violations",1,harm) :- action(0,A), harmful(A).'
VALUES = {0: 2.0, 1: 1.0}


@pytest.fixture(autouse=True)
def clear_cache():
    _solver._cache.parse.cache_clear()
    yield
    _solver._cache.parse.cache_clear()


def test_static_reuse_preserves_changing_states_horizons_and_public_source():
    planners = (ASPPlanner(), ASPPlanner())
    for index, (harmful, horizon) in enumerate(((0, 3), (1, 1), (0, 2), (1, 3))):
        problem = PlanningProblem.from_parts(static=RULES, dynamic=f"harmful({harmful}).", horizon=horizon)
        plain = PlanningProblem(problem.program, horizon)
        assert problem == plain and hash(problem) == hash(plain) and repr(problem) == repr(plain)
        decision = planners[index % 2].solve(problem, VALUES)
        assert decision == ASPPlanner().solve(plain, VALUES)
        assert decision.action == 1 - harmful and len(decision.plan) == horizon
    assert _solver._cache.parse.cache_info().hits == 3
    replacement = replace(problem, program="candidate(T,0) :- time(T).")
    assert planners[0].solve(replacement, VALUES).action == 0


@pytest.mark.parametrize(
    "environment,model_type",
    [
        (lambda: StormTaxiEnv(), TaxiModel),
        (MerchantEnv, MerchantModel),
        (GardenerEnv, GardenerModel),
        (PacmanEnv, PacmanModel),
    ],
)
def test_builtin_fragments_match_complete_source_across_resets(environment, model_type):
    with closing(environment()) as env:
        for seed in (0, 1, 0):
            env.reset(seed=seed)
            state = env.labeling_state()
            model = model_type() if model_type in (MerchantModel, TaxiModel) else model_type(state, horizon=2)
            planner = ASPPlanner(objectives=model.objectives) if model_type is GardenerModel else ASPPlanner()
            values = dict.fromkeys(range(env.action_space.n), 0.0)
            for remaining in (2, 1, 2):
                problem = (
                    model.problem()
                    if model_type is TaxiModel
                    else model.problem(state)
                    if model_type is MerchantModel
                    else model.problem(state, remaining_steps=remaining)
                )
                assert planner.solve(problem, values) == planner.solve(
                    PlanningProblem(problem.program, problem.horizon), values
                )
    if model_type not in (PacmanModel, MerchantModel, TaxiModel):
        assert _solver._cache.parse.cache_info().hits > 0


def test_independent_model_reuses_static_source_with_changing_layouts_and_states():
    class CrossingModel:
        rules = (
            "candidate(T,A) :- time(T), exit(A,_). "
            'cost("violations",1,enter(T,Side)) :- action(T,A), exit(A,Side), occupied(Side).'
        )

        def __init__(self, layout):
            self.layout = layout

        def problem(self, occupied):
            return PlanningProblem.from_parts(
                static=self.rules + "\n" + self.layout, dynamic=f"occupied({occupied}).", horizon=2
            )

    first = CrossingModel("exit(0,north). exit(1,south).")
    second = CrossingModel("exit(0,south). exit(1,north).")
    planner = ASPPlanner()
    for model, occupied, expected in ((first, "north", 1), (first, "south", 0), (second, "north", 0)):
        problem = model.problem(occupied)
        decision = planner.solve(problem, VALUES)
        assert decision == planner.solve(PlanningProblem(problem.program, 2), VALUES)
        assert decision.plan == (expected, expected) and decision.costs["violations"] == 0
    assert _solver._cache.parse.cache_info().hits == 1


def test_cache_is_bounded_and_eviction_does_not_change_decisions():
    planner = ASPPlanner()
    for index in (*range(17), 0):
        problem = PlanningProblem.from_parts(static=f"candidate(0,0). marker({index}).")
        assert planner.solve(problem, VALUES).action == 0
    info = _solver._cache.parse.cache_info()
    assert info.currsize == 16 and info.misses == 18
    oversized = "candidate(0,1). %" + "x" * _solver._MAX_CACHED_SOURCE
    assert planner.solve(PlanningProblem.from_parts(static=oversized), VALUES).action == 1
    assert _solver._cache.parse.cache_info() == info


@pytest.mark.parametrize(
    "source,kwargs,values,match",
    [
        ("candidate(1,0).", {}, VALUES, "Invalid candidate"),
        ("candidate(0,1).", {}, {0: 1.0}, "Invalid candidate"),
        ('candidate(0,0). cost("violations",1,x).', {"objectives": {"policy": Objective()}}, VALUES, "Unknown cost"),
        (
            'candidate(0,0). cost("violations",2147483647,x).',
            {"objectives": {"violations": Objective(2), "policy": Objective()}},
            VALUES,
            "weighted cost",
        ),
    ],
)
def test_cached_syntax_is_validated_against_current_solve(source, kwargs, values, match):
    source = "candidate(T,0) :- time(T). " + source
    problem = PlanningProblem.from_parts(static=source, horizon=2)
    ASPPlanner().solve(problem, VALUES)
    # Keep the same cached source, but change the horizon/actions/objectives.
    problem = PlanningProblem.from_parts(static=source)
    with pytest.raises(PlanningModelError, match=match):
        ASPPlanner(**kwargs).solve(problem, values)


@pytest.mark.parametrize("source", ["not valid ASP!", "#minimize {1}.", "#script (python)\n#end."])
def test_invalid_static_fragments_are_not_cached_or_silently_accepted(source):
    problem = PlanningProblem.from_parts(static=source, dynamic="candidate(0,0).")
    for _ in range(2):
        with pytest.raises(PlanningModelError):
            ASPPlanner(on_failure="base_policy").solve(problem, VALUES)
    assert _solver._cache.parse.cache_info().currsize == 0


def test_fragment_boundaries_preserve_constants_and_program_sections():
    problem = PlanningProblem.from_parts(
        static="#const last=1. candidate(T,0..last) :- time(T). #program unused. candidate(99,99).",
        dynamic='#program base. cost("violations",1,harm) :- action(0,0).',
        horizon=2,
    )
    planner = ASPPlanner()
    assert planner.solve(problem, VALUES) == planner.solve(PlanningProblem(problem.program, 2), VALUES)


def test_custom_program_includes_are_read_on_every_solve(tmp_path):
    included = tmp_path / "candidates.lp"
    problem = PlanningProblem(f'#include "{included}".')
    planner = ASPPlanner()
    for action in (0, 1, 0):
        included.write_text(f"candidate(0,{action}).", encoding="utf-8")
        assert planner.solve(problem, VALUES).action == action
    assert _solver._cache.parse.cache_info().currsize == 0


def test_dynamic_includes_stay_current_and_preserve_program_scope(tmp_path):
    included = tmp_path / "current.lp"
    problem = PlanningProblem.from_parts(
        static="candidate(0,0..1). #program unused.",
        dynamic=f'#include "{included}". #program base.',
    )
    planner = ASPPlanner()
    for harmful in (0, 1, 0):
        included.write_text(
            f'candidate(99,99). #program base. cost("violations",1,harm) :- action(0,{harmful}).',
            encoding="utf-8",
        )
        decision = planner.solve(problem, VALUES)
        assert decision.action == 1 - harmful
        assert decision == planner.solve(PlanningProblem(problem.program), VALUES)


def test_empty_static_source_still_solves():
    static = ""
    assert ASPPlanner().solve(PlanningProblem.from_parts(static=static, dynamic="candidate(0,1)."), VALUES).action == 1


@pytest.mark.parametrize("static,dynamic", [("candidate(0,", "0)."), ("%* unfinished", "*% candidate(0,0).")])
def test_both_sources_must_contain_complete_statements(static, dynamic):
    with pytest.raises(PlanningModelError):
        ASPPlanner().solve(PlanningProblem.from_parts(static=static, dynamic=dynamic), VALUES)


def test_threads_keep_separate_syntax_trees():
    barrier = Barrier(2)

    def solve():
        parsed = _solver._cache.parse(RULES)
        barrier.wait(timeout=10)
        problem = PlanningProblem.from_parts(static=RULES, dynamic="harmful(0).", horizon=2)
        return parsed, ASPPlanner().solve(problem, VALUES)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(lambda _: solve(), range(2)))
    assert first[0] is not second[0]
    assert first[1] == second[1]
