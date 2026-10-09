"""Public ASP source assembly works without loading the optional solver."""

from dataclasses import FrozenInstanceError, replace

import pytest

from npc_gym.policy_fixes import PlanningProblem


def test_source_order_defaults_and_immutable_value_semantics():
    problem = PlanningProblem.from_parts(static="first.\nsecond.", dynamic="current.")
    assert problem.program == "first.\nsecond.\ncurrent."
    assert problem.horizon == 1
    plain = PlanningProblem(problem.program)
    assert problem == plain and hash(problem) == hash(plain) and repr(problem) == repr(plain)
    assert replace(problem, horizon=3).program == problem.program
    with pytest.raises(FrozenInstanceError):
        problem.program = "changed."


def test_empty_static_source_is_allowed():
    static = ""
    problem = PlanningProblem.from_parts(static=static, dynamic="candidate(0,0).")
    assert problem.program.strip() == "candidate(0,0)."
    assert PlanningProblem.from_parts(static=static).program.strip() == ""


@pytest.mark.parametrize("static", [None, 1, (), ("fact.",), ["fact."], ("fact.", 1), (None,), {"fact."}])
def test_invalid_static_types(static):
    with pytest.raises(TypeError, match="static must"):
        PlanningProblem.from_parts(static=static)


@pytest.mark.parametrize("dynamic", [None, 1, ("fact.",)])
def test_invalid_dynamic_types(dynamic):
    with pytest.raises(TypeError, match="dynamic must"):
        PlanningProblem.from_parts(static="fact.", dynamic=dynamic)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_invalid_horizons(horizon):
    with pytest.raises((TypeError, ValueError), match="horizon"):
        PlanningProblem.from_parts(static="fact.", horizon=horizon)


@pytest.mark.parametrize(
    "source",
    [
        '#include "missing.lp".',
        'fact. #include "missing.lp".',
        "#include <incmode>.",
        '% comment\n#include "missing.lp".',
        '%* outer %* inner *% *% #include "missing.lp".',
        r'note("escaped\" quote"). #include "missing.lp".',
    ],
)
def test_static_includes_fail_before_reading_any_files(source):
    with pytest.raises(ValueError, match="static ASP must not use #include"):
        PlanningProblem.from_parts(static="first.\n" + source)


@pytest.mark.parametrize(
    "source",
    [
        'note("#include").',
        r'note("escaped\" #include").',
        '% #include "missing.lp".\nfact.',
        '%* outer %* inner *% #include "missing.lp". *% fact.',
        '%* " % line\n *% note("#include").',
    ],
)
def test_mentions_of_includes_in_strings_and_comments_are_allowed(source):
    assert PlanningProblem.from_parts(static=source).program == source + "\n"


def test_dynamic_includes_and_syntax_validation_are_deferred_to_solving():
    assert "#include" in PlanningProblem.from_parts(static="", dynamic='#include "missing.lp".').program
    assert PlanningProblem.from_parts(static="unfinished(").program == "unfinished(\n"


def test_external_input_source_is_detached_and_construction_is_dependency_free():
    inputs = ["occupied(north)"]
    problem = PlanningProblem.from_externals(static="#external occupied(north;south).", true_atoms=inputs)
    inputs.clear()
    assert problem.program.endswith("#program base.\noccupied(north).")
    assert problem == PlanningProblem(problem.program)
    assert replace(problem)._externals is None
    assert PlanningProblem.from_externals(static="").horizon == 1


@pytest.mark.parametrize("atoms", ["occupied(north)", [None], [1], [""], ["   "]])
def test_invalid_external_input_types(atoms):
    with pytest.raises(TypeError, match="true_atoms"):
        PlanningProblem.from_externals(static="", true_atoms=atoms)


def test_external_source_types_includes_and_horizons():
    with pytest.raises(TypeError, match="static"):
        PlanningProblem.from_externals(static=("fact.",))
    with pytest.raises(ValueError, match="static ASP must not use #include"):
        PlanningProblem.from_externals(static='#include "missing.lp".')
    with pytest.raises(ValueError, match="horizon"):
        PlanningProblem.from_externals(static="", horizon=0)
