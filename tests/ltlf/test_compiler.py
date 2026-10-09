import importlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from npc_gym.monitors.ltlf import LTLfCompilationError, MonaCompiler


@pytest.mark.parametrize("formula", ["", " ", "G (", "a &", '"MixedCase"'])
def test_invalid_formulas(formula, compiler):
    with pytest.raises(ValueError):
        compiler.compile(formula)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        MonaCompiler(timeout=timeout)


def test_missing_python_extra(monkeypatch):
    original = importlib.import_module

    def blocked(name):
        if name.startswith("ltlf2dfa"):
            raise ModuleNotFoundError("blocked for test", name="ltlf2dfa")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(ImportError, match=r"npc-gym\[ltlf\].*MONA"):
        MonaCompiler().compile("F a")


def test_missing_mona(compiler):
    with pytest.raises(FileNotFoundError, match="MONA.*not found"):
        MonaCompiler(executable="npcgym-nonexistent-mona").compile("F a")


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "malformed", "bad_guard", "incomplete", "overlap"])
def test_compiler_failures_clean_up_private_files(failure, monkeypatch, compiler):
    paths = []

    def failed_run(command, **kwargs):
        path = Path(command[-1])
        paths.append(path)
        assert path.parent == Path(kwargs["cwd"])
        assert path.read_text().startswith("#")
        assert kwargs["timeout"] == 0.25
        assert "shell" not in kwargs
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "nonzero":
            return subprocess.CompletedProcess(command, 7, "", "compilation failed")
        output = "malformed"
        if failure in {"bad_guard", "incomplete", "overlap"}:
            edges = {
                "bad_guard": "State 1: Y -> state 1",
                "incomplete": "State 1: 0 -> state 1",
                "overlap": "State 1: X -> state 1\nState 1: 1 -> state 1",
            }[failure]
            output = (
                "DFA for formula with free variables: A\nInitial state: 0\nAccepting states: 1\nState 0: X -> state 1\n"
                + edges
            )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(LTLfCompilationError):
        MonaCompiler(timeout=0.25).compile("F a")
    assert len(paths) == 1
    assert not paths[0].parent.exists()


def test_concurrent_compilations_use_distinct_directories(monkeypatch, compiler):
    original = subprocess.run
    paths = []

    def record(command, **kwargs):
        paths.append(Path(command[-1]))
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    backend = MonaCompiler()
    with ThreadPoolExecutor(max_workers=4) as pool:
        definitions = list(pool.map(backend.compile, ["F a", "G a", "X a", "!a"] * 2))
    assert len(definitions) == len(paths) == len({path.parent for path in paths}) == 8
    assert all(not path.parent.exists() for path in paths)
    assert all(definition.atoms == ("a",) for definition in definitions)


@pytest.mark.parametrize("formula", ["true", "false", "F a", "G(a -> X b)"])
def test_default_minimization_and_explicit_opt_out(formula, compiler):
    from itertools import product

    original = MonaCompiler(minimize=False).compile(formula)
    minimal = MonaCompiler().compile(formula)
    assert len(minimal.states) <= len(original.states)
    assert minimal.minimized() is minimal
    letters = (frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"}))
    for word in product(letters, repeat=4):
        left, right = original.initial_state, minimal.initial_state
        assert (left in original.final_states) == (right in minimal.final_states)
        for labels in word:
            left, right = original.transition(labels, left), minimal.transition(labels, right)
            assert (left in original.final_states) == (right in minimal.final_states)
