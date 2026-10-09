"""Lazy LTLf2DFA/MONA adapter; no compiler types cross this boundary."""

import importlib
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from npc_gym.automata import CompiledDFA


class LTLfCompilationError(RuntimeError):
    """The formula compiler failed, timed out, or returned an invalid DFA."""


class LTLfCompiler(Protocol):
    """Replaceable translator to an NPC Gym automaton, independent of reporting."""

    def compile(self, formula: str) -> CompiledDFA: ...


# The dependency has no py.typed marker. Keep its untyped surface local and
# describe only the two operations used, without disabling import checking.
class _ParsedFormula(Protocol):
    def find_labels(self) -> list[str]: ...


class _MonaProgram(Protocol):
    def mona_program(self) -> str: ...


def _read_dfa(output: str, atoms: tuple[str, ...]) -> CompiledDFA:
    def field(name: str) -> str:
        found = re.search(rf"^{name}:[ \t]*([^\n]*)$", output, re.MULTILINE)
        if found is None:
            raise ValueError(f"Missing MONA field: {name}")
        return found.group(1).strip()

    names = tuple(field("DFA for formula with free variables").lower().split())
    if set(names) != set(atoms) or len(names) != len(atoms):
        raise ValueError("MONA returned unexpected proposition names")
    initial = int(field("Initial state"))
    finals = frozenset(map(int, field("Accepting states").split()))
    transitions: dict[int, list[tuple[str, int]]] = {}
    for line in output.splitlines():
        if not line.startswith("State "):
            continue
        match = re.fullmatch(r"State (\d+):[ \t]*([01X]*)[ \t]*-> state (\d+)[ \t]*", line)
        if match is None:
            raise ValueError("Malformed MONA transition")
        source, guard, target = match.groups()
        if len(guard) != len(names):
            raise ValueError("MONA guard width differs from its proposition count")
        ordered_guard = "".join(guard[names.index(atom)] for atom in atoms)
        transitions.setdefault(int(source), []).append((ordered_guard, int(target)))
    # MONA consumes a dummy start position before the first trace letter.
    # Validate the whole graph before removing that marker from the runtime.
    raw = CompiledDFA(atoms, initial, finals, transitions)
    starts = {target for _, target in raw.transitions[initial]}
    if len(starts) != 1:
        raise ValueError("MONA initial marker must have one destination")
    return CompiledDFA(atoms, starts.pop(), finals, transitions)


class MonaCompiler:
    """Translate LTLf using the ``ltlf`` extra and the external MONA executable.

    Atoms are lowercase identifiers (``[a-z][a-z0-9_]*``). Each compilation
    runs in a private temporary directory, with a finite timeout and cleanup on
    failure. This avoids the upstream convenience API's shared site-packages
    output file. Minimize after removing the dummy start position unless
    ``minimize=False``. Importing this module never imports the optional dependency.
    """

    def __init__(self, *, executable: str = "mona", timeout: float = 30.0, minimize: bool = True) -> None:
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a bool")
        self.minimize = minimize
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.executable = executable
        self.timeout = timeout

    def compile(self, formula: str) -> CompiledDFA:
        if not isinstance(formula, str):
            raise TypeError("formula must be a string")
        if not formula.strip():
            raise ValueError("formula must not be empty")
        try:
            parser_module = importlib.import_module("ltlf2dfa.parser.ltlf")
            program_module = importlib.import_module("ltlf2dfa.base")
        except ModuleNotFoundError as error:
            if error.name and error.name.split(".")[0] == "ltlf2dfa":
                raise ImportError("Install npc-gym[ltlf] and MONA to compile LTLf formulas") from error
            raise
        parser = cast(Callable[[], Callable[[str], _ParsedFormula]], parser_module.LTLfParser)()
        program = cast(Callable[[_ParsedFormula], _MonaProgram], program_module.MonaProgram)
        syntax_error = cast(type[Exception], importlib.import_module("lark.exceptions").UnexpectedInput)
        try:
            parsed = parser(formula)
        except syntax_error as error:
            raise ValueError(f"Invalid LTLf formula: {error}") from error
        atoms = tuple(sorted(parsed.find_labels()))
        if any(re.fullmatch(r"[a-z][a-z0-9_]*", atom) is None for atom in atoms):
            raise ValueError("LTLf atoms must match [a-z][a-z0-9_]*")
        executable = shutil.which(self.executable)
        if executable is None:
            raise FileNotFoundError(f"MONA executable {self.executable!r} not found; install MONA and add it to PATH")
        with tempfile.TemporaryDirectory(prefix="npc-gym-ltlf-") as directory:
            source = Path(directory) / "formula.mona"
            source.write_text(program(parsed).mona_program(), encoding="utf-8")
            try:
                result = subprocess.run(
                    [executable, "-q", "-u", "-w", str(source)],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise LTLfCompilationError(f"MONA exceeded the {self.timeout:g}-second timeout") from error
        if result.returncode:
            raise LTLfCompilationError(f"MONA exited with status {result.returncode}: {result.stderr or result.stdout}")
        try:
            definition = _read_dfa(result.stdout, atoms)
            return definition.minimized() if self.minimize else definition
        except ValueError as error:
            raise LTLfCompilationError(f"Invalid MONA output: {error}") from error
