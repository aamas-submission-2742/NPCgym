"""Clingo backend; imported only when constructing an ASPPlanner."""

import re
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from threading import local

from clingo import Control, Function, Number, String, Symbol, SymbolType, parse_term
from clingo.ast import AST, ASTType, ProgramBuilder, Sign, parse_string

from npc_gym.policy_fixes.core import (
    NoPlanError,
    Objective,
    PlanningDecision,
    PlanningLimitError,
    PlanningModelError,
    PlanningProblem,
    integer,
)

_SHARED = files("npc_gym.policy_fixes").joinpath("planning.lp").read_text(encoding="utf-8")
_MAX_CACHED_SOURCE = 256_000
_FRAMEWORK_ARITIES = {"time": 1, "available_action": 1, "preference": 2, "action": 2}
_RESERVED_NAME = re.compile(r"\b(?:" + "|".join(_FRAMEWORK_ARITIES) + r")\b|\bnpc_|#include")


def _check_head(head: AST) -> None:
    """Reject domain definitions of framework predicates, allowing body references."""
    if not _RESERVED_NAME.search(str(head)):
        return
    if head.ast_type == ASTType.Literal and head.sign == Sign.NoSign:
        _check_head(head.atom)
    elif head.ast_type == ASTType.SymbolicAtom:
        symbol = head.symbol
        positive = symbol.ast_type == ASTType.Function
        if symbol.ast_type == ASTType.UnaryOperation:
            symbol = symbol.argument
        if symbol.ast_type == ASTType.Function and (
            symbol.name.startswith("npc_") or positive and _FRAMEWORK_ARITIES.get(symbol.name) == len(symbol.arguments)
        ):
            raise PlanningModelError(
                f"Domain programs must not define framework predicate {symbol.name}/{len(symbol.arguments)}"
            )
    elif head.ast_type in (ASTType.Aggregate, ASTType.Disjunction):
        for element in head.elements:
            _check_head(element.literal)
    elif head.ast_type == ASTType.HeadAggregate:
        for element in head.elements:
            _check_head(element.condition.literal)


def _parse(source: str) -> tuple[AST, ...]:
    statements: list[AST] = []
    parse_string(source, statements.append)
    # Most changing inputs are domain facts. Inspect heads only if the source
    # can mention a reserved name; includes may supply such names from a file.
    check_heads = _RESERVED_NAME.search(source) is not None
    for statement in statements:
        if statement.ast_type in (ASTType.Minimize, ASTType.Script):
            raise PlanningModelError("Domain programs must supply costs, not optimization directives or scripts")
        if check_heads and statement.ast_type == ASTType.Rule:
            _check_head(statement.head)
        elif check_heads and statement.ast_type == ASTType.External:
            _check_head(statement.atom)
    return tuple(statements)


class _ParseCache(local):
    """Bound retained syntax trees and keep Clingo AST objects thread-local."""

    def __init__(self) -> None:
        self.parse = lru_cache(maxsize=16)(_parse)


_cache = _ParseCache()


def _static_statements(source: str) -> tuple[AST, ...]:
    parse = _cache.parse if len(source) <= _MAX_CACHED_SOURCE else _parse
    return parse(source)


def _domain_statements(problem: PlanningProblem) -> tuple[AST, ...]:
    if problem._parts is None:
        return _parse(problem.program)
    static, dynamic = problem._parts
    # Drop the second implicit #program base to preserve the static source's scope.
    return _static_statements(static) + _parse(dynamic)[1:]


def _fact(name: str, *arguments: Symbol) -> str:
    return f"{Function(name, list(arguments))}."


class Solver(local):
    """Keep one prepared solver per planner and thread; ordinary calls stay fresh."""

    def __init__(self, *, reuse_solver: bool = True) -> None:
        self.reuse_solver = reuse_solver
        self.key: tuple[object, ...] | None = None
        self.prepared: _Prepared | None = None

    def __call__(
        self,
        problem: PlanningProblem,
        ranks: Mapping[int, int],
        allowed: set[int],
        proposal: int,
        objectives: Mapping[str, Objective],
        conflict_limit: int | None,
    ) -> PlanningDecision:
        if problem._externals is None:
            ctl, relevant = _ground(problem, ranks, allowed, objectives, conflict_limit)
        else:
            source, inputs = problem._externals
            key = (source, problem.horizon, tuple(sorted(ranks)), tuple(sorted(objectives.items())), conflict_limit)
            if not self.reuse_solver:
                prepared = _Prepared(problem, ranks, objectives, conflict_limit)
            elif key != self.key:
                self.key = self.prepared = None
                self.prepared = _Prepared(problem, ranks, objectives, conflict_limit)
                self.key = key
                prepared = self.prepared
            else:
                assert self.prepared is not None
                prepared = self.prepared
            prepared.assign(inputs, ranks, allowed)
            ctl, relevant = prepared.ctl, prepared.relevant
        return _decision(ctl, relevant, problem, allowed, proposal, objectives)


class _Prepared:
    def __init__(
        self,
        problem: PlanningProblem,
        ranks: Mapping[int, int],
        objectives: Mapping[str, Objective],
        conflict_limit: int | None,
    ) -> None:
        self.ctl, self.relevant = _ground(problem, ranks, set(), objectives, conflict_limit, prepared=True)
        self.inputs = {
            str(atom.symbol): atom.literal
            for atom in self.ctl.symbolic_atoms
            if atom.is_external and atom.symbol.name not in {"preference", "npc_allowed"}
        }
        self.preferences = {
            (atom.symbol.arguments[0].number, atom.symbol.arguments[1].number): atom.literal
            for atom in self.ctl.symbolic_atoms.by_signature("preference", 2)
        }
        self.allowed = {
            atom.symbol.arguments[0].number: atom.literal
            for atom in self.ctl.symbolic_atoms.by_signature("npc_allowed", 1)
        }
        self.active: set[int] = set()

    def assign(self, inputs: tuple[str, ...], ranks: Mapping[int, int], allowed: set[int]) -> None:
        active = set()
        # Validate the complete update before mutating solver state. Clingo's
        # assign_external silently ignores unknown atoms, so check explicitly.
        for text in inputs:
            if text in self.inputs:
                active.add(self.inputs[text])
                continue
            try:
                symbol = parse_term(text)
            except RuntimeError as error:
                raise PlanningModelError(
                    f"Invalid external input {text!r}: expected a ground atom without a period"
                ) from error
            if str(symbol) != text:
                raise PlanningModelError(f"External input {text!r} must use canonical atom syntax: {str(symbol)!r}")
            raise PlanningModelError(f"Input {text!r} is not a declared domain external in the grounded program")
        active.update(self.preferences[action, rank] for action, rank in ranks.items())
        active.update(self.allowed[action] for action in allowed)
        for literal in self.active - active:
            self.ctl.assign_external(literal, False)
        for literal in active - self.active:
            self.ctl.assign_external(literal, True)
        self.active = active


def _ground(
    problem: PlanningProblem,
    ranks: Mapping[int, int],
    allowed: set[int],
    objectives: Mapping[str, Objective],
    conflict_limit: int | None,
    *,
    prepared: bool = False,
) -> tuple[Control, list[tuple[Symbol, int]]]:
    arguments = ["--opt-mode=opt", "--models=0", "--seed=0", "--parallel-mode=1", "--warn=none"]
    if conflict_limit is not None:
        arguments.append(f"--solve-limit={conflict_limit}")
    ctl = Control(arguments)
    facts = [f"time(0..{problem.horizon - 1})."]
    for action, rank in ranks.items():
        facts.append(_fact("available_action", Number(action)))
        if not prepared:
            facts.append(_fact("preference", Number(action), Number(rank)))
    if prepared:
        facts.extend(
            (
                f"#external preference(A,0..{len(ranks) - 1}) : available_action(A).",
                "#external npc_allowed(A) : available_action(A).",
            )
        )
    facts.extend(_fact("npc_allowed", Number(action)) for action in sorted(allowed))
    facts.extend(_fact("npc_objective", String(n), Number(t.weight), Number(t.priority)) for n, t in objectives.items())
    try:
        if prepared:
            assert problem._externals is not None
            statements = _static_statements(problem._externals[0])
            for statement in statements:
                if statement.ast_type == ASTType.External and str(statement.external_type) != "false":
                    raise PlanningModelError("Prepared external declarations must use the default false value")
        else:
            statements = _domain_statements(problem)
        ctl.add("base", [], "\n".join((*facts, _SHARED)))
        # Reuse the validated syntax trees instead of parsing the domain twice.
        with ProgramBuilder(ctl) as builder:
            for statement in statements:
                builder.add(statement)
        ctl.ground([("base", [])])
    except RuntimeError as error:
        raise PlanningModelError(f"Cannot parse or ground the planning problem: {error}") from error
    relevant = _validate_ground_program(ctl, problem, ranks, objectives)
    return ctl, relevant


def _decision(
    ctl: Control,
    relevant: list[tuple[Symbol, int]],
    problem: PlanningProblem,
    allowed: set[int],
    proposal: int,
    objectives: Mapping[str, Objective],
) -> PlanningDecision:
    atoms: list[Symbol] | None = None
    # Read only validated action/cost literals, independently of domain #show
    # directives. Model views expire when requesting the next solution.
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            atoms = [symbol for symbol, literal in relevant if model.is_true(literal)]
        result = handle.get()
    if not result.exhausted:
        raise PlanningLimitError("Planning stopped before proving an optimum; increase or remove conflict_limit")
    if result.unsatisfiable:
        raise NoPlanError("No feasible plan: check domain constraints, candidates and the planning horizon")
    if atoms is None:
        raise PlanningModelError("Solver reported completion without a plan")
    action_atoms = [s for s in atoms if s.match("action", 2)]
    actions = {s.arguments[0].number: s.arguments[1].number for s in action_atoms}
    if len(action_atoms) != problem.horizon or set(actions) != set(range(problem.horizon)) or actions[0] not in allowed:
        raise PlanningModelError("Domain rules must leave exactly one admissible action at each planning time")
    costs = dict.fromkeys(objectives, 0)
    keys: set[tuple[str, Symbol]] = set()
    for symbol in atoms:
        if not symbol.match("cost", 3):
            continue
        name, value, key = symbol.arguments
        identity = (name.string, key)
        if identity in keys:
            raise PlanningModelError(f"Cost key {key} has multiple values for objective {name.string!r}")
        keys.add(identity)
        costs[name.string] += value.number
    totals: dict[int, int] = {}
    for objective_name, term in objectives.items():
        totals[term.priority] = totals.get(term.priority, 0) + term.weight * costs[objective_name]
    plan = tuple(actions[t] for t in range(problem.horizon))
    return PlanningDecision(proposal, plan[0], plan, costs, totals, "optimal")


def _validate_ground_program(
    ctl: Control, problem: PlanningProblem, ranks: Mapping[int, int], objectives: Mapping[str, Objective]
) -> list[tuple[Symbol, int]]:
    """Validate the domain interface and retain only decision-relevant literals."""
    atoms = ctl.symbolic_atoms
    for predicate, arity, positive in atoms.signatures:
        expected = {"candidate": 2, "action": 2, "cost": 3}.get(predicate)
        if expected is not None and arity != expected:
            symbol = next(atoms.by_signature(predicate, arity, positive)).symbol
            raise PlanningModelError(f"{predicate} must have arity {expected}; got {symbol}")
    relevant = []
    for predicate in ("candidate", "action"):
        for atom in atoms.by_signature(predicate, 2):
            symbol = atom.symbol
            time, action = symbol.arguments
            if (
                time.type != SymbolType.Number
                or action.type != SymbolType.Number
                or time.number not in range(problem.horizon)
                or action.number not in ranks
            ):
                raise PlanningModelError(f"Invalid {predicate} atom {symbol}: use modeled times and action IDs")
            if predicate == "action":
                relevant.append((symbol, atom.literal))
    for atom in atoms.by_signature("cost", 3):
        symbol = atom.symbol
        name, value, _key = symbol.arguments
        if name.type != SymbolType.String or name.string not in objectives:
            raise PlanningModelError(f"Unknown cost objective in {symbol}; configure every named objective")
        if value.type != SymbolType.Number:
            raise PlanningModelError(f"Cost must be an integer in {symbol}")
        try:
            integer("weighted cost", value.number * objectives[name.string].weight, minimum=-(2**31 - 1))
        except ValueError as error:
            raise PlanningModelError(str(error)) from error
        relevant.append((symbol, atom.literal))
    return relevant
