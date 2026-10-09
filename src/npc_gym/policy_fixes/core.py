"""Dependency-free contracts for ASP policy fixing.

Domain encoders own the dynamics and normative model. The planner owns action
selection, objective configuration, solver limits and detached diagnostics.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeVar

_INT_MAX = 2**31 - 1
_ASP_TOKEN = re.compile(r'%\*|%[^\r\n]*|"(?:\\[\s\S]|[^"\\])*"|#include\b')
_COMMENT_DELIMITER = re.compile(r"%\*|\*%")
ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)


class PlanningError(RuntimeError):
    """A planning operation could not produce a certified decision."""


class NoPlanError(PlanningError):
    """The planning program has no answer set."""


class PlanningLimitError(PlanningError):
    """The solver stopped before proving optimality or unsatisfiability."""


class PlanningModelError(PlanningError):
    """The domain program violates the planning interface or cannot be grounded."""


def integer(name: str, value: object, *, minimum: int = 0) -> int:
    """Validate an integer representable in Clingo's numeric terms."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer; got {value!r}")
    if not minimum <= int(value) <= _INT_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT_MAX}]; got {value!r}")
    return int(value)


def _check_static_source(source: str) -> None:
    """Reject includes before Clingo can read them; ignore strings and comments."""
    if "#include" not in source:
        return
    position = 0
    while match := _ASP_TOKEN.search(source, position):
        position = match.end()
        if match[0] == "#include":
            raise ValueError("static ASP must not use #include; read the file into a string or use dynamic source")
        if match[0] == "%*":
            depth = 1
            while depth:
                delimiter = _COMMENT_DELIMITER.search(source, position)
                if delimiter is None:
                    return  # Clingo will report the unfinished comment at solve time.
                depth += 1 if delimiter[0] == "%*" else -1
                position = delimiter.end()


@dataclass(frozen=True, slots=True)
class Objective:
    """Multiply a named cost by weight; larger positive priorities dominate.

    Weight zero disables a term. Costs may be negative to represent bonuses.
    Priorities below one are reserved for deterministic plan tie-breaking.
    """

    weight: int = 1
    priority: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", integer("weight", self.weight))
        object.__setattr__(self, "priority", integer("priority", self.priority, minimum=1))


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    """Trusted domain ASP source for exactly ``horizon`` future actions.

    Define ``candidate(T,A)`` and optional ``cost("name",Value,Key)`` atoms.
    The shared program chooses ``action(T,A)`` for each zero-based time. Use
    distinct keys for costs that must be counted separately. Domain rules may
    restrict these choices and compute predicted costs, but must not define
    framework predicates or add their own optimization directives.

    The domain owns uncertainty handling, history, termination and any spatial
    abstraction. Represent post-terminal positions by cost-free padding. Passing
    a planning problem does not itself establish conservative model fidelity.
    """

    program: str
    horizon: int = 1
    _parts: tuple[str, str] | None = field(default=None, init=False, repr=False, compare=False)
    _externals: tuple[str, tuple[str, ...]] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.program, str):
            raise TypeError("program must be ASP source text")
        object.__setattr__(self, "horizon", integer("horizon", self.horizon, minimum=1))

    @classmethod
    def from_parts(cls, *, static: str, dynamic: str = "", horizon: int = 1) -> PlanningProblem:
        """Assemble reusable ASP source followed by the current state.

        Both inputs are ASP strings containing complete rules and facts.
        ``static`` uses a bounded parsing cache; ``dynamic`` is parsed afresh.
        ``program`` joins ``static`` and ``dynamic`` with a newline. Empty strings are allowed.
        Each solve still grounds and validates the full problem.

        Static ``#include`` directives raise ValueError; invalid types raise
        TypeError. ASP syntax is checked when solving. Construction needs no
        Clingo. Reconstructing or using ``dataclasses.replace`` discards parsing
        hints, not source. Changing text requires no cache invalidation.
        """
        if not isinstance(static, str):
            raise TypeError("static must be an ASP string")
        if not isinstance(dynamic, str):
            raise TypeError("dynamic must be an ASP string")
        _check_static_source(static)
        problem = cls(static + "\n" + dynamic, horizon)
        object.__setattr__(problem, "_parts", (static, dynamic))
        return problem

    @classmethod
    def from_externals(cls, *, static: str, true_atoms: Iterable[str] = (), horizon: int = 1) -> PlanningProblem:
        """Reuse a grounded program whose changing inputs are declared ``#external``.

        ``static`` contains the complete rules, fixed facts and external declarations.
        ``true_atoms`` contains canonical ground atom strings without trailing
        periods; all other externals are false. External declarations must use
        the default false value. Unknown or non-external inputs fail at solve time.

        By default, each planner retains its most recent prepared solver per thread. Changes
        to source, horizon, action IDs, objectives or conflict limit rebuild it.
        Inputs, action ranks and admissibility are replaced on every solve.
        Construction needs no Clingo; includes are forbidden as in ``from_parts``.
        ``program`` also includes the true inputs as facts for ordinary solving.
        """
        if not isinstance(static, str):
            raise TypeError("static must be an ASP string")
        _check_static_source(static)
        if isinstance(true_atoms, (str, bytes)):
            raise TypeError("true_atoms must be an iterable of ground atom strings")
        atoms = tuple(true_atoms)
        if any(not isinstance(atom, str) or not atom.strip() for atom in atoms):
            raise TypeError("true_atoms must contain nonempty ground atom strings")
        program = static + "\n#program base.\n" + "\n".join(f"{atom}." for atom in atoms)
        problem = cls(program, horizon)
        object.__setattr__(problem, "_externals", (static, atoms))
        return problem


@dataclass(frozen=True, slots=True)
class PlanningDecision:
    """Detached result; fallback decisions have no plan or predicted costs.

    ``costs`` contains unweighted sums by name; ``objective`` contains weighted
    sums by priority, excluding deterministic tie-break terms. These are model
    predictions, not actual monitor events. Only optimal decisions may be used
    as expert evidence or for disagreement-based replay filtering.

    Plan ties are deterministic; answer-set ties need not be. If multiple optimal
    answer sets have the same plan and objective totals, their per-name ``costs``
    may differ, including between fresh and reused solvers. Disabled objectives
    can also have unspecified costs. Equality compares all fields, including costs.
    """

    proposed_action: int
    action: int
    plan: tuple[int, ...]
    costs: Mapping[str, int]
    objective: Mapping[int, int]
    status: Literal["optimal", "unsatisfiable", "limit"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "costs", MappingProxyType(dict(self.costs)))
        object.__setattr__(self, "objective", MappingProxyType(dict(self.objective)))

    @property
    def changed(self) -> bool:
        return self.action != self.proposed_action

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"

    @property
    def expert_eligible(self) -> bool:
        return self.optimal


class ActionValuePolicy(Protocol[ObservationT_contra]):
    """Read-only action values, keyed by the environment's integer action IDs."""

    def __call__(self, observation: ObservationT_contra, info: Mapping[str, Any], /) -> Mapping[int, float]: ...


class ASPPlanner:
    """Optimize domain problems, optionally reusing a prepared solver.

    The default objectives are ``violations`` (priority 2) and ``policy``
    (priority 1). The policy term ranks the first action: best is zero, with
    action-ID ties. A supplied proposal is promoted to rank zero, so exploration
    can be checked against the same fixer. Other actions retain their ranking.

    ``conflict_limit`` bounds solver conflicts, not elapsed time or grounding.
    ``on_failure="base_policy"`` explicitly allows returning the proposal on
    unsatisfiability or an incomplete solve. Input/model errors never fall back.
    Ordinary problems use fresh solvers. ``PlanningProblem.from_externals``
    reuses the most recent prepared solver per thread, replacing all inputs.
    Set ``reuse_solver=False`` to build a fresh solver on every call, including
    external-input problems. Input validation and objectives are unchanged.
    No environment or normative history is retained.
    """

    def __init__(
        self,
        *,
        objectives: Mapping[str, Objective] | None = None,
        conflict_limit: int | None = None,
        on_failure: Literal["raise", "base_policy"] = "raise",
        reuse_solver: bool = True,
    ) -> None:
        configured = (
            dict(objectives)
            if objectives is not None
            else {
                "violations": Objective(),
                "policy": Objective(priority=1),
            }
        )
        if "policy" not in configured:
            raise ValueError("objectives must include 'policy' (use weight=0 to disable it)")
        for name, term in configured.items():
            if not isinstance(name, str) or not name:
                raise ValueError("objective names must be nonempty strings")
            if not isinstance(term, Objective):
                raise TypeError(f"objective {name!r} must be an Objective")
        if on_failure not in ("raise", "base_policy"):
            raise ValueError("on_failure must be 'raise' or 'base_policy'")
        if not isinstance(reuse_solver, bool):
            raise TypeError("reuse_solver must be a Boolean")
        limit = None if conflict_limit is None else integer("conflict_limit", conflict_limit)
        # Loading the public module and data types does not import Clingo.
        try:
            from npc_gym.policy_fixes._solver import Solver
        except ModuleNotFoundError as error:
            if error.name != "clingo":
                raise
            raise ImportError("ASP planning requires Clingo; install `pip install 'npc-gym[asp]'`.") from error
        self._solve = Solver(reuse_solver=reuse_solver)
        self.objectives: Mapping[str, Objective] = MappingProxyType(configured)
        self.conflict_limit = limit
        self.on_failure = on_failure

    def solve(
        self,
        problem: PlanningProblem,
        action_values: Mapping[int, float],
        *,
        proposed_action: int | None = None,
        allowed_actions: Iterable[int] | None = None,
    ) -> PlanningDecision:
        """Choose the first action of a proven optimal plan without executing it.

        Values must include every modeled action. ``allowed_actions`` restricts
        only the current action; future admissibility belongs to the model.
        All inputs are validated before invoking Clingo. Calling repeatedly does
        not advance episode history; the caller supplies the current problem.
        """
        if not isinstance(problem, PlanningProblem):
            raise TypeError("problem must be a PlanningProblem")
        if not isinstance(action_values, Mapping) or not action_values:
            raise ValueError("action_values must be a nonempty mapping of action IDs to finite values")
        values: dict[int, float] = {}
        for action, value in action_values.items():
            key = integer("action ID", action, minimum=-_INT_MAX)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"action value for {key} must be a non-Boolean real number")
            try:
                numeric = float(value)
            except OverflowError as error:
                raise ValueError(f"action value for {key} must be finite and real") from error
            if not isfinite(numeric):
                raise ValueError(f"action value for {key} must be finite and real")
            values[key] = numeric
        allowed = (
            set(values)
            if allowed_actions is None
            else {integer("allowed action", action, minimum=-_INT_MAX) for action in allowed_actions}
        )
        if not allowed or not allowed <= values.keys():
            raise ValueError("allowed_actions must be a nonempty subset of action_values")
        ranking = sorted(values, key=lambda action: (-values[action], action))
        proposal = next(action for action in ranking if action in allowed)
        if proposed_action is not None:
            proposal = integer("proposed_action", proposed_action, minimum=-_INT_MAX)
            if proposal not in allowed:
                raise ValueError("proposed_action must be allowed and have an action value")
        ranking.remove(proposal)
        ranking.insert(0, proposal)
        ranks = {action: rank for rank, action in enumerate(ranking)}
        try:
            return self._solve(problem, ranks, allowed, proposal, self.objectives, self.conflict_limit)
        except (NoPlanError, PlanningLimitError) as error:
            if self.on_failure == "raise":
                raise
            status: Literal["unsatisfiable", "limit"] = "unsatisfiable" if isinstance(error, NoPlanError) else "limit"
            return PlanningDecision(proposal, proposal, (), {}, {}, status)


class FixedPolicy:
    """Adapt action values and a problem builder to the evaluation policy API.

    Both callbacks receive the original observation and public info. A builder
    must obtain its snapshot/history from those inputs, not mutate the live
    environment. The builder owns domain-specific lifecycle handling. This
    adapter retains only ``last_decision`` for diagnostics; every call replans.
    """

    def __init__(
        self,
        planner: ASPPlanner,
        action_values: ActionValuePolicy[Any],
        problem: Callable[[Any, Mapping[str, Any]], PlanningProblem],
        *,
        allowed_actions: Callable[[Any, Mapping[str, Any]], Iterable[int]] | None = None,
    ) -> None:
        if not isinstance(planner, ASPPlanner):
            raise TypeError("planner must be an ASPPlanner")
        if not callable(action_values) or not callable(problem):
            raise TypeError("action_values and problem must be callable")
        if allowed_actions is not None and not callable(allowed_actions):
            raise TypeError("allowed_actions must be callable or None")
        self.planner = planner
        self.action_values = action_values
        self.problem = problem
        self.allowed_actions = allowed_actions
        self.last_decision: PlanningDecision | None = None

    def __call__(self, observation: Any, info: Mapping[str, Any], /) -> int:
        self.last_decision = None
        allowed = None if self.allowed_actions is None else self.allowed_actions(observation, info)
        decision = self.planner.solve(
            self.problem(observation, info), self.action_values(observation, info), allowed_actions=allowed
        )
        self.last_decision = decision
        return decision.action
