"""Immutable specifications for reward-bearing finite-trace automata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING

from npc_gym.automata import CompiledDFA
from npc_gym.monitors import Proposition, Reporting

if TYPE_CHECKING:
    from npc_gym.monitors.ltlf import LTLfCompiler
    from npc_gym.monitors.regex import RegexLimits


def _finite_reward(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-Boolean real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class BoltSpec:
    """Share a DFA and stateless bindings; each wrapper owns its execution state.

    ``reward`` is paid on every accepting transition, including self-loops.
    ``prefix`` retains the reached state; ``restart`` retains the initial state
    after acceptance without replaying the input. Initial labels are consumed
    by default, applying the same restart policy, but reset never pays a reward.
    A normative bolt is simply a configuration with a strictly negative reward.
    Specifications compare and hash by identity, like their compiled definitions.
    """

    definition: CompiledDFA
    propositions: Mapping[str, Proposition]
    reward: float
    reporting: Reporting | str = Reporting.PREFIX
    consume_initial: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.definition, CompiledDFA):
            raise TypeError("definition must be a CompiledDFA")
        if not isinstance(self.propositions, Mapping):
            raise TypeError("propositions must be a mapping")
        missing = set(self.definition.atoms) - self.propositions.keys()
        if missing:
            raise ValueError(f"Missing proposition bindings: {', '.join(sorted(missing))}")
        bindings = {atom: self.propositions[atom] for atom in self.definition.atoms}
        if any(not callable(predicate) for predicate in bindings.values()):
            raise TypeError("Proposition bindings must be callable")
        policy = Reporting(self.reporting)
        if policy is Reporting.EPISODE_END:
            raise ValueError("Bolts support only prefix or restart reporting")
        if not isinstance(self.consume_initial, bool):
            raise TypeError("consume_initial must be a bool")
        object.__setattr__(self, "reward", _finite_reward(self.reward, "Bolt reward"))
        object.__setattr__(self, "reporting", policy)
        object.__setattr__(self, "propositions", MappingProxyType(bindings))


def make_regex_bolt(
    expression: str,
    *,
    propositions: Mapping[str, Proposition],
    reward: float,
    reporting: Reporting | str = Reporting.PREFIX,
    consume_initial: bool = True,
    limits: RegexLimits | None = None,
    minimize: bool = True,
) -> BoltSpec:
    """Compile a Boolean trace regex into a bolt with a minimal DFA by default."""
    from npc_gym.monitors.regex import compile_regex

    return BoltSpec(
        compile_regex(expression, limits=limits, minimize=minimize), propositions, reward, reporting, consume_initial
    )


def make_ltlf_bolt(
    formula: str,
    *,
    propositions: Mapping[str, Proposition],
    reward: float,
    reporting: Reporting | str = Reporting.PREFIX,
    consume_initial: bool = True,
    compiler: LTLfCompiler | None = None,
    minimize: bool = True,
) -> BoltSpec:
    """Compile LTLf using an injectable compiler; optional dependencies load lazily."""
    from npc_gym.monitors.ltlf import MonaCompiler

    if not isinstance(minimize, bool):
        raise TypeError("minimize must be a bool")
    definition = (compiler if compiler is not None else MonaCompiler(minimize=False)).compile(formula)
    if minimize:
        definition = definition.minimized()
    return BoltSpec(definition, propositions, reward, reporting, consume_initial)


def make_builtin_bolts(
    monitor_id: str, *, reward: float, rewards: Mapping[str, float] | None = None, minimize: bool = True
) -> Mapping[str, BoltSpec]:
    """Build independent regex bolts with optional named reward overrides.

    DFAs are minimized by default; ``minimize=False`` preserves the original
    state IDs and observation dimensions. Each component receives its own reward. Derived counts are
    not additional reward events. Composed expressions without a single DFA
    (Evolving and SolutionGuiltMaximum) cannot expose bolt state features.
    """
    from npc_gym.monitors.bindings import BUILTIN_PROPOSITIONS
    from npc_gym.monitors.builtins import builtin_regex_components

    adjustment = _finite_reward(reward, "Bolt reward")
    components = builtin_regex_components(monitor_id)
    environment = monitor_id.split("/")[0]
    keys = {f"{monitor_id}/{name}": recipe for name, recipe in components.items()}
    if rewards is not None and not isinstance(rewards, Mapping):
        raise TypeError("rewards must be a mapping of component IDs to finite rewards")
    overrides = {}
    for key, value in (rewards or {}).items():
        if key not in keys:
            raise ValueError(f"Unknown reward component ID {key!r}; available: {', '.join(keys)}")
        overrides[key] = _finite_reward(value, f"Reward for {key!r}")
    return MappingProxyType(
        {
            key: BoltSpec(
                recipe.compile(minimize=minimize),
                BUILTIN_PROPOSITIONS[environment],
                overrides.get(key, adjustment),
                recipe.reporting,
                recipe.consume_initial,
            )
            for key, recipe in keys.items()
        }
    )


__all__ = ["BoltSpec", "make_builtin_bolts", "make_ltlf_bolt", "make_regex_bolt"]
