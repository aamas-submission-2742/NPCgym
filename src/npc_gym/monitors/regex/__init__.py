"""Dependency-free Boolean trace regex compilation and simple monitors."""

from collections.abc import Mapping

from npc_gym.monitors.automaton import AutomatonMonitor, Proposition, Reporting

from .compiler import RegexCompilationError, RegexLimits, RegexSyntaxError, compile_regex

__all__ = ["RegexCompilationError", "RegexLimits", "RegexSyntaxError", "compile_regex", "from_regex"]


def from_regex(
    expression: str,
    *,
    propositions: Mapping[str, Proposition],
    reporting: Reporting | str = Reporting.PREFIX,
    consume_initial: bool = True,
    limits: RegexLimits | None = None,
    minimize: bool = True,
) -> AutomatonMonitor:
    """Compile a whole-trace regex into a Boolean monitor, minimized by default."""
    return AutomatonMonitor(
        compile_regex(expression, limits=limits, minimize=minimize),
        propositions=propositions,
        reporting=reporting,
        consume_initial=consume_initial,
    )
