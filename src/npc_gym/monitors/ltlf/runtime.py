"""Optional LTLf construction of Boolean simple monitors."""

from collections.abc import Mapping

from npc_gym.monitors.automaton import AutomatonMonitor, Proposition, Reporting

from .compiler import LTLfCompiler, MonaCompiler


def from_ltlf(
    formula: str,
    *,
    propositions: Mapping[str, Proposition],
    reporting: Reporting | str = Reporting.PREFIX,
    consume_initial: bool = True,
    compiler: LTLfCompiler | None = None,
    minimize: bool = True,
) -> AutomatonMonitor:
    """Compile a finite-trace formula and create a fresh monitor.

    Pass an alternative ``compiler`` to replace LTLf2DFA/MONA. Bindings may
    include unused atoms so one domain vocabulary can serve several formulas.
    Minimize the returned definition unless ``minimize=False``.
    """
    if not isinstance(minimize, bool):
        raise TypeError("minimize must be a bool")
    policy = Reporting(reporting)
    definition = (compiler if compiler is not None else MonaCompiler(minimize=False)).compile(formula)
    if minimize:
        definition = definition.minimized()
    return AutomatonMonitor(
        definition,
        propositions=propositions,
        reporting=policy,
        consume_initial=consume_initial,
    )
