"""Built-in regex events and explicit Boolean/count compositions.

Expressions consume original, stateless domain propositions. Repetition helpers
below only construct regex text; no recipe has private runtime state.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from npc_gym.automata import CompiledDFA
from npc_gym.monitors.automaton import AutomatonMonitor
from npc_gym.monitors.bindings import BUILTIN_PROPOSITIONS
from npc_gym.monitors.events import ComplexMonitor, MultiMonitor
from npc_gym.monitors.regex import RegexLimits, compile_regex
from npc_gym.monitors.registry import UnknownMonitorIDError


def _repeat(expression: str, n: int) -> str:
    return expression * n or "eps"


def _union(*expressions: str) -> str:
    return "(" + "|".join(expressions) + ")"


def _active_after(n: int, *, active: bool) -> str:
    """Exactly n early inputs ending with/without an extraction obligation.

    A non-tree letter sets inactive, a non-extracting tree sets active, and
    tree+extract toggles. Enumerate the last setting letter and the parity of
    the remaining toggles. This captures no-replay behavior across input 14.
    """
    toggle = "[tree & extract]"
    alternatives = [_repeat(toggle, n)] if bool(n % 2) == active else []
    for index in range(1, n + 1):
        setting = "[tree & !extract]" if (bool((n - index) % 2) != active) else "[!tree]"
        alternatives.append(_repeat(".", index - 1) + setting + _repeat(toggle, n - index))
    return _union(*alternatives)


@dataclass(frozen=True)
class RegexRecipe:
    """One regex, acceptance policy, and initial-input convention."""

    expression: str
    reporting: str = "prefix"
    consume_initial: bool = True

    def compile(self, *, minimize: bool = True) -> CompiledDFA:
        """Return the shared immutable DFA for this recipe."""
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a bool")
        return _compile(self.expression, minimize)

    def make(self, environment: str, *, minimize: bool = True) -> AutomatonMonitor:
        return AutomatonMonitor(
            self.compile(minimize=minimize),
            propositions=BUILTIN_PROPOSITIONS[environment],
            reporting=self.reporting,
            consume_initial=self.consume_initial,
        )


@cache
def _compile(expression: str, minimize: bool) -> CompiledDFA:
    return compile_regex(expression, limits=RegexLimits(max_nfa_states=2048), minimize=minimize)


def _recipes() -> dict[str, dict[str, RegexRecipe]]:
    recipes: dict[str, dict[str, RegexRecipe]] = {domain: {} for domain in ("taxi", "merchant", "gardener", "pacman")}

    def add(domain: str, name: str, expression: str, reporting: str = "prefix") -> None:
        recipes[domain][name] = RegexRecipe(expression, reporting, domain not in ("taxi", "gardener"))

    add("merchant", "Danger", ".*[danger]")
    add("merchant", "CTD", ".*[danger][fight]")
    add("merchant", "Delivery", ".*([home & sundown]|[home & !market & !sundown][!market & !sundown]*[sundown])")
    add("merchant", "EnvFriendly", ".*[tree & wood][extract]")
    add("merchant", "early", ".*[tree][extract]", "restart")
    add("merchant", "early_phase", "." + ".?" * 13)
    toggle = "[tree & wood & extract]"
    pairs = f"({toggle}{toggle})*"
    # From input 15, either carry the state at 14 or use a later setting letter.
    late = (
        _union(
            _active_after(14, active=True) + pairs,
            _active_after(14, active=False) + toggle + pairs,
            "." * 14 + ".*" + _union("[tree & wood & !extract]", "[!(tree & wood)]" + toggle) + pairs,
        )
        + "[extract]"
    )
    add("merchant", "late", late)
    add("gardener", "CollectOne", "[!collected]*[!collected & end]")
    for name, atom in (("VegetarianBlue", "blue"), ("VegetarianOrange", "orange"), ("Cautious", "pellet")):
        add("pacman", name, f".*[{atom}]")
    for color in ("blue", "orange"):
        add("pacman", f"ConditionalVegetarian{color.title()}", f".*[{color} & !west]")
        add("pacman", f"FirstTaste{color.title()}", f"[!eat]*[{color}]")
        add("pacman", f"LaterTaste{color.title()}", f".*[eat].*[{color}]")
    add("pacman", "Switch", _union(".?" * 39 + "[blue]", "." * 40 + ".*[orange]"))
    add("pacman", "CTD", ".*[eat][!still]")
    add("pacman", "CTD3", ".*[eat].?.?[!still]")
    add("pacman", "Trapped", ".*([zero & !high & !west]|[zero & !high][!high]*[!high & !west])")
    for name, target in (("Hungr", "eat"), ("Errand", "corner"), ("Visit", "southeast")):
        add(
            "pacman",
            name,
            f".*([zero & deadline & !{target}]|[zero & !deadline & !{target}][!deadline & !{target}]*[deadline & !{target}])",
        )
    activation = "[adjacent & !blue]"
    clear = "[!adjacent | blue]"
    due = activation + "[!blue]" * 3
    cycle = _union(clear, activation + "[!blue]?[!blue]?[blue]", due + activation + "*" + clear)
    add("pacman", "OblBlue", cycle + "*" + due + activation + "*")
    add("taxi", "Warn Violations", "([rain]|.*[!rain][rain])[!warn]")
    tails = ["[!safe & !rain & !new_hurricane]" * 7]
    for before_rain in range(7):
        tails.append(
            "[!safe & !rain & !new_hurricane]" * before_rain
            + "[rain & !safe & !new_hurricane]"
            + "[!safe]" * (6 - before_rain)
        )
    add(
        "taxi",
        "Seven-Step Safety Violations",
        "[!rain]*"
        + _union("[new_hurricane & rain & !safe]" + "[!safe]" * 7, "[new_hurricane & !rain & !safe]" + _union(*tails)),
    )
    add("taxi", "Three-Step Safety Violations", ".*[rain].*[new_hurricane & !safe]" + "[!new_hurricane & !safe]" * 3)
    add("taxi", "Stay Violations", "([!new_hurricane]*|.*[safe][!new_hurricane]*)[hurricane & !safe & !new_hurricane]")
    return recipes


RECIPES: Mapping[str, Mapping[str, RegexRecipe]] = MappingProxyType(
    {domain: MappingProxyType(recipes) for domain, recipes in _recipes().items()}
)


def _sum_counts(counts: Mapping[str, int]) -> int:
    return sum(counts.values())


@dataclass(frozen=True)
class _CollectionRecipe:
    members: Mapping[str, str]
    derived: Mapping[str, Callable[[Mapping[str, int]], int]]


def _collection(names: tuple[str, ...], total: str) -> _CollectionRecipe:
    return _CollectionRecipe({"CTD" if name == "CTD3" else name: name for name in names}, {total: _sum_counts})


# Both monitor and bolt factories use these named components.
_COLLECTIONS = {
    "taxi/emergency-v0": _CollectionRecipe(
        {
            name: name
            for name in (
                "Warn Violations",
                "Stay Violations",
                "Seven-Step Safety Violations",
                "Three-Step Safety Violations",
            )
        },
        {
            "Safety Violations": lambda c: c["Seven-Step Safety Violations"] + c["Three-Step Safety Violations"],
            "Emergency Violations": _sum_counts,
        },
    ),
    "merchant/pacifist-v0": _collection(("Danger", "CTD"), "Pacifist(total)"),
    "merchant/delivery-pacifist-v0": _collection(("Delivery", "Danger", "CTD"), "DeliveryPacifist"),
    "pacman/vegan-v0": _collection(("VegetarianBlue", "VegetarianOrange"), "Vegan"),
    "pacman/conditional-vegan-v0": _collection(
        ("ConditionalVegetarianBlue", "ConditionalVegetarianOrange"), "ConditionalVegan"
    ),
    "pacman/all-or-nothing-v0": _collection(("FirstTasteBlue", "FirstTasteOrange"), "AllOrNothing"),
    "pacman/one-taste-v1": _collection(("LaterTasteBlue", "LaterTasteOrange"), "OneTaste"),
    "pacman/vegan-preference-v0": _collection(("VegetarianBlue", "VegetarianOrange"), "VeganPreference"),
    "pacman/vegan-conflict-v1": _collection(("VegetarianBlue", "VegetarianOrange", "OblBlue"), "VeganConflict"),
    "pacman/hungry-vegan-v0": _collection(("VegetarianBlue", "VegetarianOrange", "Hungr"), "HungryVegan"),
    "pacman/hungry-vegetarian-v0": _collection(("VegetarianOrange", "Hungr"), "HungryVegetarian"),
    "pacman/hungry-vegan-penalty-v1": _collection(
        ("VegetarianBlue", "VegetarianOrange", "Hungr", "CTD"), "HungryVeganPenalty"
    ),
    "pacman/maximum-v2": _collection(("VegetarianBlue", "VegetarianOrange", "Hungr", "CTD", "Trapped"), "Maximum"),
    "pacman/penalty-v0": _collection(("VegetarianBlue", "VegetarianOrange", "CTD"), "Penalty(total)"),
    "pacman/penalty-1-v0": _collection(("VegetarianBlue", "VegetarianOrange", "CTD"), "Penalty1(total)"),
    "pacman/penalty-3-v0": _collection(("VegetarianBlue", "VegetarianOrange", "CTD3"), "Penalty3(total)"),
}
_SINGLES = {
    "merchant/danger-v0": "Danger",
    "merchant/delivery-v0": "Delivery",
    "merchant/env-friendly-v0": "EnvFriendly",
    "gardener/collect-one-v0": "CollectOne",
    "pacman/vegetarian-blue-v0": "VegetarianBlue",
    "pacman/vegetarian-orange-v0": "VegetarianOrange",
    "pacman/cautious-v0": "Cautious",
    "pacman/obligation-blue-v1": "OblBlue",
    "pacman/switch-v0": "Switch",
    "pacman/trapped-v1": "Trapped",
    "pacman/hungry-v0": "Hungr",
    "pacman/errand-v0": "Errand",
    "pacman/visit-v0": "Visit",
}
BUILTIN_IDS = tuple(sorted((*_SINGLES, *_COLLECTIONS, "merchant/evolving-v0", "pacman/solution-guilt-maximum-v2")))


def make_builtin_monitor(identifier: str, *, minimize: bool = True) -> ComplexMonitor | MultiMonitor:
    """Create a fresh event graph with minimal DFAs unless ``minimize=False``.

    Minimization preserves all events and counts, but changes DFA state IDs.
    Unsupported/deferred IDs fail explicitly.
    """
    if identifier not in BUILTIN_IDS:
        raise UnknownMonitorIDError(f"Unknown monitor ID {identifier!r}; available: {', '.join(BUILTIN_IDS)}")
    environment = identifier.split("/")[0]

    def event(name: str) -> AutomatonMonitor:
        return RECIPES[environment][name].make(environment, minimize=minimize)

    if identifier in _SINGLES:
        return event(_SINGLES[identifier])
    if identifier == "merchant/evolving-v0":
        return (event("early") & event("early_phase")) | event("late")
    if identifier == "pacman/solution-guilt-maximum-v2":
        blue, orange, hungry, ctd, trapped = (
            event(n) for n in ("VegetarianBlue", "VegetarianOrange", "Hungr", "CTD", "Trapped")
        )
        vegetarian = orange | hungry
        guilt = blue | vegetarian | ctd
        return MultiMonitor(
            {"HungryVegetarian": vegetarian, "HungryVeganPenalty": guilt, "Maximum": guilt | trapped},
            derived={"SolutionGuiltMaximum": lambda c: sum(c.values())},
        )
    recipe = _COLLECTIONS[identifier]
    return MultiMonitor({name: event(component) for name, component in recipe.members.items()}, derived=recipe.derived)


def builtin_regex_components(identifier: str) -> Mapping[str, RegexRecipe]:
    """Return immutable named regex components for built-in bolt construction.

    Collection keys match member count keys; derived totals are excluded.
    Simple recipes use their catalogue name. Boolean compositions without
    independent DFA components raise ValueError; unknown IDs raise UnknownMonitorIDError.
    """
    if identifier not in BUILTIN_IDS:
        raise UnknownMonitorIDError(f"Unknown monitor ID {identifier!r}")
    if identifier in _SINGLES:
        name = _SINGLES[identifier]
        members = {name: name}
    elif identifier in _COLLECTIONS:
        members = dict(_COLLECTIONS[identifier].members)
    else:
        raise ValueError(f"{identifier!r} uses Boolean composition; independent DFA bolts are unsupported")
    environment = identifier.split("/")[0]
    return MappingProxyType({name: RECIPES[environment][component] for name, component in members.items()})
