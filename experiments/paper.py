"""The configurations represented in the paper's environment tables."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import gymnasium as gym

from experiments import specifications as specs
from experiments.pacman_baselines import MONITORS
from experiments.pacman_bolts import LAYOUT, make_environment
from npc_gym.monitors import make_builtin_monitor
from npc_gym.monitors.gardener_monitors import make_gardener_monitor

GARDENER_MONITORS = (
    "gardener/collect-one-v0",
    "gardener/rescue-v2",
    "gardener/permission-aware-v0",
    "gardener/drain-v0",
    "gardener/no-collect-v0",
)

PACMAN_ENVIRONMENT_ID = "pacman/small-classic-v0"
PACMAN_VEGAN_OFTEN_ID = "pacman-dqn-often-v2"
PACMAN_VEGETARIAN_OFTEN_ID = "pacman-dqn-often-vegetarian-v2"
PACMAN_TRAPPED_OFTEN_ID = "pacman-dqn-often-trapped-v3"
_PACMAN_RECIPES = {
    "pacman-dqn-unconstrained-v0": specs.PACMAN_DQN_EXPERIMENT_ID,
    "pacman-ppo-unconstrained-v1": specs.PACMAN_EXPERIMENT_ID,
    PACMAN_VEGAN_OFTEN_ID: specs.PACMAN_OFTEN_EXPERIMENT_ID,
    PACMAN_VEGETARIAN_OFTEN_ID: specs.PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID,
    PACMAN_TRAPPED_OFTEN_ID: specs.PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID,
}
EXPERIMENTS_BY_ENVIRONMENT = {
    "taxi": (
        specs.TAXI_EXPERIMENT_ID,
        specs.TAXI_POLICY_FIX_EXPERIMENT_ID,
        specs.TAXI_WARN_BOLT_EXPERIMENT_ID,
        specs.TAXI_BOLT_EXPERIMENT_ID,
    ),
    "merchant": (
        specs.MERCHANT_EXPERIMENT_ID,
        specs.MERCHANT_POLICY_FIX_EXPERIMENT_ID,
        *specs.MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS,
    ),
    "gardener": (
        specs.GARDENER_EXPERIMENT_ID,
        specs.GARDENER_PPO_EXPERIMENT_ID,
        specs.GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID,
        specs.GARDENER_DRAIN_OFTEN_EXPERIMENT_ID,
        specs.GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID,
        specs.GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID,
        *specs.GARDENER_BOLT_EXPERIMENT_IDS,
    ),
    "pacman": (
        *_PACMAN_RECIPES,
        *(
            f"pacman-smallclassic-{algorithm}-bolts-{norm}-v0"
            for norm in (
                "vegan",
                "vegetarian",
                "hungry-vegan",
                "hungry-vegan-penalty",
                "vegan-conflict-5m",
                "trapped-20m",
            )
            for algorithm in ("dqn", "ppo")
        ),
    ),
}


def _smallclassic(render_mode: Any, *, layout: str, features: str, dfas: None) -> gym.Env[Any, Any]:
    if layout != "smallClassic" or dfas is not None:
        raise ValueError("Paper Pacman requires smallClassic and no environment DFAs")
    return make_environment(render_mode, layout=LAYOUT, features=features, ghost_behavior="random")


def make_registry() -> specs.ExperimentRegistry:
    """Select paper methods with common monitors for every policy in each environment.

    Budgets, penalties and learner settings come from the standard recipes.
    All evaluations use 1,000 episodes. Pilots and image experiments are excluded.
    """
    selected = {identifier for identifiers in EXPERIMENTS_BY_ENVIRONMENT.values() for identifier in identifiers}
    experiments = [item for item in specs.EXPERIMENTS if item.id in selected]
    for identifier, template in _PACMAN_RECIPES.items():
        source = specs.REGISTRY.resolve(template).specification
        experiments.append(
            replace(
                source,
                id=identifier,
                environment_id=PACMAN_ENVIRONMENT_ID,
                scenario_id="pacman/paper-norms-v0" if source.often is None else source.scenario_id,
            )
        )
    environment = specs.EnvironmentSpecification(
        PACMAN_ENVIRONMENT_ID,
        "pacman",
        _smallclassic,
        specs.KeywordArguments.from_mapping({"layout": "smallClassic", "features": "complete", "dfas": None}),
    )

    # The same observation/reward factories and learners support both layouts.
    def support_paper(component: Any) -> Any:
        if specs.PACMAN_ENVIRONMENT_ID in component.environment_ids:
            return replace(component, environment_ids=component.environment_ids | {PACMAN_ENVIRONMENT_ID})
        return component

    scenario = specs.ScenarioSpecification(
        "pacman/paper-norms-v0",
        frozenset({PACMAN_ENVIRONMENT_ID}),
        tuple((norm, partial(make_builtin_monitor, norm)) for norm in MONITORS),
    )
    pacman_scenarios = {
        item.scenario_id
        for item in experiments
        if item.environment_id in {PACMAN_ENVIRONMENT_ID, specs.PACMAN_BOLTS_ENVIRONMENT_ID}
    }
    gardener_scenarios = {item.scenario_id for item in experiments if item.id in EXPERIMENTS_BY_ENVIRONMENT["gardener"]}
    gardener_monitors = tuple(
        (norm, partial(make_gardener_monitor, norm, num_frogs=2, num_puddles=4, minimize=True))
        for norm in GARDENER_MONITORS
    )
    scenarios = tuple(
        replace(item, monitor_factories=scenario.monitor_factories)
        if item.id in pacman_scenarios
        else replace(item, monitor_factories=gardener_monitors)
        if item.id in gardener_scenarios
        else item
        for item in map(support_paper, specs.SCENARIOS)
    )
    return specs.ExperimentRegistry(
        environments=(*specs.ENVIRONMENTS, environment),
        wrappers=tuple(map(support_paper, specs.WRAPPERS)),
        scenarios=(*scenarios, scenario),
        algorithms=tuple(map(support_paper, specs.ALGORITHMS)),
        techniques=specs.TECHNIQUES,
        experiments=experiments,
    )
