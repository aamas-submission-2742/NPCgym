"""Validate built distributions without importing from the source checkout."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

REQUIRED_SDIST_PATHS = {
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "environment.yml",
    "environment-paper.yml",
    "pyproject.toml",
    "shell.nix",
    "docs/conf.py",
    "docs/api.rst",
    "docs/environments.rst",
    "docs/evaluation.rst",
    "docs/examples.rst",
    "docs/experiments.rst",
    "docs/index.rst",
    "docs/integrations.rst",
    "docs/labeling_functions.rst",
    "docs/monitors.rst",
    "docs/monitor_catalogue.rst",
    "docs/monitor_specifications.rst",
    "docs/restraining_bolts.rst",
    "docs/policy_fixes.rst",
    "docs/often.rst",
    "src/npc_gym/algorithms/tabular_often.py",
    "tests/policy_fixes/test_tabular_often.py",
    "src/npc_gym/policy_fixes/__init__.py",
    "src/npc_gym/policy_fixes/core.py",
    "src/npc_gym/policy_fixes/_solver.py",
    "src/npc_gym/policy_fixes/planning.lp",
    "src/npc_gym/policy_fixes/taxi.py",
    "src/npc_gym/policy_fixes/taxi.lp",
    "src/npc_gym/policy_fixes/merchant.py",
    "src/npc_gym/policy_fixes/merchant.lp",
    "tests/policy_fixes/test_taxi_fixes.py",
    "tests/policy_fixes/test_merchant_fixes.py",
    "src/npc_gym/policy_fixes/gardener.py",
    "src/npc_gym/policy_fixes/gardener.lp",
    "tests/policy_fixes/test_gardener_fixes.py",
    "src/npc_gym/policy_fixes/pacman.py",
    "src/npc_gym/policy_fixes/pacman.lp",
    "src/npc_gym/policy_fixes/pacman_trapped.lp",
    "src/npc_gym/wrappers/pacman_wrappers.py",
    "tests/wrappers/test_pacman_wrappers.py",
    "tests/policy_fixes/test_pacman_trapped_fixes.py",
    "tests/policy_fixes/test_pacman_fixes.py",
    "tests/policy_fixes/fixtures/often_pacman.lp",
    "tests/integrations/fixtures/often_update.json",
    "tests/policy_fixes/test_planner.py",
    "tests/policy_fixes/test_asp_imports.py",
    "docs/catalogue/taxi.rst",
    "docs/catalogue/merchant.rst",
    "docs/catalogue/gardener.rst",
    "docs/catalogue/pacman.rst",
    "examples/custom_environment.py",
    "examples/gardener.py",
    "examples/merchant.py",
    "examples/merchant_policy_fixes.py",
    "tests/examples/test_merchant_policy_fixes.py",
    "examples/pacman.py",
    "examples/play_pacman.py",
    "tools/pacman_reference.py",
    "tools/pacman_transition_parity.py",
    "tools/pacman_benchmark.py",
    "examples/pacman_often.py",
    "tests/examples/test_pacman_often.py",
    "tests/examples/test_sb3_examples.py",
    "examples/pacman_images.py",
    "examples/taxi.py",
    "examples/taxi_policy_fixes.py",
    "tests/examples/test_taxi_policy_fixes.py",
    "experiments/__init__.py",
    "experiments/execution.py",
    "experiments/pacman_bolts.py",
    "experiments/parallel.py",
    "tests/experiments/test_pacman_bolts.py",
    "tests/experiments/test_parallel.py",
    "experiments/benchmarks.py",
    "experiments/report.py",
    "tests/experiments/test_reports.py",
    "experiments/policy_fixes.py",
    "experiments/plot_learning_curves.py",
    "experiments/run.py",
    "experiments/results.py",
    "experiments/paper.py",
    "experiments/generate_paper_tables.py",
    "experiments/validate_learning.py",
    "experiments/specifications.py",
    "tests/conftest.py",
    "tests/docs/test_documentation.py",
    "tests/docs/test_provenance.py",
    "tests/package_smoke.py",
    "tests/test_public_api.py",
    "tests/typing/contracts.py",
    "src/npc_gym/envs/pacman/state.py",
    "src/npc_gym/automata/__init__.py",
    "src/npc_gym/automata/compiled.py",
    "src/npc_gym/monitors/automaton.py",
    "tests/automata/test_compiled.py",
    "tests/monitors/test_automaton.py",
    "src/npc_gym/monitors/regex/__init__.py",
    "src/npc_gym/monitors/regex/compiler.py",
    "tests/regex/test_regex_compiler.py",
    "tests/regex/test_monitors.py",
    "tests/regex/test_imports.py",
    "tests/ltlf/test_regex_parity.py",
    "src/npc_gym/envs/gardener/__init__.py",
    "src/npc_gym/envs/merchant/__init__.py",
    "tests/algorithms/test_tabular_q_learning.py",
    "tests/automata/test_pacman_dfas.py",
    "tests/envs/test_authority_state.py",
    "tests/envs/test_gardener.py",
    "tests/envs/test_merchant.py",
    "tests/envs/test_pacman.py",
    "tests/envs/test_storm_taxi.py",
    "tests/envs/test_storm_characterization.py",
    "tests/storm_taxi_characterization.py",
    "tests/envs/test_composed_storm_taxi.py",
    "tests/envs/test_composed_storm_rendering.py",
    "src/npc_gym/envs/taxi/storm_taxi.py",
    "src/npc_gym/envs/taxi/_taxi_backend.py",
    "src/npc_gym/envs/taxi/_storm_weather.py",
    "src/npc_gym/envs/taxi/_storm_rendering.py",
    "tests/envs/test_taxi.py",
    "tests/envs/test_taxi_labels.py",
    "tests/evaluation/test_evaluation.py",
    "tests/evaluation/test_plot_learning_curves.py",
    "tests/evaluation/test_terminal_transitions.py",
    "tests/evaluation/test_writers.py",
    "tests/examples/test_examples.py",
    "tests/examples/test_pacman_images.py",
    "tests/examples/test_custom_environment.py",
    "tests/examples/test_conceptual_examples.py",
    "tests/experiments/test_execution.py",
    "tests/experiments/test_policy_fixes.py",
    "tests/experiments/test_validate_learning.py",
    "tests/experiments/test_specifications.py",
    "tests/integrations/test_optuna.py",
    "tests/integrations/test_dqn_often.py",
    "src/npc_gym/integrations/sb3/often.py",
    "tests/integrations/test_optional_imports.py",
    "tests/integrations/test_sb3.py",
    "tests/integrations/test_sb3_isolation.py",
    "tests/monitors/test_contracts.py",
    "tests/monitors/test_builtin_boundaries.py",
    "tests/monitors/episode_counts.json",
    "src/npc_gym/monitors/events.py",
    "src/npc_gym/monitors/builtins.py",
    "src/npc_gym/monitors/bindings.py",
    "tests/wrappers/test_labeling.py",
    "tests/wrappers/test_monitor_wrapper.py",
    "src/npc_gym/wrappers/monitoring.py",
    "src/npc_gym/bolts.py",
    "src/npc_gym/wrappers/restraining_bolts.py",
    "tests/wrappers/test_restraining_bolts.py",
    "tests/ltlf/test_bolt_factories.py",
    "src/npc_gym/monitors/collection.py",
}

LTLF_MODULES = {"__init__", "compiler", "runtime"}
REQUIRED_SDIST_PATHS.update(f"src/npc_gym/monitors/ltlf/{module}.py" for module in LTLF_MODULES)
REQUIRED_SDIST_PATHS.update(
    f"tests/ltlf/{module}.py" for module in ("conftest", "test_runtime", "test_compiler", "test_integration")
)

RETIRED_SDIST_PATHS = {
    "experiments/pacman_calibration.py",
    "tests/experiments/test_pacman_calibration.py",
    "IMPLEMENTATION_PLAN.md",
    "src/npc_gym/envs/taxi/taxi.py",
    "src/npc_gym/envs/taxi/kernel.py",
    "STORM_TAXI_IMPLEMENTATION_PLAN.md",
    "examples/pacman_dqn.py",
    "src/npc_gym/monitors/_collection.py",
    "tests/monitors/count_fingerprints.json",
    "docs/labels_and_monitors.rst",
    "docs/automaton_monitors.rst",
    "src/npc_gym/monitors/ltlf/specifications.py",
    "src/npc_gym/monitors/ltlf/builtins.py",
    "src/npc_gym/monitors/ltlf/bindings.py",
    "src/npc_gym/monitors/ltlf/automaton.py",
    "experiments/train_gardener.py",
    "experiments/train_merchant.py",
    "experiments/train_pacman.py",
    "experiments/train_taxi.py",
    "src/npc_gym/algorithms/qlearning.py",
    "src/npc_gym/evaluation/csvwriter.py",
    "src/npc_gym/evaluation/stats.py",
    "src/npc_gym/monitors/monitor.py",
    "tests/algorithms/test_qlearning.py",
    "tests/utils/test_stats.py",
}

WHEEL_SMOKE = """
import importlib.abc
import importlib.util
import sys

class BlockCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "clingo":
            raise ModuleNotFoundError(fullname, name="clingo")
        if fullname.split(".")[0] in {"ltlf2dfa", "lark", "sympy"}:
            raise AssertionError("Core import required an optional compiler: " + fullname)

sys.meta_path.insert(0, BlockCompiler())
from pathlib import Path

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))

import gymnasium as gym
import npc_gym
from npc_gym.algorithms import TabularQLearning
from npc_gym.envs import TaxiAuthorityState
from npc_gym.evaluation import evaluate
import npc_gym.integrations.optuna
import npc_gym.integrations.sb3
from npc_gym.automata import CompiledDFA
from npc_gym.monitors import AutomatonMonitor, SimpleMonitor, MonitorInput
from npc_gym.policy_fixes import ASPPlanner, PlanningProblem, Objective
assert "clingo" not in sys.modules
assert PlanningProblem("candidate(0,0).").horizon == 1
assert PlanningProblem.from_parts(static="candidate(0,0).").horizon == 1
try:
    ASPPlanner()
except ImportError as error:
    assert "npc-gym[asp]" in str(error)
else:
    raise AssertionError("Clingo should be unavailable to this core-only check")
assert not any(name.startswith("npc_gym.monitors.ltlf") for name in sys.modules)
definition = CompiledDFA((), 0, frozenset({0}), {0: [("", 0)]})
first, second = [
    AutomatonMonitor(definition, propositions={}, reporting="prefix")
    for _ in range(2)
]
assert first.reset(MonitorInput(frozenset())) == True
assert first.update(MonitorInput(frozenset())) == True
assert second.count == 0
from npc_gym.monitors.regex import compile_regex, from_regex
regex = compile_regex(".*[a][b]")
assert regex.atoms == ("a", "b")
monitor = from_regex(
    ".*[a][b]", reporting="prefix", consume_initial=False,
    propositions={"a": lambda i: "a" in i.labels, "b": lambda i: "b" in i.labels},
)
assert monitor.reset(MonitorInput(frozenset())) is False
assert monitor.update(MonitorInput(frozenset({"a", "b"}))) is False
assert monitor.update(MonitorInput(frozenset({"a", "b"}))) == True
assert not any(name.startswith("npc_gym.monitors.ltlf") for name in sys.modules)
from npc_gym.bolts import make_regex_bolt
from npc_gym.wrappers import RestrainingBoltWrapper
bolt_env = RestrainingBoltWrapper(
    gym.make("npc_gym/StormTaxi-v0", max_episode_steps=2),
    bolts={"bonus": make_regex_bolt(".*", propositions={}, reward=1)},
)
try:
    observation, info = bolt_env.reset(seed=0)
    assert bolt_env.observation_space.contains(observation)
    assert info["restraining_bolts"]["reward_adjustments"] == {"bonus": 0}
    observation, reward, _, _, info = bolt_env.step(0)
    assert bolt_env.observation_space.contains(observation)
    assert reward == info["restraining_bolts"]["wrapped_reward"] + 1
finally:
    bolt_env.close()
assert not any(name.startswith("npc_gym.monitors.ltlf") for name in sys.modules)
from npc_gym.monitors.ltlf import from_ltlf
from npc_gym.monitors.merchant_monitors import make_merchant_monitor

builtin = make_merchant_monitor("merchant/danger-v0")
assert builtin.reset(MonitorInput(frozenset({"atDanger"}))) == True
from npc_gym.wrappers import LabelingWrapper, MonitorWrapper

assert Path(npc_gym.__file__).resolve().is_relative_to(target)
assert npc_gym.integrations.optuna.__name__.endswith("optuna")
assert npc_gym.integrations.sb3.__name__.endswith("sb3")

if importlib.util.find_spec("stable_baselines3") is None:
    try:
        from npc_gym.integrations.sb3 import SB3EvaluationCallback
    except ImportError as error:
        assert "npc-gym[sb3]" in str(error)
    else:
        raise AssertionError("SB3 adapter loaded without its optional extra")

from npc_gym.integrations.optuna import OptunaReporter

class Trial:
    def report(self, value, step):
        pass

    def should_prune(self):
        return False

if importlib.util.find_spec("optuna") is None:
    try:
        OptunaReporter(Trial())
    except ImportError as error:
        assert "npc-gym[optuna]" in str(error)
    else:
        raise AssertionError("Optuna adapter loaded without its optional extra")

from npc_gym.envs import StormTaxiEnv

with StormTaxiEnv() as storm:
    assert storm.reset(seed=7)[0] == 217552
    assert storm.step(0)[:4] == (288304, -1, False, False)
assert "pygame" not in sys.modules
assert "npc_gym/Taxi-v1" not in gym.registry
with gym.make("npc_gym/StormTaxi-v0") as registered:
    assert type(registered.unwrapped) is StormTaxiEnv
    assert registered.spec.max_episode_steps == 50

for environment_id in (
    "npc_gym/StormTaxi-v0",
    "npc_gym/Merchant-v2",
    "npc_gym/Gardener-v0",
    "npc_gym/Pacman-v1",
    "npc_gym/PacmanMedium-v1",
    "npc_gym/PacmanLarge-v1",
):
    env = gym.make(environment_id)
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
    finally:
        env.close()

for name, retired, current in (("Merchant", 0, 2), ("Merchant", 1, 2)):
    try:
        gym.make(f"npc_gym/{name}-v{retired}")
    except gym.error.DeprecatedEnv as error:
        assert f"{name}-v{current}" in str(error)
    else:
        raise AssertionError(f"retired {name}-v{retired} unexpectedly constructed")

env = gym.make("npc_gym/Merchant-v2", layout="cycle")
try:
    env.reset(seed=7)
    for action in [1] * 2 + [2] * 4 + [0] * 2 + [2] * 3:
        observation, _, _, _, info = env.step(action)
    assert observation[:2] == (8, 3)
    assert info["action_mask"].tolist() == [0, 0, 0, 1, 0, 0, 0]
    model = TabularQLearning(env, seed=7, use_action_mask=True, log_interval=None)
    observation, reward, terminated, truncated, _ = env.step(model.predict(observation, info))
    assert observation[:2] == (7, 3)
    assert (reward, terminated, truncated) == (0, False, False)
finally:
    env.close()

env = gym.make("npc_gym/StormTaxi-v0")
try:
    summary = evaluate(env, lambda observation, info: 0, seed=0)
    assert len(summary.episodes) == 1
    assert summary.episodes[0].metrics == {"success": 0}
finally:
    env.close()

class ExternalMonitor(SimpleMonitor):
    def __init__(self):
        super().__init__()
        self.seen = False
    def reset_history(self):
        self.seen = False
    def detect(self, input):
        occurred = not self.seen and "external" in input.labels
        self.seen = self.seen or occurred
        return occurred

env = LabelingWrapper(
    gym.make("npc_gym/StormTaxi-v0"),
    lambda transition: frozenset({"external"}),
    mode="replace",
)
try:
    _, info = env.reset(seed=0)
    assert info["labels"] == frozenset({"external"})
    assert isinstance(env.unwrapped.labeling_state(), TaxiAuthorityState)
    summary = evaluate(env, lambda observation, info: 0, monitors={"external/norm-v0": ExternalMonitor}, seed=0)
    assert summary.episodes[0].monitor_counts == {"external/norm-v0": {"count": 1}}
    env = MonitorWrapper(env, monitors={"external/norm-v0": ExternalMonitor})
    recorded = evaluate(env, lambda observation, info: 0, monitor_source="wrapper", seed=0)
    assert recorded == summary
finally:
    env.close()

env = gym.make("npc_gym/StormTaxi-v0")
try:
    model = TabularQLearning(env, seed=0, use_action_mask=True)
    model.learn(2)
    model_path = target.parent / "tabular-model.zip"
    model.save(model_path)
    loaded = TabularQLearning.load(model_path, env=env)
    assert loaded.num_timesteps == 2
finally:
    env.close()
"""


STORM_RENDER_SMOKE = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import npc_gym
from npc_gym.envs import StormTaxiEnv
assert Path(npc_gym.__file__).resolve().is_relative_to(Path(sys.argv[1]))
with StormTaxiEnv(render_mode="rgb_array") as first, StormTaxiEnv(render_mode="rgb_array") as second:
    first.reset(seed=7)
    second.reset(seed=2)
    assert first.render().shape == second.render().shape == (350, 550, 3)
    first.close()
    second.step(6)
    assert second.render().shape == (350, 550, 3)
"""


LTLF_WHEEL_SMOKE = """
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import npc_gym.monitors.ltlf as ltlf
from npc_gym.monitors import AutomatonMonitor, MonitorInput

assert Path(ltlf.__file__).resolve().is_relative_to(target)
monitor = ltlf.from_ltlf(
    "F(a & !(X true))",
    propositions={"a": lambda input: "a" in input.labels}, reporting="prefix",
)
assert type(monitor) is AutomatonMonitor
assert monitor.reset(MonitorInput(frozenset({"a"}))) == True
assert monitor.update(MonitorInput(frozenset())) is False
assert monitor.count == 1
from npc_gym.bolts import make_builtin_bolts, make_ltlf_bolt
bolts = make_builtin_bolts("pacman/hungry-vegan-penalty-v1", reward=-1)
assert len(bolts) == 4
assert all(spec.reward == -1 for spec in bolts.values())
bolt = make_ltlf_bolt("a", propositions={"a": lambda value: True}, reward=-1)
assert bolt.definition.atoms == ("a",)

"""


ASP_WHEEL_SMOKE = """
import sys
from pathlib import Path
from importlib.resources import files

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import npc_gym.policy_fixes as fixes
assert Path(fixes.__file__).resolve().is_relative_to(target)
assert files(fixes).joinpath("planning.lp").is_file()
problem = fixes.PlanningProblem(
    'candidate(0,0..1). cost("violations",1,harm) :- action(0,0).'
)
decision = fixes.ASPPlanner().solve(problem, {0: 10.0, 1: 0.0})
assert decision.action == 1 and decision.optimal
assert decision.costs == {"violations": 0, "policy": 1}
rules = 'candidate(0,0..1). cost("violations",1,harm) :- action(0,A), harmful(A).'
for harmful in (0, 1, 0):
    problem = fixes.PlanningProblem.from_parts(static=rules, dynamic=f"harmful({harmful}).")
    assert fixes.ASPPlanner().solve(problem, {0: 10.0, 1: 0.0}).action == 1 - harmful
planner = fixes.ASPPlanner()
for harmful in (0, 1, 0):
    problem = fixes.PlanningProblem.from_externals(
        static=rules + " #external harmful(0..1).", true_atoms=[f"harmful({harmful})"]
    )
    assert planner.solve(problem, {0: 10.0, 1: 0.0}).action == 1 - harmful
from npc_gym.envs import PacmanEnv
assert files(fixes).joinpath("pacman.lp").is_file()
assert files(fixes).joinpath("pacman_trapped.lp").is_file()
env = PacmanEnv(features="essential")
try:
    env.reset(seed=17)
    model = fixes.PacmanModel(env.labeling_state(), horizon=2)
    for _ in range(3):
        decision = fixes.ASPPlanner().solve(model.problem(env.labeling_state()), dict.fromkeys(range(5), 0.0))
        assert decision.optimal and len(decision.plan) == 2
        _, _, terminated, truncated, _ = env.step(decision.action)
        if terminated or truncated:
            break
finally:
    env.close()
from npc_gym.envs import GardenerEnv
from npc_gym.monitors.contracts import MonitorInput
from npc_gym.monitors.gardener_monitors import NO_COLLECT_NORM_ID
assert files(fixes).joinpath("gardener.lp").is_file()
env = GardenerEnv(size=5)
try:
    env.reset(seed=17)
    model = fixes.GardenerModel(env.labeling_state(), norm_id=NO_COLLECT_NORM_ID, horizon=2)
    planner = fixes.ASPPlanner(objectives=model.objectives)
    for t in range(3):
        decision = planner.solve(
            model.problem(env.labeling_state(), remaining_steps=3-t),
            dict.fromkeys(range(5), 0.0),
        )
        assert decision.optimal
        _, _, terminated, truncated, info = env.step(decision.action)
        if terminated or truncated:
            break
finally:
    env.close()
from npc_gym.envs import StormTaxiEnv, MerchantEnv
from npc_gym.monitors import MonitorInput
from npc_gym.wrappers.pacman_wrappers import TrappedObservation
with TrappedObservation(PacmanEnv(features="complete")) as env:
    _, info = env.reset(seed=17)
    assert env.observation_space.shape == (64,)
    model = fixes.PacmanTrappedModel(MonitorInput(info["labels"]))
    planner = fixes.ASPPlanner()
    for _ in range(3):
        decision = planner.solve(model.problem(env.unwrapped.labeling_state()), dict.fromkeys(range(5), 0.0))
        assert decision.optimal
        _, _, terminated, truncated, info = env.step(decision.action)
        model.advance(MonitorInput(info["labels"], terminated, truncated))
        if terminated or truncated:
            break
for env, model_type, resource in (
    (StormTaxiEnv(), fixes.TaxiModel, "taxi.lp"),
    (MerchantEnv(), fixes.MerchantModel, "merchant.lp"),
):
    assert files(fixes).joinpath(resource).is_file()
    try:
        env.reset(seed=17)
        model = model_type()
        for t in range(3):
            decision = fixes.ASPPlanner().solve(
                model.problem(env.labeling_state()) if model_type is fixes.MerchantModel
                else model.problem(),
                dict.fromkeys(range(env.action_space.n), 0.0),
            )
            assert decision.optimal
            _, _, terminated, truncated, info = env.step(decision.action)
            if model_type is not fixes.MerchantModel:
                model.advance(MonitorInput(info["labels"], terminated, truncated or t == 2))
            if terminated or truncated:
                break
    finally:
        env.close()
# An installed trainer must solve, update, save and run inference without Clingo.
import gymnasium as gym
import numpy as np
import tempfile
import importlib.abc
from npc_gym.algorithms import TabularQLearning, TabularOFTEN, TeachingSession
class TeachingEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Discrete(1)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}
    def step(self, action):
        return 0, 1.0, True, False, {}
def session(obs, info):
    return TeachingSession(fixes.ASPPlanner(), lambda obs, info: problem)
for weight, filtering in ((1, True), (0, True), (1, False)):
    ordinary, expert = TeachingEnv(), TeachingEnv()
    try:
        policy = TabularQLearning(ordinary, learning_rate=0.1, seed=7, log_interval=None)
        trainer = TabularOFTEN(policy, expert, ordinary_session=session, expert_session=session,
                              batch_size=2, expert_weight=weight, filter_replay=filtering, seed=7).learn(4)
        assert trainer.stats.total_steps == 4 and trainer.stats.updates > 0
        assert np.any(policy.q_values(0) != 0)
    finally:
        ordinary.close()
        expert.close()
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'clingo', 'torch', 'stable_baselines3'}:
            raise ModuleNotFoundError(fullname, name=fullname)
for name in tuple(sys.modules):
    if name.split('.')[0] == 'clingo':
        del sys.modules[name]
sys.meta_path.insert(0, BlockOptional())
with tempfile.TemporaryDirectory() as directory:
    saved = Path(directory) / 'taught.zip'
    policy.save(saved)
    restored = TabularQLearning.load(saved, env=TeachingEnv())
    assert np.array_equal(restored.q_values(0), policy.q_values(0))
    assert restored.predict(0) == policy.predict(0)
assert "torch" not in sys.modules
assert "stable_baselines3" not in sys.modules
"""


OFTEN_WHEEL_SMOKE = """
import sys
from pathlib import Path
import tempfile
import importlib.abc
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

torch.set_num_threads(1)
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
from npc_gym.integrations.sb3 import DQNOFTEN, SB3Policy
from npc_gym.algorithms import TeachingSession
from npc_gym.policy_fixes import ASPPlanner, PlanningProblem
import npc_gym.integrations.sb3.often as often
assert Path(often.__file__).resolve().is_relative_to(target)
class Env(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Dict({'state': gym.spaces.Box(0, 1, (1,), dtype=np.float32)})
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {'state': np.zeros(1, dtype=np.float32)}, {}
    def step(self, action):
        return {'state': np.ones(1, dtype=np.float32)}, 1.0, False, True, {}
problem = PlanningProblem('candidate(0,0..1). cost("violations",1,harm) :- action(0,0).')
def session(obs, info):
    return TeachingSession(ASPPlanner(), lambda obs, info: problem)
for weight, filtering in ((1, True), (0, True), (1, False)):
    ordinary, expert = Env(), Env()
    try:
        model = DQN('MultiInputPolicy', ordinary, device='cpu', seed=7, buffer_size=8,
                    policy_kwargs={'net_arch': [8]}, learning_rate=0.01, batch_size=2,
                    train_freq=1, learning_starts=0, exploration_initial_eps=0, exploration_final_eps=0)
        teacher = DQNOFTEN(model, ordinary, expert, ordinary_session=session, expert_session=session,
                          seed=7, initial_expert_weight=weight, final_expert_weight=weight,
                          filter_replay=filtering)
        observation = {'state': np.zeros(1, dtype=np.float32)}
        with torch.no_grad():
            model.q_net.q_net[-1].bias[1] = 10
        before = teacher.q_values(observation)
        teacher.learn(4)
        assert teacher.stats.total_steps == 4 and teacher.stats.updates > 0
        assert not np.array_equal(before, teacher.q_values(observation))
    finally:
        ordinary.close()
        expert.close()
class BlockASP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'clingo':
            raise ModuleNotFoundError(fullname, name=fullname)
for name in tuple(sys.modules):
    if name.split('.')[0] == 'clingo':
        del sys.modules[name]
sys.meta_path.insert(0, BlockASP())
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'taught.zip'
    model.save(path)
    loaded = DQN.load(path, device='cpu')
    assert SB3Policy(loaded)(observation, {}) == SB3Policy(model)(observation, {})
    for name, value in model.q_net.state_dict().items():
        assert torch.equal(value, loaded.q_net.state_dict()[name])
"""


def archive_payload_paths(names: list[str]) -> set[str]:
    """Strip the versioned top-level directory from sdist member names."""
    return {name.split("/", 1)[1] for name in names if "/" in name}


def check_sdist(sdist: Path) -> None:
    """Check that the source archive is complete but excludes model artifacts."""
    with tarfile.open(sdist, "r:gz") as archive:
        paths = archive_payload_paths(archive.getnames())

    missing = REQUIRED_SDIST_PATHS - paths
    assert not missing, f"sdist is missing required files: {sorted(missing)}"
    assert not any(path.startswith("src/npc_gym/envs/taxi/img/") for path in paths)
    assert not any("/pacman/_engine" in path for path in paths)
    assert "envs.png" not in paths
    assert {path.rsplit("/", 1)[-1] for path in paths if "/pacman/layouts/" in path} == {
        "small.lay",
        "medium.lay",
        "large.lay",
    }
    present = RETIRED_SDIST_PATHS & paths
    assert not present, f"sdist contains retired files: {sorted(present)}"
    assert not any(path.startswith("experiments/models/") for path in paths), "sdist contains model artifacts"
    assert not any(path.startswith("experiments/output/") for path in paths), "sdist contains local run artifacts"
    assert "experiments/layouts/smallClassic.lay" not in paths, "sdist contains temporary Berkeley research layout"
    assert not any("__pycache__" in path or path.endswith((".pyc", ".pyo")) for path in paths)


def check_wheel(
    wheel: Path,
    workdir: Path,
    *,
    ltlf: bool = False,
    asp: bool = False,
    often: bool = False,
    image_example: bytes | None = None,
    often_examples: dict[str, bytes] | None = None,
    policy_examples: dict[str, bytes] | None = None,
) -> None:
    """Install the wheel and smoke-test its public environments outside the checkout."""
    with zipfile.ZipFile(wheel) as archive:
        paths = set(archive.namelist())
        metadata_path = next(path for path in paths if path.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        clingo_requirements = [value for value in metadata.get_all("Requires-Dist", []) if value.startswith("clingo")]
        assert len(clingo_requirements) == 1 and 'extra == "asp"' in clingo_requirements[0]
    assert any(path.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for path in paths)
    assert {f"npc_gym/monitors/ltlf/{module}.py" for module in LTLF_MODULES} <= paths
    assert not {path.removeprefix("src/") for path in RETIRED_SDIST_PATHS if path.startswith("src/")} & paths
    assert "npc_gym/py.typed" in paths
    assert not any(path.endswith("often_update.json") for path in paths)
    assert {
        "npc_gym/policy_fixes/__init__.py",
        "npc_gym/policy_fixes/core.py",
        "npc_gym/policy_fixes/_solver.py",
        "npc_gym/policy_fixes/planning.lp",
        "npc_gym/policy_fixes/taxi.py",
        "npc_gym/policy_fixes/taxi.lp",
        "npc_gym/policy_fixes/merchant.py",
        "npc_gym/policy_fixes/merchant.lp",
        "npc_gym/policy_fixes/gardener.py",
        "npc_gym/policy_fixes/gardener.lp",
        "npc_gym/policy_fixes/pacman.py",
        "npc_gym/policy_fixes/pacman.lp",
        "npc_gym/policy_fixes/pacman_trapped.lp",
        "npc_gym/wrappers/pacman_wrappers.py",
    } <= paths
    assert "npc_gym/automata/__init__.py" in paths
    assert "npc_gym/automata/compiled.py" in paths
    assert "npc_gym/monitors/automaton.py" in paths
    assert "npc_gym/monitors/regex/__init__.py" in paths
    assert "npc_gym/monitors/regex/compiler.py" in paths
    assert "npc_gym/envs/gardener/__init__.py" in paths
    assert "npc_gym/envs/merchant/__init__.py" in paths
    assert "npc_gym/envs/merchant/merchant_layouts/basic.txt" in paths
    assert "npc_gym/envs/pacman/layouts/small.lay" in paths
    assert "npc_gym/envs/taxi/storm_taxi.py" in paths
    assert "npc_gym/envs/taxi/_taxi_backend.py" in paths
    assert "npc_gym/envs/taxi/taxi.py" not in paths
    assert "npc_gym/envs/taxi/kernel.py" not in paths
    assert not any(path.startswith("npc_gym/envs/taxi/img/") for path in paths)
    assert "npc_gym/integrations/sb3/often.py" in paths
    assert "npc_gym/integrations/optuna/reporter.py" in paths
    assert "npc_gym/integrations/sb3/callback.py" in paths
    assert "npc_gym/algorithms/tabular_q_learning.py" in paths
    assert "npc_gym/algorithms/tabular_often.py" in paths
    assert "npc_gym/wrappers/labeling.py" in paths
    assert "npc_gym/wrappers/monitoring.py" in paths
    assert "npc_gym/bolts.py" in paths
    assert "npc_gym/wrappers/restraining_bolts.py" in paths
    assert "npc_gym/monitors/collection.py" in paths
    assert "npc_gym/envs/pacman/state.py" in paths
    assert not any("/pacman/_engine" in path for path in paths)
    assert {path.rsplit("/", 1)[-1] for path in paths if "/pacman/layouts/" in path} == {
        "small.lay",
        "medium.lay",
        "large.lay",
    }
    assert "npc_gym/envs/pacman/_observation_types.py" not in paths
    assert "npc_gym/algorithms/qlearning.py" not in paths
    assert "npc_gym/evaluation/csvwriter.py" not in paths
    assert "npc_gym/evaluation/stats.py" not in paths
    assert "npc_gym/monitors/monitor.py" not in paths
    assert not any(path.startswith(("docs/", "examples/", "experiments/", "tests/")) for path in paths)

    target = workdir / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        check=True,
    )
    subprocess.run([sys.executable, "-c", WHEEL_SMOKE, str(target)], cwd=workdir, check=True)
    subprocess.run(
        [sys.executable, "-c", STORM_RENDER_SMOKE, str(target)],
        cwd=workdir,
        check=True,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
    )
    if ltlf:
        subprocess.run([sys.executable, "-c", LTLF_WHEEL_SMOKE, str(target)], cwd=workdir, check=True)
    if asp:
        subprocess.run([sys.executable, "-c", ASP_WHEEL_SMOKE, str(target)], cwd=workdir, check=True)
    if often:
        subprocess.run([sys.executable, "-c", OFTEN_WHEEL_SMOKE, str(target)], cwd=workdir, check=True)
    for name, source in (policy_examples or {}).items():
        script = workdir / f"{name}.py"
        script.write_bytes(source)
        command = (
            "import sys, runpy; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "import npc_gym; assert Path(npc_gym.__file__).is_relative_to(Path(sys.argv[1])); "
            "example = runpy.run_path(sys.argv[2]); summaries = example['main'](); "
            "assert set(summaries) == {'base', 'fixed'}; "
            "assert all(len(summary.episodes) == 2 for summary in summaries.values())"
        )
        subprocess.run(
            [sys.executable, "-c", command, str(target), str(script)],
            cwd=workdir,
            check=True,
            env={
                **{key: value for key, value in os.environ.items() if not key.startswith("NPC_GYM_")},
                "NPC_GYM_TRAINING_STEPS": "32",
                "NPC_GYM_EVALUATION_EPISODES": "2",
                "NPC_GYM_MAX_EPISODE_STEPS": "8",
                "NPC_GYM_OUTPUT_DIR": str(workdir / f"{name}-results"),
                "NPC_GYM_SEED": "0",
            },
        )
    if often_examples:
        for name, source in often_examples.items():
            (workdir / f"{name}.py").write_bytes(source)
        command = (
            "import sys, runpy; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "import npc_gym; assert Path(npc_gym.__file__).is_relative_to(Path(sys.argv[1])); "
            "runpy.run_path('pacman.py', run_name='__main__'); "
            "example = runpy.run_path('pacman_often.py'); summaries = example['main'](); "
            "assert tuple(summaries) == ('base', 'often', 'continued'); "
            "assert all(len(summary.episodes) == 2 for summary in summaries.values())"
        )
        subprocess.run(
            [sys.executable, "-c", command, str(target)],
            cwd=workdir,
            check=True,
            env={
                **{key: value for key, value in os.environ.items() if not key.startswith("NPC_GYM_")},
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "NPC_GYM_TRAINING_STEPS": "8",
                "NPC_GYM_TEACHING_STEPS": "4",
                "NPC_GYM_EVALUATION_EPISODES": "2",
                "NPC_GYM_MAX_EPISODE_STEPS": "4",
                "NPC_GYM_PLANNING_HORIZON": "1",
                "NPC_GYM_SEED": "0",
                "NPC_GYM_MODEL_PATH": str(workdir / "pacman-base.zip"),
            },
        )
    if image_example is not None:
        script = workdir / "pacman_images.py"
        script.write_bytes(image_example)
        command = (
            "import sys, runpy; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "import npc_gym; assert Path(npc_gym.__file__).is_relative_to(Path(sys.argv[1])); "
            "runpy.run_path(sys.argv[2], run_name='__main__')"
        )
        subprocess.run(
            [sys.executable, "-c", command, str(target), str(script)],
            cwd=workdir,
            check=True,
            env={
                **os.environ,
                "SDL_VIDEODRIVER": "dummy",
                "SDL_AUDIODRIVER": "dummy",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NPC_GYM_DEVICE": "cpu",
                "NPC_GYM_TRAINING_STEPS": "16",
                "NPC_GYM_EVALUATION_EPISODES": "2",
                "NPC_GYM_MAX_EPISODE_STEPS": "40",
                "NPC_GYM_DQN_LEARNING_STARTS": "4",
                "NPC_GYM_SEED": "0",
            },
        )


def main() -> None:
    dist_dir = Path(sys.argv[1]).resolve()
    sdists = list(dist_dir.glob("*.tar.gz"))
    wheels = list(dist_dir.glob("*.whl"))
    assert len(sdists) == 1, f"expected one sdist, found {sdists}"
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"

    check_sdist(sdists[0])
    policy_examples = {}
    if "--asp" in sys.argv[2:]:
        with tarfile.open(sdists[0], "r:gz") as archive:
            for name in ("taxi_policy_fixes", "merchant_policy_fixes"):
                member = next(path for path in archive.getnames() if path.endswith(f"/examples/{name}.py"))
                source = archive.extractfile(member)
                assert source is not None
                policy_examples[name] = source.read()
    often_examples = {}
    if "--often" in sys.argv[2:]:
        with tarfile.open(sdists[0], "r:gz") as archive:
            for name in ("pacman", "pacman_often"):
                member = next(path for path in archive.getnames() if path.endswith(f"/examples/{name}.py"))
                source = archive.extractfile(member)
                assert source is not None
                often_examples[name] = source.read()
    image_example = None
    if "--images" in sys.argv[2:]:
        with tarfile.open(sdists[0], "r:gz") as archive:
            member = next(name for name in archive.getnames() if name.endswith("/examples/pacman_images.py"))
            source = archive.extractfile(member)
            assert source is not None
            image_example = source.read()
    with tempfile.TemporaryDirectory(prefix="npc-gym-wheel-") as directory:
        check_wheel(
            wheels[0],
            Path(directory),
            ltlf="--ltlf" in sys.argv[2:],
            asp="--asp" in sys.argv[2:],
            often="--often" in sys.argv[2:],
            image_example=image_example,
            often_examples=often_examples,
            policy_examples=policy_examples,
        )


if __name__ == "__main__":
    main()
