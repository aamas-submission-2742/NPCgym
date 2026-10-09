"""Static regressions run by Tox's typing gate, without executing environments."""

from collections.abc import Mapping
from typing import assert_type

import gymnasium as gym
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from optuna import Trial
from pandas import DataFrame
from PIL.Image import Image
from pygame import Surface
from stable_baselines3.common.callbacks import BaseCallback
from torch import Tensor

from npc_gym.automata import CompiledDFA
from npc_gym.bolts import BoltSpec, make_builtin_bolts, make_ltlf_bolt, make_regex_bolt
from npc_gym.envs import GardenerEnv, MerchantEnv, PacmanEnv, StormTaxiEnv
from npc_gym.envs.pacman.labels import PacmanAuthorityState
from npc_gym.envs.pacman.observations import Observation
from npc_gym.envs.pacman.state import PacmanSnapshot
from npc_gym.monitors import AutomatonMonitor, MonitorInput, Proposition, Reporting
from npc_gym.monitors.ltlf import LTLfCompiler, from_ltlf
from npc_gym.monitors.regex import RegexLimits, compile_regex, from_regex
from npc_gym.policy_fixes import PlanningProblem
from npc_gym.wrappers import MonitorWrapper, RestrainingBoltWrapper
from npc_gym.wrappers.gardener_wrappers import IllegalActionPenaltyWrapper, StateFeatureObsWrapper
from npc_gym.wrappers.merchant_wrappers import IgnoreTimeObservation
from npc_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant


def dependency_contracts(
    table: DataFrame,
    axes: Axes,
    surface: Surface,
    image: Image,
    trial: Trial,
    callback: BaseCallback,
    tensor: Tensor,
) -> None:
    # Member results must have real types even if an imported class becomes Any.
    assert_type(table.shape, tuple[int, int])
    assert_type(axes.plot([0, 1], [1, 2]), list[Line2D])
    assert_type(surface.get_size(), tuple[int, int])
    assert_type(image.size, tuple[int, int])
    assert_type(trial.number, int)
    assert_type(callback.num_timesteps, int)
    assert_type(tensor.numel(), int)
    assert_type(np.zeros(3, dtype=np.float32).dtype, np.dtype[np.float32])
    assert_type(gym.spaces.Discrete[np.int64](5).sample(), np.int64)


def planning_contracts() -> None:
    assert_type(PlanningProblem.from_parts(static="candidate(0,0)."), PlanningProblem)
    problem = PlanningProblem.from_parts(static="rules. fixed_fact.", dynamic="state.", horizon=3)
    assert_type(problem, PlanningProblem)
    assert_type(problem.program, str)
    assert_type(PlanningProblem.from_externals(static="#external x.", true_atoms=["x"]), PlanningProblem)


def environment_contracts(taxi: StormTaxiEnv, merchant: MerchantEnv, gardener: GardenerEnv, pacman: PacmanEnv) -> None:
    assert_type(taxi.reset()[0], int)
    assert_type(MonitorWrapper(taxi, monitors={}).reset()[0], int)
    assert_type(MonitorWrapper(merchant, monitors={}).step(0)[0], tuple[int, ...])
    assert_type(IgnoreWeatherRelevant(taxi).reset()[0], int)
    assert_type(merchant.reset()[0], tuple[int, ...])
    assert_type(IgnoreTimeObservation(merchant).reset()[0], tuple[int, ...])
    assert_type(gardener.reset()[0], tuple[int, ...])
    assert_type(IllegalActionPenaltyWrapper(gardener).step(0)[0], tuple[int, ...])
    features = StateFeatureObsWrapper(gardener)
    assert_type(features.reset()[0], NDArray[np.float32])
    assert_type(IllegalActionPenaltyWrapper(features).step(0)[0], NDArray[np.float32])
    assert_type(pacman.action_space.sample(), np.int64)
    assert_type(pacman.reset()[0], Observation)
    assert_type(pacman.labeling_state(), PacmanAuthorityState)
    assert_type(pacman.state(), PacmanSnapshot)


def automaton_contracts(compiler: LTLfCompiler, input: MonitorInput) -> None:
    definition = compiler.compile("a")
    assert_type(definition, CompiledDFA)
    predicate: Proposition = lambda value: "a" in value.labels
    monitor = AutomatonMonitor(definition, propositions={"a": predicate}, reporting=Reporting.PREFIX)
    assert_type(monitor.state, int)
    assert_type(monitor.reset(input), bool)
    assert_type(monitor.update(input), bool)
    assert_type(
        from_ltlf("a", propositions={"a": predicate}, reporting="prefix", compiler=compiler),
        AutomatonMonitor,
    )


def regex_contracts() -> None:
    limits = RegexLimits(max_atoms=4)
    assert_type(compile_regex("[a]", limits=limits), CompiledDFA)
    assert_type(
        from_regex(
            "[a]",
            propositions={"a": lambda value: "a" in value.labels},
            reporting="prefix",
            limits=limits,
        ),
        AutomatonMonitor,
    )


def bolt_contracts(env: StormTaxiEnv, compiler: LTLfCompiler) -> None:
    bolt = make_regex_bolt(".*", propositions={}, reward=1)
    assert_type(bolt, BoltSpec)
    assert_type(make_builtin_bolts("pacman/hungry-vegan-penalty-v1", reward=-1), Mapping[str, BoltSpec])
    assert_type(make_ltlf_bolt("true", propositions={}, reward=-1, compiler=compiler), BoltSpec)
    wrapped = RestrainingBoltWrapper(env, bolts={"a": bolt})
    assert_type(wrapped.step(0)[1], float)
    assert_type(wrapped.bolt_metadata["a"].state_indices[0], int)


def storm_taxi_contracts(env: StormTaxiEnv) -> None:
    assert_type(env.reset()[0], int)
    assert_type(env.step(6)[0], int)
    assert_type(env.action_mask(), NDArray[np.int8])
    assert_type(IgnoreWeatherRelevant(env).reset()[0], int)
    assert_type(MonitorWrapper(env, monitors={}).reset()[0], int)
