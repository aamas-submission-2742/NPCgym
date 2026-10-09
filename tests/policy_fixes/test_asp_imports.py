"""Core and data-contract imports work when the ASP extra is absent."""

import subprocess
import sys


def test_optional_imports_and_missing_extra():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class WithoutOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"clingo", "torch", "stable_baselines3"}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, WithoutOptional())
import npc_gym
from npc_gym.algorithms import TabularQLearning, TabularOFTEN, TeachingSession
from npc_gym.monitors.gardener_monitors import RescueMonitor
from npc_gym.policy_fixes import ASPPlanner, GardenerModel, MerchantModel, Objective, PacmanModel, PlanningProblem, TaxiModel
from npc_gym.policy_fixes import PacmanTrappedModel
from npc_gym.monitors import MonitorInput
from npc_gym.envs import GardenerEnv, MerchantEnv, PacmanEnv, StormTaxiEnv
assert "clingo" not in sys.modules
assert "torch" not in sys.modules
assert Objective().weight == 1
assert PlanningProblem("candidate(0,0).").horizon == 1
assert PlanningProblem.from_parts(static="candidate(T,0) :- time(T). marker.", dynamic="current.").horizon == 1
assert PlanningProblem.from_externals(static="#external x.", true_atoms=["x"]).horizon == 1
env = PacmanEnv(features="essential")
try:
    _, info = env.reset(seed=7)
    assert PacmanModel(env.labeling_state()).problem(env.labeling_state()).horizon == 1
    assert PacmanTrappedModel(MonitorInput(info["labels"])).problem(env.labeling_state()).horizon == 1
finally:
    env.close()
env = GardenerEnv(size=5)
try:
    env.reset(seed=7)
    from npc_gym.wrappers.gardener_wrappers import StateFeatureObsWrapper
    wrapped = StateFeatureObsWrapper(env, include_frogs=True)
    observation, _ = wrapped.reset(seed=7)
    assert observation.shape == (27,) and wrapped.observation_space.contains(observation)
    model = GardenerModel(env.labeling_state())
    assert model.problem(env.labeling_state()).horizon == 1
finally:
    env.close()
for env, model_type in ((StormTaxiEnv(), TaxiModel), (MerchantEnv(), MerchantModel)):
    try:
        env.reset(seed=7)
        model = model_type()
        problem = model.problem(env.labeling_state()) if model_type is MerchantModel else model.problem()
        assert problem.horizon == 1
    finally:
        env.close()
try:
    ASPPlanner()
except ImportError as error:
    assert "npc-gym[asp]" in str(error)
else:
    raise AssertionError("Planner constructed without Clingo")
""",
        ],
        check=True,
    )
