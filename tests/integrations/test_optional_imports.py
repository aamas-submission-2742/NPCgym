import importlib
import subprocess
import sys

import pytest

LAZY_IMPORT_CHECK = """
import sys

import npc_gym.integrations.optuna
import npc_gym.integrations.sb3

assert "stable_baselines3" not in sys.modules
assert "optuna" not in sys.modules
"""


def test_integration_packages_do_not_eagerly_import_optional_dependencies():
    subprocess.run([sys.executable, "-c", LAZY_IMPORT_CHECK], check=True)


def test_environments_explain_missing_render_extra_and_remain_usable_headlessly():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class WithoutPygame(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pygame" or fullname.startswith("pygame."):
            raise ModuleNotFoundError("No module named 'pygame'", name="pygame")

sys.meta_path.insert(0, WithoutPygame())
from gymnasium.error import DependencyNotInstalled
from npc_gym.envs import StormTaxiEnv, MerchantEnv, GardenerEnv, PacmanEnv

for constructor in (StormTaxiEnv, MerchantEnv, GardenerEnv, PacmanEnv):
    peer = constructor()
    try:
        peer.reset(seed=0)
        for mode in constructor.metadata["render_modes"]:
            if mode == "ansi":
                continue
            rendered = None
            try:
                rendered = constructor(render_mode=mode)
                rendered.reset(seed=0)
                rendered.render()
            except DependencyNotInstalled as error:
                assert "npc-gym[render]" in str(error), str(error)
            else:
                raise AssertionError(f"{constructor.__name__} {mode} loaded without pygame")
            finally:
                if rendered is not None:
                    rendered.close()
            observation, *_ = peer.step(0)
            assert peer.observation_space.contains(observation)
    finally:
        peer.close()
assert "pygame" not in sys.modules
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("name", ["SB3EvaluationCallback", "DQNOFTEN"])
def test_sb3_public_names_explain_how_to_install_the_missing_extra(monkeypatch, name):
    sb3 = importlib.import_module("npc_gym.integrations.sb3")
    monkeypatch.delitem(sb3.__dict__, name, raising=False)
    monkeypatch.setattr(sb3, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match=r"npc-gym\[sb3\]"):
        sb3.__getattr__(name)
