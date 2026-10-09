import inspect
from importlib.resources import files

import gymnasium as gym
import pytest

import npc_gym
from npc_gym.envs import GardenerEnv, MerchantEnv, PacmanEnv, StormTaxiEnv

EXPECTED_SPECS = {
    "npc_gym/StormTaxi-v0": (50, {}),
    "npc_gym/Merchant-v2": (150, {"layout": "basic"}),
    "npc_gym/Gardener-v0": (1_000, {"size": 15}),
    "npc_gym/Pacman-v1": (300, {"layout": "small", "features": "complete"}),
    "npc_gym/PacmanMedium-v1": (500, {"layout": "medium", "features": "complete"}),
    "npc_gym/PacmanLarge-v1": (800, {"layout": "large", "features": "complete"}),
}


def test_import_registers_namespaced_environments():
    assert npc_gym.__version__ == "0.1.0"
    for environment_id, (time_limit, kwargs) in EXPECTED_SPECS.items():
        spec = gym.spec(environment_id)
        assert spec.max_episode_steps == time_limit
        assert spec.kwargs == kwargs


@pytest.mark.parametrize("name, retired, current", [("Merchant", 0, 2), ("Merchant", 1, 2)])
def test_retired_environment_id_is_rejected_with_current_version(name, retired, current):
    assert f"npc_gym/{name}-v{retired}" not in gym.registry
    with (
        pytest.warns(DeprecationWarning, match=f"{name}-v{retired}"),
        pytest.raises(gym.error.DeprecatedEnv, match=f"{name}-v{current}"),
    ):
        gym.make(f"npc_gym/{name}-v{retired}")


@pytest.mark.parametrize("environment_id", EXPECTED_SPECS)
def test_registered_environments_use_the_documented_defaults(environment_id):
    env = gym.make(environment_id)
    try:
        assert env.spec.id == environment_id
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
    finally:
        env.close()


@pytest.mark.parametrize("constructor", [StormTaxiEnv, MerchantEnv, GardenerEnv, PacmanEnv])
def test_public_constructors_are_explicit_and_usable(constructor):
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(constructor).parameters.values()
    )
    env = constructor()
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
    finally:
        env.close()
        env.close()


@pytest.mark.parametrize("constructor", [StormTaxiEnv, MerchantEnv, GardenerEnv, PacmanEnv])
def test_public_constructors_reject_unknown_render_modes(constructor):
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        constructor(render_mode="printer")


def test_package_resources_are_available():
    assert (files("npc_gym.envs.merchant") / "merchant_layouts/basic.txt").is_file()
    assert (files("npc_gym.envs.pacman") / "layouts/small.lay").is_file()
    assert not (files("npc_gym.envs.taxi") / "img").is_dir()


def test_plain_taxi_is_not_part_of_npc_gym():
    from npc_gym import envs

    assert not hasattr(envs, "TaxiEnv")
    for environment_id in ("npc_gym/Taxi-v0", "npc_gym/Taxi-v1"):
        assert environment_id not in gym.registry
        with pytest.raises(gym.error.NameNotFound):
            gym.make(environment_id)
