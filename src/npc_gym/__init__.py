from gymnasium.envs.registration import register

__version__ = "0.1.0"

register(
    id="npc_gym/StormTaxi-v0",
    entry_point="npc_gym.envs.taxi.storm_taxi:StormTaxiEnv",
    max_episode_steps=50,
)

register(
    id="npc_gym/Merchant-v2",
    entry_point="npc_gym.envs.merchant.merchant:MerchantEnv",
    kwargs={"layout": "basic"},
    max_episode_steps=150,
)

register(
    id="npc_gym/Gardener-v0",
    entry_point="npc_gym.envs.gardener.gardener:GardenerEnv",
    kwargs={"size": 15},
    max_episode_steps=1_000,
)

for _name, _layout, _limit in (
    ("Pacman", "small", 300),
    ("PacmanMedium", "medium", 500),
    ("PacmanLarge", "large", 800),
):
    register(
        id=f"npc_gym/{_name}-v1",
        entry_point="npc_gym.envs.pacman.pacman_env:PacmanEnv",
        kwargs={"layout": _layout, "features": "complete"},
        max_episode_steps=_limit,
    )
