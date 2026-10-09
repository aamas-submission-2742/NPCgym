import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/npcgym-matplotlib")


@pytest.fixture(scope="session")
def storm_taxi():
    """A storm Taxi shared by tests; reset it before stepping it."""
    from npc_gym.envs.taxi.storm_taxi import StormTaxiEnv

    return StormTaxiEnv()
