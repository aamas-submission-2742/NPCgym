"""The integration gate requires real compilation; ordinary core installs may skip it."""

import importlib.util
import os
import shutil
from functools import cache

import pytest

from npc_gym.monitors.ltlf import MonaCompiler


@pytest.fixture(scope="session")
def compiler():
    if importlib.util.find_spec("ltlf2dfa") is None or shutil.which("mona") is None:
        message = "LTLf tests require npc-gym[ltlf] and MONA on PATH"
        if os.environ.get("NPC_GYM_REQUIRE_LTLF") == "1":
            pytest.fail(message)
        pytest.skip(message)

    class CachedCompiler:
        compile = staticmethod(cache(MonaCompiler().compile))

    return CachedCompiler()
