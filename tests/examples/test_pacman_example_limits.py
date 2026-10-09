"""Feature and image examples select the requested map and its time limit."""

import runpy
from contextlib import closing
from pathlib import Path

import pytest
from gymnasium.wrappers import TimeLimit

pytest.importorskip("stable_baselines3")
pytest.importorskip("pygame")


@pytest.mark.parametrize(("name", "limit"), [("small", 300), ("medium", 500), ("large", 800)])
@pytest.mark.parametrize("script", ["pacman.py", "pacman_images.py"])
@pytest.mark.parametrize("override", [None, 913])
def test_examples_share_map_defaults_and_honor_explicit_limits(name, limit, script, override, monkeypatch):
    monkeypatch.setenv("NPC_GYM_LAYOUT", name)
    monkeypatch.delenv("NPC_GYM_MAX_EPISODE_STEPS", raising=False)
    if override is not None:
        monkeypatch.setenv("NPC_GYM_MAX_EPISODE_STEPS", str(override))
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / script))
    for training in (False, True) if script == "pacman.py" else (False,):
        env = namespace["make_env"](training=training) if script == "pacman.py" else namespace["make_env"]({})
        with closing(env):
            assert env.unwrapped.layout.name == name
            inner = env
            while not isinstance(inner, TimeLimit):
                inner = inner.env
            assert inner._max_episode_steps == (limit if override is None else override)
            observation, _ = env.reset(seed=0)
            assert env.observation_space.contains(observation)
            observation, *_ = env.step(0)
            assert env.observation_space.contains(observation)
