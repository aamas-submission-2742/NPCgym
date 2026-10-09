"""Small real learning runs for both algorithm choices in the feature examples."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")


@pytest.fixture(params=["pacman", "gardener"])
def example(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    module = importlib.import_module(request.param)
    for name, value in {
        "TRAINING_STEPS": 128,
        "EVALUATION_EPISODES": 2,
        "MAX_EPISODE_STEPS": 6,
        "SEED": 0,
    }.items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.delenv("NPC_GYM_MODEL_PATH", raising=False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(previous)


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_algorithms_update_reload_evaluate_and_refuse_overwrite(example, algorithm, monkeypatch, tmp_path):
    monkeypatch.setattr(example, "ALGORITHM", algorithm)
    path = tmp_path / "model.zip"
    monkeypatch.setenv("NPC_GYM_MODEL_PATH", str(path))
    learner = example.DQN if algorithm == "dqn" else example.PPO
    learn = learner.learn
    updated = []

    def learn_and_check(self, *args, **kwargs):
        if example.__name__ == "pacman":
            assert self.gamma == (0.95 if algorithm == "dqn" else 0.99)
        before = deepcopy(self.policy.state_dict())
        result = learn(self, *args, **kwargs)
        updated.append(any(not torch.equal(value, self.policy.state_dict()[key]) for key, value in before.items()))
        return result

    monkeypatch.setattr(learner, "learn", learn_and_check)
    summary, steps = example.main()
    assert updated == [True]
    assert steps == 128 and len(summary.episodes) == 2
    assert all(episode.length <= 6 for episode in summary.episodes)
    assert list(tmp_path.iterdir()) == [path]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        example.main()
    assert path.read_bytes() == original


def test_unknown_algorithm_fails_before_creating_environments(example, monkeypatch):
    monkeypatch.setattr(example, "ALGORITHM", "a2c")

    def fail(**kwargs):
        raise AssertionError("must reject the algorithm before making an environment")

    monkeypatch.setattr(example, "make_env", fail)
    with pytest.raises(ValueError, match="NPC_GYM_ALGORITHM must be dqn or ppo"):
        example.main()
