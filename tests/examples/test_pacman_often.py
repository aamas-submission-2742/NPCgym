"""Small real Pacman OFTEN runs from standard SB3 checkpoints."""

import importlib
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("NPC_GYM_REQUIRE_OFTEN") == "1":
    import clingo  # noqa: F401
    import torch
else:
    pytest.importorskip("clingo")
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")


@pytest.fixture
def example(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("pacman_often")
    for key, value in {
        "TRAINING_STEPS": 128,
        "EVALUATION_EPISODES": 2,
        "MAX_EPISODE_STEPS": 6,
        "SEED": 0,
        "ALGORITHM": "dqn",
    }.items():
        monkeypatch.setattr(module.task, key, value)
    monkeypatch.setattr(module, "TEACHING_STEPS", 256)
    monkeypatch.setattr(module, "EVALUATION_EPISODES", 2)
    monkeypatch.setattr(module, "HORIZON", 1)
    for variable in ("NPC_GYM_MODEL_PATH", "NPC_GYM_SMOKE"):
        monkeypatch.delenv(variable, raising=False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(previous)


def test_saved_base_teaching_control_rewards_and_paired_evaluation(example, monkeypatch, tmp_path, capsys):
    example.task.main()
    path = tmp_path / "pacman-dqn.zip"
    original = path.read_bytes()
    base = example.DQN.load(path, device="cpu")
    snapshots, seeds, teachers = [], [], []
    assess, evaluate, teacher_type = example.assess, example.evaluate, example.DQNOFTEN

    def record_assess(model):
        if not snapshots:
            for key, value in base.policy.state_dict().items():
                assert torch.equal(value, model.policy.state_dict()[key])
            expected = base.policy.optimizer.state_dict()
            actual = model.policy.optimizer.state_dict()
            assert actual["param_groups"] == expected["param_groups"]
            for parameter, state in expected["state"].items():
                for key, value in state.items():
                    assert torch.equal(value, actual["state"][parameter][key])
        snapshots.append(deepcopy(model.q_net.state_dict()))
        return assess(model)

    def record_evaluate(*args, **kwargs):
        seeds.append(kwargs["seed"])
        env = args[0]
        _, before = env.reset(seed=19)
        _, reward, _, _, after = env.step(0)
        assert reward == after["episode_metrics"]["score"] - before["episode_metrics"]["score"]
        return evaluate(*args, **kwargs)

    def make_teacher(model, ordinary, expert, **kwargs):
        assert ordinary.unwrapped is not expert.unwrapped
        for stream in (ordinary, expert):
            _, before = stream.reset(seed=19)
            _, reward, _, _, after = stream.step(0)
            assert reward == pytest.approx(
                (after["episode_metrics"]["score"] - before["episode_metrics"]["score"]) / 100
            )
        teacher = teacher_type(model, ordinary, expert, **kwargs)
        assert teacher.margin == 0.5
        assert model.gamma == 0.95
        assert model.batch_size == 32 and model.train_freq.frequency == 4
        teachers.append(teacher)
        return teacher

    teach = teacher_type.learn

    def record_teaching(teacher, total_timesteps, **kwargs):
        assert kwargs["schedule_timesteps"] == 5_000_000
        return teach(teacher, total_timesteps, **kwargs)

    monkeypatch.setattr(teacher_type, "learn", record_teaching)
    learn = example.DQN.learn

    def continue_from_base(model, *args, **kwargs):
        for key, value in snapshots[0].items():
            assert torch.equal(value, model.q_net.state_dict()[key])
        env = model.get_env()
        env.seed(19)
        env.reset()
        score = env.envs[0].unwrapped.labeling_state().score
        _, rewards, _, infos = env.step([0])
        assert rewards[0] == pytest.approx((infos[0]["episode_metrics"]["score"] - score) / 100)
        result = learn(model, *args, **kwargs)
        assert model.num_timesteps == 256
        return result

    monkeypatch.setattr(example, "assess", record_assess)
    monkeypatch.setattr(example, "evaluate", record_evaluate)
    monkeypatch.setattr(example, "DQNOFTEN", make_teacher)
    monkeypatch.setattr(example.DQN, "learn", continue_from_base)
    summaries = example.main()
    assert tuple(summaries) == ("base", "often", "continued")
    assert seeds == [50_000, 50_001] * 3
    assert all(len(summary.episodes) == 2 for summary in summaries.values())
    for trained in snapshots[1:]:
        assert any(not torch.equal(value, trained[key]) for key, value in snapshots[0].items())
    stats = teachers[0].stats
    assert stats.ordinary_steps == stats.expert_steps == 128
    assert stats.updates > 0 and stats.ordinary_fallbacks == stats.expert_fallbacks == 0
    assert teachers[0].progress_remaining > 0.9999  # Short runs retain the full teaching schedule.
    output = capsys.readouterr().out
    assert f"Teaching statistics: {stats}" in output
    for name, summary in summaries.items():
        assert f"{name}: return={summary.mean_return:.2f}; length={summary.mean_length:.2f};" in output
    assert output.count("Episode endings: {'truncated': 2}") == 3
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_missing_checkpoint_and_explicit_smoke(example, monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="first train a DQN"):
        example.main()
    monkeypatch.setenv("NPC_GYM_SMOKE", "1")
    assert tuple(example.main()) == ("base", "often", "continued")
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("NPC_GYM_MODEL_PATH", "missing.zip")
    with pytest.raises(FileNotFoundError):
        example.main()


def test_sessions_track_actual_steps_and_preserve_dynamics(example):
    with example.task.make_env() as left, example.task.make_env() as right:
        obs, info = left.reset(seed=4)
        right.reset(seed=4)
        start = example.sessions(left)
        session = start(obs, info)
        assert session.planner is not example.sessions(right)(obs, info).planner
        initial = left.unwrapped.labeling_state()
        for step in range(6):
            problem = session.problem(obs, info)
            assert problem.horizon == min(example.HORIZON, 6 - step)
            result = session.planner.solve(problem, dict.fromkeys(range(5), 0.0))
            assert result.optimal
            obs, reward, term, trunc, info = left.step(result.action)
            other = right.step(result.action)
            np.testing.assert_array_equal(obs, other[0])
            assert (reward, term, trunc) == other[1:4]
            assert info["labels"] == other[4]["labels"]
            session.advance(example.MonitorInput(info["labels"], term, trunc))
        with pytest.raises(ValueError):
            session.problem(obs, info)
        obs, info = left.reset(seed=4)
        assert left.unwrapped.labeling_state() == initial
        next_session = start(obs, info)
        assert next_session.planner is session.planner
        assert next_session.problem(obs, info).horizon == min(example.HORIZON, 6)
        with pytest.raises(ValueError):
            session.problem(obs, info)


def test_teaching_failure_closes_environments(example, monkeypatch):
    monkeypatch.setenv("NPC_GYM_SMOKE", "1")
    closed = []
    close = example.PacmanEnv.close

    def record_close(self):
        closed.append(self)
        close(self)

    def fail(*args, **kwargs):
        raise RuntimeError("teaching failed")

    monkeypatch.setattr(example.PacmanEnv, "close", record_close)
    monkeypatch.setattr(example, "DQNOFTEN", fail)
    with pytest.raises(RuntimeError, match="teaching failed"):
        example.main()
    assert len({id(env) for env in closed}) == 3


@pytest.mark.parametrize("setting", ["TEACHING_STEPS", "HORIZON", "EVALUATION_EPISODES"])
def test_invalid_budget(example, monkeypatch, setting):
    monkeypatch.setenv("NPC_GYM_SMOKE", "1")
    monkeypatch.setattr(example, setting, 0)
    with pytest.raises(ValueError, match="positive"):
        example.main()


def test_learned_inference_does_not_plan(example, monkeypatch):
    with example.task.make_env() as env:
        model = example.DQN("MlpPolicy", env, device="cpu", buffer_size=8)

        def fail(*args, **kwargs):
            raise AssertionError("inference must not construct a planner")

        monkeypatch.setattr(example, "ASPPlanner", fail)
        assert len(example.assess(model).episodes) == 2
