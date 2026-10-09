"""Exercise the FrozenLake example through public observations and monitor contracts."""

import importlib.util
from contextlib import closing
from pathlib import Path

import pytest

from npc_gym.evaluation import evaluate
from npc_gym.labels import Transition
from npc_gym.monitors import MonitorInput

VIA_CHECKPOINT = [2, 2, 1, 1, 1, 2]
WITHOUT_CHECKPOINT = [1, 1, 2, 1, 2, 2]


@pytest.fixture
def example(monkeypatch):
    for name in ("TRAINING_STEPS", "EVALUATION_EPISODES", "MAX_EPISODE_STEPS", "SEED"):
        monkeypatch.delenv(f"NPC_GYM_{name}", raising=False)
    path = Path(__file__).resolve().parents[2] / "examples" / "custom_environment.py"
    spec = importlib.util.spec_from_file_location("custom_environment_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("state", "terminated", "truncated", "labels"),
    [
        (0, False, False, set()),
        (6, False, False, {"checkpoint"}),
        (5, True, False, set()),
        (15, False, True, set()),
        (15, True, False, {"goal"}),
        (15, True, True, {"goal"}),
    ],
)
def test_labels_use_tile_and_lifecycle_flags(example, state, terminated, truncated, labels):
    transition = Transition(previous_state=0, action=2, state=state, terminated=terminated, truncated=truncated)
    assert example.label_transition(transition) == frozenset(labels)


@pytest.mark.parametrize(
    ("actions", "limit", "reward", "termination", "violations"),
    [
        (VIA_CHECKPOINT, 100, 1, "terminated", 0),
        (WITHOUT_CHECKPOINT, 100, 1, "terminated", 1),
        ([1, 2], 100, 0, "terminated", 0),
        ([0, 0, 0], 3, 0, "truncated", 0),
        (VIA_CHECKPOINT[:3], 3, 0, "truncated", 0),
        (WITHOUT_CHECKPOINT, 6, 1, "terminated_and_truncated", 1),
    ],
)
def test_real_environment_routes(example, monkeypatch, actions, limit, reward, termination, violations):
    monkeypatch.setattr(example, "MAX_EPISODE_STEPS", limit)
    with closing(example.make_env()) as env:
        observation, info = env.reset(seed=0)
        assert observation == 0
        assert info["labels"] == frozenset()
        for action in actions:
            observation, _, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(observation)
        assert terminated or truncated
        assert ("goal" in info["labels"]) == bool(reward)
        route = iter(actions)
        summary = evaluate(
            env,
            lambda observation, info: next(route),
            seed=10_000,
            monitors={example.CHECKPOINT_NORM_ID: example.make_checkpoint_monitor},
        )
    result = summary.episodes[0]
    assert result.episode_return == reward
    assert result.length == len(actions)
    assert result.termination == termination
    assert result.monitor_counts == {example.CHECKPOINT_NORM_ID: {"count": violations}}


def test_evaluation_resets_counts_between_episodes(example):
    route = iter(VIA_CHECKPOINT + WITHOUT_CHECKPOINT + VIA_CHECKPOINT)
    with closing(example.make_env()) as env:
        summary = evaluate(
            env,
            lambda observation, info: next(route),
            episodes=3,
            seed=10_000,
            monitors={example.CHECKPOINT_NORM_ID: example.make_checkpoint_monitor},
        )
    assert [episode.monitor_counts[example.CHECKPOINT_NORM_ID]["count"] for episode in summary.episodes] == [0, 1, 0]


def test_monitor_consumes_initial_labels_and_requires_a_prior_visit(example):
    monitor = example.make_checkpoint_monitor()
    goal = MonitorInput(frozenset({"goal"}), terminated=True)
    assert monitor.reset(MonitorInput(frozenset({"checkpoint"}))) is False
    assert monitor.update(goal) is False
    assert monitor.reset(MonitorInput(frozenset())) is False
    assert monitor.update(MonitorInput(frozenset({"checkpoint", "goal"}), terminated=True)) is True
    monitor.reset(MonitorInput(frozenset()))
    assert monitor.count == 0


def test_default_example_learns_successfully(example, monkeypatch, capsys):
    assert len(Path(example.__file__).read_text().splitlines()) < 100
    assert example.TRAINING_STEPS == 10_000
    assert example.EVALUATION_EPISODES == 100
    assert example.MAX_EPISODE_STEPS == 100
    assert example.SEED == 0
    results = []

    def record_evaluation(env, model, **kwargs):
        assert env is not model.env
        assert model.num_timesteps == 10_000
        assert kwargs["seed"] == 10_000
        result = evaluate(env, model, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(example, "evaluate", record_evaluation)
    example.main()
    assert results[0].mean_return == 1.0
    assert len(results[0].episodes) == 100
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "FrozenLake success rate (mean task return): 1.00",
        f"Mean norm violations: {dict(results[0].mean_monitor_counts)}",
    ]
