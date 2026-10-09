"""Small real runs of the saved-base Pacman and Gardener OFTEN experiments."""

import json
from dataclasses import replace

import pytest

from experiments import execution, run
from experiments.specifications import (
    ALGORITHMS,
    ENVIRONMENTS,
    GARDENER_DRAIN_OFTEN_EXPERIMENT_ID,
    GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID,
    GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID,
    GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID,
    PACMAN_OFTEN_EXPERIMENT_ID,
    PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID,
    PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID,
    REGISTRY,
    SCENARIOS,
    TECHNIQUES,
    WRAPPERS,
    ExperimentRegistry,
    GardenerOFTENSettings,
    KeywordArguments,
    OFTENSettings,
    PacmanOFTENSettings,
    SpecificationError,
)


def tiny_registry(*, frequency=128, experiment_id=PACMAN_OFTEN_EXPERIMENT_ID):
    resolved = REGISTRY.resolve(experiment_id)
    settings = replace(
        resolved.specification.run,
        training_steps=256,
        max_episode_steps=6,
        intermediate_evaluation_frequency=frequency,
        intermediate_evaluation_episodes=2,
        final_evaluation_episodes=2,
    )
    often = replace(resolved.specification.often, pretraining_steps=128)
    if isinstance(often, PacmanOFTENSettings):
        often = replace(
            often,
            pretraining_max_episode_steps=6,
            evaluation_max_episode_steps=6,
            pretraining_evaluation_frequency=frequency,
            pretraining_evaluation_seed_offset=50000,
        )
    specification = replace(resolved.specification, run=settings, often=often)
    kwargs = resolved.algorithm.constructor_kwargs.to_dict()
    kwargs["buffer_size"] = 1024
    algorithm = replace(resolved.algorithm, constructor_kwargs=KeywordArguments.from_mapping(kwargs))
    return ExperimentRegistry(
        environments=ENVIRONMENTS,
        wrappers=WRAPPERS,
        scenarios=SCENARIOS,
        algorithms=tuple(algorithm if item.id == algorithm.id else item for item in ALGORITHMS),
        techniques=TECHNIQUES,
        experiments=(specification,),
    )


@pytest.fixture
def dependencies():
    pytest.importorskip("clingo")
    pytest.importorskip("stable_baselines3")
    return pytest.importorskip("torch")


def test_benchmark_configuration_and_paired_seeds(tmp_path):
    plans = execution.plan_training_batch(tmp_path, experiment_ids=[PACMAN_OFTEN_EXPERIMENT_ID], seeds=range(8))
    assert len(plans) == 8
    for plan in plans:
        resolved = plan.resolved
        config = execution.resolved_configuration(resolved)
        assert config["often"] == {
            "norm_id": "pacman/vegan-v0",
            "pretraining_steps": 5_000_000,
            "teaching_seed_offset": 20_000,
            "evaluation_seed": 50_000,
            "margin": 0.5,
            "horizon": 1,
            "radius": 5,
            "pretraining_max_episode_steps": 300,
            "evaluation_max_episode_steps": 300,
            "pretraining_evaluation_frequency": 250_000,
            "pretraining_evaluation_seed_offset": 10_000,
        }
        assert resolved.specification.run.training_steps == 10_000_000
        assert resolved.specification.run.intermediate_evaluation_frequency == 1_000_000
        assert resolved.specification.run.final_evaluation_episodes == 1000
        assert resolved.specification.run.intermediate_evaluation_episodes == 1000
        assert plan.intermediate_evaluation_seed == plan.final_evaluation_seed == 50_000
        assert resolved.algorithm.constructor_kwargs.to_dict()["gamma"] == 0.99
        assert list(resolved.make_monitors()) == ["pacman/vegan-v0"]
        assert execution.resolved_configuration(execution.resolved_from_configuration(config)) == config
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_cli_dry_run_and_missing_asp(tmp_path, capsys, monkeypatch, dependencies):
    args = [
        "train",
        "--experiment",
        PACMAN_OFTEN_EXPERIMENT_ID,
        "--output",
        str(tmp_path / "runs"),
        "--seed",
        "0",
        "--dry-run",
    ]
    assert run.main(args) == 0
    assert json.loads(capsys.readouterr().out)[0]["configuration"]["often"]["margin"] == 0.5
    assert not (tmp_path / "runs").exists()
    real = execution.importlib.util.find_spec
    monkeypatch.setattr(
        execution.importlib.util, "find_spec", lambda name, real=real: None if name == "clingo" else real(name)
    )
    assert run.main(args) == 2
    assert "asp,sb3" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "experiment_id",
    [
        PACMAN_OFTEN_EXPERIMENT_ID,
        PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID,
        PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID,
        GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID,
        GARDENER_DRAIN_OFTEN_EXPERIMENT_ID,
        GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID,
        GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID,
    ],
)
def test_train_reload_paired_evaluation_and_curve_invariance(tmp_path, dependencies, monkeypatch, experiment_id):
    from stable_baselines3 import DQN

    from experiments import often

    torch = dependencies
    models = []
    documents = []
    previous_threads = torch.get_num_threads()
    seeds = []
    evaluate = execution.evaluate

    def record_evaluation(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(execution, "evaluate", record_evaluation)
    monkeypatch.setattr(often, "evaluate", record_evaluation)
    for frequency in (128, 256):
        registry = tiny_registry(frequency=frequency, experiment_id=experiment_id)
        (plan,) = execution.plan_training_batch(tmp_path / str(frequency), seeds=[0], registry=registry)
        (result,) = execution.execute_training_batch([plan])
        assert torch.get_num_threads() == previous_threads
        assert result.actual_timesteps == 256
        document = json.loads(plan.run_path.read_text())
        documents.append(document)
        assert document["status"] == "complete"
        assert document["pretraining"]["actual_steps"] == 128
        assert document["pretraining"]["updates"] == 7
        assert document["pretraining"]["seconds"] > 0
        stats = document["teaching"]["stats"]
        assert document["teaching"]["seed"] == 20_000
        assert document["teaching"]["seconds"] > 0
        assert stats["ordinary_steps"] == stats["expert_steps"] == 128
        gardener = isinstance(plan.resolved.specification.often, GardenerOFTENSettings)
        assert stats["updates"] == (28 if gardener else 7)
        assert stats["ordinary_fallbacks"] == stats["expert_fallbacks"] == 0
        base = DQN.load(plan.run_directory / "base.zip", device="cpu")
        final = DQN.load(plan.model_path, device="cpu")
        assert base.gamma == final.gamma == (0.95 if gardener else 0.99)
        assert base.batch_size == 32 and base.gradient_steps == 1
        assert final.batch_size == (256 if gardener else 32)
        assert final.gradient_steps == (-1 if gardener else 1)
        if gardener:
            assert base.observation_space.shape == final.observation_space.shape == (27,)
        elif experiment_id == PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID:
            assert base.observation_space.shape == final.observation_space.shape == (64,)
        assert base.num_timesteps == final.num_timesteps == 128  # Teaching has separate counters.
        assert any(not torch.equal(v, final.policy.state_dict()[k]) for k, v in base.policy.state_dict().items())
        models.append(final)
        initial = json.loads((plan.run_directory / "evaluations/intermediate-0/summary.json").read_text())
        assert initial["metadata"]["seed"] == (10_000 if gardener else 50_000)
        assert initial["metadata"]["episode_seed_rule"] == "seed + episode index"
        assert len(initial["episodes"]) == 2
        evaluation = execution.plan_evaluation_batch(plan.output_root, registry=registry, name="repeat", seeds=[0])
        # Paired evaluations need ASP for the base-with-fixes comparison.
        real = execution.importlib.util.find_spec
        with monkeypatch.context() as context:
            context.setattr(
                execution.importlib.util, "find_spec", lambda name, real=real: None if name == "clingo" else real(name)
            )
            with pytest.raises(execution.PreflightError, match="asp,sb3"):
                execution.execute_evaluation_batch(evaluation)
            assert not evaluation[0].output_directory.exists()
        (again,) = execution.execute_evaluation_batch(evaluation)
        assert again.summary == result.final_evaluation
        references = []
        for output in sorted((plan.run_directory / "evaluations").iterdir()):
            if output.name == "base":
                continue
            reference = json.loads((output / "base-fixed/summary.json").read_text())
            references.append(reference)
            assert (output / "base-fixed/episodes.csv").is_file()
            metadata = reference["metadata"]
            assert metadata["variant"] == "base-fixed"
            assert metadata["model_sha256"] == document["artifacts"]["base"]["sha256"]
            assert metadata["seed"] == initial["metadata"]["seed"]
            assert metadata["timesteps"] == 128
            assert metadata["timesteps_unit"] == "base DQN environment steps"
            assert metadata["policy_fix"]["norm_id"] == plan.resolved.specification.often.norm_id
            assert metadata["policy_fix"]["horizon"] == plan.resolved.specification.often.horizon
            assert metadata["policy_fix"]["radius"] == plan.resolved.specification.often.radius
            assert metadata["policy_fix"]["objectives"]["violations"] == {
                "weight": 5 if gardener else 1,
                "priority": 1 if gardener else 2,
            }
            assert metadata["fallbacks"] == 0
            assert metadata["decisions"] == sum(episode["length"] for episode in reference["episodes"])
            assert 0 <= metadata["interventions"] <= metadata["decisions"]
            assert metadata["planning_seconds"] > 0
            assert metadata["evaluation_seconds"] >= metadata["planning_seconds"]
        assert all(reference["episodes"] == references[0]["episodes"] for reference in references)
        assert execution.sha256_file(plan.run_directory / "base.zip") == document["artifacts"]["base"]["sha256"]
        if experiment_id == GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID:
            assert set(initial["episodes"][0]["monitor_counts"]) == {
                "gardener/permission-aware-v0",
                "gardener/drain-v0",
                "gardener/rescue-v1",
            }
        with pytest.raises(execution.PreflightError, match="already exists"):
            execution.execute_training_batch([plan])
    first_seed = 10_000 if gardener else 50_000
    if gardener:
        assert seeds == [first_seed, first_seed + 1] * (len(seeds) // 2)
    else:
        # Native base evaluations use one initial seed; paired teaching/fixes use one seed per episode.
        assert set(seeds) == {50000, 50001}
        assert seeds.count(50000) > seeds.count(50001)
    assert documents[0]["teaching"]["stats"] == documents[1]["teaching"]["stats"]
    for key, value in models[0].policy.state_dict().items():
        assert torch.equal(value, models[1].policy.state_dict()[key]), key


def test_overwrite_ownership_and_base_integrity(tmp_path, dependencies):
    registry = tiny_registry()
    (plan,) = execution.plan_training_batch(tmp_path, seeds=[0], registry=registry)
    execution.execute_training_batch([plan])
    note = plan.run_directory / "notes.txt"
    note.write_text("keep")
    base = plan.run_directory / "base.zip"
    execution.execute_training_batch([plan], overwrite=True)
    assert note.read_text() == "keep"
    document = json.loads(plan.run_path.read_text())
    assert execution.sha256_file(base) == document["artifacts"]["base"]["sha256"]
    base.write_bytes(b"corrupt")
    with pytest.raises(execution.SavedRunError, match="Base checkpoint checksum"):
        execution.plan_evaluation_batch(tmp_path, registry=registry, name="repeat")
    base.unlink()
    base.symlink_to(note)
    with pytest.raises(execution.PreflightError, match="unsafe run artifact"):
        execution.preflight_training([plan], overwrite=True)
    assert note.read_text() == "keep"


def test_failed_teaching_is_incomplete_and_closes_streams(tmp_path, dependencies, monkeypatch):
    from npc_gym.envs import PacmanEnv
    from npc_gym.integrations.sb3 import DQNOFTEN

    registry = tiny_registry()
    (plan,) = execution.plan_training_batch(tmp_path, seeds=[0], registry=registry)
    closed = []
    close = PacmanEnv.close

    def record_close(env):
        closed.append(env)
        close(env)

    def fail(*args, **kwargs):
        raise RuntimeError("teaching failed")

    monkeypatch.setattr(PacmanEnv, "close", record_close)
    monkeypatch.setattr(DQNOFTEN, "learn", fail)
    with pytest.raises(RuntimeError, match="teaching failed"):
        execution.execute_training_batch([plan])
    document = json.loads(plan.run_path.read_text())
    assert document["status"] == "incomplete"
    assert document["timesteps"]["actual"] == 0
    assert document["pretraining"]["actual_steps"] == 128
    assert document["teaching"]["stats"]["ordinary_steps"] == 0
    assert len({id(env) for env in closed}) == 7
    assert not plan.model_path.exists()


@pytest.mark.parametrize("changes", [{"pretraining_steps": 0}, {"margin": float("nan")}, {"radius": True}])
def test_invalid_teaching_settings(changes):
    with pytest.raises(SpecificationError):
        OFTENSettings(**changes)


def test_saved_teaching_settings_and_seed_bounds(tmp_path):
    resolved = REGISTRY.resolve(PACMAN_OFTEN_EXPERIMENT_ID)
    config = execution.resolved_configuration(resolved)
    config["often"]["horizon"] = 0
    with pytest.raises(execution.SavedRunError, match="Invalid saved OFTEN"):
        execution.resolved_from_configuration(config)
    with pytest.raises(execution.SelectionError, match="seeds exceed"):
        execution.plan_training_batch(tmp_path, experiment_ids=[PACMAN_OFTEN_EXPERIMENT_ID], seeds=[2**32 - 20_000])


def test_vegetarian_configuration_selects_the_reference_norm():
    vegan = REGISTRY.resolve(PACMAN_OFTEN_EXPERIMENT_ID)
    vegetarian = REGISTRY.resolve(PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID)
    assert vegetarian.algorithm == vegan.algorithm
    assert (
        replace(vegetarian.specification.often, norm_id=vegan.specification.often.norm_id) == vegan.specification.often
    )
    assert vegetarian.specification.run == vegan.specification.run
    assert list(vegetarian.make_monitors()) == ["pacman/vegetarian-orange-v0"]


def test_trapped_configuration_and_horizon(tmp_path):
    vegan = REGISTRY.resolve(PACMAN_OFTEN_EXPERIMENT_ID)
    trapped = REGISTRY.resolve(PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID)
    assert trapped.algorithm == vegan.algorithm
    assert replace(trapped.specification.often, norm_id=vegan.specification.often.norm_id) == vegan.specification.often
    assert trapped.specification.run == vegan.specification.run
    assert list(trapped.make_monitors()) == ["pacman/trapped-v1"]
    invalid = replace(trapped.specification, often=replace(trapped.specification.often, horizon=2))
    with pytest.raises(execution.PreflightError, match="horizon=1"):
        execution._validate_execution_settings(replace(trapped, specification=invalid))


def test_trapped_sessions_keep_separate_history_and_consume_actual_steps(dependencies, monkeypatch):
    from experiments.often import _sessions
    from npc_gym.envs.pacman.labels import PacmanLabel as P
    from npc_gym.monitors import MonitorInput

    resolved = REGISTRY.resolve(PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID)
    with resolved.make_env(training=False) as env:
        obs, _ = env.reset(seed=7)
        snapshot = env.unwrapped.labeling_state()
        position = next(
            (x, y)
            for x in range(snapshot.layout.width // 2 + 1, snapshot.layout.width)
            for y in range(snapshot.layout.height)
            if (x, y) not in snapshot.layout.walls
        )
        outside = replace(snapshot, player=replace(snapshot.player, position=position))
        monkeypatch.setattr(env.unwrapped, "labeling_state", lambda: outside)
        start = _sessions(env, resolved.specification.often, 300, "pacman/trapped-v1")
        first = start(obs, {"labels": frozenset({P.SCORE_0})})
        second = start(obs, {"labels": frozenset({P.SCORE_0})})

        def cost(session):
            return session.planner.solve(
                session.problem(obs, {}), dict.fromkeys(range(5), 0.0), allowed_actions=[0]
            ).costs["violations"]

        assert cost(first) == cost(second) == 1
        first.advance(MonitorInput(frozenset({P.SCORE_GREATER_100})))
        assert cost(first) == cost(second) == 1
        first.advance(MonitorInput(frozenset({P.SCORE_GREATER_400})))
        assert cost(first) == 0 and cost(second) == 1
        first.advance(MonitorInput(frozenset({P.SCORE_0}), truncated=True))
        with pytest.raises(ValueError, match="ended"):
            cost(first)
        assert cost(start(obs, {"labels": frozenset({P.SCORE_0})})) == 1


@pytest.mark.parametrize(
    "experiment_id,model_name,expected",
    [
        (PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID, "PacmanModel", "pacman/vegetarian-orange-v0"),
        (GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID, "GardenerModel", "gardener/permission-drain-v0"),
    ],
)
def test_teaching_uses_the_scenario_for_both_planners(
    tmp_path, dependencies, monkeypatch, experiment_id, model_name, expected
):
    from experiments import often

    constructor = getattr(often, model_name)
    norms = []

    class RecordingModel(constructor):
        def __init__(self, *args, **kwargs):
            norms.append(kwargs["norm_id"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(often, model_name, RecordingModel)
    registry = tiny_registry(experiment_id=experiment_id)
    (plan,) = execution.plan_training_batch(tmp_path, registry=registry, seeds=[0])
    execution.execute_training_batch([plan])
    assert len(norms) >= 2
    assert set(norms) == {expected}


@pytest.mark.parametrize(
    "experiment_id,norm_ids",
    [
        (GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID, ["gardener/no-collect-v0"]),
        (GARDENER_DRAIN_OFTEN_EXPERIMENT_ID, ["gardener/drain-v0"]),
        (GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID, ["gardener/permission-aware-v0"]),
        (GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID, ["gardener/permission-aware-v0", "gardener/drain-v0"]),
    ],
)
def test_gardener_validated_configuration(tmp_path, experiment_id, norm_ids):
    (plan,) = execution.plan_training_batch(tmp_path, experiment_ids=[experiment_id], seeds=[0])
    resolved = plan.resolved
    settings = resolved.specification.often
    assert isinstance(settings, GardenerOFTENSettings)
    assert settings.pretraining_steps == resolved.specification.run.training_steps == 250_000
    assert settings.margin == 5 and settings.final_expert_weight == 0.5
    assert settings.horizon == 1 and settings.radius == 4
    assert settings.batch_size == 256 and settings.gradient_steps == -1
    assert plan.intermediate_evaluation_seed == plan.final_evaluation_seed == 10_000
    assert resolved.specification.run.final_evaluation_episodes == 1000
    assert list(resolved.make_monitors()) == ["gardener/permission-aware-v0", "gardener/drain-v0", "gardener/rescue-v1"]
    assert settings.norm_id == (norm_ids[0] if len(norm_ids) == 1 else "gardener/permission-drain-v0")
    with resolved.make_env(training=True) as env:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (27,)
        assert env.unwrapped.num_frogs == 2 and env.unwrapped.num_puddles == 4
    config = execution.resolved_configuration(resolved)
    assert execution.resolved_configuration(execution.resolved_from_configuration(config)) == config


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"gradient_steps": -2},
        {"final_expert_weight": float("nan")},
        {"final_expert_weight": -1},
    ],
)
def test_invalid_gardener_teaching_settings(changes):
    with pytest.raises(SpecificationError):
        GardenerOFTENSettings(**changes)


def test_base_fixed_named_seeds_overwrite_and_atomic_failure(tmp_path, dependencies, monkeypatch):
    registry = tiny_registry(experiment_id=GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID)
    (plan,) = execution.plan_training_batch(tmp_path, registry=registry, seeds=[0])
    execution.execute_training_batch([plan])
    checksums = {name: execution.sha256_file(plan.run_directory / name) for name in ("base.zip", "model.zip")}
    evaluation = execution.plan_evaluation_batch(
        tmp_path, registry=registry, name="paired", episodes=3, evaluation_seed=321
    )
    execution.execute_evaluation_batch(evaluation)
    target = evaluation[0].output_directory
    for name in ("summary.json", "base-fixed/summary.json", "base/summary.json"):
        summary = json.loads((target / name).read_text())
        assert summary["metadata"]["seed"] == 321
        assert summary["metadata"]["episode_count"] == len(summary["episodes"]) == 3
    base = json.loads((target / "base/summary.json").read_text())
    assert base["metadata"]["model_sha256"] == checksums["base.zip"]
    assert base["metadata"]["variant"] == "base"
    previous = {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    with pytest.raises(execution.PreflightError, match="already exists"):
        execution.execute_evaluation_batch(evaluation)
    write = execution._write_evaluation_directory

    def fail_fixed(path, *args, **kwargs):
        if path.name == "base-fixed":
            raise RuntimeError("fixed writer failed")
        return write(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(execution, "_write_evaluation_directory", fail_fixed)
        with pytest.raises(RuntimeError, match="fixed writer failed"):
            execution.execute_evaluation_batch(evaluation, overwrite=True)
    assert previous == {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    assert not list(target.parent.glob(".*.tmp"))
    changed = execution.plan_evaluation_batch(
        tmp_path, registry=registry, name="paired", episodes=1, evaluation_seed=322
    )
    execution.execute_evaluation_batch(changed, overwrite=True)
    for name in ("summary.json", "base-fixed/summary.json", "base/summary.json"):
        summary = json.loads((target / name).read_text())
        assert summary["metadata"]["seed"] == 322
        assert summary["metadata"]["episode_count"] == len(summary["episodes"]) == 1
    assert checksums == {name: execution.sha256_file(plan.run_directory / name) for name in checksums}


def test_fixed_evaluation_advances_trapped_history_through_end_and_reset(dependencies):
    from experiments.often import _FixEvaluationEnv

    resolved = tiny_registry(experiment_id=PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID).resolve(
        PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID
    )
    with _FixEvaluationEnv(resolved.make_env(training=False), resolved) as env:
        observation, info = env.reset(seed=7)
        first = env.session
        first.problem(observation, info)
        for _ in range(resolved.specification.run.max_episode_steps):
            observation, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        assert terminated or truncated
        with pytest.raises(ValueError, match="ended"):
            first.problem(observation, info)
        observation, info = env.reset(seed=8)
        assert env.session is not first
        env.session.problem(observation, info)


def test_reference_evaluation_does_not_change_teaching_rng(tmp_path, dependencies, monkeypatch):
    import random

    import numpy as np
    from stable_baselines3 import DQN

    from experiments import often

    reference = often.evaluate_base_fixed
    registry = tiny_registry()
    models = []
    stats = []

    def noisy_reference(*args, **kwargs):
        random.random()
        np.random.random(20)
        dependencies.rand(20)
        return reference(*args, **kwargs)

    for index, evaluator in enumerate((reference, noisy_reference)):
        monkeypatch.setattr(often, "evaluate_base_fixed", evaluator)
        (plan,) = execution.plan_training_batch(tmp_path / str(index), registry=registry, seeds=[0])
        execution.execute_training_batch([plan])
        models.append(DQN.load(plan.model_path, device="cpu").policy.state_dict())
        stats.append(json.loads(plan.run_path.read_text())["teaching"]["stats"])
    assert stats[0] == stats[1]
    assert all(dependencies.equal(value, models[1][key]) for key, value in models[0].items())


@pytest.mark.parametrize("identifier", ["pacman-dqn-often-v2", "pacman-dqn-often-vegetarian-v2"])
def test_paper_often_matches_main_dqn_base_and_separates_phase_limits(tmp_path, dependencies, monkeypatch, identifier):
    from stable_baselines3 import DQN

    from experiments.paper import make_registry
    from experiments.specifications import PacmanOFTENSettings

    source = make_registry()
    resolved = source.resolve(identifier)
    main = source.resolve("pacman-dqn-unconstrained-v0")
    assert resolved.algorithm == main.algorithm
    assert resolved.wrappers == main.wrappers
    settings = resolved.specification.often
    assert isinstance(settings, PacmanOFTENSettings)
    assert settings.pretraining_steps == 5_000_000
    assert settings.pretraining_max_episode_steps == 300
    assert settings.evaluation_max_episode_steps == 300
    assert resolved.specification.run.training_steps == 10_000_000
    assert resolved.specification.run.max_episode_steps == 500
    assert resolved.algorithm.constructor_kwargs.to_dict()["gamma"] == 0.99
    config = execution.resolved_configuration(resolved)
    assert execution.resolved_configuration(execution.resolved_from_configuration(config, registry=source)) == config
    trapped = source.resolve("pacman-dqn-often-trapped-v3")
    assert trapped.specification.run.max_episode_steps == 500
    assert trapped.specification.run.training_steps == 10_000_000
    assert trapped.algorithm == main.algorithm

    specification = replace(
        resolved.specification,
        run=replace(
            resolved.specification.run,
            training_steps=256,
            max_episode_steps=6,
            intermediate_evaluation_frequency=128,
            intermediate_evaluation_episodes=2,
            final_evaluation_episodes=2,
        ),
        often=replace(settings, pretraining_steps=128, pretraining_max_episode_steps=3, evaluation_max_episode_steps=4),
    )
    algorithm = replace(
        resolved.algorithm,
        constructor_kwargs=KeywordArguments.from_mapping(
            resolved.algorithm.constructor_kwargs.to_dict() | {"buffer_size": 1024, "verbose": 0}
        ),
    )
    registry = ExperimentRegistry(
        environments=[resolved.environment],
        wrappers=[replace(w, environment_ids=frozenset({resolved.environment.id})) for w in resolved.wrappers],
        scenarios=[replace(resolved.scenario, environment_ids=frozenset({resolved.environment.id}))],
        algorithms=[replace(algorithm, environment_ids=frozenset({resolved.environment.id}))],
        techniques=[resolved.technique],
        experiments=[specification],
    )
    original = execution._make_sb3_model
    training_limits = []

    def make_model(resolved, env, seed):
        training_limits.append(env.get_wrapper_attr("_max_episode_steps"))
        return original(resolved, env, seed)

    monkeypatch.setattr(execution, "_make_sb3_model", make_model)
    previous_threads = dependencies.get_num_threads()
    dependencies.set_num_threads(1)
    try:
        (plan,) = execution.plan_training_batch(tmp_path / "runs", registry=registry, seeds=[0])
        execution.execute_training_batch([plan])
        assert training_limits == [3]
        with plan.resolved.make_env(training=True) as teaching, plan.resolved.make_env(training=False) as evaluation:
            assert teaching.get_wrapper_attr("_max_episode_steps") == 6
            assert evaluation.get_wrapper_attr("_max_episode_steps") == 4
        for path in (plan.run_directory / "evaluations").glob("intermediate-*/summary.json"):
            result = json.loads(path.read_text())
            assert result["metadata"]["episode_limit"] == 4
            assert max(episode["length"] for episode in result["episodes"]) <= 4
        from experiments.often import _FixEvaluationEnv
        from npc_gym.policy_fixes import PacmanModel

        remaining = []
        problem = PacmanModel.problem

        def record_deadline(model, state, *, remaining_steps):
            remaining.append(remaining_steps)
            return problem(model, state, remaining_steps=remaining_steps)

        monkeypatch.setattr(PacmanModel, "problem", record_deadline)
        with _FixEvaluationEnv(plan.resolved.make_env(training=False), plan.resolved) as env:
            observation, info = env.reset(seed=0)
            assert env.get_wrapper_attr("_max_episode_steps") == 4
            env.session.problem(observation, info)
            assert remaining == [4]
            observation, _, _, _, info = env.step(4)
            env.session.problem(observation, info)
            assert remaining == [4, 3]
        base = DQN.load(plan.run_directory / "base.zip", device="cpu")
        final = DQN.load(plan.model_path, device="cpu")
        assert base.gamma == final.gamma == 0.99
        assert base.observation_space.shape == final.observation_space.shape == (63,)
        assert base.action_space.n == final.action_space.n == 5
        document = json.loads(plan.run_path.read_text())
        assert document["pretraining"]["actual_steps"] == 128
        assert document["teaching"]["stats"]["ordinary_steps"] == 128
        assert document["teaching"]["stats"]["expert_steps"] == 128
        assert document["teaching"]["stats"]["updates"] == 7
        base_recipe = replace(
            main,
            algorithm=algorithm,
            specification=replace(main.specification, run=replace(main.specification.run, max_episode_steps=3)),
        )
        with base_recipe.make_env(training=True) as env:
            reference = original(base_recipe, env, 0)
            reference.learn(128)
        for name, value in base.q_net.state_dict().items():
            assert dependencies.equal(value, reference.q_net.state_dict()[name]), name
    finally:
        dependencies.set_num_threads(previous_threads)


@pytest.mark.parametrize("value", [0, -1, True, 3.5])
@pytest.mark.parametrize("field", ["pretraining_max_episode_steps", "evaluation_max_episode_steps"])
def test_pacman_often_limit_validation(value, field):
    from experiments.specifications import PacmanOFTENSettings

    with pytest.raises(SpecificationError, match=field):
        PacmanOFTENSettings(**{field: value})
