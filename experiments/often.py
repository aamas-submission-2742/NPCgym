"""OFTEN's two training phases for the versioned experiment runner."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import gymnasium as gym

from experiments import execution
from experiments.specifications import GardenerOFTENSettings, OFTENSettings, PacmanOFTENSettings, ResolvedExperiment
from npc_gym.algorithms import TeachingSession, TeachingStep
from npc_gym.envs import GardenerEnv, PacmanEnv
from npc_gym.evaluation import EvaluationSummary, LearningCurveCSVWriter, evaluate
from npc_gym.monitors import MonitorInput
from npc_gym.monitors.pacman_monitors import TRAPPED_NORM_ID
from npc_gym.policy_fixes import ASPPlanner, GardenerModel, PacmanModel, PacmanTrappedModel, PlanningProblem

_LOGGER = logging.getLogger(__name__)


def _sessions(
    env: gym.Env[Any, Any], settings: OFTENSettings, episode_limit: int, norm_id: str
) -> Callable[[Any, Mapping[str, Any]], TeachingSession]:
    raw = cast(PacmanEnv | GardenerEnv, env.unwrapped)

    planner: ASPPlanner | None = None

    def start(observation: Any, info: Mapping[str, Any]) -> TeachingSession:
        nonlocal planner
        model: GardenerModel | PacmanModel | PacmanTrappedModel
        if isinstance(raw, GardenerEnv):
            model = GardenerModel(
                raw.labeling_state(),
                norm_id=norm_id,
                horizon=settings.horizon,
                radius=settings.radius,
                puddle_respawn=raw.puddle_respawn,
                frog_freeze=raw.frog_freeze,
            )
        elif norm_id == TRAPPED_NORM_ID:
            model = PacmanTrappedModel(MonitorInput(info["labels"]))
        else:
            model = PacmanModel(raw.labeling_state(), norm_id=norm_id, horizon=settings.horizon, radius=settings.radius)
        if planner is None:
            planner = ASPPlanner(objectives=model.objectives) if isinstance(model, GardenerModel) else ASPPlanner()
        remaining = episode_limit

        def problem(observation: Any, info: Mapping[str, Any]) -> PlanningProblem:
            if isinstance(model, GardenerModel):
                assert isinstance(raw, GardenerEnv)
                return model.problem(raw.labeling_state(), remaining_steps=remaining)
            assert isinstance(raw, PacmanEnv)
            if isinstance(model, PacmanTrappedModel):
                return model.problem(raw.labeling_state())
            return model.problem(raw.labeling_state(), remaining_steps=remaining)

        def advance(actual: MonitorInput) -> None:
            nonlocal remaining
            remaining -= 1
            if isinstance(model, PacmanTrappedModel):
                model.advance(actual)

        return TeachingSession(planner, problem, advance)

    return start


class _FixEvaluationEnv(gym.Wrapper[Any, Any, Any, Any]):
    """Give runtime fixes the same reset/advance lifecycle as OFTEN teaching."""

    def __init__(self, env: gym.Env[Any, Any], resolved: ResolvedExperiment) -> None:
        super().__init__(env)
        settings = resolved.specification.often
        assert settings is not None
        limit = (
            settings.evaluation_max_episode_steps
            if isinstance(settings, PacmanOFTENSettings)
            else resolved.specification.run.max_episode_steps
        )
        self.start = _sessions(env, settings, limit, settings.norm_id)
        self.session: TeachingSession | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.session = self.start(observation, info)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        assert self.session is not None
        if self.session.advance is not None:
            self.session.advance(MonitorInput(info["labels"], terminated, truncated))
        return observation, float(reward), terminated, truncated, info


def evaluate_base_fixed(
    resolved: ResolvedExperiment, base_path: Path, *, episodes: int, seed: int, render: bool = False
) -> tuple[EvaluationSummary, dict[str, object]]:
    """Evaluate the saved base with OFTEN's fixes on independent paired seeds.

    Uses greedy base Q-values, raw task rewards, and the same planner recipes,
    objectives, horizon and radius as teaching. Records intervention counts and
    runtime without changing the saved checkpoint or training any policy.
    """
    import torch

    settings = resolved.specification.often
    assert settings is not None
    started = perf_counter()
    with closing(_FixEvaluationEnv(resolved.make_env(training=False, render=render), resolved)) as env:
        model = execution._load_model(resolved, base_path, env)
        model.policy.set_training_mode(False)
        interventions = fallbacks = decisions = 0
        planning_seconds = 0.0

        def choose(observation: Any, info: Mapping[str, Any]) -> int:
            nonlocal interventions, fallbacks, decisions, planning_seconds
            before = perf_counter()
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(observation)
                values = model.q_net(tensor).cpu().numpy()[0]
            session = env.session
            assert session is not None
            decision = session.planner.solve(session.problem(observation, info), dict(enumerate(map(float, values))))
            planning_seconds += perf_counter() - before
            interventions += int(decision.changed)
            fallbacks += int(not decision.optimal)
            decisions += 1
            return decision.action

        results = []
        for index in range(episodes):
            summary = evaluate(
                env, choose, episodes=1, monitors=dict(resolved.scenario.monitor_factories), seed=seed + index
            )
            results.append(replace(summary.episodes[0], episode=index))
        assert env.session is not None
        metadata = execution._evaluation_metadata(
            resolved,
            episodes=episodes,
            seed=seed,
            model_source="saved-base-checkpoint",
            model_checksum=execution.sha256_file(base_path),
            timesteps=model.num_timesteps,
        )
        metadata.update(
            {
                "variant": "base-fixed",
                "timesteps_unit": "base DQN environment steps",
                "policy_fix": {
                    "norm_id": settings.norm_id,
                    "horizon": settings.horizon,
                    "radius": settings.radius,
                    "use_action_mask": False,
                    "reuse_solver": True,
                    "objectives": {
                        name: asdict(objective) for name, objective in env.session.planner.objectives.items()
                    },
                },
                "interventions": interventions,
                "fallbacks": fallbacks,
                "decisions": decisions,
                "planning_seconds": planning_seconds,
                "evaluation_seconds": perf_counter() - started,
            }
        )
        return EvaluationSummary.from_episodes(results), metadata


def train_often(
    plan: execution.TrainingPlan,
    document: dict[str, Any],
    curve: LearningCurveCSVWriter,
    pretraining_env: gym.Env[Any, Any],
) -> Any:
    """Train/save/reload a base, then teach it; curve steps count teaching only.

    Persist phase times and teacher counters even on failure. Evaluations preserve
    RNGs and network modes, and their duration is excluded from teaching time.
    The caller owns the base environment, final model and completion metadata.
    """
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from npc_gym.integrations.sb3 import DQNOFTEN
    from npc_gym.integrations.sb3.callback import _preserve_training_state

    resolved = plan.resolved
    run = resolved.specification.run
    settings = resolved.specification.often
    assert settings is not None
    norm_id = settings.norm_id
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _LOGGER.info(
            "%s seed %s: pretraining %s steps", resolved.specification.id, plan.seed, settings.pretraining_steps
        )
        model = execution._make_sb3_model(resolved, pretraining_env, plan.seed)
        pretraining_steps = settings.pretraining_steps
        base_resolved = resolved
        base_seed = plan.intermediate_evaluation_seed
        base_frequency = run.intermediate_evaluation_frequency
        if isinstance(settings, PacmanOFTENSettings):
            base_seed = plan.seed + settings.pretraining_evaluation_seed_offset
            base_frequency = settings.pretraining_evaluation_frequency
            base_resolved = replace(
                resolved,
                specification=replace(
                    resolved.specification,
                    often=None,
                    run=replace(run, max_episode_steps=settings.pretraining_max_episode_steps),
                ),
            )
        base_evaluation_seconds = 0.0
        base_steps: set[int] = set()

        with LearningCurveCSVWriter(plan.run_directory / "base_learning_curve.csv") as base_curve:

            def record_base() -> None:
                nonlocal base_evaluation_seconds
                before = perf_counter()
                steps = model.num_timesteps
                with _preserve_training_state(model):
                    summary = execution._evaluate_model(
                        base_resolved,
                        model,
                        episodes=run.intermediate_evaluation_episodes,
                        seed=base_seed,
                    )
                metadata = execution._evaluation_metadata(
                    base_resolved,
                    episodes=len(summary.episodes),
                    seed=base_seed,
                    model_source="in-memory-base-policy",
                    model_checksum=None,
                    timesteps=steps,
                )
                metadata["timesteps_unit"] = "base DQN environment steps"
                execution._write_evaluation_directory(
                    plan.run_directory / "evaluations" / "base" / f"intermediate-{steps}",
                    summary,
                    metadata,
                    overwrite=False,
                )
                base_curve.write(steps, summary)
                base_steps.add(steps)
                base_evaluation_seconds += perf_counter() - before

            class BaseCurveCallback(BaseCallback):
                def _on_step(self) -> bool:
                    if self.num_timesteps < pretraining_steps and self.num_timesteps % base_frequency == 0:
                        record_base()
                    return True

            record_base()
            base_evaluation_seconds = 0.0
            started = perf_counter()
            try:
                model.learn(settings.pretraining_steps, log_interval=None, callback=BaseCurveCallback())
                if model.num_timesteps not in base_steps:
                    record_base()
            finally:
                document["pretraining"] = {
                    "requested_steps": settings.pretraining_steps,
                    "actual_steps": model.num_timesteps,
                    "updates": model._n_updates,
                    "seconds": perf_counter() - started - base_evaluation_seconds,
                }
                execution._write_json_atomic(plan.run_path, document)
        base = plan.run_directory / "base.zip"
        execution._save_model_staged(model, base)
        document["artifacts"]["base"] = {"path": base.name, "sha256": execution.sha256_file(base)}
        del model

        with closing(resolved.make_env(training=True)) as ordinary, closing(resolved.make_env(training=True)) as expert:
            model = execution._load_model(resolved, base, ordinary)
            final_expert_weight = 1.0
            if isinstance(settings, GardenerOFTENSettings):
                model.batch_size = settings.batch_size
                model.gradient_steps = settings.gradient_steps
                final_expert_weight = settings.final_expert_weight
            with _preserve_training_state(model):
                base_fixed = evaluate_base_fixed(
                    resolved,
                    base,
                    episodes=run.intermediate_evaluation_episodes,
                    seed=plan.intermediate_evaluation_seed,
                )
            evaluation_seconds = 0.0

            def record(steps: int) -> None:
                nonlocal evaluation_seconds
                before = perf_counter()
                with _preserve_training_state(model):
                    summary = execution._evaluate_model(
                        resolved,
                        model,
                        episodes=run.intermediate_evaluation_episodes,
                        seed=plan.intermediate_evaluation_seed,
                    )
                execution._record_intermediate(
                    plan, document, curve, (steps,), steps, summary, base_fixed_evaluation=base_fixed
                )
                evaluation_seconds += perf_counter() - before

            record(0)
            teacher = DQNOFTEN(
                model,
                ordinary,
                expert,
                ordinary_session=_sessions(ordinary, settings, run.max_episode_steps, norm_id),
                expert_session=_sessions(expert, settings, run.max_episode_steps, norm_id),
                margin=settings.margin,
                final_expert_weight=final_expert_weight,
                seed=plan.seed + settings.teaching_seed_offset,
            )
            evaluation_seconds = 0.0
            started = perf_counter()

            def progress(step: TeachingStep) -> None:
                total = step.stats.total_steps
                if total % run.intermediate_evaluation_frequency == 0:
                    record(total)
                    _LOGGER.info(
                        "%s seed %s: OFTEN %s/%s transitions",
                        resolved.specification.id,
                        plan.seed,
                        total,
                        run.training_steps,
                    )

            try:
                teacher.learn(run.training_steps, callback=progress)
            finally:
                document["timesteps"]["actual"] = teacher.stats.total_steps
                document["teaching"] = {
                    "seed": plan.seed + settings.teaching_seed_offset,
                    "seconds": perf_counter() - started - evaluation_seconds,
                    "stats": asdict(teacher.stats),
                }
                execution._write_json_atomic(plan.run_path, document)
            return model
    finally:
        torch.set_num_threads(previous_threads)
