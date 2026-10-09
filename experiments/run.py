"""Command-line entry point for safe NPC Gym experiment execution."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.execution import (
    EvaluationPlan,
    ExecutionError,
    TrainingPlan,
    execute_evaluation_batch,
    execute_training_batch,
    plan_as_dict,
    plan_evaluation_batch,
    plan_training_batch,
    preflight_evaluation,
    preflight_training,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the import-safe train/evaluate command parser."""
    parser = argparse.ArgumentParser(description="Run versioned NPC Gym experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilots = subparsers.add_parser("pacman-bolt-pilots", help="queue budget pilots and Hungry Vegan Penalty final runs")
    pilots.add_argument("--output", required=True, type=Path, help="fresh experiment output root")
    pilots.add_argument("--cpus", nargs="+", required=True, type=int, help="one worker per listed Linux CPU")
    pilots.add_argument("--dry-run", action="store_true", help="validate and print all 52 plans without writing")

    baselines = subparsers.add_parser(
        "pacman-baselines", help="evaluate existing smallClassic baselines with the six paper norm bases"
    )
    baselines.add_argument(
        "--source", required=True, type=Path, help="original complete baseline runs with checkpoints"
    )
    baselines.add_argument("--output", required=True, type=Path, help="fresh evaluation-only output directory")
    baselines.add_argument("--cpus", nargs="+", required=True, type=int, help="one evaluation worker per Linux CPU")

    train = subparsers.add_parser("train", help="train, save, reload, and evaluate selected experiments")
    _add_common_arguments(train)
    train.add_argument("--seed", action="append", required=True, type=int, help="explicit training seed; repeatable")
    train.add_argument("--cpus", nargs="+", type=int, help="run one single-threaded worker per listed Linux CPU")
    train.add_argument(
        "--threads", type=int, help="Torch threads for sequential training; defaults to the active setup"
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate existing saved checkpoints")
    _add_common_arguments(evaluate)
    evaluate.add_argument("--seed", action="append", type=int, default=[], help="saved training seed; repeatable")
    evaluate.add_argument("--name", required=True, help="safe result name; base, final and intermediate-* are reserved")
    evaluate.add_argument("--evaluation-seed", type=int, help="override the saved final-evaluation seed")
    evaluate.add_argument("--episodes", type=int, help="override the current recipe's final-evaluation episode count")
    evaluate.add_argument(
        "--current-monitors", action="store_true", help="evaluate saved policies with the current recipe's monitors"
    )
    evaluate.add_argument("--render", action="store_true", help="render evaluation episodes")
    report = subparsers.add_parser("report", help="compare methods by environment and norm base")
    report.add_argument(
        "--input", required=True, type=Path, help="completed experiment output or retained results root"
    )
    report.add_argument("--outdir", required=True, type=Path, help="new directory for tables and figures")
    report.add_argument(
        "--evaluation-name",
        default="final",
        help="saved evaluation to report; named evaluations show final points only",
    )
    report.add_argument("--benchmark", action="append", default=[], help="benchmark ID; repeatable")
    retain = subparsers.add_parser("retain", help="export completed paper results without checkpoints or logs")
    retain.add_argument("--input", action="append", required=True, type=Path, help="working run root; repeatable")
    retain.add_argument("--outdir", required=True, type=Path, help="fresh results root; adds an environment directory")
    retain.add_argument("--environment", required=True, choices=("taxi", "merchant", "gardener", "pacman"))
    retain.add_argument("--experiment", action="append", default=[], help="paper experiment ID; defaults to all")
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper", action="store_true", help="select only the configurations in the paper tables")
    parser.add_argument("--output", required=True, type=Path, help="experiment output root")
    parser.add_argument("--experiment", action="append", default=[], help="exact experiment ID; repeatable")
    parser.add_argument("--environment", action="append", default=[], help="exact environment ID filter; repeatable")
    parser.add_argument("--wrapper", action="append", default=[], help="required wrapper ID filter; repeatable")
    parser.add_argument("--scenario", action="append", default=[], help="exact scenario ID filter; repeatable")
    parser.add_argument("--algorithm", action="append", default=[], help="exact algorithm ID filter; repeatable")
    parser.add_argument("--technique", action="append", default=[], help="exact technique ID filter; repeatable")
    parser.add_argument("--overwrite", action="store_true", help="replace only selected, run-owned outputs")
    parser.add_argument("--dry-run", action="store_true", help="preflight and print plans without writing or training")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI command and return its process status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "retain":
        from experiments.results import retain_results

        try:
            target = retain_results(args.input, args.outdir, args.environment, experiment_ids=args.experiment)
            print(f"retained completed paper results: {target}")
            return 0
        except (ExecutionError, OSError, ValueError, TypeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    if args.command == "report":
        try:
            from experiments.report import create_report

            output = create_report(
                args.input, args.outdir, benchmark_ids=tuple(args.benchmark), evaluation_name=args.evaluation_name
            )
            print(f"created comparison tables and curves: {output / 'index.md'}")
            return 0
        except ImportError:
            print("error: comparison reports require 'npc-gym[plots]'", file=sys.stderr)
            return 2
        except (OSError, ValueError, TypeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    if args.command == "pacman-baselines":
        from experiments.pacman_baselines import execute_evaluations

        try:
            execute_evaluations(args.source, args.output, args.cpus)
            print(f"completed baseline evaluations: {args.output / 'batch.json'}")
            return 0
        except (OSError, ValueError, TypeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    if args.command == "pacman-bolt-pilots":
        from experiments.pacman_bolts import FINAL_SEEDS, PILOT_NORMS, PILOT_SEEDS
        from experiments.parallel import execute_parallel_training, validate_cpus
        from experiments.specifications import PACMAN_BOLT_STUDY_CONFIGURATIONS

        try:
            validate_cpus(args.cpus)
            plans = tuple(
                plan
                for norm, _algorithm, _steps, identifier in PACMAN_BOLT_STUDY_CONFIGURATIONS
                for plan in plan_training_batch(
                    args.output,
                    seeds=PILOT_SEEDS if norm in PILOT_NORMS else FINAL_SEEDS,
                    experiment_ids=[identifier],
                )
            )
            preflight_training(plans)
            if args.dry_run:
                _print_plans(plans)
            else:
                execute_parallel_training(plans, args.cpus)
                print(f"completed pilots and Hungry Vegan Penalty: {args.output / 'batch.json'}")
            return 0
        except (ExecutionError, OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    filters = {
        "experiment_ids": args.experiment,
        "environment_ids": args.environment,
        "wrapper_ids": args.wrapper,
        "scenario_ids": args.scenario,
        "algorithm_ids": args.algorithm,
        "technique_ids": args.technique,
    }
    if args.paper:
        from experiments.paper import make_registry

        filters["registry"] = make_registry()
    try:
        if args.command == "train":
            if args.cpus is not None:
                from experiments.parallel import validate_cpus

                validate_cpus(args.cpus)
                if args.overwrite or args.threads not in (None, 1):
                    raise ValueError("--cpus requires fresh batch output and uses one thread per worker")
            if args.threads is not None and args.threads < 1:
                raise ValueError("--threads must be positive")
            training_plans = plan_training_batch(args.output, seeds=args.seed, **filters)
            preflight_training(training_plans, overwrite=args.overwrite)
            if args.dry_run:
                _print_plans(training_plans)
                return 0
            if args.cpus is not None:
                from experiments.parallel import execute_parallel_training

                execute_parallel_training(training_plans, args.cpus, paper=args.paper)
                print(f"completed parallel batch: {args.output / 'batch.json'}")
                return 0
            if args.threads is not None and any(p.resolved.algorithm.execution_path == "sb3" for p in training_plans):
                import torch

                torch.set_num_threads(args.threads)
                if torch.get_num_interop_threads() != 1:
                    torch.set_num_interop_threads(1)
            training_results = execute_training_batch(training_plans, overwrite=args.overwrite)
            for training_result in training_results:
                print(
                    f"completed {training_result.run_directory}: {training_result.actual_timesteps} timesteps, "
                    f"return={training_result.final_evaluation.mean_return:g}"
                )
            return 0

        evaluation_plans = plan_evaluation_batch(
            args.output,
            seeds=args.seed,
            name=args.name,
            evaluation_seed=args.evaluation_seed,
            episodes=args.episodes,
            current_monitors=args.current_monitors,
            **filters,
        )
        preflight_evaluation(evaluation_plans, overwrite=args.overwrite)
        if args.dry_run:
            _print_plans(evaluation_plans)
            return 0
        evaluation_results = execute_evaluation_batch(evaluation_plans, overwrite=args.overwrite, render=args.render)
        for evaluation_result in evaluation_results:
            print(f"evaluated {evaluation_result.output_directory}: return={evaluation_result.summary.mean_return:g}")
        return 0
    except (ExecutionError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _print_plans(plans: Sequence[TrainingPlan | EvaluationPlan]) -> None:
    print(json.dumps([plan_as_dict(plan) for plan in plans], indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
