"""Report completed NPC Gym evaluations to Optuna."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from numbers import Integral
from typing import Protocol, cast

from npc_gym.evaluation import EvaluationSummary


class _Trial(Protocol):
    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...


_Objective = Callable[[EvaluationSummary], float]


def _mean_return(summary: EvaluationSummary) -> float:
    return summary.mean_return


class OptunaReporter:
    """Report an objective derived from each supplied evaluation summary.

    The reporter performs no evaluation and stores no summaries. By default the
    intermediate objective is ``summary.mean_return``; callers may supply a
    different scalar projection. Optuna's ``TrialPruned`` is raised when the
    trial requests pruning.
    """

    def __init__(self, trial: _Trial, *, objective: _Objective = _mean_return) -> None:
        if not callable(getattr(trial, "report", None)) or not callable(getattr(trial, "should_prune", None)):
            raise TypeError("trial must provide callable report() and should_prune() methods")
        if not callable(objective):
            raise TypeError("objective must be callable")
        self._trial = trial
        self._objective = objective
        self._trial_pruned = _load_trial_pruned()

    def __call__(self, step: int, summary: EvaluationSummary) -> None:
        """Report ``summary`` at a non-negative environment-transition count."""
        if not isinstance(summary, EvaluationSummary):
            raise TypeError(f"summary must be an EvaluationSummary, got {type(summary).__name__}")
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError(f"step must be a non-negative integer; got {step!r}")
        self._trial.report(float(self._objective(summary)), int(step))
        if self._trial.should_prune():
            raise self._trial_pruned(f"Trial pruned at evaluation step {int(step)}")


def _load_trial_pruned() -> type[Exception]:
    try:
        optuna = import_module("optuna")
    except ModuleNotFoundError as error:
        if error.name == "optuna":
            raise ImportError(
                "Optuna integration requires the 'optuna' extra; install it with `pip install 'npc-gym[optuna]'`."
            ) from None
        raise
    return cast(type[Exception], optuna.TrialPruned)


__all__ = ["OptunaReporter"]
