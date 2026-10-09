import optuna
import pytest

from npc_gym.evaluation import EpisodeResult, EvaluationSummary, TerminationClass
from npc_gym.integrations.optuna import OptunaReporter


def make_summary() -> EvaluationSummary:
    return EvaluationSummary.from_episodes(
        (EpisodeResult(0, 5.0, 2, TerminationClass.TERMINATED, {"score": 8}, {"test/norm-v0": {"count": 3}}),)
    )


class FakeTrial:
    def __init__(self, *, prune: bool = False) -> None:
        self.prune = prune
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune


def test_reporter_consumes_a_summary_with_a_caller_selected_objective():
    trial = FakeTrial()
    reporter = OptunaReporter(trial, objective=lambda summary: -summary.mean_monitor_counts["test/norm-v0"]["count"])

    reporter(12, make_summary())

    assert trial.reports == [(-3.0, 12)]


def test_reporter_uses_mean_return_by_default_and_honors_pruning():
    trial = FakeTrial(prune=True)
    reporter = OptunaReporter(trial)

    with pytest.raises(optuna.TrialPruned, match="step 7"):
        reporter(7, make_summary())

    assert trial.reports == [(5.0, 7)]


@pytest.mark.parametrize("step", [True, -1, 1.5])
def test_reporter_rejects_invalid_steps(step):
    with pytest.raises(ValueError, match="non-negative integer"):
        OptunaReporter(FakeTrial())(step, make_summary())


def test_reporter_requires_the_optuna_extra_when_constructed(monkeypatch):
    from npc_gym.integrations.optuna import reporter

    def missing_optuna(name):
        raise ModuleNotFoundError("No module named 'optuna'", name="optuna")

    monkeypatch.setattr(reporter, "import_module", missing_optuna)

    with pytest.raises(ImportError, match=r"npc-gym\[optuna\]"):
        OptunaReporter(FakeTrial())
