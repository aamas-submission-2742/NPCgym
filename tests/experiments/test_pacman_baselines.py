"""Shared baseline evaluation keeps the paper's six monitored norm bases."""

import pytest

from experiments.pacman_baselines import MONITORS, compare_episodes, plan_evaluations
from experiments.pacman_bolts import RECIPES
from npc_gym.evaluation import EpisodeResult, EvaluationSummary
from npc_gym.evaluation._recording import TerminationClass


def test_paper_monitor_selection_and_separate_output(tmp_path):
    assert MONITORS == tuple(RECIPES.values())
    assert len(MONITORS) == 6
    with pytest.raises(ValueError, match="separate"):
        plan_evaluations(tmp_path, tmp_path / "new")
    with pytest.raises(ValueError, match="separate"):
        plan_evaluations(tmp_path / "runs", tmp_path)


def test_replay_comparison_checks_common_norms_and_task_trajectory():
    current = EvaluationSummary.from_episodes(
        (EpisodeResult(0, 42.0, 3, TerminationClass.TERMINATED, {"won": 1}, {name: {"count": 2} for name in MONITORS}),)
    )
    old = {
        "episodes": [
            {
                "return": 42.0,
                "length": 3,
                "termination": "terminated",
                "metrics": {"won": 1},
                "monitor_counts": {MONITORS[0]: {"count": 2}, "pacman/obsolete-v0": {"count": 1}},
            }
        ]
    }
    assert compare_episodes(current, old) == {"task_mismatched_episodes": [], "norm_mismatched_episode_counts": {}}
    altered = {"episodes": [{**old["episodes"][0], "return": 43.0, "monitor_counts": {MONITORS[0]: {"count": 3}}}]}
    assert compare_episodes(current, altered) == {
        "task_mismatched_episodes": [0],
        "norm_mismatched_episode_counts": {MONITORS[0]: 1},
    }
    with pytest.raises(ValueError, match="counts differ"):
        compare_episodes(current, {"episodes": []})
