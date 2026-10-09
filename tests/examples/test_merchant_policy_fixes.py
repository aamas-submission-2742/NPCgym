"""Exercise the Merchant comparison's real planning and result artifacts."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from npc_gym.algorithms import TabularQLearning
from npc_gym.policy_fixes import ASPPlanner

if os.environ.get("NPC_GYM_REQUIRE_ASP") == "1":
    import clingo  # noqa: F401
else:
    pytest.importorskip("clingo")


@pytest.fixture
def example(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "examples" / "merchant_policy_fixes.py"
    spec = importlib.util.spec_from_file_location("merchant_fixes_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in {
        "TRAINING_STEPS": 32,
        "EVALUATION_EPISODES": 2,
        "MAX_EPISODE_STEPS": 8,
        "SEED": 0,
    }.items():
        monkeypatch.setattr(module, key, value)
    monkeypatch.delenv("NPC_GYM_OUTPUT_DIR", raising=False)
    return module


def test_real_solves_learning_persistence_and_paired_artifacts(example, monkeypatch, tmp_path):
    output = tmp_path / "results"
    monkeypatch.setenv("NPC_GYM_OUTPUT_DIR", str(output))
    monkeypatch.chdir(tmp_path)
    saved = []
    decisions = []
    save = TabularQLearning.save
    solve = ASPPlanner.solve

    def record_save(self, path):
        assert self.use_action_mask
        assert self.num_timesteps == 32
        assert any(np.any(row != 0) for row in self.q_table.values())
        saved.append(Path(path))
        return save(self, path)

    def record_solve(self, *args, **kwargs):
        decision = solve(self, *args, **kwargs)
        assert decision.optimal and decision.costs["violations"] == 0
        decisions.append(decision)
        return decision

    monkeypatch.setattr(TabularQLearning, "save", record_save)
    monkeypatch.setattr(ASPPlanner, "solve", record_solve)
    summaries = example.main()
    assert set(summaries) == {"base", "fixed"}
    assert len(decisions) == sum(result.length for result in summaries["fixed"].episodes)
    assert saved and all(not path.exists() for path in saved)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["results"]
    assert sorted(path.name for path in output.iterdir()) == ["base.json", "fixed.json"]
    for variant, summary in summaries.items():
        document = json.loads((output / f"{variant}.json").read_text())
        metadata = document["metadata"]
        assert metadata["evaluation_seeds"] == [10_000, 10_001]
        assert metadata["training_steps"] == 32
        assert metadata["teaching_steps"] == 0
        assert metadata["norm_id"] == "merchant/env-friendly-v0"
        assert metadata["horizon"] == 1
        assert metadata["evaluation_steps"] == sum(result.length for result in summary.episodes)
        assert metadata["interventions"] == (sum(d.changed for d in decisions) if variant == "fixed" else 0)
        assert document["aggregate"]["monitor_counts"]
        assert [result["episode"] for result in document["episodes"]] == [0, 1]
    before = (output / "base.json").read_bytes()
    with pytest.raises(FileExistsError):
        example.main()
    assert (output / "base.json").read_bytes() == before


def test_failure_cleans_up_temporary_policy_and_environments(example, monkeypatch, tmp_path):
    paths, closed = [], []
    save = TabularQLearning.save
    close = example.MerchantEnv.close

    def record_save(self, path):
        paths.append(Path(path))
        save(self, path)

    def record_close(self):
        closed.append(id(self))
        close(self)

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(TabularQLearning, "save", record_save)
    monkeypatch.setattr(example.MerchantEnv, "close", record_close)
    monkeypatch.setattr(example, "paired_evaluation", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        example.main()
    assert len(set(closed)) == 3
    assert paths and all(not path.exists() for path in paths)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("setting", ["TRAINING_STEPS", "EVALUATION_EPISODES", "MAX_EPISODE_STEPS", "SEED"])
def test_invalid_configuration_fails_before_creating_output(example, monkeypatch, tmp_path, setting):
    monkeypatch.setattr(example, setting, -1)
    monkeypatch.setenv("NPC_GYM_OUTPUT_DIR", str(tmp_path / "results"))
    with pytest.raises(ValueError, match="NPC_GYM_"):
        example.main()
    assert list(tmp_path.iterdir()) == []
