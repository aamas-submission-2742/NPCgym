"""The development timing utility has a tiny, output-free smoke gate."""

import json

import pytest

from experiments.benchmark_storm_taxi import main


def test_storm_benchmark_smoke(capsys):
    main(["--steps", "8", "--resets", "1", "--constructions", "1", "--repeats", "1"])
    result = json.loads(capsys.readouterr().out)
    assert result["medians"]["storm"]["hot_step_us"] > 0
    assert result["memory"]["storm"]["construction_peak_bytes"] > 0


def test_storm_benchmark_rejects_empty_workloads():
    with pytest.raises(SystemExit):
        main(["--steps", "0"])
