"""Import-safety checks for the standalone examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_importing_examples_has_no_runtime_side_effects(tmp_path: Path) -> None:
    example_dir = Path(__file__).resolve().parents[2] / "examples"
    script_paths = sorted(example_dir.glob("*.py"))
    command = (
        "import runpy, sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]).parent)); "
        "[runpy.run_path(path, run_name='npc_gym_example_import') for path in sys.argv[1:]]"
    )

    result = subprocess.run(
        [sys.executable, "-c", command, *map(str, script_paths)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert [path.name for path in script_paths] == [
        "custom_environment.py",
        "gardener.py",
        "merchant.py",
        "merchant_policy_fixes.py",
        "pacman.py",
        "pacman_images.py",
        "pacman_often.py",
        "play_pacman.py",
        "plot_learning_curves.py",
        "taxi.py",
        "taxi_policy_fixes.py",
    ]
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
