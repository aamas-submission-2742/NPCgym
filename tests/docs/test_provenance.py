"""Check bundled map separation and source-only experiment assets."""

from pathlib import Path


def test_library_maps_are_separate_from_paper_layout():
    root = Path("src/npc_gym/envs/pacman")
    assert not list((root / "_engine").glob("*.py"))
    assert {path.name for path in (root / "layouts").glob("*.lay")} == {"small.lay", "medium.lay", "large.lay"}
    assert Path("experiments/layouts/smallClassic.lay").is_file()
    assert Path("envs.png").is_file()
    assert "![NPC Gym environments](envs.png)" in Path("README.md").read_text(encoding="utf-8")
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")
    assert "exclude envs.png" in manifest
    assert "exclude experiments/layouts/smallClassic.lay" in manifest
