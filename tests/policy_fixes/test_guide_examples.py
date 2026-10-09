"""Execute the documented examples in the ASP gate, including their assertions."""

import os
import re
from pathlib import Path

import pytest

if os.environ.get("NPC_GYM_REQUIRE_ASP") == "1":
    import clingo  # noqa: F401
else:
    pytest.importorskip("clingo")

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("reuse_solver", [True, False])
@pytest.mark.parametrize(
    "page,section,count",
    [
        ("policy_fixes.rst", "", 4),
        ("catalogue/pacman.rst", ".. _catalogue-pacman-policy-fixes:", 3),
        ("catalogue/gardener.rst", ".. _catalogue-gardener-policy-fixes:", 2),
        ("catalogue/taxi.rst", ".. _catalogue-taxi-policy-fixes:", 1),
        ("catalogue/merchant.rst", ".. _catalogue-merchant-policy-fixes:", 1),
    ],
)
def test_policy_fix_guide_examples(page, section, count, reuse_solver):
    source = (ROOT / "docs" / page).read_text(encoding="utf-8")
    if section:
        source = source.split(section, 1)[1]
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:   [^\n]*\n|\n)+)", source)
    assert len(blocks) == count  # Fail if document structure silently hides an example.
    namespace = {}
    for block in blocks:
        code = "\n".join(line[3:] for line in block.splitlines())
        if not reuse_solver:
            code = code.replace("ASPPlanner(", "ASPPlanner(reuse_solver=False, ")
        exec(compile(code, page, "exec"), namespace)  # noqa: S102 -- trusted, repository-owned examples
