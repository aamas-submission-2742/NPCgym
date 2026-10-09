import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from npc_gym.monitors.gardener_monitors import GARDENER_NORM_IDS
from npc_gym.monitors.merchant_monitors import merchant_monitor_registry
from npc_gym.monitors.pacman_monitors import pacman_monitor_registry
from npc_gym.monitors.taxi_monitors import taxi_monitor_registry

MONITOR_ID = re.compile(r"``((?:taxi|merchant|gardener|pacman)/[a-z0-9-]+-v[0-9]+)``")


def test_monitor_catalogue_matches_public_factories():
    pages = [Path("docs/monitor_catalogue.rst"), *sorted(Path("docs/catalogue").glob("*.rst"))]
    catalogue = "\n".join(page.read_text(encoding="utf-8") for page in pages)
    documented_ids = MONITOR_ID.findall(catalogue)
    registered_ids = {
        monitor_id
        for registry in (
            taxi_monitor_registry,
            merchant_monitor_registry,
            pacman_monitor_registry,
        )
        for monitor_id in registry.ids
    }
    # Gardener's configurable object recipes require sizes at construction.
    registered_ids.update(GARDENER_NORM_IDS)

    assert len(documented_ids) == len(set(documented_ids))
    assert set(documented_ids) == registered_ids


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.append(dict(attrs).get("href", ""))


@pytest.fixture
def sphinx_source(tmp_path):
    pytest.importorskip("sphinx", reason="Sphinx link checks run in the Tox docs environment")
    source = tmp_path / "source"
    (source / "guide").mkdir(parents=True)
    (source / "conf.py").write_text(Path("docs/conf.py").read_text(encoding="utf-8"), encoding="utf-8")
    (source / "index.rst").write_text(
        "Documentation\n=============\n\n.. toctree::\n\n   guide/page\n   target\n", encoding="utf-8"
    )
    (source / "target.rst").write_text(
        "Target\n======\n\n.. _details:\n\nDetails\n-------\n\nTarget content.\n", encoding="utf-8"
    )
    (source / "guide/page.rst").write_text(
        "Guide\n=====\n\n.. _local:\n\nLocal\n-----\n\n"
        "`Page <../target.rst>`__\n\n"
        "`Details <../target.rst#details>`__\n\n"
        "`Local section <#local>`__\n\n"
        "`README <../../README.md>`__\n\n"
        "`External <https://example.org/reference.rst#section>`__\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Download content.\n", encoding="utf-8")
    return source


@pytest.mark.parametrize("builder", ["html", "dirhtml"])
def test_sphinx_resolves_portable_links_and_copies_downloads(sphinx_source, tmp_path, builder):
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-q", "-W", "-b", builder, str(sphinx_source), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    page = output / ("guide/page.html" if builder == "html" else "guide/page/index.html")
    links = _Links()
    links.feed(page.read_text(encoding="utf-8"))
    target = "../target.html" if builder == "html" else "../../target/"
    assert target in links.hrefs
    assert f"{target}#details" in links.hrefs
    assert "#local" in links.hrefs
    assert "https://example.org/reference.rst#section" in links.hrefs
    downloads = [href for href in links.hrefs if "_downloads/" in href]
    assert len(downloads) == 1
    assert (page.parent / urlsplit(downloads[0]).path).read_text(encoding="utf-8") == "Download content.\n"


@pytest.mark.parametrize(
    ("link", "warning"),
    [
        ("../missing.rst", "document link target not found"),
        ("../target.rst#missing", "document link anchor not found"),
        ("#missing", "document link anchor not found"),
        ("../../missing.md", "download file not readable"),
    ],
)
def test_sphinx_rejects_broken_portable_links(sphinx_source, tmp_path, link, warning):
    with (sphinx_source / "guide/page.rst").open("a", encoding="utf-8") as page:
        page.write(f"\n`Broken <{link}>`__\n")
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-q", "-W", "-b", "html", str(sphinx_source), str(tmp_path / "output")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert warning in result.stderr
