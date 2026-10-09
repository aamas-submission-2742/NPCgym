import posixpath
from urllib.parse import unquote, urlsplit

from docutils import nodes
from sphinx import addnodes
from sphinx.util import logging
from sphinx.util.nodes import make_refnode

project = "NPC Gym"
copyright = "2026, NPC Gym contributors"
extensions = []
exclude_patterns = ["_build"]
html_theme = "alabaster"


def _prepare_downloads(app, doctree):
    """Include relative file links as downloads before Sphinx collects assets."""
    for node in list(doctree.findall(nodes.reference)):
        uri = urlsplit(node.get("refuri", ""))
        if uri.scheme or uri.netloc or not uri.path or uri.path.startswith("/") or uri.path.endswith(".rst"):
            continue
        download = addnodes.download_reference(node.rawsource, reftarget=unquote(uri.path))
        download += nodes.literal("", node.astext(), classes=["xref", "download"])
        download.source, download.line = node.source, node.line
        node.replace_self(download)


def _resolve_document_links(app, doctree, docname):
    """Resolve portable RST paths and anchors using the active Sphinx builder."""
    logger = logging.getLogger(__name__)
    for node in list(doctree.findall(nodes.reference)):
        uri = urlsplit(node.get("refuri", ""))
        if uri.scheme or uri.netloc or uri.path.startswith("/"):
            continue
        if uri.path.endswith(".rst"):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(docname), unquote(uri.path)))[:-4]
        elif not uri.path and uri.fragment:
            target = docname
        else:
            continue
        anchor = unquote(uri.fragment)
        if target not in app.env.found_docs:
            logger.warning("document link target not found: %s", node["refuri"], location=node)
            continue
        if anchor and anchor not in app.env.get_doctree(target).ids:
            logger.warning("document link anchor not found: %s", node["refuri"], location=node)
            continue
        content = nodes.inline("", "", *node.children, classes=["std", "std-ref"] if anchor else ["doc"])
        node.replace_self(make_refnode(app.builder, docname, target, anchor, content))


def setup(app):
    """Keep source links readable in Pandoc and functional in Sphinx output."""
    app.connect("doctree-read", _prepare_downloads, priority=100)
    app.connect("doctree-resolved", _resolve_document_links)
