"""Tests for KnowledgeGraph — NetworkX wikilink graph."""
from __future__ import annotations

from pathlib import Path

import pytest

from vault_rag.engine.graph import KnowledgeGraph
from vault_rag.ingest.scanner import ScannedNote


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _note(
    title: str,
    path: str,
    links: list[str] | None = None,
    tags: list[str] | None = None,
) -> ScannedNote:
    return ScannedNote(
        path=Path(path),
        relative_path=path,
        title=title,
        content="",
        tags=tags or [],
        links=links or [],
        modified=0.0,
        content_hash="",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_graph_from_notes() -> None:
    """A->B, B->A, B->C produces 3 nodes and 3 edges."""
    notes = [
        _note("A", "a.md", links=["b"]),
        _note("B", "b.md", links=["a", "c"]),
        _note("C", "c.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)

    assert g.node_count() == 3
    assert g.edge_count() == 3


def test_orphan_detection() -> None:
    """A->B + Orphan (no links). orphan.md must appear in orphans()."""
    notes = [
        _note("A", "a.md", links=["b"]),
        _note("B", "b.md"),
        _note("Orphan", "orphan.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)

    orphans = g.find_orphans()
    assert "orphan.md" in orphans
    assert "a.md" not in orphans
    assert "b.md" not in orphans


def test_hub_nodes() -> None:
    """Hub links to A, B, C, D, E — degree 5. Must appear in find_hubs(min_degree=3)."""
    notes = [
        _note("Hub", "hub.md", links=["a", "b", "c", "d", "e"]),
        _note("A", "a.md"),
        _note("B", "b.md"),
        _note("C", "c.md"),
        _note("D", "d.md"),
        _note("E", "e.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)

    hubs = g.find_hubs(min_degree=3)
    assert "hub.md" in hubs


def test_neighbors() -> None:
    """A links to B and C. neighbors('a.md') must equal {'b.md', 'c.md'}."""
    notes = [
        _note("A", "a.md", links=["b", "c"]),
        _note("B", "b.md"),
        _note("C", "c.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)

    assert set(g.neighbors("a.md")) == {"b.md", "c.md"}


def test_save_and_load(tmp_path: Path) -> None:
    """Build, save to tmp_path, load into fresh graph — node/edge counts match."""
    notes = [
        _note("A", "a.md", links=["b"]),
        _note("B", "b.md", links=["a"]),
        _note("C", "c.md"),
    ]
    g = KnowledgeGraph()
    g.build(notes)

    save_file = tmp_path / "graph.json"
    g.save(save_file)

    g2 = KnowledgeGraph()
    g2.load(save_file)

    assert g2.node_count() == g.node_count()
    assert g2.edge_count() == g.edge_count()


def test_clusters() -> None:
    """A<->B and X<->Y form 2 clusters (each size > 1)."""
    notes = [
        _note("A", "a.md", links=["b"]),
        _note("B", "b.md", links=["a"]),
        _note("X", "x.md", links=["y"]),
        _note("Y", "y.md", links=["x"]),
    ]
    g = KnowledgeGraph()
    g.build(notes)

    clusters = g.find_clusters()
    assert len(clusters) == 2
    assert {"a.md", "b.md"} in clusters
    assert {"x.md", "y.md"} in clusters
