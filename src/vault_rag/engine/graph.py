"""KnowledgeGraph: NetworkX-backed wikilink graph for an Obsidian vault."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from vault_rag.ingest.scanner import ScannedNote


class KnowledgeGraph:
    """Directed graph where nodes are note relative_paths and edges are wikilinks."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._title_to_path: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, notes: list[ScannedNote]) -> None:
        """Clear graph, add nodes (relative_path as id), resolve wikilinks to edges."""
        self._graph.clear()
        self._title_to_path.clear()

        # First pass: add all nodes and populate the title→path mapping
        for note in notes:
            self._graph.add_node(
                note.relative_path,
                title=note.title,
                tags=list(note.tags),
            )
            # Map lowercase title and lowercase stem to relative_path
            self._title_to_path[note.title.lower()] = note.relative_path
            stem = Path(note.relative_path).stem.lower()
            self._title_to_path[stem] = note.relative_path

        # Second pass: add edges
        for note in notes:
            for link in note.links:
                target = self._resolve_link(link)
                if target is not None:
                    self._graph.add_edge(note.relative_path, target)

    # ------------------------------------------------------------------
    # Link resolution
    # ------------------------------------------------------------------

    def _resolve_link(self, link_text: str) -> str | None:
        """Resolve wikilink text to relative_path.

        Tries (in order):
        1. Exact lowercase match
        2. Lowercase with spaces replaced by hyphens
        3. Lowercase with hyphens replaced by spaces
        """
        key = link_text.lower()
        if key in self._title_to_path:
            return self._title_to_path[key]

        hyphenated = key.replace(" ", "-")
        if hyphenated in self._title_to_path:
            return self._title_to_path[hyphenated]

        spaced = key.replace("-", " ")
        if spaced in self._title_to_path:
            return self._title_to_path[spaced]

        return None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def find_orphans(self) -> list[str]:
        """Return nodes with degree == 0 (no incoming or outgoing edges)."""
        return [n for n in self._graph.nodes if self._graph.degree(n) == 0]

    def find_hubs(self, min_degree: int = 5) -> list[str]:
        """Return nodes with degree >= min_degree."""
        return [n for n in self._graph.nodes if self._graph.degree(n) >= min_degree]

    def neighbors(self, node_id: str) -> list[str]:
        """Return successors (outgoing neighbors) of a node."""
        return list(self._graph.successors(node_id))

    def find_clusters(self) -> list[set[str]]:
        """Return connected components (undirected) with size > 1."""
        undirected = self._graph.to_undirected()
        return [
            component for component in nx.connected_components(undirected) if len(component) > 1
        ]

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize graph to JSON via nx.node_link_data."""
        data = nx.node_link_data(self._graph)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Deserialize graph from JSON via nx.node_link_graph."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self._graph = nx.node_link_graph(data, directed=True, multigraph=False)
        # Rebuild title→path mapping from node attributes
        self._title_to_path.clear()
        for node_id, attrs in self._graph.nodes(data=True):
            title = attrs.get("title", "")
            if title:
                self._title_to_path[title.lower()] = node_id
            stem = Path(node_id).stem.lower()
            self._title_to_path[stem] = node_id
