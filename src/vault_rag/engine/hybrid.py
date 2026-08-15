"""Hybrid retrieval: BM25 + vector + graph, fused with reciprocal rank.

A single `index.md` catalogue stops working somewhere in the low hundreds of
pages, and this vault holds thousands. Vector search alone misses exact terms
(error codes, library names); keyword search alone misses paraphrase; neither
follows the link structure that makes a wiki a wiki.

Each retriever produces its own ranking and reciprocal rank fusion combines
them, so a document that several retrievers like beats one that a single
retriever loves. Fusion needs only the ranks, which means the vector leg can be
absent -- offline or without an API key the other two still answer.

No new dependency: BM25 here is the standard Okapi formulation over
whitespace/punctuation tokens, which handles Korean acceptably because Korean
is space-delimited at the phrase level.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from vault_rag.ingest.scanner import ScannedNote

_RE_TOKEN = re.compile(r"[0-9A-Za-z_]+|[가-힣]+")

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric and Hangul runs."""
    return [t.lower() for t in _RE_TOKEN.findall(text)]


@dataclass(frozen=True)
class ScoredNote:
    """One retrieval result."""

    path: str
    score: float
    sources: tuple[str, ...]


class BM25Index:
    """Okapi BM25 over note title + body."""

    def __init__(self, notes: list[ScannedNote]) -> None:
        self._paths: list[str] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for note in notes:
            tokens = tokenize(f"{note.title}\n{note.content}")
            counts = Counter(tokens)
            self._paths.append(note.relative_path)
            self._tfs.append(counts)
            self._lengths.append(len(tokens))
            document_frequency.update(counts.keys())

        self._n = len(notes)
        self._avg_len = (sum(self._lengths) / self._n) if self._n else 0.0
        self._idf = {
            term: math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """Return (relative_path, score) for the best matches, score-descending."""
        terms = tokenize(query)
        if not terms or self._n == 0:
            return []

        scored: list[tuple[str, float]] = []
        for i, counts in enumerate(self._tfs):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                norm = 1 - BM25_B + BM25_B * (self._lengths[i] / self._avg_len or 1)
                score += self._idf.get(term, 0.0) * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * norm)
            if score > 0:
                scored.append((self._paths[i], score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]


def graph_expand(
    seeds: list[str],
    notes: list[ScannedNote],
    limit: int = 50,
) -> list[str]:
    """Return notes one hop from *seeds* along wikilinks, closest first.

    Ordered by how many seeds reach them, so a note several results agree on
    outranks one reached from a single seed.
    """
    by_stem: dict[str, str] = {}
    for note in notes:
        by_stem.setdefault(
            note.relative_path.rsplit("/", 1)[-1].removesuffix(".md").lower(), note.relative_path
        )
        by_stem.setdefault(note.title.lower(), note.relative_path)

    outbound: dict[str, set[str]] = defaultdict(set)
    for note in notes:
        for link in note.links:
            target = by_stem.get(link.split("#", 1)[0].strip().lower())
            if target and target != note.relative_path:
                outbound[note.relative_path].add(target)
                outbound[target].add(note.relative_path)  # treat as undirected

    seed_set = set(seeds)
    hits: Counter[str] = Counter()
    for seed in seeds:
        for neighbour in outbound.get(seed, ()):
            if neighbour not in seed_set:
                hits[neighbour] += 1

    return [path for path, _ in hits.most_common(limit)]


# Graph expansion answers "what is connected to the hits", not "what matches the
# query". Unweighted it is as strong as the relevance legs: a note ranked 6th by
# BM25 plus 1st by graph outscores the exact-title match, which is a worse result
# than plain BM25. Measured over three live queries, weight 1.0 pushed the top
# BM25 hit out of the fused top 3 twice, 0.3 once, and 0.15 never -- so graph
# stays as a recall aid at 0.15.
DEFAULT_LEG_WEIGHTS: dict[str, float] = {"bm25": 1.0, "vector": 1.0, "graph": 0.15}


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    limit: int = 20,
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[ScoredNote]:
    """Fuse named rankings by weighted reciprocal rank.

    Args:
        rankings: retriever name -> ranked list of note paths (best first).
        limit: how many fused results to return.
        k: RRF damping; larger k flattens the advantage of top ranks.
        weights: per-leg multiplier; defaults to DEFAULT_LEG_WEIGHTS, and any
            leg not listed there contributes at full weight.
    """
    leg_weights = DEFAULT_LEG_WEIGHTS if weights is None else weights
    totals: dict[str, float] = defaultdict(float)
    contributors: dict[str, list[str]] = defaultdict(list)

    for name, ranked in rankings.items():
        weight = leg_weights.get(name, 1.0)
        for rank, path in enumerate(ranked, start=1):
            totals[path] += weight / (k + rank)
            contributors[path].append(name)

    fused = [
        ScoredNote(path=path, score=score, sources=tuple(contributors[path]))
        for path, score in totals.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.path))
    return fused[:limit]


def fuse_with_graph_recall(
    relevance: dict[str, list[str]],
    graph: list[str],
    limit: int = 20,
) -> list[ScoredNote]:
    """Rank by relevance legs, then append what only the graph found.

    Weighting the graph as a fused leg cannot be made safe: RRF only sees ranks,
    and the gap between BM25 rank 1 and rank 6 is smaller than any graph weight
    large enough to matter, so a mid-pack note that happens to be well connected
    displaces the exact match.

    Graph therefore never reorders the relevance ranking. It only appends notes
    the relevance legs missed entirely -- which is the recall it was added for.
    """
    ranked = reciprocal_rank_fusion(relevance, limit=limit)
    seen = {item.path for item in ranked}

    for rank, path in enumerate(graph, start=1):
        if len(ranked) >= limit:
            break
        if path in seen:
            continue
        seen.add(path)
        ranked.append(ScoredNote(path=path, score=1.0 / (RRF_K + rank), sources=("graph",)))

    return ranked[:limit]
