"""Tests for hybrid retrieval: BM25, graph expansion, reciprocal rank fusion."""

from __future__ import annotations

from pathlib import Path

from vault_rag.engine.hybrid import (
    BM25Index,
    fuse_with_graph_recall,
    graph_expand,
    reciprocal_rank_fusion,
    tokenize,
)
from vault_rag.ingest.scanner import ScannedNote


def _note(name: str, content: str = "", links: list[str] | None = None) -> ScannedNote:
    return ScannedNote(
        path=Path(f"/v/{name}.md"),
        relative_path=f"{name}.md",
        title=name,
        content=content,
        tags=[],
        links=links or [],
        modified=0.0,
        content_hash="x",
    )


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_splits_hangul_and_ascii() -> None:
    assert tokenize("Supabase RLS 정책 적용") == ["supabase", "rls", "정책", "적용"]


def test_tokenize_drops_punctuation() -> None:
    assert tokenize("error_code: 128 (fatal!)") == ["error_code", "128", "fatal"]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_ranks_the_on_topic_note_first() -> None:
    notes = [
        _note("unrelated", "고양이와 강아지에 대한 잡담"),
        _note("target", "Supabase RLS 정책을 테이블마다 다르게 적용하는 방법"),
        _note("mentions-once", "지나가듯 Supabase 를 한 번 언급"),
    ]
    ranked = BM25Index(notes).search("Supabase RLS 정책")
    assert ranked[0][0] == "target.md"


def test_bm25_ignores_notes_without_any_term() -> None:
    notes = [_note("a", "완전히 다른 내용"), _note("b", "gradle 빌드 실패")]
    ranked = BM25Index(notes).search("gradle")
    assert [p for p, _ in ranked] == ["b.md"]


def test_bm25_rare_term_outweighs_common_term() -> None:
    """A term in every note carries no signal; a rare one decides the ranking."""
    notes = [_note(f"n{i}", "common common common") for i in range(5)]
    notes.append(_note("rare", "common 희귀단어"))
    ranked = BM25Index(notes).search("common 희귀단어")
    assert ranked[0][0] == "rare.md"


def test_bm25_empty_query_returns_nothing() -> None:
    assert BM25Index([_note("a", "text")]).search("   ") == []


def test_bm25_on_empty_corpus_returns_nothing() -> None:
    assert BM25Index([]).search("anything") == []


# ---------------------------------------------------------------------------
# graph expansion
# ---------------------------------------------------------------------------


def test_graph_expand_finds_neighbours_of_seeds() -> None:
    notes = [
        _note("hub", links=["leaf-a", "leaf-b"]),
        _note("leaf-a"),
        _note("leaf-b"),
        _note("far"),
    ]
    assert set(graph_expand(["hub.md"], notes)) == {"leaf-a.md", "leaf-b.md"}


def test_graph_expand_is_undirected() -> None:
    """A note that links to the seed is just as related as one the seed links to."""
    notes = [_note("cites", links=["seed"]), _note("seed")]
    assert graph_expand(["seed.md"], notes) == ["cites.md"]


def test_graph_expand_ranks_shared_neighbours_first() -> None:
    notes = [
        _note("s1", links=["shared", "only1"]),
        _note("s2", links=["shared"]),
        _note("shared"),
        _note("only1"),
    ]
    assert graph_expand(["s1.md", "s2.md"], notes)[0] == "shared.md"


def test_graph_expand_excludes_the_seeds_themselves() -> None:
    notes = [_note("a", links=["b"]), _note("b", links=["a"])]
    assert graph_expand(["a.md", "b.md"], notes) == []


# ---------------------------------------------------------------------------
# reciprocal rank fusion
# ---------------------------------------------------------------------------


def test_rrf_prefers_agreement_over_a_single_top_hit() -> None:
    """Second place in two retrievers beats first place in only one."""
    fused = reciprocal_rank_fusion(
        {
            "bm25": ["solo.md", "agreed.md"],
            "vector": ["other.md", "agreed.md"],
        }
    )
    assert fused[0].path == "agreed.md"


def test_rrf_reports_which_retrievers_contributed() -> None:
    fused = reciprocal_rank_fusion({"bm25": ["a.md"], "graph": ["a.md"]})
    assert set(fused[0].sources) == {"bm25", "graph"}


def test_rrf_tolerates_a_missing_retriever() -> None:
    """Vector leg absent (no API key / empty store) must not break fusion."""
    fused = reciprocal_rank_fusion({"bm25": ["a.md", "b.md"], "vector": []})
    assert [f.path for f in fused] == ["a.md", "b.md"]


def test_graph_never_reorders_the_relevance_ranking() -> None:
    """Graph is a recall aid; it must not displace the best textual match.

    Fusing graph as a weighted leg cannot be made safe: the RRF gap between
    BM25 rank 1 and rank 6 is smaller than any graph weight big enough to
    matter, so a well-connected mid-pack note wins. Recall-append avoids it.
    """
    bm25 = [f"r{i}.md" for i in range(1, 11)]  # r1 is the best textual match
    graph = ["r6.md", "r7.md", "r8.md"]

    fused = fuse_with_graph_recall({"bm25": bm25}, graph, limit=5)

    assert [f.path for f in fused] == ["r1.md", "r2.md", "r3.md", "r4.md", "r5.md"]


def test_graph_only_hits_are_appended_after_relevance() -> None:
    fused = fuse_with_graph_recall({"bm25": ["a.md", "b.md"]}, ["c.md"], limit=5)

    assert [f.path for f in fused] == ["a.md", "b.md", "c.md"]
    assert fused[-1].sources == ("graph",)


def test_graph_recall_respects_limit() -> None:
    fused = fuse_with_graph_recall({"bm25": ["a.md"]}, ["b.md", "c.md"], limit=2)
    assert len(fused) == 2


def test_rrf_weights_are_overridable() -> None:
    fused = reciprocal_rank_fusion(
        {"bm25": ["a.md"], "graph": ["b.md"]},
        weights={"bm25": 0.1, "graph": 10.0},
    )
    assert fused[0].path == "b.md"


def test_rrf_respects_limit() -> None:
    fused = reciprocal_rank_fusion({"bm25": [f"n{i}.md" for i in range(10)]}, limit=3)
    assert len(fused) == 3
