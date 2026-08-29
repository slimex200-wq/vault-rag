"""Tests for VaultScanner (Task 2: Data Ingest Layer)."""

from pathlib import Path

from vault_rag.config import VaultConfig
from vault_rag.ingest.scanner import ScannedNote, VaultScanner


def _note(notes: list[ScannedNote], filename: str) -> ScannedNote:
    """Pick the scanned note whose file is *filename*.

    Matching a substring of the full path is not safe here. pytest names the
    temporary directory after the test function, so a selector like
    ``"backslash"`` also matches every note inside
    ``.../test_scan_strips_backslash_fro0/`` and ``next()`` returns whichever
    file the scan yielded first. That passed on Windows and failed on Linux
    purely on directory ordering.
    """
    return next(n for n in notes if n.path.name == filename)

# ---------------------------------------------------------------------------
# 1. scan finds expected .md files
# ---------------------------------------------------------------------------


def test_scan_finds_md_files(cfg: VaultConfig) -> None:
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    relative_paths = {n.relative_path for n in notes}
    # project-alpha.md and concepts.md must be found
    assert any("project-alpha.md" in p for p in relative_paths)
    assert any("concepts.md" in p for p in relative_paths)


# ---------------------------------------------------------------------------
# 2. excluded dirs — .obsidian
# ---------------------------------------------------------------------------


def test_scan_excludes_obsidian_dir(tmp_vault: Path, cfg: VaultConfig) -> None:
    (tmp_vault / ".obsidian" / "hidden.md").write_text(
        "# Hidden\n\nShould not appear.", encoding="utf-8"
    )
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert not any(".obsidian" in n.relative_path for n in notes)


# ---------------------------------------------------------------------------
# 3. excluded dirs — _trash
# ---------------------------------------------------------------------------


def test_scan_excludes_trash(tmp_vault: Path, cfg: VaultConfig) -> None:
    (tmp_vault / "_trash" / "deleted.md").write_text("# Deleted\n\nTrash note.", encoding="utf-8")
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert not any("_trash" in n.relative_path for n in notes)


def test_scan_excludes_dedupe_trash_prefix(tmp_vault: Path, cfg: VaultConfig) -> None:
    dedupe_trash = tmp_vault / ".dedupe-trash-2026-04-25"
    dedupe_trash.mkdir()
    (dedupe_trash / "duplicate.md").write_text(
        "# Duplicate\n\nOld duplicate note.", encoding="utf-8"
    )

    scanner = VaultScanner(cfg)
    notes = scanner.scan()

    assert not any(".dedupe-trash-" in n.relative_path for n in notes)


# ---------------------------------------------------------------------------
# 4. title extracted from first # heading
# ---------------------------------------------------------------------------


def test_scanned_note_extracts_title(cfg: VaultConfig) -> None:
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    alpha = next(n for n in notes if "project-alpha.md" in n.relative_path)
    assert alpha.title == "Project Alpha"


# ---------------------------------------------------------------------------
# 5. YAML frontmatter tags
# ---------------------------------------------------------------------------


def test_scanned_note_extracts_tags_from_frontmatter(tmp_vault: Path) -> None:
    note_path = tmp_vault / "Projects" / "project-alpha.md"
    note_path.write_text(
        "---\ntags: [project, ai]\n---\n# Project Alpha\n\nBody text.",
        encoding="utf-8",
    )
    cfg = VaultConfig(vault_path=tmp_vault)
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    alpha = next(n for n in notes if "project-alpha.md" in n.relative_path)
    assert "project" in alpha.tags
    assert "ai" in alpha.tags


# ---------------------------------------------------------------------------
# 6. inline #tags from body
# ---------------------------------------------------------------------------


def test_scanned_note_extracts_inline_tags(cfg: VaultConfig) -> None:
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    # concepts.md has "tags: #knowledge #concepts" in its body
    concepts = next(n for n in notes if "concepts.md" in n.relative_path)
    assert "knowledge" in concepts.tags
    assert "concepts" in concepts.tags


# ---------------------------------------------------------------------------
# 7. wikilinks extracted
# ---------------------------------------------------------------------------


def test_scanned_note_extracts_wikilinks(cfg: VaultConfig) -> None:
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    alpha = next(n for n in notes if "project-alpha.md" in n.relative_path)
    # project-alpha.md contains [[Knowledge/concepts]]
    assert "Knowledge/concepts" in alpha.links


# ---------------------------------------------------------------------------
# 8. every note has a non-empty content hash
# ---------------------------------------------------------------------------


def test_scanned_note_has_content_hash(cfg: VaultConfig) -> None:
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert all(len(n.content_hash) > 0 for n in notes)


# ---------------------------------------------------------------------------
# 9. incremental scan — all unchanged → 0 results
# ---------------------------------------------------------------------------


def test_incremental_scan_skips_unchanged(cfg: VaultConfig) -> None:
    scanner = VaultScanner(cfg)
    first = scanner.scan()
    known_hashes = {n.relative_path: n.content_hash for n in first}
    second = scanner.scan(known_hashes=known_hashes)
    assert len(second) == 0


# ---------------------------------------------------------------------------
# 10. incremental scan — modified file detected
# ---------------------------------------------------------------------------


def test_incremental_scan_detects_changes(tmp_vault: Path) -> None:
    cfg = VaultConfig(vault_path=tmp_vault)
    scanner = VaultScanner(cfg)
    first = scanner.scan()
    known_hashes = {n.relative_path: n.content_hash for n in first}

    # Modify one file
    note_path = tmp_vault / "Projects" / "project-alpha.md"
    note_path.write_text("# Project Alpha\n\nUpdated content.", encoding="utf-8")

    second = scanner.scan(known_hashes=known_hashes)
    assert len(second) == 1
    assert "project-alpha.md" in second[0].relative_path


# ---------------------------------------------------------------------------
# 11. code blocks — wikilinks inside ``` should be ignored
# ---------------------------------------------------------------------------


def test_scan_ignores_wikilinks_in_code_blocks(cfg: VaultConfig, tmp_vault: Path) -> None:
    """Code blocks with [[ should not produce wikilinks."""
    (tmp_vault / "code-note.md").write_text(
        "# Code\n\n```lua\nlocal t = [[\nmultiline\n]]\n```\n\nSee [[real-link]].\n",
        encoding="utf-8",
    )
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    code_note = _note(notes, "code-note.md")
    assert "real-link" in code_note.links
    assert not any("\n" in link for link in code_note.links)  # No multiline "links"


# ---------------------------------------------------------------------------
# 12. trailing backslash — [[link\]] should be stripped to "link"
# ---------------------------------------------------------------------------


def test_scan_strips_backslash_from_wikilinks(cfg: VaultConfig, tmp_vault: Path) -> None:
    """Trailing backslash in [[link\\]] should be stripped."""
    (tmp_vault / "backslash-note.md").write_text(
        "# BS\n\nSee [[some-target\\]].\n",
        encoding="utf-8",
    )
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    bs_note = _note(notes, "backslash-note.md")
    assert "some-target" in bs_note.links


# ---------------------------------------------------------------------------
# 13. YAML list format tags
# ---------------------------------------------------------------------------


def test_scanned_note_extracts_tags_from_yaml_list(cfg: VaultConfig, tmp_vault: Path) -> None:
    """Tags in YAML list format (tags:\\n  - tag) should be extracted."""
    (tmp_vault / "yaml-tags.md").write_text(
        "---\ntags:\n  - alpha\n  - beta\n---\n\n# YAML Tags\n\nContent.\n",
        encoding="utf-8",
    )
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    note = _note(notes, "yaml-tags.md")
    assert "alpha" in note.tags
    assert "beta" in note.tags


# ---------------------------------------------------------------------------
# 14. frontmatter scalars
# ---------------------------------------------------------------------------


def test_scanned_note_exposes_frontmatter_scalars(cfg: VaultConfig, tmp_vault: Path) -> None:
    """Top-level scalars are exposed; lists and nested blocks are skipped."""
    (tmp_vault / "fm-note.md").write_text(
        "---\n"
        'status: "active"\n'
        "quality_tier: HIGH\n"
        "tags: [a, b]\n"
        "nested:\n"
        "  key: value\n"
        "---\n\n# FM\n\nContent.\n",
        encoding="utf-8",
    )
    scanner = VaultScanner(cfg)
    note = _note(scanner.scan(), "fm-note.md")

    assert note.frontmatter["status"] == "active"  # quotes stripped
    assert note.frontmatter["quality_tier"] == "HIGH"
    assert "tags" not in note.frontmatter  # inline list, not a scalar
    assert "nested" not in note.frontmatter  # block header has no scalar value
    assert "key" not in note.frontmatter  # indented child is not top level


def test_unclosed_tag_bracket_does_not_swallow_later_lines(
    cfg: VaultConfig, tmp_vault: Path
) -> None:
    """An unclosed `tags: [` must not turn the summary sentence into a tag."""
    (tmp_vault / "unclosed.md").write_text(
        "---\n"
        "tags: [alpha, beta\n"
        "summary: 'a sentence with a bracket ] inside it'\n"
        "---\n\n# Unclosed\n\nBody.\n",
        encoding="utf-8",
    )
    note = _note(VaultScanner(cfg).scan(), "unclosed.md")

    assert not any("summary" in t for t in note.tags)
    assert not any(len(t) > 40 for t in note.tags)


def test_quoted_inline_tags_are_unquoted(cfg: VaultConfig, tmp_vault: Path) -> None:
    """`tags: ["ocr", 'RLS']` yields ocr and RLS, not '"ocr"' and "'RLS'"."""
    (tmp_vault / "quoted-tags.md").write_text(
        "---\ntags: [\"ocr\", 'RLS', plain]\n---\n\n# Quoted\n\nBody.\n",
        encoding="utf-8",
    )
    note = _note(VaultScanner(cfg).scan(), "quoted-tags.md")

    assert note.tags == ["ocr", "RLS", "plain"]


def test_root_operational_journal_is_not_a_note(cfg: VaultConfig, tmp_vault: Path) -> None:
    """log.md is the tool's own journal; indexing it lets maintenance move its metrics."""
    (tmp_vault / "log.md").write_text(
        "---\ntags: [system, log]\n---\n# Wiki Log\n", encoding="utf-8"
    )
    (tmp_vault / "Dev").mkdir(exist_ok=True)
    (tmp_vault / "Dev" / "log.md").write_text("# A real note named log\n", encoding="utf-8")

    paths = {n.relative_path for n in VaultScanner(cfg).scan()}

    assert "log.md" not in paths
    assert "Dev/log.md" in paths  # only the root-level journal is exempt


def test_inline_tags_ignore_code_fences(cfg: VaultConfig, tmp_vault: Path) -> None:
    """Hex colours and sample hashtags inside fences are not vault tags."""
    (tmp_vault / "fenced.md").write_text(
        "# Fenced\n\nSee #realtag here.\n\n"
        "```css\n.a { color: #f8f9fa; }\n```\n\n"
        "```\n설명: #lofi #playlist\n```\n",
        encoding="utf-8",
    )
    scanner = VaultScanner(cfg)
    note = _note(scanner.scan(), "fenced.md")

    assert "realtag" in note.tags
    assert "f8f9fa" not in note.tags
    assert "lofi" not in note.tags
    assert "playlist" not in note.tags


def test_frontmatter_is_empty_without_a_block(cfg: VaultConfig, tmp_vault: Path) -> None:
    (tmp_vault / "no-fm.md").write_text("# Plain\n\nNo frontmatter.\n", encoding="utf-8")
    scanner = VaultScanner(cfg)
    note = _note(scanner.scan(), "no-fm.md")
    assert note.frontmatter == {}
