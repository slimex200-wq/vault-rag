"""Tests for VaultScanner (Task 2: Data Ingest Layer)."""
from pathlib import Path

import pytest

from vault_rag.config import VaultConfig
from vault_rag.ingest.scanner import ScannedNote, VaultScanner


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
    (tmp_vault / "_trash" / "deleted.md").write_text(
        "# Deleted\n\nTrash note.", encoding="utf-8"
    )
    scanner = VaultScanner(cfg)
    notes = scanner.scan()
    assert not any("_trash" in n.relative_path for n in notes)


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
    note_path.write_text(
        "# Project Alpha\n\nUpdated content.", encoding="utf-8"
    )

    second = scanner.scan(known_hashes=known_hashes)
    assert len(second) == 1
    assert "project-alpha.md" in second[0].relative_path
