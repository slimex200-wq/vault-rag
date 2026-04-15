"""Tests for the Image Analyzer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vault_rag.ingest.image_analyzer import ImageAnalysisResult, ImageAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_anthropic(text: str) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = text

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_message

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    return mock_client


_SAMPLE_RESPONSE = """Title: Dark Mode Crypto Dashboard
Style: dark mode, card-based, split comparison
Colors: dark background, charcoal cards, yellow CTA
Components: Swap/Pool tabs, token selector, amount input
Layout: 2-column symmetric cards
Tags: design-ref, ui, crypto, dark-mode, dashboard
Description: A DEX swap interface with dual card layout for send/receive."""


def _create_test_image(tmp_path: Path) -> Path:
    """Create a minimal PNG file for testing."""
    img_path = tmp_path / "test.png"
    # Minimal valid PNG (1x1 pixel)
    img_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return img_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analyze_returns_result(tmp_path: Path) -> None:
    client = _mock_anthropic(_SAMPLE_RESPONSE)
    analyzer = ImageAnalyzer(client=client, model="test", vault_path=tmp_path)
    img = _create_test_image(tmp_path)

    result = analyzer.analyze(img)

    assert isinstance(result, ImageAnalysisResult)
    assert result.title == "Dark Mode Crypto Dashboard"
    assert "dark mode" in result.style
    assert "crypto" in result.tags
    client.messages.create.assert_called_once()


def test_analyze_sends_image_as_base64(tmp_path: Path) -> None:
    client = _mock_anthropic(_SAMPLE_RESPONSE)
    analyzer = ImageAnalyzer(client=client, model="test", vault_path=tmp_path)
    img = _create_test_image(tmp_path)

    analyzer.analyze(img)

    call_kwargs = client.messages.create.call_args[1]
    content = call_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"


def test_analyze_and_save_creates_note(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    client = _mock_anthropic(_SAMPLE_RESPONSE)
    analyzer = ImageAnalyzer(client=client, model="test", vault_path=vault)
    img = _create_test_image(tmp_path)

    result, note_path = analyzer.analyze_and_save(img, folder="Reference/design-ref")

    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "Dark Mode Crypto Dashboard" in content
    assert "![[test.png]]" in content
    assert "dark mode" in content


def test_analyze_and_save_copies_image(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    client = _mock_anthropic(_SAMPLE_RESPONSE)
    analyzer = ImageAnalyzer(client=client, model="test", vault_path=vault)
    img = _create_test_image(tmp_path)

    analyzer.analyze_and_save(img, folder="Reference/design-ref")

    copied = vault / "Reference" / "design-ref" / "test.png"
    assert copied.exists()


def test_parse_handles_missing_fields(tmp_path: Path) -> None:
    client = _mock_anthropic("Title: Simple\nTags: minimal")
    analyzer = ImageAnalyzer(client=client, model="test", vault_path=tmp_path)
    img = _create_test_image(tmp_path)

    result = analyzer.analyze(img)

    assert result.title == "Simple"
    assert result.tags == ["minimal"]
    assert result.style == ""
    assert result.description == ""
