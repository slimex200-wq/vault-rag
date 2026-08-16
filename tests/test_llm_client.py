"""Tests for the Anthropic client factory (subscription OAuth vs API key)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vault_rag.llm_client import (
    CLAUDE_CODE_SYSTEM,
    OAUTH_BETA_HEADER,
    OAUTH_TOKEN_ENV,
    OAuthAnthropic,
    make_anthropic_client,
    using_oauth,
)


@pytest.fixture()
def anthropic_cls() -> MagicMock:
    """Stand-in for anthropic.Anthropic that records constructor kwargs."""
    return MagicMock()


def test_api_key_path_when_no_token(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock
) -> None:
    monkeypatch.delenv(OAUTH_TOKEN_ENV, raising=False)

    client = make_anthropic_client(anthropic_cls=anthropic_cls)

    anthropic_cls.assert_called_once_with()
    assert not isinstance(client, OAuthAnthropic)
    assert using_oauth() is False


def test_oauth_path_sets_bearer_and_beta_header(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock
) -> None:
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "tok-123")

    client = make_anthropic_client(anthropic_cls=anthropic_cls)

    kwargs = anthropic_cls.call_args.kwargs
    assert kwargs["auth_token"] == "tok-123"
    assert kwargs["default_headers"]["anthropic-beta"] == OAUTH_BETA_HEADER
    assert isinstance(client, OAuthAnthropic)
    assert using_oauth() is True


def test_oauth_injects_claude_code_system_block(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock
) -> None:
    """Without this block the OAuth endpoint rejects the request."""
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "tok")
    client = make_anthropic_client(anthropic_cls=anthropic_cls)

    client.messages.create(model="m", max_tokens=1, messages=[])

    sent = anthropic_cls.return_value.messages.create.call_args.kwargs
    assert sent["system"] == [{"type": "text", "text": CLAUDE_CODE_SYSTEM}]


def test_oauth_keeps_caller_system_prompt_after_the_marker(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock
) -> None:
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "tok")
    client = make_anthropic_client(anthropic_cls=anthropic_cls)

    client.messages.create(model="m", max_tokens=1, messages=[], system="be terse")

    sent = anthropic_cls.return_value.messages.create.call_args.kwargs
    assert sent["system"][0]["text"] == CLAUDE_CODE_SYSTEM
    assert sent["system"][1] == {"type": "text", "text": "be terse"}


def test_oauth_does_not_duplicate_the_marker(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock
) -> None:
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "tok")
    client = make_anthropic_client(anthropic_cls=anthropic_cls)
    marker = {"type": "text", "text": CLAUDE_CODE_SYSTEM}

    client.messages.create(model="m", max_tokens=1, messages=[], system=[marker])

    sent = anthropic_cls.return_value.messages.create.call_args.kwargs
    assert sent["system"] == [marker]


def test_oauth_wrapper_forwards_other_attributes(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock
) -> None:
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "tok")
    anthropic_cls.return_value.models = "inner-models"

    client = make_anthropic_client(anthropic_cls=anthropic_cls)

    assert client.models == "inner-models"
