"""Tests for the Anthropic client factory (subscription OAuth vs API key)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vault_rag.llm_client import (
    AGENT_DIR_ENV,
    CLAUDE_CODE_SYSTEM,
    OAUTH_BETA_HEADER,
    OAUTH_TOKEN_ENV,
    OAuthAnthropic,
    make_anthropic_client,
    using_oauth,
)

HOUR_MS = 60 * 60 * 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_store(agent_dir: Path, rows: list[tuple[str, str, int, str | None]]) -> None:
    """Build a GJC-shaped credential store: (provider, token, expires_ms, cause)."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    db = agent_dir / "agent.db"
    db.unlink(missing_ok=True)

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE auth_credentials ("
        "id INTEGER PRIMARY KEY, provider TEXT, credential_type TEXT, data TEXT, "
        "disabled_cause TEXT, identity_key TEXT, created_at INTEGER, updated_at INTEGER)"
    )
    for provider, token, expires, cause in rows:
        data = json.dumps({"access": token, "refresh": "refresh-value", "expires": expires})
        conn.execute(
            "INSERT INTO auth_credentials (provider, credential_type, data, disabled_cause) "
            "VALUES (?, 'oauth', ?, ?)",
            (provider, data, cause),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Path:
    return tmp_path / "gjc-agent"


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch, agent_dir: Path) -> None:
    """Keep the developer's real store and shell token out of these tests."""
    monkeypatch.setenv(AGENT_DIR_ENV, str(agent_dir))
    monkeypatch.delenv(OAUTH_TOKEN_ENV, raising=False)


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


def test_stored_token_beats_stale_env_token(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock, agent_dir: Path
) -> None:
    """The env copy goes stale within a day; the store is what stays current."""
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "stale-env-token")
    _write_store(agent_dir, [("anthropic", "live-store-token", _now_ms() + HOUR_MS, None)])

    make_anthropic_client(anthropic_cls=anthropic_cls)

    assert anthropic_cls.call_args.kwargs["auth_token"] == "live-store-token"


def test_stored_token_alone_is_enough(anthropic_cls: MagicMock, agent_dir: Path) -> None:
    _write_store(agent_dir, [("anthropic", "store-token", _now_ms() + HOUR_MS, None)])

    client = make_anthropic_client(anthropic_cls=anthropic_cls)

    assert using_oauth() is True
    assert isinstance(client, OAuthAnthropic)
    assert anthropic_cls.call_args.kwargs["auth_token"] == "store-token"


def test_newest_live_credential_wins(anthropic_cls: MagicMock, agent_dir: Path) -> None:
    now = _now_ms()
    _write_store(
        agent_dir,
        [
            ("anthropic", "older", now + HOUR_MS, None),
            ("anthropic", "newer", now + 8 * HOUR_MS, None),
        ],
    )

    make_anthropic_client(anthropic_cls=anthropic_cls)

    assert anthropic_cls.call_args.kwargs["auth_token"] == "newer"


@pytest.mark.parametrize(
    ("expires_offset_ms", "cause", "provider"),
    [
        (-HOUR_MS, None, "anthropic"),  # expired
        (HOUR_MS, "oauth refresh failed: 401", "anthropic"),  # soft-disabled
        (HOUR_MS, None, "openai-codex"),  # another provider entirely
    ],
)
def test_unusable_rows_fall_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_cls: MagicMock,
    agent_dir: Path,
    expires_offset_ms: int,
    cause: str | None,
    provider: str,
) -> None:
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "env-token")
    _write_store(agent_dir, [(provider, "unusable", _now_ms() + expires_offset_ms, cause)])

    make_anthropic_client(anthropic_cls=anthropic_cls)

    assert anthropic_cls.call_args.kwargs["auth_token"] == "env-token"


def test_unreadable_store_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, anthropic_cls: MagicMock, agent_dir: Path
) -> None:
    monkeypatch.setenv(OAUTH_TOKEN_ENV, "env-token")
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.db").write_text("not a database", encoding="utf-8")

    make_anthropic_client(anthropic_cls=anthropic_cls)

    assert anthropic_cls.call_args.kwargs["auth_token"] == "env-token"


def test_missing_store_and_env_falls_back_to_api_key(anthropic_cls: MagicMock) -> None:
    make_anthropic_client(anthropic_cls=anthropic_cls)

    anthropic_cls.assert_called_once_with()


def test_rotated_store_token_is_used_on_the_next_call(
    anthropic_cls: MagicMock, agent_dir: Path
) -> None:
    """A client built before a refresh must not keep sending the retired token."""
    _write_store(agent_dir, [("anthropic", "token-a", _now_ms() + HOUR_MS, None)])
    client = make_anthropic_client(anthropic_cls=anthropic_cls)
    inner = anthropic_cls.return_value

    client.messages.create(model="m", max_tokens=1, messages=[])
    assert inner.auth_token == "token-a"

    _write_store(agent_dir, [("anthropic", "token-b", _now_ms() + 8 * HOUR_MS, None)])
    client.messages.create(model="m", max_tokens=1, messages=[])

    assert inner.auth_token == "token-b"
    assert inner.messages.create.call_count == 2
