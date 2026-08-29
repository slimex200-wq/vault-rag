"""Anthropic client factory: subscription OAuth when available, API key otherwise.

Chat subscriptions bill nothing per call, so routing compile/ask/lint through an
OAuth token instead of ANTHROPIC_API_KEY removes the only meaningful recurring
cost in this pipeline. Embeddings cannot go the same way -- no chat subscription
covers the embeddings endpoint -- which is why that side moved to a local model
instead.

The OAuth path has three hard requirements. Miss any one and the API rejects the
call, so they are encapsulated here rather than repeated at each call site:

1. `Authorization: Bearer <token>` instead of `x-api-key`.
2. `anthropic-beta: oauth-2025-04-20`.
3. The first system block must identify the caller as Claude Code.

The token itself is resolved per call, not per process, in this order:

1. The GJC credential store (`~/.gjc/agent/agent.db`), which the agent runtime
   keeps refreshed. Access tokens there live about eight hours and rotate on
   every refresh, so a copy pasted into a shell variable is dead by the next
   day; reading the store at call time is what keeps long-running commands and
   already-open terminals working.
2. `ANTHROPIC_OAUTH_TOKEN`, for machines that have no such store.
3. Neither: fall back to the metered API-key client.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
AGENT_DIR_ENV = "GJC_CODING_AGENT_DIR"
AGENT_DB_NAME = "agent.db"
OAUTH_BETA_HEADER = "oauth-2025-04-20"
CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

# Refuse a stored token this close to its expiry: the call still has to travel.
EXPIRY_SKEW_MS = 60_000


def _agent_db_path() -> Path:
    """Location of the GJC credential store, honouring its own directory var."""
    configured = os.environ.get(AGENT_DIR_ENV)
    root = Path(configured) if configured else Path.home() / ".gjc" / "agent"
    return root / AGENT_DB_NAME


def _agent_db_token(now_ms: int | None = None) -> str | None:
    """Newest live anthropic access token in the GJC store, or None.

    Every failure here -- no store, unreadable file, schema drift, only expired
    or disabled rows -- means "no token" rather than an exception, because the
    caller still has the environment variable and the API key behind it. The
    store is opened read-only: this process must never rotate or disable a
    credential the agent runtime owns.
    """
    path = _agent_db_path()
    if not path.is_file():
        return None

    import json
    import sqlite3

    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = conn.execute(
                "SELECT data FROM auth_credentials "
                "WHERE provider = 'anthropic' AND disabled_cause IS NULL"
            ).fetchall()
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError):
        return None

    now = now_ms if now_ms is not None else int(time.time() * 1000)
    cutoff = now + EXPIRY_SKEW_MS
    best: tuple[int, str] | None = None

    for (data,) in rows:
        try:
            entry = json.loads(data)
            token = entry["access"]
            expires = int(entry["expires"])
        except (TypeError, ValueError, KeyError):
            continue
        if not isinstance(token, str) or not token or expires <= cutoff:
            continue
        if best is None or expires > best[0]:
            best = (expires, token)

    return best[1] if best else None


def resolve_oauth_token() -> str | None:
    """Subscription token for the next call, freshest source first."""
    return _agent_db_token() or os.environ.get(OAUTH_TOKEN_ENV) or None


class _OAuthMessages:
    """messages proxy that refreshes the token and prepends the Claude Code block."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        # The runtime rotates the stored token roughly every eight hours; picking
        # it up here is what keeps a long-lived client from dying mid-session.
        token = resolve_oauth_token()
        if token:
            self._client.auth_token = token

        marker = {"type": "text", "text": CLAUDE_CODE_SYSTEM}
        system = kwargs.get("system")

        if system is None:
            kwargs["system"] = [marker]
        elif isinstance(system, str):
            kwargs["system"] = [marker, {"type": "text", "text": system}]
        else:
            blocks = list(system)
            first = blocks[0] if blocks else None
            already = isinstance(first, dict) and first.get("text") == CLAUDE_CODE_SYSTEM
            kwargs["system"] = blocks if already else [marker, *blocks]

        return self._client.messages.create(**kwargs)


class OAuthAnthropic:
    """Thin wrapper exposing the `.messages.create` surface the callers use."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.messages = _OAuthMessages(inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def using_oauth() -> bool:
    """True when a subscription token is reachable, so calls are not metered."""
    return resolve_oauth_token() is not None


def make_anthropic_client(anthropic_cls: Any = None) -> Any:
    """Return an OAuth-backed client when a token exists, else the API-key client.

    Args:
        anthropic_cls: injection point for tests; defaults to anthropic.Anthropic.
    """
    if anthropic_cls is None:
        from anthropic import Anthropic

        anthropic_cls = Anthropic

    token = resolve_oauth_token()
    if not token:
        return anthropic_cls()

    inner = anthropic_cls(
        auth_token=token,
        default_headers={"anthropic-beta": OAUTH_BETA_HEADER},
    )
    return OAuthAnthropic(inner)
