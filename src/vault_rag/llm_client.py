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
"""

from __future__ import annotations

import os
from typing import Any

OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
OAUTH_BETA_HEADER = "oauth-2025-04-20"
CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."


class _OAuthMessages:
    """messages proxy that prepends the required Claude Code system block."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
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

        return self._inner.create(**kwargs)


class OAuthAnthropic:
    """Thin wrapper exposing the `.messages.create` surface the callers use."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.messages = _OAuthMessages(inner.messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def using_oauth() -> bool:
    """True when a subscription token is present, so calls are not metered."""
    return bool(os.environ.get(OAUTH_TOKEN_ENV))


def make_anthropic_client(anthropic_cls: Any = None) -> Any:
    """Return an OAuth-backed client when a token exists, else the API-key client.

    Args:
        anthropic_cls: injection point for tests; defaults to anthropic.Anthropic.
    """
    if anthropic_cls is None:
        from anthropic import Anthropic

        anthropic_cls = Anthropic

    token = os.environ.get(OAUTH_TOKEN_ENV)
    if not token:
        return anthropic_cls()

    inner = anthropic_cls(
        auth_token=token,
        default_headers={"anthropic-beta": OAUTH_BETA_HEADER},
    )
    return OAuthAnthropic(inner)
