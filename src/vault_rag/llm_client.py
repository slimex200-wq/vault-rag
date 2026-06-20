"""Anthropic client factory.

LLM 호출(compile / ask / generate 등)을 Claude 구독 OAuth 토큰으로 보내,
종량제 API 키 대신 Claude Max 구독에서 비용이 빠지게 한다.

우선순위:
  1. ANTHROPIC_OAUTH_TOKEN  -> Claude 구독 (API 키 과금 없음)
  2. ANTHROPIC_API_KEY      -> 표준 종량제 API (토큰 없을 때 폴백)

임베딩은 의도적으로 건드리지 않는다. 임베딩 엔드포인트는 어떤 채팅 구독에도
포함되지 않으므로 OpenAI 종량제(indexer.create_openai_embed_fn)를 그대로 둔다.
"""

from __future__ import annotations

import os
from typing import Any

# Claude 구독 OAuth 는 Claude Code 신원 시스템 프롬프트를 요구한다.
# 첫 system 블록이 이 문자열이 아니면 엔드포인트가 401 로 거절한다.
_CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."
_OAUTH_BETA = "oauth-2025-04-20"


class _Messages:
    """`messages.create` 호출마다 Claude Code 신원 system 블록을 앞에 끼워 넣는
    얇은 래퍼."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, *, system: Any = None, **kwargs: Any) -> Any:
        ident = {"type": "text", "text": _CLAUDE_CODE_SYSTEM}
        if system is None:
            merged: list[dict[str, Any]] = [ident]
        elif isinstance(system, str):
            merged = [ident, {"type": "text", "text": system}]
        else:
            merged = [ident, *system]
        return self._inner.create(system=merged, **kwargs)


class _OAuthAnthropic:
    """vault-rag 가 사용하는 부분(`.messages.create`)만 노출하는 구독-OAuth 래퍼."""

    def __init__(self, token: str) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": _OAUTH_BETA},
        )
        self.messages = _Messages(self._client.messages)


def make_anthropic_client() -> Any:
    """OAuth 토큰이 있으면 구독 클라이언트를, 없으면 표준 API 클라이언트를 반환한다."""
    from anthropic import Anthropic

    token = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
    if token:
        return _OAuthAnthropic(token)
    return Anthropic()
