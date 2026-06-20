# Decisions

## Architecture

- `VaultConfig` stays a plain dataclass to keep CLI startup light.
- `chroma_path` and `graph_path` stay derived properties so tests can redirect `vault_path` cleanly.
- Default priority order is `Projects > Knowledge > Research > Reference > Dev`.
- Default excluded directories include `.obsidian`, `_trash`, `.git`, `docs`, and `Templates`.
- Excluded directory configuration should stay immutable/hashable unless there is a concrete need to change it.

## Model Usage

- Keep cheap embeddings separate from more expensive compile/QA model calls.
- Do not add automatic large-scale vault processing without explicit cost and safety notes.
- Default model/API changes must document required environment variables and expected cost impact.

### LLM auth: Claude 구독 OAuth (2026-06-20)

- compile / ask / generate 등 모든 Anthropic 호출은 `llm_client.make_anthropic_client()`
  팩토리를 거친다. cli.py 는 더 이상 `Anthropic()` 을 직접 생성하지 않는다.
- 인증 우선순위: `ANTHROPIC_OAUTH_TOKEN`(Claude 구독, 종량 과금 0) → 없으면
  `ANTHROPIC_API_KEY`(표준 종량제) 폴백.
- 구독 OAuth 는 Bearer 토큰 + `anthropic-beta: oauth-2025-04-20` 헤더 + 첫 system 블록
  "You are Claude Code, Anthropic's official CLI for Claude." 를 요구한다. 세 가지 모두
  `_OAuthAnthropic` 래퍼에 캡슐화돼 있다.
- 임베딩은 의도적으로 OpenAI 종량제 유지. 임베딩 엔드포인트는 어떤 채팅 구독에도
  포함되지 않으므로 `ANTHROPIC_OAUTH_TOKEN` 으로 대체 불가. `OPENAI_API_KEY` 필요.
  비용: 전체 재인덱싱 1회 ≈ $0.03~0.05(text-embedding-3-small), 검색 쿼리당 ≈ $0.
- 유효 모델 ID 는 구독에서 살아있는 것만 사용: `claude-sonnet-4-6`(qa),
  `claude-haiku-4-5-20251001`(compile). 구버전 `claude-sonnet-4-20250514` 는 404 → 폐기.
