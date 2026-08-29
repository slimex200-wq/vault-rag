# vault-rag

Obsidian Vault를 위한 개인 지식 RAG 시스템.

## AI Harness

Before asking Claude, Codex, or another coding agent to work here, read:

- `AGENTS.md` - shared AI entrypoint for Codex/OMX and Claude/OMC
- `CLAUDE.md` - thin Claude/OMC adapter that points back to the shared harness
- `PROJECT_STATE.md` - current status and next work
- `CHECKS.md` - test, lint, and formatting gates
- `DECISIONS.md` - architecture and model-usage decisions

## 왜 만들었나

Andrej Karpathy가 X에 올린 글에서 시작됐습니다:

> "It would be nice to have a single markdown-based knowledge base where an LLM is responsible for maintaining and compiling it."

매일 쌓이는 노트, 클리핑, 리서치를 **LLM이 자동으로 정리**해주고, 나중에 **질문하면 답변**해주는 시스템이 필요했습니다.

기존 Obsidian 검색은 키워드 매칭만 됩니다. vault-rag는 **의미 기반 검색**으로 "비슷한 개념"을 찾아주고, 노트 간 관계를 자동으로 연결합니다.

## 작동 원리

```
                    ┌─────────────┐
                    │  Obsidian   │
                    │   Vault     │
                    │  (.md 파일) │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌─────────┐  ┌─────────┐
         │  Scan  │  │  Clip   │  │  PDF    │
         │ (스캔) │  │ (웹저장)│  │(PDF변환)│
         └───┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             ▼            ▼            ▼
        ┌─────────────────────────────────┐
        │         Compile (LLM)           │
        │  요약 · 태그 · 링크 · 품질점수  │
        │  atom 추출 · 후속 질문 생성     │
        └───────────────┬─────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼                   ▼
        ┌──────────┐        ┌──────────┐
        │ ChromaDB │        │ NetworkX │
        │ (벡터DB) │        │ (그래프) │
        └─────┬────┘        └──────────┘
              │
     ┌────────┼────────┬──────────┐
     ▼        ▼        ▼          ▼
  Search    Ask     Generate   Watch
  (검색)   (Q&A)   (출력생성) (자동감지)
```

## 설치

```bash
cd vault-rag
pip install -e ".[dev]"
```

## 환경변수

컴파일·Q&A용 Anthropic 토큰은 **호출할 때마다** 다음 순서로 해결된다:

1. GJC 크리덴셜 스토어 (`~/.gjc/agent/agent.db`, 읽기 전용 조회). 구독 OAuth 경로라 호출당 과금이 없다. 위치는 `GJC_CODING_AGENT_DIR` 로 바꿀 수 있다.
2. `ANTHROPIC_OAUTH_TOKEN` — 그 스토어가 없는 머신용 수동 지정.
3. `ANTHROPIC_API_KEY` — 위 둘 다 없을 때만. 여기서부터 종량 과금이다.

스토어의 access 토큰은 약 8시간마다 회전한다. 셸 변수에 복사해 둔 값은 하루면 죽으므로, 매 호출마다 스토어를 다시 읽는 것이 `OAuth access token has been revoked` 401 을 막는 유일한 방법이다.

```bash
# 구독 OAuth 를 쓸 수 없을 때만 필요하다
export ANTHROPIC_API_KEY="sk-ant-..."  # 컴파일 + Q&A 폴백 (종량)
export OPENAI_API_KEY="sk-..."         # embedding_provider="openai" 로 바꿨을 때만 (기본은 로컬 모델, 무과금)
```

## 사용법

### 1단계: 노트 스캔

```bash
vault-rag scan
# Vault: C:/Users/.../Obsidian Vault
# Total notes: 342
```

### 2단계: 임베딩 인덱스 생성

```bash
vault-rag index          # 변경분만 인덱싱
vault-rag index --full   # 전체 재인덱싱
```

### 3단계: 검색 & 질문

```bash
# 의미 기반 검색
vault-rag search "AI 에이전트 패턴"

# 하이브리드 검색 (벡터 + 키워드)
vault-rag search --hybrid "RAG pipeline"

# 태그 필터링
vault-rag search --filter-tag ai --filter-tag agents "autonomous"

# 검색 결과 요약 추출 (LLM 호출)
vault-rag search --extract "프롬프트 엔지니어링"

# RAG Q&A
vault-rag ask "vault-rag의 아키텍처는?"

# 후속 질문 포함
vault-rag ask --follow-up "RAG란 무엇인가?"
```

### 4단계: 노트 컴파일

```bash
# 단일 노트 컴파일 (미리보기)
vault-rag compile note.md --dry-run

# 단일 노트 컴파일 (저장)
vault-rag compile note.md

# 미태깅 노트 일괄 컴파일
vault-rag compile-new
```

컴파일하면 노트에 자동 추가됩니다:
- **요약** (frontmatter `summary`)
- **태그** (frontmatter `tags`)
- **관련 노트** (`## Related` 섹션)
- **후속 질문** (`## Further Questions` 섹션)
- **품질 점수** (0-100, HIGH/MEDIUM/LOW)

### 5단계: 외부 콘텐츠 수집

```bash
# 웹 페이지 클리핑
vault-rag clip https://example.com/article --folder Research

# PDF 인제스트
vault-rag ingest-pdf paper.pdf --folder Research
```

### 6단계: 출력 생성

```bash
# 슬라이드 덱 생성
vault-rag generate slides "AI 에이전트 트렌드"

# 종합 리포트 생성
vault-rag generate report "지식 관리 시스템"

# 요약 브리핑 생성
vault-rag generate summary "RAG 파이프라인"
```

생성된 파일은 `vault-rag/output/` 디렉토리에 저장됩니다.

### 7단계: 자동 감지 (Watcher)

```bash
vault-rag watch
# Watching C:/.../Obsidian Vault ... (Ctrl+C to stop)
# .md 파일 생성/수정 시 자동으로 compile + index 실행
```

### 기타

```bash
vault-rag health   # 깨진 링크, 고아 노트, 미태깅 노트 점검
vault-rag audit    # 품질 등급별 노트 분류 (HIGH/MEDIUM/LOW)
```

## 비용 가이드

| 기능 | 모델 | 예상 비용/호출 |
|------|------|-------------|
| **index** | OpenAI text-embedding-3-small | ~$0.00001 |
| **compile** | Claude Sonnet ($3/$15 MTok) | ~$0.01 |
| **ask** | Claude Sonnet | ~$0.02 |
| **search --extract** | Claude Sonnet | ~$0.02 |
| **generate slides** | Claude Sonnet | ~$0.03 |
| **generate report** | Claude Sonnet | ~$0.05-0.10 |
| **generate summary** | Claude Sonnet | ~$0.02 |
| **search / health / scan / watch** | (API 호출 없음) | $0 |

하루 5회 compile + 10회 Q&A 기준 **월 ~$7.5** 예상.

## 전체 커맨드 요약

| 커맨드 | 설명 | API 비용 |
|--------|------|---------|
| `scan` | Vault 스캔 + 노트 목록 | 없음 |
| `health` | 깨진 링크/고아 노트 점검 | 없음 |
| `index [--full]` | 임베딩 인덱스 생성 | OpenAI |
| `search <query> [--hybrid] [--filter-tag] [--extract]` | 검색 | --extract만 Claude |
| `ask <question> [--follow-up]` | RAG Q&A | Claude |
| `compile <path> [--dry-run]` | 단일 노트 컴파일 | Claude |
| `compile-new [--dry-run]` | 미태깅 일괄 컴파일 | Claude |
| `clip <url> [--folder] [--no-compile]` | 웹 클리핑 | Claude (compile 시) |
| `ingest-pdf <path> [--folder] [--no-compile]` | PDF 인제스트 | Claude (compile 시) |
| `audit` | 품질 등급 분류 | 없음 |
| `watch` | 파일 변경 자동 감지 | Claude (변경 시) |
| `generate slides <topic>` | 슬라이드 생성 | Claude |
| `generate report <topic>` | 리포트 생성 | Claude |
| `generate summary <topic>` | 요약 생성 | Claude |

## 개발

```bash
# 테스트
python -m pytest tests/ -q

# 커버리지
python -m pytest tests/ --cov=vault_rag --cov-report=term-missing -q

# 린트
ruff check . --fix

# 포맷
ruff format .
```
