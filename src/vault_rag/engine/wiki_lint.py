"""WikiLint: LLM-powered health checks — contradictions, stale claims, data gaps."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from vault_rag.ingest.scanner import ScannedNote

logger = logging.getLogger(__name__)

_LINT_PROMPT = (
    "You are a wiki quality auditor. Analyze these wiki pages and find issues.\n"
    "Return JSON with these arrays:\n"
    '{{"contradictions": [{{"page_a": "title", "page_b": "title", "issue": "description"}}], '
    '"stale_claims": [{{"page": "title", "claim": "the outdated claim"}}], '
    '"missing_pages": ["concept that should have its own page but doesn\'t"], '
    '"data_gaps": ["question that the wiki should answer but can\'t"]}}\n\n'
    "Wiki pages:\n{pages}\n\n"
    "Return ONLY valid JSON, no markdown fences."
)


@dataclass(frozen=True)
class LintResult:
    """Result of a wiki lint pass."""

    contradictions: list[dict] = field(default_factory=list)
    stale_claims: list[dict] = field(default_factory=list)
    missing_pages: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    orphan_count: int = 0
    broken_link_count: int = 0
    untagged_count: int = 0
    pages_examined: int = 0
    next_offset: int = 0

    @property
    def total_issues(self) -> int:
        return (
            len(self.contradictions)
            + len(self.stale_claims)
            + len(self.missing_pages)
            + len(self.data_gaps)
            + self.orphan_count
            + self.broken_link_count
            + self.untagged_count
        )


class WikiLinter:
    """Run LLM-powered lint checks on the wiki."""

    def __init__(self, client: object, model: str) -> None:
        self._client = client
        self.model = model

    def lint(
        self,
        notes: list[ScannedNote],
        health_report: dict | None = None,
        max_pages: int = 50,
        offset: int = 0,
    ) -> LintResult:
        """Run lint on a rotating window of wiki pages.

        Combines structural checks (from HealthChecker) with
        LLM-powered semantic checks (contradictions, stale claims, gaps).

        A vault far larger than *max_pages* cannot be audited in one pass, so
        the window advances by *offset* and wraps around. Persist the returned
        ``next_offset`` and feed it back on the following run; consecutive runs
        then sweep the whole vault instead of re-reading the same head slice.
        """
        # Structural stats from health report
        orphan_count = health_report.get("orphan_count", 0) if health_report else 0
        broken_link_count = health_report.get("broken_link_count", 0) if health_report else 0
        untagged_count = len(health_report.get("untagged_notes", [])) if health_report else 0

        # Sample pages for LLM analysis (prioritize compiled pages)
        compiled = [n for n in notes if "compiled:" in n.content[:500]]
        pool = compiled if compiled else notes

        if not pool or max_pages <= 0:
            return LintResult(
                orphan_count=orphan_count,
                broken_link_count=broken_link_count,
                untagged_count=untagged_count,
                next_offset=offset,
            )

        start = offset % len(pool)
        take = min(max_pages, len(pool))
        sample = (pool + pool)[start : start + take]
        next_offset = (start + take) % len(pool)

        # Build context for LLM
        pages_text = "\n\n---\n\n".join(f"[{n.title}]\n{n.content[:500]}" for n in sample)

        prompt = _LINT_PROMPT.format(pages=pages_text)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        llm_result = self._parse_response(message.content[0].text)

        return LintResult(
            contradictions=llm_result.get("contradictions", []),
            stale_claims=llm_result.get("stale_claims", []),
            missing_pages=llm_result.get("missing_pages", []),
            data_gaps=llm_result.get("data_gaps", []),
            orphan_count=orphan_count,
            broken_link_count=broken_link_count,
            untagged_count=untagged_count,
            pages_examined=len(sample),
            next_offset=next_offset,
        )

    def _parse_response(self, text: str) -> dict:
        """Parse JSON response from LLM."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Lint JSON parse failed: %s", exc)
            return {}
