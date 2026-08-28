"""Incremental, evidence-validated trend reports."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from domain.models import ProfileCitation, TrendClaim, TrendReport
from utils.prompt_loader import load_trends_prompt
from workflows.comparison import _canonical_evidence, _escape, _norm
from workflows.relevance import relevance_groups


class _RawClaim(BaseModel):
    kind: str
    text: str
    paper_ids: list[str] = Field(default_factory=list)
    evidence: list[ProfileCitation] = Field(default_factory=list)


class _RawTrends(BaseModel):
    claims: list[_RawClaim] = Field(default_factory=list)


def _is_water_colour_candidate(paper) -> bool:
    """Exclude candidates that are clearly about another remote-sensing domain."""
    groups = relevance_groups(paper)
    return not groups or "target" in groups


def _counts(profiles) -> dict[str, dict[str, int]]:
    output = {}
    for name in ("prediction_target", "sensors", "models", "datasets", "metrics", "limitations"):
        values = []
        for profile in profiles:
            raw = getattr(profile, name)
            fields = raw if isinstance(raw, list) else [raw]
            values.extend(item.value for item in fields if item.value and item.evidence)
        output[name] = dict(sorted(Counter(values).items()))
    return output


class TrendService:
    def __init__(self, repository, model, *, historical_limit: int = 50) -> None:
        if historical_limit <= 0:
            raise ValueError("trend_invalid_historical_limit")
        self.repository = repository
        self.model = model
        self.historical_limit = historical_limit

    def generate(self, since: datetime) -> TrendReport:
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("trend_since_must_be_timezone_aware")
        new_papers = sorted(
            (
                paper
                for paper in self.repository.list_papers_discovered_after(since)
                if _is_water_colour_candidate(paper)
            ),
            key=lambda paper: (paper.first_seen_at, paper.paper_id),
        )
        now = datetime.now(UTC)
        if not new_papers:
            return TrendReport(generated_at=now, new_papers=[], claims=[])
        new_ids = {p.paper_id for p in new_papers}
        candidates = [
            paper
            for paper in self.repository.list_papers(limit=self.historical_limit + len(new_ids))
            if _is_water_colour_candidate(paper)
        ]
        historical = sorted(
            (p for p in candidates if p.paper_id not in new_ids),
            key=lambda paper: (paper.first_seen_at, paper.paper_id),
            reverse=True,
        )[: self.historical_limit]
        new_profiles = [(p, self.repository.get_profile(p.paper_id)) for p in new_papers]
        historical_profiles = [(p, self.repository.get_profile(p.paper_id)) for p in historical]
        new_profiles = [(p, profile) for p, profile in new_profiles if profile is not None]
        historical_profiles = [(p, profile) for p, profile in historical_profiles if profile is not None]
        canonical = [c for p, profile in new_profiles + historical_profiles for c in _canonical_evidence(p.paper_id, profile)]
        context = {
            "new_paper_ids": [p.paper_id for p in new_papers],
            "historical_paper_ids": [p.paper_id for p in historical],
            "new_counts": _counts([profile for _, profile in new_profiles]),
            "historical_counts": _counts([profile for _, profile in historical_profiles]),
            "evidence": [c.model_dump() for c in canonical],
        }
        request = [SystemMessage(content=load_trends_prompt()), HumanMessage(content="BEGIN UNTRUSTED PROFILE DATA\n" + _escape(json.dumps(context, ensure_ascii=False, sort_keys=True)) + "\nEND UNTRUSTED PROFILE DATA")]
        try:
            raw = _RawTrends.model_validate(self.model.with_structured_output(_RawTrends).invoke(request))
        except Exception:
            raw = _RawTrends()
        allowed = {(c.paper_id, c.page_number, _norm(c.quote)): c for c in canonical}
        claims = []
        for item in raw.claims:
            if item.kind not in {"direct", "synthesis", "suggestion"}:
                continue
            evidence, seen = [], set()
            for proposed in item.evidence:
                key = (proposed.paper_id, proposed.page_number, _norm(proposed.quote))
                if key in allowed and key not in seen:
                    seen.add(key)
                    evidence.append(allowed[key])
            supported_papers = {c.paper_id for c in evidence if c.paper_id in item.paper_ids}
            if item.kind == "direct" and len(supported_papers) < 1:
                continue
            if item.kind == "synthesis" and len(supported_papers) < 2:
                continue
            text = item.text
            if item.kind == "suggestion":
                text = text if text.startswith("可验证假设：") else f"可验证假设：{text}"
            canonical_paper_ids = sorted(supported_papers)
            evidence = [citation for citation in evidence if citation.paper_id in supported_papers]
            claims.append(TrendClaim(kind=item.kind, text=text, paper_ids=canonical_paper_ids, evidence=evidence))
        return TrendReport(generated_at=now, new_papers=new_papers, claims=claims)
