"""Deterministic paper comparison with validated profile citations."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from domain.models import ComparisonReport, ComparisonRow, PaperProfile, ProfileCitation
from utils.prompt_loader import load_comparison_prompt

MISSING = "未报告"
INSUFFICIENT = "证据不足，无法形成可靠的跨论文综合。"


class _SynthesisClaim(BaseModel):
    text: str
    citations: list[ProfileCitation] = Field(default_factory=list)


class _Synthesis(BaseModel):
    claims: list[_SynthesisClaim] = Field(default_factory=list)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _escape(text: str) -> str:
    return text.replace("BEGIN UNTRUSTED PROFILE DATA", "[ESCAPED BEGIN PROFILE MARKER]").replace(
        "END UNTRUSTED PROFILE DATA", "[ESCAPED END PROFILE MARKER]"
    )


def _one(field) -> str:
    return field.value if field.value and field.evidence else MISSING


def _many(fields) -> str:
    values = [item.value for item in fields if item.value and item.evidence]
    return "；".join(values) if values else MISSING


def _canonical_evidence(paper_id: str, profile: PaperProfile) -> list[ProfileCitation]:
    citations = []
    for name in type(profile).model_fields:
        value = getattr(profile, name)
        fields = value if isinstance(value, list) else [value]
        for item in fields:
            for evidence in item.evidence:
                citations.append(ProfileCitation(paper_id=paper_id, **evidence.model_dump()))
    return citations


class ComparisonService:
    def __init__(self, repository, model) -> None:
        self.repository = repository
        self.model = model

    def compare(self, paper_ids: list[str]) -> ComparisonReport:
        if not 2 <= len(paper_ids) <= 5:
            raise ValueError("comparison_requires_2_to_5_papers")
        if len(set(paper_ids)) != len(paper_ids):
            raise ValueError("comparison_duplicate_paper_ids")
        papers, profiles = [], []
        for paper_id in paper_ids:
            paper = self.repository.get_paper(paper_id)
            if paper is None:
                raise ValueError("comparison_paper_not_found")
            profile = self.repository.get_profile(paper_id)
            if profile is None:
                raise ValueError("comparison_profile_not_found")
            papers.append(paper)
            profiles.append(profile)
        rows = [self._row(paper.paper_id, paper.title, profile) for paper, profile in zip(papers, profiles, strict=True)]
        canonical = [citation for paper, profile in zip(papers, profiles, strict=True) for citation in _canonical_evidence(paper.paper_id, profile)]
        context = {
            "rows": [row.model_dump() for row in rows],
            "evidence": [item.model_dump() for item in canonical],
        }
        request = [
            SystemMessage(content=load_comparison_prompt()),
            HumanMessage(content="BEGIN UNTRUSTED PROFILE DATA\n" + _escape(json.dumps(context, ensure_ascii=False, sort_keys=True)) + "\nEND UNTRUSTED PROFILE DATA"),
        ]
        try:
            proposed = _Synthesis.model_validate(self.model.with_structured_output(_Synthesis).invoke(request))
        except Exception:
            proposed = _Synthesis()
        allowed = {(c.paper_id, c.page_number, _norm(c.quote)): c for c in canonical}
        surviving = []
        surviving_text = []
        seen = set()
        for claim in proposed.claims:
            claim_citations = []
            for citation in claim.citations:
                key = (citation.paper_id, citation.page_number, _norm(citation.quote))
                canonical_citation = allowed.get(key)
                if canonical_citation is not None:
                    claim_citations.append((key, canonical_citation))
            if not claim_citations:
                continue
            surviving_text.append(claim.text)
            for key, canonical_citation in claim_citations:
                if key not in seen:
                    seen.add(key)
                    surviving.append(canonical_citation)
        return ComparisonReport(
            rows=rows,
            synthesis_markdown="\n\n".join(surviving_text) if surviving_text else INSUFFICIENT,
            citations=surviving,
        )

    @staticmethod
    def _row(paper_id: str, title: str, p: PaperProfile) -> ComparisonRow:
        return ComparisonRow(
            paper_id=paper_id, title=title, prediction_target=_one(p.prediction_target),
            sensors=_many(p.sensors), study_area=_one(p.study_area), time_span=_one(p.time_span),
            sample_size=_one(p.sample_size), preprocessing=_many(p.preprocessing), models=_many(p.models),
            baselines=_many(p.baselines), datasets=_many(p.datasets), metrics=_many(p.metrics), limitations=_many(p.limitations),
        )
