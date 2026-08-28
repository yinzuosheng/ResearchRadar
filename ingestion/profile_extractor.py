"""Evidence-linked structured paper profile extraction."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from domain.models import PageText, PaperProfile
from retrieval.citations import validate_evidence
from utils.prompt_loader import load_paper_profile_prompt


class ProfileExtractionError(RuntimeError):
    """Stable, secret-safe failure raised at the extraction boundary."""


class PaperProfileExtractor:
    """Ask a structured model for a profile, then validate every citation."""

    def __init__(self, model, *, character_budget: int = 12_000) -> None:
        if character_budget <= 0:
            raise ProfileExtractionError("profile_extraction_invalid_budget")
        self.model = model
        self.character_budget = character_budget

    def extract(self, pages: list[PageText]) -> PaperProfile:
        if not pages:
            raise ProfileExtractionError("profile_extraction_empty_pages")
        page_text = self._render_pages(pages)
        request = [
            SystemMessage(content=load_paper_profile_prompt()),
            HumanMessage(
                content=(
                    "BEGIN UNTRUSTED DOCUMENT DATA\n"
                    f"{page_text}\n"
                    "END UNTRUSTED DOCUMENT DATA"
                )
            ),
        ]
        try:
            proposed = self.model.with_structured_output(PaperProfile).invoke(request)
            profile = PaperProfile.model_validate(proposed)
        except Exception:
            raise ProfileExtractionError("profile_extraction_invalid_output") from None
        return validate_evidence(profile, pages)

    def _render_pages(self, pages: list[PageText]) -> str:
        prioritized = [
            page
            for _, page in sorted(
                enumerate(pages),
                key=lambda item: (
                    0
                    if item[1].page_number == 0
                    else 1
                    if item[1].page_number == 1
                    else 2,
                    item[0],
                ),
            )
        ]
        mandatory = [page for page in prioritized if page.page_number in (0, 1)]
        regular = [page for page in prioritized if page.page_number not in (0, 1)]

        delimiters = [f"--- PAGE {page.page_number} ---\n" for page in mandatory]
        used = sum(map(len, delimiters)) + max(0, len(delimiters) - 1) * 2
        if used > self.character_budget:
            raise ProfileExtractionError("profile_extraction_budget_exhausted")

        remaining = self.character_budget - used
        mandatory_text = [self._escape_untrusted_text(page.text) for page in mandatory]
        bodies = ["" for _ in mandatory]
        oversized: list[int] = []
        for index, text in enumerate(mandatory_text):
            if len(text) <= remaining:
                bodies[index] = text
                remaining -= len(text)
            else:
                oversized.append(index)
        for index in oversized:
            text = mandatory_text[index]
            prefix_length = min(len(text), remaining)
            bodies[index] = text[:prefix_length]
            remaining -= prefix_length
        rendered = [
            delimiter + body for delimiter, body in zip(delimiters, bodies, strict=True)
        ]

        for page in regular:
            block = (
                f"--- PAGE {page.page_number} ---\n"
                f"{self._escape_untrusted_text(page.text)}"
            )
            required = len(block) + (2 if rendered else 0)
            if required > remaining:
                continue
            rendered.append(block)
            remaining -= required

        if not rendered:
            raise ProfileExtractionError("profile_extraction_budget_exhausted")
        return "\n\n".join(rendered)

    @staticmethod
    def _escape_untrusted_text(text: str) -> str:
        escaped = text.replace(
            "BEGIN UNTRUSTED DOCUMENT DATA",
            "[ESCAPED BEGIN DOCUMENT MARKER]",
        ).replace(
            "END UNTRUSTED DOCUMENT DATA",
            "[ESCAPED END DOCUMENT MARKER]",
        )
        return re.sub(
            r"(?m)^([ \t]*)--- PAGE (-?\d+) ---[ \t]*\r?$",
            lambda match: (
                f"{match.group(1)}[ESCAPED PAGE MARKER {match.group(2)}]"
            ),
            escaped,
        )
