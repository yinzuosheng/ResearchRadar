"""Extract PDF text without losing source page numbers."""

from pathlib import Path

from pypdf import PdfReader

from domain.models import PageText


class PdfParser:
    def parse(self, path: Path) -> list[PageText]:
        reader = PdfReader(path)
        pages: list[PageText] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(page_number=page_number, text=text))
        return pages
