from pathlib import Path

from domain.models import ExtractedField, PageText, PaperCandidate, PaperProfile
from domain.statuses import FAILED, PARSED, PROFILED
from storage.database import ResearchDatabase
from workflows.profile import ProfileService


def paper(number: int) -> PaperCandidate:
    return PaperCandidate(
        source="openalex",
        source_id=f"W{number}",
        title=f"Paper {number}",
        year=2024,
        doi=f"10.1000/{number}",
    )


def profile() -> PaperProfile:
    return PaperProfile(
        prediction_target=ExtractedField(value="chlorophyll-a"),
        study_area=ExtractedField(value="lake"),
        time_span=ExtractedField(value="2020-2024"),
        sample_size=ExtractedField(value="100"),
    )


class FakeParser:
    def parse(self, path: Path):
        return [PageText(page_number=1, text="paper evidence")]


class FakeExtractor:
    def extract(self, pages):
        return profile()


def test_profile_service_saves_profile_and_marks_paper_profiled(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    record = database.upsert_candidate(paper(1))
    database.update_status(record.paper_id, PARSED)
    pdf = tmp_path / "papers" / f"{record.paper_id}.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"pdf")

    report = ProfileService(
        database=database,
        pdf_dir=tmp_path / "papers",
        parser=FakeParser(),
        extractor=FakeExtractor(),
    ).run()

    assert report == {"processed": 1, "profiled": 1, "failed": {}}
    assert database.get_profile(record.paper_id) == profile()
    assert database.get_paper(record.paper_id).status == PROFILED


def test_profile_service_reports_missing_pdf_without_changing_status(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    record = database.upsert_candidate(paper(1))
    database.update_status(record.paper_id, PARSED)

    report = ProfileService(
        database=database,
        pdf_dir=tmp_path / "papers",
        parser=FakeParser(),
        extractor=FakeExtractor(),
    ).run()

    assert report == {"processed": 0, "profiled": 0, "failed": {"pdf_not_found": 1}}
    assert database.get_paper(record.paper_id).status == PARSED


def test_profile_service_continues_after_one_extraction_failure(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    records = [database.upsert_candidate(paper(1)), database.upsert_candidate(paper(2))]
    for record in records:
        database.update_status(record.paper_id, PARSED)
        pdf = tmp_path / "papers" / f"{record.paper_id}.pdf"
        pdf.parent.mkdir(exist_ok=True)
        pdf.write_bytes(b"pdf")

    class SelectiveExtractor(FakeExtractor):
        def __init__(self):
            self.calls = 0

        def extract(self, pages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model unavailable")
            return profile()

    report = ProfileService(
        database=database,
        pdf_dir=tmp_path / "papers",
        parser=FakeParser(),
        extractor=SelectiveExtractor(),
    ).run()

    assert report["processed"] == 2
    assert report["profiled"] == 1
    assert report["failed"] == {"profile_extraction_failed": 1}
    assert sum(database.get_paper(item.paper_id).status == FAILED for item in records) == 1
