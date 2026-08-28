"""Generate evidence-linked profiles from stored abstracts without an LLM."""

from __future__ import annotations

from ingestion.metadata_profile import MetadataProfileExtractor


class MetadataProfileService:
    def __init__(self, database, extractor=None) -> None:
        self.database = database
        self.extractor = extractor or MetadataProfileExtractor()

    def run(self, *, limit: int) -> dict[str, object]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("metadata_profile_invalid_limit")
        profiled = skipped = 0
        records = self.database.list_papers(limit=limit * 3)
        for record in records:
            if not (record.abstract or "").strip():
                skipped += 1
                continue
            if self.database.get_profile(record.paper_id) is not None:
                continue
            self.database.save_profile(record.paper_id, self.extractor.extract(record))
            profiled += 1
            if profiled >= limit:
                break
        return {"profiled": profiled, "skipped": skipped}
