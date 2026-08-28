"""Low-cost, citation-linked profiles for metadata and abstract evidence."""

from __future__ import annotations

import re

from domain.models import EvidenceRef, ExtractedField, PaperCandidate, PaperProfile

_TARGETS = (
    "chlorophyll-a", "chlorophyll a", "turbidity", "Secchi depth",
    "water quality", "algal bloom", "cyanobacteria", "phycocyanin",
    "suspended solids", "CDOM",
)
_SENSORS = (
    "Sentinel-2", "Landsat", "MODIS", "hyperspectral", "multispectral",
    "satellite imagery", "remote sensing",
)
_MODELS = (
    "Random Forest", "XGBoost", "LightGBM", "Support Vector Machine", "SVM",
    "CNN", "LSTM", "GRU", "neural network", "deep learning", "regression",
    "PLSR", "domain adaptation",
)


class MetadataProfileExtractor:
    """Extract transparent keyword candidates from title and abstract only."""

    def extract(self, candidate: PaperCandidate) -> PaperProfile:
        text = " ".join(
            part.strip()
            for part in (candidate.title or "", candidate.abstract or "")
            if isinstance(part, str) and part.strip()
        )
        return PaperProfile(
            prediction_target=self._field(text, _TARGETS),
            sensors=self._fields(text, _SENSORS),
            study_area=ExtractedField(),
            time_span=ExtractedField(),
            sample_size=ExtractedField(),
            preprocessing=[],
            models=self._fields(text, _MODELS),
            baselines=[],
            datasets=[],
            metrics=[],
            conclusions=[],
            limitations=[],
            future_work=[],
        )

    @staticmethod
    def _quote(text: str, term: str) -> str:
        match = re.search(re.escape(term), text, flags=re.IGNORECASE)
        if not match:
            return text[:240]
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 140)
        return text[start:end].strip()

    def _field(self, text: str, terms: tuple[str, ...]) -> ExtractedField:
        for term in terms:
            match = re.search(re.escape(term), text, flags=re.IGNORECASE)
            if match:
                return ExtractedField(
                    value=match.group(0),
                    evidence=[EvidenceRef(page_number=0, quote=self._quote(text, term))],
                )
        return ExtractedField()

    def _fields(self, text: str, terms: tuple[str, ...]) -> list[ExtractedField]:
        fields: list[ExtractedField] = []
        seen: set[str] = set()
        for term in terms:
            match = re.search(re.escape(term), text, flags=re.IGNORECASE)
            if not match or match.group(0).casefold() in seen:
                continue
            seen.add(match.group(0).casefold())
            fields.append(
                ExtractedField(
                    value=match.group(0),
                    evidence=[EvidenceRef(page_number=0, quote=self._quote(text, term))],
                )
            )
        return fields
