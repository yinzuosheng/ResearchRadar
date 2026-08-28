"""Deterministic relevance screening for the water-colour literature corpus."""

from __future__ import annotations

from domain.models import PaperCandidate


TERM_GROUPS = {
    "target": (
        "chlorophyll", "turbidity", "secchi", "water quality", "algal bloom",
        "cyanobacter", "phycocyanin", "suspended solids", "cdom",
    ),
    "sensor": (
        "sentinel-2", "sentinel 2", "landsat", "modis", "hyperspectral",
        "remote sensing", "satellite", "multispectral", "earth observation",
    ),
    "method": (
        "machine learning", "deep learning", "random forest", "xgboost", "lstm",
        "neural network", "support vector", "regression", "prediction", "forecast",
        "estimation", "domain adaptation",
    ),
}


def relevance_groups(candidate: PaperCandidate) -> set[str]:
    text = " ".join(
        part.casefold()
        for part in (candidate.title or "", candidate.abstract or "")
        if isinstance(part, str)
    )
    return {
        group
        for group, terms in TERM_GROUPS.items()
        if any(term in text for term in terms)
    }


def is_relevant(candidate: PaperCandidate, *, minimum_groups: int = 0) -> bool:
    if minimum_groups <= 0:
        return True
    return len(relevance_groups(candidate)) >= minimum_groups
