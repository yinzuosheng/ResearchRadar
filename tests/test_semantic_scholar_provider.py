import json

import pytest

from domain.models import PaperCandidate


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def candidate():
    return PaperCandidate(
        source="openalex",
        source_id="W123",
        title="Water Colour from Space",
        doi="10.1000/example",
    )


def test_semantic_scholar_resolves_only_declared_open_access_pdf(monkeypatch):
    from providers.semantic_scholar import SemanticScholarResolver

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(
            {
                "openAccessPdf": {
                    "url": "https://repository.example/s2-paper.pdf",
                    "status": "GREEN",
                }
            }
        )

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-s2-key")
    monkeypatch.setattr("providers.semantic_scholar.get_once", fake_get)

    locations = SemanticScholarResolver().resolve(candidate())

    assert [(item.provider, item.url, item.is_oa) for item in locations] == [
        ("semantic_scholar", "https://repository.example/s2-paper.pdf", True)
    ]
    assert captured["headers"] == {"x-api-key": "test-s2-key"}
    assert "/paper/DOI:10.1000%2Fexample" in captured["url"]
    assert "test-s2-key" not in json.dumps(locations, default=str)


def test_semantic_scholar_ignores_missing_or_non_oa_pdf(monkeypatch):
    from providers.semantic_scholar import SemanticScholarResolver

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-s2-key")
    payloads = [
        {"openAccessPdf": None},
        {"openAccessPdf": {"url": "https://publisher.example/paywall", "status": "CLOSED"}},
    ]
    monkeypatch.setattr(
        "providers.semantic_scholar.get_once",
        lambda *args, **kwargs: FakeResponse(payloads.pop(0)),
    )
    resolver = SemanticScholarResolver()

    assert resolver.resolve(candidate()) == []
    assert resolver.resolve(candidate()) == []


def test_semantic_scholar_requires_key_without_echoing_value(monkeypatch):
    from providers.semantic_scholar import SemanticScholarResolver

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    with pytest.raises(ValueError) as error:
        SemanticScholarResolver().resolve(candidate())

    assert str(error.value) == (
        "Missing required environment variable: SEMANTIC_SCHOLAR_API_KEY"
    )
