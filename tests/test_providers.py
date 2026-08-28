import json
from pathlib import Path

import pytest

from domain.models import PaperCandidate


FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def candidate(**changes) -> PaperCandidate:
    values = {
        "source": "openalex",
        "source_id": "W123",
        "title": "Water Colour from Space",
        "doi": "10.1000/example",
    }
    values.update(changes)
    return PaperCandidate(**values)


def test_openalex_search_normalizes_fixture_and_reconstructs_abstract(monkeypatch):
    from providers import openalex

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(fixture("openalex_search.json"))

    monkeypatch.setenv("OPENALEX_API_KEY", "test-openalex-key")
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setattr(openalex, "get_with_retry", fake_get)

    provider = openalex.OpenAlexProvider()
    result = provider.search("water colour", from_year=2020, max_results=150)[0]

    assert result.doi == "10.1000/example"
    assert result.source_id == "W123"
    assert result.abstract == "Water colour retrieval works"
    assert result.pdf_url == "https://repository.example/water-colour.pdf"
    assert result.license == "cc-by"
    assert captured["params"] == {
        "api_key": "test-openalex-key",
        "search": "water colour",
        "filter": "open_access.is_oa:true,from_publication_date:2020-01-01",
        "sort": "relevance_score:desc",
        "per_page": 100,
    }


def test_openalex_search_follows_cursor_pages(monkeypatch):
    from providers import openalex

    calls = []
    first = fixture("openalex_search.json")
    second = fixture("openalex_search.json")
    second["results"][0]["id"] = "https://openalex.org/W124"
    payloads = [
        {"results": first["results"], "meta": {"next_cursor": "cursor-2"}},
        {"results": second["results"], "meta": {"next_cursor": None}},
    ]

    def fake_get(url, **kwargs):
        calls.append(kwargs["params"])
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr(openalex, "get_with_retry", fake_get)
    provider = openalex.OpenAlexProvider()
    results = provider.search("water", from_year=None, max_results=2)

    assert [item.source_id for item in results] == ["W123", "W124"]
    assert "cursor" not in calls[0]
    assert calls[1]["cursor"] == "cursor-2"


def test_openalex_resolution_prefers_licensed_oa_and_includes_cached_content(monkeypatch):
    from providers import openalex

    monkeypatch.setenv("OPENALEX_API_KEY", "test-openalex-key")
    monkeypatch.setattr(
        openalex,
        "get_with_retry",
        lambda *args, **kwargs: FakeResponse(fixture("openalex_search.json")),
    )
    provider = openalex.OpenAlexProvider()
    paper = provider.search("water", from_year=None, max_results=1)[0]

    locations = provider.resolve(paper)

    assert [location.priority for location in locations] == [10, 20, 30]
    assert locations[0].url == "https://repository.example/water-colour.pdf"
    assert locations[0].is_oa is True
    assert locations[0].license == "cc-by"
    assert locations[-1].url == "https://content.openalex.org/works/W123.pdf"
    assert locations[-1].license == "cc-by"


def test_openalex_does_not_offer_unlicensed_or_non_oa_pdf(monkeypatch):
    from providers import openalex

    payload = fixture("openalex_search.json")
    work = payload["results"][0]
    work["best_oa_location"]["license"] = None
    work["locations"][0]["is_oa"] = False
    monkeypatch.setenv("OPENALEX_API_KEY", "test-openalex-key")
    monkeypatch.setattr(openalex, "get_with_retry", lambda *args, **kwargs: FakeResponse(payload))
    provider = openalex.OpenAlexProvider()
    paper = provider.search("water", from_year=None, max_results=1)[0]

    assert paper.pdf_url is None
    assert provider.resolve(paper) == []


def test_openalex_resolves_rehydrated_direct_oa_pdf_without_cached_work():
    from providers.openalex import OpenAlexProvider

    paper = candidate(
        pdf_url="https://repository.example/direct.pdf",
        license=None,
    )

    locations = OpenAlexProvider().resolve(paper)

    assert [(location.url, location.license, location.is_oa) for location in locations] == [
        ("https://repository.example/direct.pdf", None, True)
    ]


def test_openalex_refreshes_work_by_doi_when_cache_is_empty(monkeypatch):
    from providers import openalex

    calls = []
    payload = fixture("openalex_search.json")

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("params", {})))
        return FakeResponse(payload)

    monkeypatch.setenv("OPENALEX_API_KEY", "test-openalex-key")
    monkeypatch.setattr(openalex, "get_with_retry", fake_get)
    paper = candidate(pdf_url=None)
    locations = openalex.OpenAlexProvider().resolve(paper)

    assert locations[0].url == "https://repository.example/water-colour.pdf"
    assert calls[0][0].endswith("/works/https%3A%2F%2Fdoi.org%2F10.1000%2Fexample")


def test_unpaywall_returns_only_open_pdf_locations(monkeypatch):
    from providers import unpaywall

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(fixture("unpaywall_record.json"))

    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.test")
    monkeypatch.setattr(unpaywall, "get_with_retry", fake_get)

    locations = unpaywall.UnpaywallResolver().resolve(candidate())

    assert len(locations) == 1
    assert locations[0].is_oa is True
    assert locations[0].license == "cc-by"
    assert captured["url"].endswith("/10.1000/example")
    assert captured["params"] == {"email": "researcher@example.test"}


def test_unpaywall_accepts_nonempty_url_for_pdf_without_filename_suffix(monkeypatch):
    from providers import unpaywall

    payload = fixture("unpaywall_record.json")
    payload["best_oa_location"]["url_for_pdf"] = "https://repository.example/download/123"
    payload["oa_locations"] = []
    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.test")
    monkeypatch.setattr(
        unpaywall, "get_with_retry", lambda *args, **kwargs: FakeResponse(payload)
    )

    locations = unpaywall.UnpaywallResolver().resolve(candidate())

    assert [location.url for location in locations] == [
        "https://repository.example/download/123"
    ]


def test_core_search_normalizes_pdf_and_rejects_landing_page(monkeypatch):
    from providers import core

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(fixture("core_search.json"))

    monkeypatch.setenv("CORE_API_KEY", "test-core-key")
    monkeypatch.setattr(core, "get_with_retry", fake_get)

    results = core.CoreProvider().search("water", from_year=2020, max_results=2)

    assert results[0].doi == "10.1000/core"
    assert results[0].pdf_url.endswith(".pdf")
    assert results[1].pdf_url is None
    assert captured["url"].endswith("/search/works")
    assert captured["params"] == {"q": "water", "limit": 2}
    assert captured["headers"] == {"Authorization": "Bearer test-core-key"}


def test_core_rehydrates_official_download_url_without_pdf_suffix(monkeypatch):
    from providers import core

    payload = fixture("core_search.json")
    payload["results"] = [payload["results"][0]]
    payload["results"][0]["downloadUrl"] = "https://core.ac.uk/download/123"
    monkeypatch.setenv("CORE_API_KEY", "test-core-key")
    monkeypatch.setattr(
        core, "get_with_retry", lambda *args, **kwargs: FakeResponse(payload)
    )

    persisted = core.CoreProvider().search(
        "water", from_year=None, max_results=1
    )[0]
    locations = core.CoreProvider().resolve(persisted)

    assert [(location.url, location.is_oa, location.license) for location in locations] == [
        ("https://core.ac.uk/download/123", True, None)
    ]


def test_core_resolves_rehydrated_explicit_repository_pdf():
    from providers.core import CoreProvider

    locations = CoreProvider().resolve(
        candidate(source="core", pdf_url="https://repository.example/files/paper.pdf")
    )

    assert [location.url for location in locations] == [
        "https://repository.example/files/paper.pdf"
    ]


def test_core_rejects_rehydrated_non_pdf_record_url():
    from providers.core import CoreProvider

    record_url = candidate(source="core", pdf_url="https://repo.test/record/1")

    assert CoreProvider().resolve(record_url) == []


def test_core_rejects_landing_only_candidate():
    from providers.core import CoreProvider

    landing_only = candidate(
        source="core",
        pdf_url=None,
        landing_url="https://repo.test/record/1",
    )

    assert CoreProvider().resolve(landing_only) == []


def test_crossref_fills_missing_metadata_without_erasing_existing_values(monkeypatch):
    from providers import crossref

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(fixture("crossref_record.json"))

    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.test")
    monkeypatch.setattr(crossref, "get_with_retry", fake_get)
    original = candidate(title="Existing title", abstract="Existing abstract", venue=None)

    result = crossref.CrossrefEnricher().enrich(original)

    assert result.title == "Existing title"
    assert result.abstract == "Existing abstract"
    assert result.venue == "Remote Sensing of Environment"
    assert captured["url"].endswith("/10.1000%2Fexample")
    assert captured["params"] == {"mailto": "researcher@example.test"}


def test_crossref_can_discover_candidates_for_openalex_fallback(monkeypatch):
    from providers import crossref

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/fallback",
                            "title": ["Fallback Paper"],
                            "container-title": ["Remote Sensing Journal"],
                            "author": [{"given": "Ada", "family": "Lake"}],
                            "published": {"date-parts": [[2024, 1, 1]]},
                            "URL": "https://doi.org/10.1000/fallback",
                            "is-referenced-by-count": 12,
                        }
                    ]
                }
            }
        )

    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.test")
    monkeypatch.setattr(crossref, "get_with_retry", fake_get)

    result = crossref.CrossrefEnricher().search(
        "water remote sensing", from_year=2020, max_results=5
    )

    assert [item.doi for item in result] == ["10.1000/fallback"]
    assert result[0].source == "crossref"
    assert result[0].cited_by_count == 12
    assert captured["params"]["mailto"] == "researcher@example.test"
    assert captured["params"]["rows"] == 5


@pytest.mark.parametrize(
    ("module_name", "class_name", "method", "env_name", "paper"),
    [
        ("providers.unpaywall", "UnpaywallResolver", "resolve", "UNPAYWALL_EMAIL", candidate()),
        ("providers.core", "CoreProvider", "search", "CORE_API_KEY", None),
        ("providers.crossref", "CrossrefEnricher", "enrich", "UNPAYWALL_EMAIL", candidate()),
    ],
)
def test_missing_credentials_name_only_the_environment_variable(
    monkeypatch, module_name, class_name, method, env_name, paper
):
    import importlib

    monkeypatch.delenv(env_name, raising=False)
    provider = getattr(importlib.import_module(module_name), class_name)()

    with pytest.raises(ValueError) as error:
        if method == "search":
            getattr(provider, method)("water", from_year=None, max_results=1)
        else:
            getattr(provider, method)(paper)

    assert str(error.value) == f"Missing required environment variable: {env_name}"


def test_openalex_anonymous_access_uses_polite_mailto(monkeypatch):
    from providers.openalex import OpenAlexProvider

    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.test")
    captured = {}

    class Response:
        def json(self):
            return {"results": []}

    def fake_get(url, *, params=None, headers=None, timeout=None):
        captured.update(params or {})
        return Response()

    monkeypatch.setattr("providers.openalex.get_with_retry", fake_get)
    OpenAlexProvider().search("water remote sensing", from_year=2020, max_results=5)

    assert "api_key" not in captured
    assert captured["mailto"] == "researcher@example.test"


def test_registry_falls_back_deduplicates_and_preserves_provider_order(monkeypatch):
    from providers.base import FullTextLocation
    from providers.registry import ProviderRegistry

    duplicate = candidate(source="core", source_id="core-1", doi="doi:10.1000/example")
    second = candidate(source="core", source_id="core-2", title="Second Paper", doi=None)

    class Discovery:
        def __init__(self, results):
            self.results = results
            self.calls = []

        def search(self, query, *, from_year, max_results):
            self.calls.append((query, from_year, max_results))
            return self.results[:max_results]

    class Resolver:
        def __init__(self, provider):
            self.provider = provider

        def resolve(self, paper):
            return [FullTextLocation(provider=self.provider, url=f"https://{self.provider}.test/a.pdf", is_oa=True, priority=1)]

    class Enricher:
        def enrich(self, paper):
            return paper.model_copy(update={"venue": "Enriched"})

    openalex_provider = Discovery([candidate()])
    core_provider = Discovery([duplicate, second])
    registry = ProviderRegistry(
        openalex=openalex_provider,
        unpaywall=Resolver("unpaywall"),
        core=core_provider,
        crossref=Enricher(),
    )

    discovered = registry.discover("water", from_year=2020, max_results=2)
    resolved = registry.resolve_full_text(candidate())

    assert [paper.source_id for paper in discovered] == ["W123", "core-2"]
    assert core_provider.calls == [("water", 2020, 2)]
    assert [location.provider for location in resolved] == ["unpaywall"]
    assert registry.enrich(candidate()).venue == "Enriched"
    assert registry.enrich(candidate(doi=None)).venue is None


def test_registry_uses_crossref_when_openalex_is_rate_limited(monkeypatch):
    from providers.crossref import CrossrefEnricher
    from providers.openalex import OpenAlexProvider
    from providers.registry import ProviderRegistry
    from utils.http import RetryableHttpError

    class RateLimitedOpenAlex:
        def search(self, query, *, from_year, max_results):
            raise RetryableHttpError(429)

    class CrossrefDiscovery:
        def search(self, query, *, from_year, max_results):
            return [candidate(source="crossref", source_id="10.1000/fallback")]

    registry = ProviderRegistry(
        openalex=RateLimitedOpenAlex(),
        unpaywall=object(),
        core=object(),
        crossref=CrossrefDiscovery(),
    )

    discovered = registry.discover("water", from_year=2020, max_results=1)

    assert [paper.source for paper in discovered] == ["crossref"]


def test_registry_prefers_direct_oa_pdf_without_calling_fallback_resolvers():
    from providers.base import FullTextLocation
    from providers.registry import ProviderRegistry

    class DirectResolver:
        def __init__(self):
            self.calls = 0

        def resolve(self, paper):
            self.calls += 1
            return [
                FullTextLocation(
                    provider="openalex",
                    url="https://repo.test/direct.pdf",
                    is_oa=True,
                    priority=10,
                )
            ]

    class FallbackResolver:
        def __init__(self):
            self.calls = 0

        def resolve(self, paper):
            self.calls += 1
            return [
                FullTextLocation(
                    provider="unpaywall",
                    url="https://repo.test/fallback.pdf",
                    is_oa=True,
                    priority=20,
                )
            ]

    direct = DirectResolver()
    fallback = FallbackResolver()
    registry = ProviderRegistry(openalex=direct, unpaywall=fallback, use_core=False)

    locations = registry.resolve_full_text(candidate(pdf_url="https://repo.test/direct.pdf"))

    assert [location.provider for location in locations] == ["openalex"]
    assert direct.calls == 1
    assert fallback.calls == 0


def test_registry_uses_semantic_scholar_after_doi_fallbacks_when_enabled():
    from providers.base import FullTextLocation
    from providers.registry import ProviderRegistry

    class EmptyResolver:
        def resolve(self, paper):
            return []

    class SemanticResolver:
        def __init__(self):
            self.calls = 0

        def resolve(self, paper):
            self.calls += 1
            return [
                FullTextLocation(
                    provider="semantic_scholar",
                    url="https://repo.test/s2.pdf",
                    is_oa=True,
                    priority=50,
                )
            ]

    semantic = SemanticResolver()
    registry = ProviderRegistry(
        openalex=EmptyResolver(),
        unpaywall=EmptyResolver(),
        semantic_scholar=semantic,
        use_core=False,
        use_semantic_scholar=True,
    )

    locations = registry.resolve_full_text(candidate())

    assert [location.provider for location in locations] == ["semantic_scholar"]
    assert semantic.calls == 1


def test_registry_skips_crossref_for_complete_candidate():
    from providers.registry import ProviderRegistry

    class Crossref:
        def __init__(self):
            self.calls = 0

        def enrich(self, paper):
            self.calls += 1
            return paper

    crossref = Crossref()
    registry = ProviderRegistry(
        openalex=object(), unpaywall=object(), crossref=crossref, use_core=False
    )

    complete = candidate(
        authors=["Ada Lake"],
        year=2024,
        venue="Remote Sensing Journal",
    )

    assert registry.enrich(complete) == complete
    assert crossref.calls == 0


def test_legacy_search_delegates_to_openalex_and_keeps_dict_shape(monkeypatch):
    from utils import search

    class Provider:
        def search(self, query, *, from_year, max_results):
            assert (query, from_year, max_results) == ("water", None, 3)
            return [candidate(landing_url="https://example.test/record")]

    monkeypatch.setattr(search, "OpenAlexProvider", Provider)

    results = search.search_papers("water", max_results=3)

    assert results == [
        {
            "title": "Water Colour from Space",
            "abstract": None,
            "authors": [],
            "year": None,
            "venue": None,
            "url": "https://example.test/record",
            "pdf_url": None,
            "source": "openalex",
            "query": "water",
        }
    ]
