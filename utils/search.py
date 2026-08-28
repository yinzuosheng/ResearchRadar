from providers.openalex import OpenAlexProvider


def search_papers(query: str, provider: str = "openalex", max_results: int = 10) -> list[dict]:
    provider = provider.lower()
    if provider != "openalex":
        raise ValueError(f"unsupported paper provider: {provider}")
    return _search_openalex(query, max_results)


def _search_openalex(query: str, max_results: int) -> list[dict]:
    candidates = OpenAlexProvider().search(
        query, from_year=None, max_results=max_results
    )
    return [
        {
            "title": candidate.title,
            "abstract": candidate.abstract,
            "authors": candidate.authors,
            "year": candidate.year,
            "venue": candidate.venue,
            "url": candidate.landing_url or candidate.doi or candidate.source_id,
            "pdf_url": candidate.pdf_url,
            "source": candidate.source,
            "query": query,
        }
        for candidate in candidates
    ]
