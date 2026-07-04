import os
import xml.etree.ElementTree as ET

from utils.config import load_tools_config
from utils.http import get, post_json


def search_web(query: str, provider: str = "tavily", max_results: int = 5) -> list[dict]:
    provider = provider.lower()
    if provider == "bing":
        return _search_bing(query, max_results)
    return _search_tavily(query, max_results)


def search_papers(query: str, provider: str = "semantic_scholar", max_results: int = 10) -> list[dict]:
    provider = provider.lower()
    if provider in {"semantic_scholar", "semanticscholar", "s2"}:
        return _search_semantic_scholar(query, max_results)
    if provider == "openalex":
        return _search_openalex(query, max_results)
    if provider == "arxiv":
        return _search_arxiv(query, max_results)
    raise ValueError(f"unsupported paper provider: {provider}")


def _search_tavily(query: str, max_results: int) -> list[dict]:
    cfg = load_tools_config().get("tavily", {})
    key = os.getenv(cfg.get("api_key_env", "TAVILY_API_KEY"), "")
    if not key:
        raise RuntimeError("Tavily API key is missing")
    endpoint = cfg.get("endpoint", "https://api.tavily.com/search")
    payload = {
        "api_key": key,
        "query": query,
        "search_depth": cfg.get("search_depth", "advanced"),
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    resp = post_json(endpoint, payload)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("content"),
            "source": "tavily",
            "query": query,
        }
        for item in results
    ]


def _search_bing(query: str, max_results: int) -> list[dict]:
    cfg = load_tools_config().get("bing", {})
    key = os.getenv(cfg.get("api_key_env", "BING_API_KEY"), "")
    if not key:
        raise RuntimeError("Bing API key is missing")
    endpoint = cfg.get("endpoint", "https://api.bing.microsoft.com/v7.0/search")
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"q": query, "count": max_results, "textDecorations": False, "textFormat": "Raw"}
    resp = get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    items = resp.json().get("webPages", {}).get("value", [])
    return [
        {
            "title": item.get("name"),
            "url": item.get("url"),
            "snippet": item.get("snippet"),
            "source": "bing",
            "query": query,
        }
        for item in items
    ]


def _search_semantic_scholar(query: str, max_results: int) -> list[dict]:
    cfg = load_tools_config().get("semantic_scholar", {})
    endpoint = cfg.get("endpoint", "https://api.semanticscholar.org/graph/v1/paper/search")
    key = os.getenv(cfg.get("api_key_env", "SEMANTIC_SCHOLAR_API_KEY"), "")
    headers = {"x-api-key": key} if key else {}
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,venue,url,openAccessPdf",
    }
    resp = get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    results = resp.json().get("data", [])
    papers = []
    for item in results:
        authors = [author.get("name") for author in item.get("authors", [])]
        pdf_info = item.get("openAccessPdf") or {}
        papers.append(
            {
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "authors": authors,
                "year": item.get("year"),
                "venue": item.get("venue"),
                "url": item.get("url"),
                "pdf_url": pdf_info.get("url"),
                "source": "semantic_scholar",
                "query": query,
            }
        )
    return papers


def _search_openalex(query: str, max_results: int) -> list[dict]:
    cfg = load_tools_config().get("openalex", {})
    endpoint = cfg.get("endpoint", "https://api.openalex.org/works")
    email = os.getenv(cfg.get("email_env", "OPENALEX_EMAIL"), "")
    params = {
        "search": query,
        "per-page": max_results,
        "sort": "publication_date:desc",
    }
    if email:
        params["mailto"] = email
    resp = get(endpoint, params=params)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    papers = []
    for item in results:
        authors = []
        for auth in item.get("authorships", []):
            name = auth.get("author", {}).get("display_name")
            if name:
                authors.append(name)
        papers.append(
            {
                "title": item.get("title"),
                "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
                "authors": authors,
                "year": item.get("publication_year"),
                "venue": (item.get("primary_location", {}) or {}).get("source", {}).get("display_name"),
                "url": (item.get("primary_location", {}) or {}).get("landing_page_url")
                or item.get("doi")
                or item.get("id"),
                "pdf_url": (item.get("open_access", {}) or {}).get("oa_url"),
                "source": "openalex",
                "query": query,
            }
        )
    return papers


def _openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        for pos in positions:
            if pos > max_pos:
                max_pos = pos
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    return " ".join(word for word in words if word)


def _search_arxiv(query: str, max_results: int) -> list[dict]:
    cfg = load_tools_config().get("arxiv", {})
    endpoint = cfg.get("endpoint", "http://export.arxiv.org/api/query")
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = get(endpoint, params=params)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = _get_text(entry, "atom:title", ns)
        summary = _get_text(entry, "atom:summary", ns)
        published = _get_text(entry, "atom:published", ns)
        url = _get_text(entry, "atom:id", ns)
        authors = [
            author.findtext("atom:name", default="", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
        year = published[:4] if published else None
        papers.append(
            {
                "title": title,
                "abstract": summary,
                "authors": [name for name in authors if name],
                "year": year,
                "venue": "arXiv",
                "url": url,
                "pdf_url": pdf_url,
                "source": "arxiv",
                "query": query,
            }
        )
    return papers


def _get_text(node: ET.Element, path: str, ns: dict) -> str:
    found = node.find(path, ns)
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())
