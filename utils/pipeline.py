import hashlib
import json
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService
from utils.config import load_agent_config, load_rag_config
from utils.content_loader import load_pdf_text, load_url_text
from utils.logger import logger
from utils.search import search_papers, search_web
from utils.text_splitter import split_text
from utils.push import push_message

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = DATA_DIR / "reports"
SOURCES_PATH = DATA_DIR / "sources.json"
PAPER_SOURCES_PATH = DATA_DIR / "paper_sources.json"
PAPER_CACHE_DIR = DATA_DIR / "paper_cache"


def _normalize_topics(topics: list[str] | None) -> list[str]:
    if topics:
        return topics
    cfg = load_agent_config()
    return cfg.get("topics", [])


def _normalize_paper_queries(queries: list[str] | None) -> list[str]:
    if queries:
        return queries
    cfg = load_agent_config()
    return cfg.get("paper_queries", [])


def _normalize_paper_providers(provider: str | None) -> list[str]:
    if provider:
        return [provider]
    cfg = load_agent_config()
    return cfg.get("papers", {}).get("providers", ["semantic_scholar"])


def collect_sources(topics: list[str], provider: str, max_results: int) -> list[dict]:
    sources = []
    seen = set()
    for topic in topics:
        results = search_web(topic, provider, max_results)
        for item in results:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            item["topic"] = topic
            sources.append(item)
    return sources


def save_sources(sources: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_PATH, "w", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=True, indent=2)


def save_paper_sources(sources: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPER_SOURCES_PATH, "w", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=True, indent=2)


def ingest_sources(sources: list[dict]) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = VectorStoreService()
    ingested = 0
    for src in sources:
        url = src.get("url", "")
        if not url:
            continue
        cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.txt"

        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            try:
                text = load_url_text(url)
            except Exception as exc:
                logger.warning("failed to load %s: %s", url, exc)
                continue
            cache_path.write_text(text, encoding="utf-8")

        if not text:
            continue

        chunks = split_text(text)
        metadata = {
            "url": url,
            "title": src.get("title"),
            "source": src.get("source"),
            "topic": src.get("topic"),
        }
        docs = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        store.add_documents(docs)
        ingested += 1
        logger.info("ingested %s chunks for %s", len(chunks), url)

    return {"sources": len(sources), "ingested": ingested}


def collect_paper_sources(
    queries: list[str],
    providers: list[str],
    max_results: int,
    min_year: int | None,
) -> list[dict]:
    sources = []
    seen = set()
    for provider in providers:
        for query in queries:
            results = search_papers(query, provider, max_results)
            for item in results:
                year = item.get("year")
                if min_year and year and int(year) < min_year:
                    continue
                url = item.get("url") or item.get("pdf_url") or ""
                key = f"{item.get('title')}|{year}|{url}"
                if key in seen:
                    continue
                seen.add(key)
                item["query"] = query
                sources.append(item)
    return sources


def _format_paper_text(paper: dict) -> str:
    parts = [
        f"Title: {paper.get('title', '')}",
        f"Authors: {', '.join(paper.get('authors') or [])}",
        f"Year: {paper.get('year', '')}",
        f"Venue: {paper.get('venue', '')}",
        f"URL: {paper.get('url', '')}",
    ]
    abstract = paper.get("abstract") or ""
    if abstract:
        parts.append("Abstract: " + abstract)
    return "\n".join(parts).strip()


def _truncate_text(text: str) -> str:
    cfg = load_rag_config()
    max_chars = int(cfg.get("max_text_chars", 20000))
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def ingest_papers(papers: list[dict], include_pdf: bool) -> dict:
    PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = VectorStoreService()
    ingested = 0

    for paper in papers:
        url = paper.get("url") or ""
        cache_key = hashlib.md5((url or paper.get("title", "")).encode("utf-8")).hexdigest()
        cache_path = PAPER_CACHE_DIR / f"{cache_key}.txt"

        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            text = _format_paper_text(paper)
            pdf_url = paper.get("pdf_url")
            if include_pdf and pdf_url:
                try:
                    pdf_text = load_pdf_text(pdf_url)
                    if pdf_text:
                        text = text + "\n\nFull Text:\n" + pdf_text
                except Exception as exc:
                    logger.warning("failed to load pdf %s: %s", pdf_url, exc)
            cache_path.write_text(text, encoding="utf-8")

        if not text:
            continue

        trimmed = _truncate_text(text)
        chunks = split_text(trimmed)
        metadata = {
            "url": paper.get("url"),
            "title": paper.get("title"),
            "source": paper.get("source"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
        }
        docs = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        store.add_documents(docs)
        ingested += 1
        logger.info("ingested %s chunks for %s", len(chunks), paper.get("title"))

    return {"sources": len(papers), "ingested": ingested}


def ingest_url_list(urls: list[str]) -> str:
    sources = [{"url": url, "source": "manual"} for url in urls]
    stats = ingest_sources(sources)
    return f"ingested_sources={stats['ingested']}"


def collect_and_ingest(
    topics: list[str] | None = None,
    provider: str | None = None,
    max_results: int | None = None,
) -> str:
    cfg = load_agent_config()
    topics = _normalize_topics(topics)
    if not topics:
        return "no topics configured"
    provider = provider or cfg.get("search", {}).get("provider", "tavily")
    max_results = max_results or int(cfg.get("search", {}).get("max_results", 5))

    sources = collect_sources(topics, provider, max_results)
    save_sources(sources)
    stats = ingest_sources(sources)
    return f"collected={len(sources)} ingested={stats['ingested']}"


def collect_papers_and_ingest(
    queries: list[str] | None = None,
    provider: str | None = None,
    max_results: int | None = None,
    include_pdf: bool | None = None,
) -> str:
    cfg = load_agent_config()
    queries = _normalize_paper_queries(queries)
    if not queries:
        return "no paper queries configured"

    providers = _normalize_paper_providers(provider)
    paper_cfg = cfg.get("papers", {})
    max_results = max_results or int(paper_cfg.get("max_results", 10))
    include_pdf = include_pdf if include_pdf is not None else bool(paper_cfg.get("include_pdf", False))
    min_year = paper_cfg.get("min_year")
    if min_year is not None:
        min_year = int(min_year)

    papers = collect_paper_sources(queries, providers, max_results, min_year)
    save_paper_sources(papers)
    stats = ingest_papers(papers, include_pdf)
    return f"collected={len(papers)} ingested={stats['ingested']}"


def generate_brief(topics: list[str] | None = None) -> str:
    topics = _normalize_topics(topics)
    if not topics:
        return "no topics configured"
    query = "Daily intelligence brief for: " + ", ".join(topics)
    rag = RagSummarizeService()
    return rag.rag_report(query)


def _save_report(text: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"brief-{stamp}.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_daily_brief(topics: list[str] | None = None, push: bool = False) -> str:
    cfg = load_agent_config().get("ingestion", {})
    if cfg.get("papers", True):
        collect_papers_and_ingest()
    if cfg.get("web", False):
        collect_and_ingest(topics)
    brief = generate_brief(topics)
    _save_report(brief)

    cfg = load_agent_config()
    if push or cfg.get("push", {}).get("enabled", False):
        push_message(brief, cfg.get("push", {}).get("channel", "feishu"))

    return brief
