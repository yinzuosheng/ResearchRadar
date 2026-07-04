import json

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.pipeline import (
    collect_and_ingest,
    collect_papers_and_ingest,
    generate_brief as build_brief,
    ingest_url_list,
)
from utils.push import push_message
from utils.search import search_papers as search_papers_api
from utils.search import search_web as search_web_api

rag = RagSummarizeService()


def _split_lines(text: str | None) -> list[str]:
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


@tool(description="Search the web for a query and return results as JSON.")
def search_web(query: str, provider: str = "tavily", max_results: int = 5) -> str:
    results = search_web_api(query, provider, max_results)
    return json.dumps(results, ensure_ascii=True)


@tool(description="Search academic sources for papers and return results as JSON.")
def search_papers(query: str, provider: str = "semantic_scholar", max_results: int = 10) -> str:
    results = search_papers_api(query, provider, max_results)
    return json.dumps(results, ensure_ascii=True)


@tool(description="Collect sources for topics and ingest them into the vector store. Topics are newline-separated.")
def collect_and_ingest_topics(topics: str = "") -> str:
    topics_list = _split_lines(topics)
    return collect_and_ingest(topics_list)


@tool(description="Collect papers for queries and ingest them into the vector store. Queries are newline-separated.")
def collect_and_ingest_papers(
    queries: str = "",
    provider: str = "",
    max_results: int = 0,
    include_pdf: bool = False,
) -> str:
    query_list = _split_lines(queries)
    provider_value = provider or None
    max_value = max_results or None
    include_pdf_value = None if include_pdf is False else True
    return collect_papers_and_ingest(query_list, provider_value, max_value, include_pdf_value)


@tool(description="Ingest URLs into the vector store. Provide one URL per line.")
def ingest_urls(urls: str) -> str:
    url_list = _split_lines(urls)
    return ingest_url_list(url_list)


@tool(description="Query the knowledge base using RAG.")
def query_knowledge_base(question: str) -> str:
    return rag.rag_summarize(question)


@tool(description="Generate a structured daily brief from the knowledge base.")
def generate_brief(topics: str = "") -> str:
    topics_list = _split_lines(topics)
    return build_brief(topics_list)


@tool(description="Push a brief to Feishu, DingTalk, or email.")
def push_brief(brief: str, channel: str = "") -> str:
    push_message(brief, channel)
    return "push completed"
