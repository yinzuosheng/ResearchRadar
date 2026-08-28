import ast
from pathlib import Path

import pytest

from utils.config import load_agent_config, load_tools_config
from utils import search


APPROVED_PROVIDERS = {"openalex", "unpaywall", "core", "crossref", "semantic_scholar"}
UNSUPPORTED_PROVIDERS = {"tavily", "bing", "arxiv"}
ROOT = Path(__file__).resolve().parents[1]


def test_only_approved_literature_providers_are_configured_or_selectable():
    configured = set(load_tools_config())

    assert APPROVED_PROVIDERS <= configured
    assert not configured & UNSUPPORTED_PROVIDERS
    assert not hasattr(search, "search_web")

    for provider in UNSUPPORTED_PROVIDERS:
        with pytest.raises(ValueError, match="unsupported paper provider"):
            search.search_papers("water color remote sensing", provider=provider)


def test_only_openalex_is_exposed_for_discovery_until_other_adapters_exist():
    assert load_agent_config()["papers"]["providers"] == ["openalex"]

    for provider in APPROVED_PROVIDERS - {"openalex"}:
        with pytest.raises(ValueError, match="unsupported paper provider"):
            search.search_papers("water color remote sensing", provider=provider)


def test_app_does_not_import_deleted_web_collection_route():
    app_module = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(app_module)
        if isinstance(node, ast.ImportFrom) and node.module == "utils.pipeline"
        for alias in node.names
    }

    assert "collect_and_ingest" not in imported_names
