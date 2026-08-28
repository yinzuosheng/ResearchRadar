from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_env_example_contains_names_but_no_supplied_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENALEX_API_KEY=" in text
    assert "UNPAYWALL_EMAIL=" in text
    assert "CORE_API_KEY=" in text
    assert "SEMANTIC_SCHOLAR_API_KEY=" in text
    for line in text.splitlines():
        if line.startswith(("OPENALEX_API_KEY=", "UNPAYWALL_EMAIL=", "CORE_API_KEY=", "SEMANTIC_SCHOLAR_API_KEY=")):
            assert line.endswith("=")


def test_gitignore_excludes_runtime_secrets():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in text.splitlines()
    assert "!.env.example" in text.splitlines()
