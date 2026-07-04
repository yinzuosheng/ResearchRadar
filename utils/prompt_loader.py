from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE_DIR / "prompts"


@lru_cache
def _load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_system_prompt() -> str:
    return _load_prompt("main_prompt.txt")


def load_rag_prompt() -> str:
    return _load_prompt("rag_summarize.txt")


def load_report_prompt() -> str:
    return _load_prompt("report_prompt.txt")
