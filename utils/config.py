from functools import lru_cache
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"


@lru_cache
def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_agent_config() -> dict:
    return _load_yaml("agent.yml")


def load_tools_config() -> dict:
    return _load_yaml("tools.yml")


def load_rag_config() -> dict:
    return _load_yaml("rag.yml")
