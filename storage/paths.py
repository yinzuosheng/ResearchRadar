"""Filesystem locations for local, generated research data."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def default_database_path() -> Path:
    return data_dir() / "research.db"
