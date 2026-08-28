"""SQLite-backed, idempotent catalog for discovered research papers."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata
from uuid import uuid4

from domain.models import (
    EvidenceChunk,
    PaperCandidate,
    PaperProfile,
    PaperRecord,
    ResearchProjectMemory,
)
from domain.statuses import DISCOVERED


class PaperIdentityConflictError(ValueError):
    """Raised when DOI and title/year keys identify different papers."""


def normalize_doi(doi: str | None) -> str | None:
    """Return the canonical comparison key for a DOI, if supplied."""
    if doi is None:
        return None
    normalized = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized or None


def normalize_title(title: str) -> str:
    """Normalize titles for the no-DOI deduplication key."""
    normalized = unicodedata.normalize("NFKC", title).lower()
    without_punctuation = "".join(
        character if not unicodedata.category(character).startswith("P") else " "
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def source_fingerprint(candidate: PaperCandidate) -> str:
    """Hash the source fields that determine whether metadata changed."""
    payload = {
        "abstract": candidate.abstract,
        "doi": candidate.doi,
        "license": candidate.license,
        "pdf_url": candidate.pdf_url,
        "source": candidate.source,
        "source_id": candidate.source_id,
        "source_updated_at": candidate.source_updated_at,
        "title": candidate.title,
        "year": candidate.year,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _deserialize_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ResearchDatabase:
    """Repository for papers, evidence chunks, profiles, and sync runs."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    doi TEXT,
                    authors_json TEXT NOT NULL,
                    year INTEGER,
                    venue TEXT,
                    abstract TEXT,
                    landing_url TEXT,
                    pdf_url TEXT,
                    license TEXT,
                    cited_by_count INTEGER NOT NULL DEFAULT 0,
                    source_updated_at TEXT,
                    normalized_doi TEXT,
                    normalized_title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section TEXT,
                    text TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS paper_profiles (
                    paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sync_runs (
                    sync_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    discovered INTEGER NOT NULL DEFAULT 0,
                    downloaded INTEGER NOT NULL DEFAULT 0,
                    indexed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_project_memory (
                    project_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_papers_doi
                ON papers(normalized_doi) WHERE normalized_doi IS NOT NULL;

                CREATE UNIQUE INDEX IF NOT EXISTS ux_papers_title_year
                ON papers(normalized_title, year) WHERE normalized_doi IS NULL;

                CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_identity
                ON chunks(paper_id, page_number, chunk_index);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    paper_id UNINDEXED,
                    title,
                    text,
                    tokenize = 'unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
                AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunk_fts(chunk_id, paper_id, title, text)
                    VALUES (new.chunk_id, new.paper_id, new.title, new.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
                AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunk_fts WHERE chunk_id = old.chunk_id;
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_update
                AFTER UPDATE ON chunks BEGIN
                    DELETE FROM chunk_fts WHERE chunk_id = old.chunk_id;
                    INSERT INTO chunk_fts(chunk_id, paper_id, title, text)
                    VALUES (new.chunk_id, new.paper_id, new.title, new.text);
                END;
                """
            )
            if self._fts_projection_needs_backfill(connection):
                connection.execute(
                    """
                    INSERT INTO chunk_fts(chunk_id, paper_id, title, text)
                    SELECT chunks.chunk_id, chunks.paper_id, chunks.title, chunks.text
                    FROM chunks
                    WHERE NOT EXISTS (
                        SELECT 1 FROM chunk_fts
                        WHERE chunk_fts.chunk_id = chunks.chunk_id
                    )
                    """
                )

    def _fts_projection_needs_backfill(self, connection=None) -> bool:
        """Check for missing or stale FTS rows without scanning chunk contents."""
        if connection is None:
            with self._connect() as owned_connection:
                return self._fts_projection_needs_backfill(owned_connection)
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
        return chunk_count != fts_count

    def upsert_candidate(self, candidate: PaperCandidate) -> PaperRecord:
        normalized_doi = normalize_doi(candidate.doi)
        normalized_title = normalize_title(candidate.title)
        fingerprint = source_fingerprint(candidate)
        now = _serialize_datetime(_utc_now())
        with self._connect() as connection:
            if normalized_doi is not None:
                doi_row = connection.execute(
                    "SELECT paper_id FROM papers WHERE normalized_doi = ?", (normalized_doi,)
                ).fetchone()
                title_year_row = connection.execute(
                    """
                    SELECT paper_id FROM papers
                    WHERE normalized_doi IS NULL
                      AND normalized_title = ?
                      AND year IS ?
                    """,
                    (normalized_title, candidate.year),
                ).fetchone()
                if (
                    doi_row is not None
                    and title_year_row is not None
                    and doi_row["paper_id"] != title_year_row["paper_id"]
                ):
                    raise PaperIdentityConflictError(
                        "DOI and title/year match different papers"
                    )
                row = doi_row or title_year_row
            else:
                row = connection.execute(
                    """
                    SELECT paper_id FROM papers
                    WHERE normalized_doi IS NULL
                      AND normalized_title = ?
                      AND year IS ?
                    """,
                    (normalized_title, candidate.year),
                ).fetchone()

            if row is None:
                paper_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO papers (
                        paper_id, source, source_id, title, doi, authors_json, year,
                        venue, abstract, landing_url, pdf_url, license, cited_by_count,
                        source_updated_at, normalized_doi, normalized_title, status,
                        source_fingerprint, first_seen_at, updated_at, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    self._candidate_values(
                        candidate,
                        paper_id,
                        normalized_doi,
                        normalized_title,
                        fingerprint,
                        now,
                    ),
                )
            else:
                paper_id = row["paper_id"]
                connection.execute(
                    """
                    UPDATE papers SET
                        source = ?, source_id = ?, title = ?, doi = ?, authors_json = ?,
                        year = ?, venue = ?, abstract = ?, landing_url = ?, pdf_url = ?,
                        license = ?, cited_by_count = ?, source_updated_at = ?,
                        normalized_doi = ?, normalized_title = ?, source_fingerprint = ?,
                        updated_at = ?
                    WHERE paper_id = ?
                    """,
                    (
                        candidate.source,
                        candidate.source_id,
                        candidate.title,
                        candidate.doi,
                        json.dumps(candidate.authors, ensure_ascii=False),
                        candidate.year,
                        candidate.venue,
                        candidate.abstract,
                        candidate.landing_url,
                        candidate.pdf_url,
                        candidate.license,
                        candidate.cited_by_count,
                        candidate.source_updated_at,
                        normalized_doi,
                        normalized_title,
                        fingerprint,
                        now,
                        paper_id,
                    ),
                )
            return self._record_from_row(
                connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
            )

    def find_candidate(self, candidate: PaperCandidate) -> PaperRecord | None:
        """Return the stored identity before an upsert changes its fingerprint."""
        normalized_doi = normalize_doi(candidate.doi)
        normalized_title = normalize_title(candidate.title)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM papers
                WHERE normalized_doi = ?
                   OR (
                       normalized_doi IS NULL
                       AND normalized_title = ?
                       AND year IS ?
                   )
                """,
                (normalized_doi, normalized_title, candidate.year),
            ).fetchall()
            doi_row = next(
                (row for row in rows if row["normalized_doi"] == normalized_doi),
                None,
            ) if normalized_doi is not None else None
            title_year_row = next(
                (row for row in rows if row["normalized_doi"] is None),
                None,
            )
            if normalized_doi is not None:
                if (
                    doi_row is not None
                    and title_year_row is not None
                    and doi_row["paper_id"] != title_year_row["paper_id"]
                ):
                    raise PaperIdentityConflictError(
                        "DOI and title/year match different papers"
                    )
                row = doi_row or title_year_row
            else:
                row = title_year_row
        return self._record_from_row(row) if row is not None else None

    def update_enriched_candidate(
        self, paper_id: str, candidate: PaperCandidate
    ) -> PaperRecord:
        """Persist enriched fields without replacing the source fingerprint."""
        normalized_doi = normalize_doi(candidate.doi)
        normalized_title = normalize_title(candidate.title)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE papers SET
                    title = ?, doi = ?, authors_json = ?, year = ?, venue = ?,
                    abstract = ?, landing_url = ?, pdf_url = ?, license = ?,
                    cited_by_count = ?, normalized_doi = ?, normalized_title = ?,
                    updated_at = ?
                WHERE paper_id = ?
                """,
                (
                    candidate.title,
                    candidate.doi,
                    json.dumps(candidate.authors, ensure_ascii=False),
                    candidate.year,
                    candidate.venue,
                    candidate.abstract,
                    candidate.landing_url,
                    candidate.pdf_url,
                    candidate.license,
                    candidate.cited_by_count,
                    normalized_doi,
                    normalized_title,
                    _serialize_datetime(_utc_now()),
                    paper_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown paper_id: {paper_id}")
        return self._record_from_row(row)

    @staticmethod
    def _candidate_values(
        candidate: PaperCandidate,
        paper_id: str,
        normalized_doi: str | None,
        normalized_title: str,
        fingerprint: str,
        now: str,
    ) -> tuple[object, ...]:
        return (
            paper_id,
            candidate.source,
            candidate.source_id,
            candidate.title,
            candidate.doi,
            json.dumps(candidate.authors, ensure_ascii=False),
            candidate.year,
            candidate.venue,
            candidate.abstract,
            candidate.landing_url,
            candidate.pdf_url,
            candidate.license,
            candidate.cited_by_count,
            candidate.source_updated_at,
            normalized_doi,
            normalized_title,
            DISCOVERED,
            fingerprint,
            now,
            now,
        )

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def list_papers(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[PaperRecord]:
        query = "SELECT * FROM papers"
        parameters: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            parameters.append(status)
        query += " ORDER BY first_seen_at DESC, paper_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record_from_row(row) for row in rows]

    def list_papers_with_doi(
        self, *, status: str, limit: int = 100
    ) -> list[PaperRecord]:
        """Return status-matched papers that can be retried by DOI."""
        if not status or isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("database_invalid_paper_query")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM papers
                WHERE status = ? AND normalized_doi IS NOT NULL
                ORDER BY
                    CASE WHEN source = 'openalex' THEN 0 ELSE 1 END,
                    CASE WHEN pdf_url IS NOT NULL AND TRIM(pdf_url) <> '' THEN 0 ELSE 1 END,
                    first_seen_at ASC,
                    paper_id ASC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def list_papers_discovered_after(self, since: datetime) -> list[PaperRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM papers WHERE first_seen_at > ?
                ORDER BY first_seen_at ASC, paper_id ASC
                """,
                (_serialize_datetime(since),),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def count_papers(self, *, status: str | None = None) -> int:
        with self._connect() as connection:
            if status is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM papers").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM papers WHERE status = ?", (status,)
                ).fetchone()
        return int(row["count"])

    def catalog_statistics(self, *, status_names: tuple[str, ...]) -> dict[str, object]:
        """Return count-only catalog statistics without reading sensitive fields."""
        if not status_names:
            raise ValueError("statistics_status_names_required")
        placeholders = ", ".join("?" for _ in status_names)
        with self._connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"]
            )
            status_rows = connection.execute(
                f"""
                SELECT status, COUNT(*) AS count
                FROM papers
                WHERE status IN ({placeholders})
                GROUP BY status
                """,
                status_names,
            ).fetchall()
            provider_rows = connection.execute(
                "SELECT source, COUNT(*) AS count FROM papers GROUP BY source ORDER BY source"
            ).fetchall()
        status_counts = {row["status"]: int(row["count"]) for row in status_rows}
        return {
            "metadata_total": total,
            **{name: status_counts.get(name, 0) for name in status_names},
            "providers": {row["source"]: int(row["count"]) for row in provider_rows},
        }

    def knowledge_statistics(self) -> dict[str, object]:
        """Return the workbench status cards and saved profile distributions."""
        status_names = (
            "pdf_ready",
            "parsed",
            "indexed",
            "abstract_only",
            "failed",
        )
        stats = self.catalog_statistics(status_names=status_names)
        with self._connect() as connection:
            paper_rows = connection.execute(
                "SELECT paper_id, year, venue, status, last_error FROM papers"
            ).fetchall()
            profile_rows = connection.execute(
                "SELECT paper_id, profile_json FROM paper_profiles"
            ).fetchall()
            indexed_rows = connection.execute(
                """
                SELECT paper_id, title, year, venue, status, doi, last_error
                FROM papers WHERE status = 'indexed'
                ORDER BY first_seen_at DESC, paper_id DESC LIMIT 500
                """
            ).fetchall()
            failure_rows = connection.execute(
                """
                SELECT paper_id, title, last_error, updated_at FROM papers
                WHERE status = 'failed' AND last_error IS NOT NULL
                ORDER BY updated_at DESC, paper_id DESC LIMIT 10
                """
            ).fetchall()
            chunks_total = int(
                connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
            )
            chunk_count_rows = connection.execute(
                "SELECT paper_id, COUNT(*) AS count FROM chunks GROUP BY paper_id"
            ).fetchall()
        stats["profiled"] = len(profile_rows)
        stats["stats"] = {
            "metadata_total": stats.pop("metadata_total"),
            "pdf_ready": stats.pop("pdf_ready", 0),
            "parsed": stats.pop("parsed", 0),
            "profiled": stats.pop("profiled", 0),
            "indexed": stats.pop("indexed", 0),
            "abstract_only": stats.pop("abstract_only", 0),
            "failed": stats.pop("failed", 0),
        }

        def count_values(values: list[str]) -> dict[str, int]:
            if not values:
                return {}
            counts: dict[str, int] = {}
            for value in values:
                label = value.strip() or "未分类"
                counts[label] = counts.get(label, 0) + 1
            return dict(sorted(counts.items()))

        years: dict[int, int] = {}
        venues: dict[str, int] = {}
        for row in paper_rows:
            if row["year"] is not None:
                years[row["year"]] = years.get(row["year"], 0) + 1
            venue = (row["venue"] or "未分类").strip() or "未分类"
            venues[venue] = venues.get(venue, 0) + 1

        targets, sensors, methods, study_areas = [], [], [], []
        profiles_by_paper: dict[str, PaperProfile] = {}
        for row in profile_rows:
            profile = PaperProfile.model_validate_json(row["profile_json"])
            profiles_by_paper[row["paper_id"]] = profile
        for row in paper_rows:
            profile = profiles_by_paper.get(row["paper_id"])
            if profile is None:
                targets.append("未分类")
                sensors.append("未分类")
                methods.append("未分类")
                study_areas.append("未分类")
                continue
            targets.append(profile.prediction_target.value or "未分类")
            sensor_values = [field.value for field in profile.sensors if field.value]
            sensors.extend(sensor_values or ["未分类"])
            method_values = [field.value for field in profile.models if field.value]
            methods.extend(method_values or ["未分类"])
            study_areas.append(profile.study_area.value or "未分类")
        return {
            "stats": stats["stats"],
            "providers": stats.get("providers", {}),
            "chunks_total": chunks_total,
            "paper_chunk_counts": {
                row["paper_id"]: int(row["count"]) for row in chunk_count_rows
            },
            "profiled_paper_ids": [row["paper_id"] for row in profile_rows],
            "years": dict(sorted(years.items())),
            "venues": dict(sorted(venues.items())),
            "prediction_targets": count_values(targets),
            "sensors": count_values(sensors),
            "methods": count_values(methods),
            "study_areas": count_values(study_areas),
            "recent_failures": [
                {
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "code": row["last_error"],
                    "updated_at": row["updated_at"],
                }
                for row in failure_rows
            ],
            "indexed_papers": [dict(row) for row in indexed_rows],
        }

    def update_status(self, paper_id: str, status: str, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE papers SET status = ?, last_error = ?, updated_at = ? WHERE paper_id = ?",
                (status, error, _serialize_datetime(_utc_now()), paper_id),
            )

    def get_project_memory(self, project_id: str = "water-color-prediction") -> ResearchProjectMemory:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_json FROM research_project_memory WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            return ResearchProjectMemory(project_id=project_id)
        return ResearchProjectMemory.model_validate_json(row["memory_json"])

    def save_project_memory(self, memory: ResearchProjectMemory) -> None:
        memory = memory.model_copy(update={"updated_at": _utc_now()})
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_project_memory (project_id, memory_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    memory_json = excluded.memory_json,
                    updated_at = excluded.updated_at
                """,
                (
                    memory.project_id,
                    memory.model_dump_json(),
                    _serialize_datetime(memory.updated_at),
                ),
            )

    def confirm_paper(self, project_id: str, paper_id: str) -> None:
        memory = self.get_project_memory(project_id)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        if exists is None:
            raise ValueError("paper_not_found")
        if paper_id not in memory.confirmed_paper_ids:
            memory.confirmed_paper_ids.append(paper_id)
        self.save_project_memory(memory)

    def replace_chunks(self, paper_id: str, chunks: list[EvidenceChunk]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            for chunk_index, chunk in enumerate(chunks):
                if chunk.paper_id != paper_id:
                    raise ValueError("all chunks must belong to the supplied paper_id")
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, paper_id, title, page_number, chunk_index, section, text, score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        paper_id,
                        chunk.title,
                        chunk.page_number,
                        chunk_index,
                        chunk.section,
                        chunk.text,
                        chunk.score,
                    ),
                )

    def get_chunks(self, paper_id: str) -> list[EvidenceChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index ASC", (paper_id,)
            ).fetchall()
        return [
            EvidenceChunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                page_number=row["page_number"],
                section=row["section"],
                text=row["text"],
                score=row["score"],
            )
            for row in rows
        ]

    def list_chunks(self) -> list[EvidenceChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks ORDER BY paper_id ASC, chunk_index ASC"
            ).fetchall()
        return [
            EvidenceChunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                page_number=row["page_number"],
                section=row["section"],
                text=row["text"],
                score=row["score"],
            )
            for row in rows
        ]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        by_id = {
            row["chunk_id"]: EvidenceChunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                page_number=row["page_number"],
                section=row["section"],
                text=row["text"],
                score=row["score"],
            )
            for row in rows
        }
        return [by_id[item] for item in chunk_ids if item in by_id]

    def get_chunks_with_context(
        self, chunk_ids: list[str], *, window: int = 1
    ) -> list[EvidenceChunk]:
        """Return selected chunks plus adjacent chunks from the same paper."""
        if not chunk_ids or window < 0:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            anchors = connection.execute(
                f"SELECT paper_id, chunk_index FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
            if not anchors:
                return []
            clauses = []
            params: list[object] = []
            for row in anchors:
                clauses.append("(paper_id = ? AND chunk_index BETWEEN ? AND ?)")
                params.extend([row["paper_id"], row["chunk_index"] - window, row["chunk_index"] + window])
            rows = connection.execute(
                "SELECT * FROM chunks WHERE " + " OR ".join(clauses) + " ORDER BY paper_id, chunk_index",
                params,
            ).fetchall()
        return [
            EvidenceChunk(
                chunk_id=row["chunk_id"], paper_id=row["paper_id"], title=row["title"],
                page_number=row["page_number"], section=row["section"], text=row["text"], score=row["score"],
            )
            for row in rows
        ]

    def save_profile(self, paper_id: str, profile: PaperProfile) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_profiles (paper_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (paper_id, profile.model_dump_json(), _serialize_datetime(_utc_now())),
            )

    def get_profile(self, paper_id: str) -> PaperProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT profile_json FROM paper_profiles WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        return PaperProfile.model_validate_json(row["profile_json"]) if row else None

    def start_sync(self, kind: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO sync_runs (kind, started_at, status) VALUES (?, ?, ?)",
                (kind, _serialize_datetime(_utc_now()), "running"),
            )
            return int(cursor.lastrowid)

    def finish_sync(
        self, sync_id: int, *, discovered: int, downloaded: int, indexed: int
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sync_runs
                SET finished_at = ?, discovered = ?, downloaded = ?, indexed = ?, status = ?
                WHERE sync_id = ?
                """,
                (
                    _serialize_datetime(_utc_now()),
                    discovered,
                    downloaded,
                    indexed,
                    "success",
                    sync_id,
                ),
            )

    def last_successful_sync(self, kind: str) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT finished_at FROM sync_runs
                WHERE kind = ? AND status = 'success'
                ORDER BY finished_at DESC, sync_id DESC LIMIT 1
                """,
                (kind,),
            ).fetchone()
        return _deserialize_datetime(row["finished_at"]) if row else None

    def recent_sync_runs(self, limit: int = 20) -> list[dict[str, object]]:
        """Return bounded, display-safe sync history for the local workbench."""
        bounded_limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sync_id, kind, started_at, finished_at,
                       discovered, downloaded, indexed, status
                FROM sync_runs
                ORDER BY sync_id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> PaperRecord:
        return PaperRecord(
            paper_id=row["paper_id"],
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            doi=row["doi"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            venue=row["venue"],
            abstract=row["abstract"],
            landing_url=row["landing_url"],
            pdf_url=row["pdf_url"],
            license=row["license"],
            cited_by_count=row["cited_by_count"],
            source_updated_at=row["source_updated_at"],
            normalized_doi=row["normalized_doi"],
            normalized_title=row["normalized_title"],
            status=row["status"],
            source_fingerprint=row["source_fingerprint"],
            first_seen_at=_deserialize_datetime(row["first_seen_at"]),
            updated_at=_deserialize_datetime(row["updated_at"]),
            last_error=row["last_error"],
        )
