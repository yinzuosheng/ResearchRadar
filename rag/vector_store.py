"""Injectable FAISS storage with stable evidence identities and guarded rebuilds."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any
from uuid import uuid4
from datetime import UTC, datetime

from langchain_core.documents import Document

from domain.models import EvidenceChunk
from retrieval.query_expansion import plan_query
from utils.config import load_rag_config
from utils.logger import logger


class UnsafeVectorStorePathError(ValueError):
    """Raised before a rebuild could delete outside its narrow data boundary."""


class VectorStoreCompatibilityError(ValueError):
    """Raised when a persisted index was built with incompatible settings."""


class EmptyRetriever:
    def invoke(self, query: str):
        return []


class VectorStoreService:
    REQUIRED_METADATA = {
        "chunk_id",
        "paper_id",
        "title",
        "page_number",
        "section",
    }
    EMBEDDING_TEXT_SCHEMA = "title-location-text-v1"

    def __init__(
        self,
        *,
        embeddings: Any | None = None,
        store: Any | None = None,
        store_path: Path | str | None = None,
        repository_root: Path | str | None = None,
        database: Any | None = None,
        faiss_cls: Any | None = None,
        config: dict | None = None,
        load_existing: bool = True,
    ) -> None:
        self.config = config or load_rag_config()
        self.repository_root = Path(
            repository_root or Path(__file__).resolve().parents[1]
        ).resolve()
        configured = (
            Path(store_path)
            if store_path is not None
            else Path(self.config.get("vector_store_path", "data/vector_store"))
        )
        self._configured_store_path = configured
        self.store_path = (
            configured if configured.is_absolute() else self.repository_root / configured
        ).resolve(strict=False)
        self.database = database
        self.faiss_cls = faiss_cls
        self.embeddings = embeddings
        self._store = store
        self._document_count = len(
            getattr(store, "index_to_docstore_id", {})
        ) if store is not None else 0
        if self._store is None:
            self.embeddings = embeddings or self._build_embeddings()
            if load_existing:
                self._load_or_create()

    def _build_embeddings(self):
        provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
            client_options = {}
            base_url = os.getenv("EMBEDDINGS_BASE_URL", "").strip() or os.getenv(
                "OPENAI_BASE_URL", ""
            ).strip()
            api_key = os.getenv("EMBEDDINGS_API_KEY", "").strip()
            if base_url:
                client_options["base_url"] = base_url
            if api_key:
                client_options["api_key"] = api_key
            return OpenAIEmbeddings(model=model, **client_options)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers for local embeddings") from exc
        model = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3")
        device = os.getenv("EMBEDDINGS_DEVICE", "cpu").strip() or "cpu"
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_or_create(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        index_file = self.store_path / "index.faiss"
        if index_file.exists():
            self._validate_manifest()
            self._store = self._get_faiss_cls().load_local(
                self._faiss_storage_path(),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self._document_count = len(
                getattr(self._store, "index_to_docstore_id", {})
            )
            logger.info("vector store loaded from %s", self.store_path)

    def _expected_manifest(self) -> dict[str, Any]:
        chunking = {
            key: self.config.get(key)
            for key in ("chunk_size", "chunk_overlap", "separators")
            if key in self.config
        }
        encoded = json.dumps(chunking, ensure_ascii=False, sort_keys=True).encode()
        return {
            "schema_version": 1,
            "embedding_text_schema": self.EMBEDDING_TEXT_SCHEMA,
            "embedding": {
                "provider": os.getenv("EMBEDDINGS_PROVIDER", "openai").lower(),
                "model": os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"),
            },
            "chunking": chunking,
            "chunking_fingerprint": hashlib.sha256(encoded).hexdigest(),
        }

    def _validate_manifest(self) -> None:
        path = self.store_path / "vector_store_manifest.json"
        if not path.exists():
            logger.warning("vector store manifest missing; loading legacy index")
            return
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise VectorStoreCompatibilityError("vector_manifest_incompatible") from None
        expected = self._expected_manifest()
        if (
            manifest.get("schema_version") != expected["schema_version"]
            or manifest.get("embedding") != expected["embedding"]
            or manifest.get("chunking_fingerprint") != expected["chunking_fingerprint"]
            or (
                "embedding_text_schema" in manifest
                and manifest.get("embedding_text_schema") != expected["embedding_text_schema"]
            )
        ):
            raise VectorStoreCompatibilityError("vector_manifest_incompatible")

    def add_documents(
        self, docs: list[Document], *, replace_existing: bool = True
    ) -> None:
        if not docs:
            return
        for document in docs:
            if not self.REQUIRED_METADATA.issubset(document.metadata):
                raise ValueError("vector_document_missing_metadata")
        docs = [self._with_embedding_text(document) for document in docs]
        ids = [str(document.metadata["chunk_id"]) for document in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("vector_document_duplicate_id")
        if self._store is None:
            self._store = self._get_faiss_cls().from_documents(
                docs, self.embeddings, ids=ids
            )
            self._document_count = len(docs)
        else:
            existing_ids = self._existing_ids_for_papers(
                {str(document.metadata["paper_id"]) for document in docs}
            )
            if replace_existing and existing_ids:
                self._store.delete(ids=existing_ids)
            self._store.add_documents(docs, ids=ids)
            deleted_count = len(existing_ids) if replace_existing else 0
            self._document_count = self._document_count - deleted_count + len(docs)
        # FAISS writes several files directly under this directory.
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(self._faiss_storage_path())
        self._write_manifest()

    @staticmethod
    def _with_embedding_text(document: Document) -> Document:
        metadata = dict(document.metadata)
        if metadata.get("embedding_text_schema") == VectorStoreService.EMBEDDING_TEXT_SCHEMA:
            return document
        title = str(metadata.get("title", "")).strip()
        section = str(metadata.get("section") or "").strip()
        page = metadata.get("page_number", 0)
        location = section or ("摘要" if int(page) == 0 else f"第 {page} 页")
        canonical = str(metadata.get("canonical_text") or document.page_content)
        metadata["canonical_text"] = canonical
        metadata["embedding_text_schema"] = VectorStoreService.EMBEDDING_TEXT_SCHEMA
        return Document(
            page_content=f"标题：{title}\n位置：{location}\n正文：{canonical}",
            metadata=metadata,
        )

    def _faiss_storage_path(self) -> str:
        """Avoid Windows FAISS failures when the repository path contains Unicode."""
        # A relative path is safe only when the process is already rooted at the repository.
        # Otherwise FAISS resolves it against an unrelated working directory.
        if Path.cwd().resolve() != self.repository_root:
            return str(self.store_path)
        try:
            return str(self.store_path.relative_to(self.repository_root))
        except ValueError:
            return str(self.store_path)

    def _existing_ids_for_papers(self, paper_ids: set[str]) -> list[str]:
        mapping = getattr(self._store, "index_to_docstore_id", {})
        docstore = getattr(self._store, "docstore", None)
        if not mapping or docstore is None:
            return []
        existing: list[str] = []
        for document_id in mapping.values():
            document = docstore.search(document_id)
            if (
                isinstance(document, Document)
                and str(document.metadata.get("paper_id")) in paper_ids
            ):
                existing.append(document_id)
        return existing

    def remove_paper(self, paper_id: str) -> int:
        """Remove all persisted vectors belonging to one paper."""
        if self._store is None:
            return 0
        existing_ids = self._existing_ids_for_papers({str(paper_id)})
        if not existing_ids:
            return 0
        self._store.delete(ids=existing_ids)
        self._document_count = max(0, self._document_count - len(existing_ids))
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(self._faiss_storage_path())
        self._write_manifest()
        return len(existing_ids)

    def _write_manifest(self) -> None:
        """Persist the inputs needed to explain or reproduce this index."""
        expected = self._expected_manifest()
        manifest = {
            **expected,
            "document_count": self._document_count,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.store_path.mkdir(parents=True, exist_ok=True)
        (self.store_path / "vector_store_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def search(
        self,
        query: str,
        *,
        k: int,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]:
        if self._store is None or k <= 0 or paper_ids == []:
            return []
        options: dict[str, Any] = {"k": k}
        if paper_ids is not None:
            options["filter"] = {"paper_id": {"$in": paper_ids}}
        matches: dict[str, tuple[Document, float]] = {}
        rank_scores: dict[str, float] = {}
        # Dense retrieval receives semantic/English variants, while BM25
        # retains the original keyword-focused wording.
        expanded_queries = plan_query(query).semantic_queries
        for query_index, expanded_query in enumerate(expanded_queries):
            query_matches = self._store.similarity_search_with_score(
                expanded_query, **options
            )
            for rank, (document, distance) in enumerate(query_matches, start=1):
                chunk_id = str(document.metadata.get("chunk_id", ""))
                if not chunk_id:
                    continue
                matches.setdefault(chunk_id, (document, distance))
                # Preserve the original query as the strongest signal while
                # allowing translated/synonym variants to contribute rank.
                weight = 1.5 if query_index == 0 else 1.0
                rank_scores[chunk_id] = rank_scores.get(chunk_id, 0.0) + weight / (60 + rank)
        results = []
        allowed_papers = set(paper_ids) if paper_ids is not None else None
        ordered_matches = sorted(
            matches.items(),
            key=lambda item: (-rank_scores[item[0]], item[0]),
        )
        for chunk_id, (document, distance) in ordered_matches:
            metadata = document.metadata
            if not self.REQUIRED_METADATA.issubset(metadata):
                continue
            if (
                allowed_papers is not None
                and str(metadata["paper_id"]) not in allowed_papers
            ):
                continue
            results.append(
                EvidenceChunk(
                    chunk_id=str(metadata["chunk_id"]),
                    paper_id=str(metadata["paper_id"]),
                    title=str(metadata["title"]),
                    page_number=int(metadata["page_number"]),
                    section=metadata.get("section"),
                    text=str(metadata.get("canonical_text") or document.page_content),
                    score=(
                        1.0 / (1.0 + max(float(distance), 0.0))
                        if len(expanded_queries) == 1
                        else rank_scores[chunk_id]
                    ),
                )
            )
        return sorted(results, key=lambda item: (-item.score, item.chunk_id))[:k]

    def get_retriever(self):
        if self._store is None:
            return EmptyRetriever()
        top_k = int(self.config.get("top_k", self.config.get("answer_k", 8)))
        return self._store.as_retriever(search_kwargs={"k": top_k})

    def rebuild_from_database(self) -> int:
        target = self._verified_rebuild_target()
        if self.database is None:
            raise ValueError("vector_rebuild_database_required")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.rebuild-build-{uuid4().hex}")
        previous_path = self.store_path
        previous_store = self._store
        previous_count = self._document_count
        self.store_path = staging
        self._store = None
        self._document_count = 0
        chunks = self.database.list_chunks()
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "canonical_text": chunk.text,
                },
            )
            for chunk in chunks
        ]
        batch_size = int(self.config.get("embedding_batch_size", 32))
        if batch_size <= 0:
            raise ValueError("embedding_batch_size_must_be_positive")
        try:
            staging.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(documents), batch_size):
                self.add_documents(
                    documents[start : start + batch_size], replace_existing=False
                )
            if not documents:
                self._write_manifest()
            self._publish_staged_snapshot(staging, target)
        except Exception:
            self.store_path = previous_path
            self._store = previous_store
            self._document_count = previous_count
            self._remove_staging_snapshot(staging)
            raise
        self.store_path = target
        return len(documents)

    def _publish_staged_snapshot(self, staging: Path, target: Path) -> None:
        """Switch a complete build into place without exposing partial FAISS files."""
        if self._verified_rebuild_target() != target:
            raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        if not staging.is_dir():
            raise OSError("vector_snapshot_missing")
        quarantine = target.with_name(f".{target.name}.rebuild-delete-{uuid4().hex}")
        had_active = target.exists()
        if had_active:
            self._reject_reparse_points(target)
            os.replace(target, quarantine)
        try:
            os.replace(staging, target)
        except Exception:
            if had_active and quarantine.exists() and not target.exists():
                os.replace(quarantine, target)
            raise
        if not had_active:
            return
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            return
        boundary = self._resolve_rebuild_path(self.repository_root / "data" / "vector_store")
        resolved = self._verified_quarantine(
            quarantine, original_target=target, boundary=boundary
        )
        shutil.rmtree(resolved)

    @staticmethod
    def _remove_staging_snapshot(staging: Path) -> None:
        if not staging.exists():
            return
        if staging.is_dir():
            shutil.rmtree(staging)
        else:
            staging.unlink(missing_ok=True)

    def _verified_rebuild_target(self) -> Path:
        configured = self._configured_store_path
        if not str(configured).strip() or ".." in configured.parts:
            raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        lexical = (
            configured
            if configured.is_absolute()
            else self.repository_root / configured
        )
        target = self._resolve_rebuild_path(lexical)
        boundary = self._resolve_rebuild_path(
            self.repository_root / "data" / "vector_store"
        )
        try:
            target.relative_to(boundary)
        except ValueError:
            raise UnsafeVectorStorePathError("unsafe_vector_store_path") from None

        self._reject_reparse_points(lexical)
        if target in {
            Path(target.anchor),
            Path.home().resolve(),
            self.repository_root,
            (self.repository_root / "data").resolve(strict=False),
        }:
            raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        return target

    def _delete_verified_target(self, expected_target: Path) -> None:
        """Rename the verified directory aside before removing its contents."""
        target = self._verified_rebuild_target()
        if target != expected_target:
            raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        if not target.exists():
            return

        quarantine = target.with_name(
            f".{target.name}.rebuild-delete-{uuid4().hex}"
        )
        os.replace(target, quarantine)
        try:
            boundary = self._resolve_rebuild_path(
                self.repository_root / "data" / "vector_store"
            )
            resolved_quarantine = self._verified_quarantine(
                quarantine, original_target=target, boundary=boundary
            )
            if (
                self._verified_quarantine(
                    quarantine, original_target=target, boundary=boundary
                )
                != resolved_quarantine
            ):
                raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        except (OSError, ValueError, UnsafeVectorStorePathError):
            if quarantine.exists() and not target.exists():
                os.replace(quarantine, target)
            raise UnsafeVectorStorePathError("unsafe_vector_store_path") from None
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            return
        try:
            shutil.rmtree(resolved_quarantine)
        except OSError:
            if quarantine.exists() and not target.exists():
                os.replace(quarantine, target)
            raise UnsafeVectorStorePathError("unsafe_vector_store_path") from None

    def _verified_quarantine(
        self,
        quarantine: Path,
        *,
        original_target: Path,
        boundary: Path,
    ) -> Path:
        resolved = self._resolve_rebuild_path(quarantine)
        expected_prefix = f".{original_target.name}.rebuild-delete-"
        if not quarantine.name.startswith(expected_prefix):
            raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        if original_target == boundary:
            if resolved.parent != boundary.parent:
                raise UnsafeVectorStorePathError("unsafe_vector_store_path")
        else:
            try:
                resolved.relative_to(boundary)
            except ValueError:
                raise UnsafeVectorStorePathError("unsafe_vector_store_path") from None
        self._reject_reparse_points(quarantine)
        return resolved

    def _reject_reparse_points(self, path: Path) -> None:
        current = path
        while True:
            is_junction = getattr(current, "is_junction", None)
            try:
                attributes = getattr(current.lstat(), "st_file_attributes", 0)
            except FileNotFoundError:
                attributes = 0
            if (
                current.is_symlink()
                or (is_junction is not None and is_junction())
                or bool(
                    attributes
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
            ):
                raise UnsafeVectorStorePathError("unsafe_vector_store_path")
            if current == self.repository_root:
                return
            if current.parent == current:
                raise UnsafeVectorStorePathError("unsafe_vector_store_path")
            current = current.parent

    @staticmethod
    def _resolve_rebuild_path(path: Path) -> Path:
        return path.resolve(strict=False)

    def _get_faiss_cls(self):
        if self.faiss_cls is None:
            from langchain_community.vectorstores import FAISS

            self.faiss_cls = FAISS
        return self.faiss_cls
