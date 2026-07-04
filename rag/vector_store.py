import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from utils.config import load_rag_config
from utils.logger import logger


class EmptyRetriever:
    def invoke(self, query: str):
        return []


class VectorStoreService:
    def __init__(self) -> None:
        self.config = load_rag_config()
        self.store_path = Path(self.config.get("vector_store_path", "data/vector_store"))
        self.store_path = Path(__file__).resolve().parents[1] / self.store_path
        self.embeddings = self._build_embeddings()
        self._store = None
        self._load_or_create()

    def _build_embeddings(self):
        provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
        if provider == "openai":
            model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model)

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers for local embeddings") from exc

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _load_or_create(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        index_file = self.store_path / "index.faiss"
        if index_file.exists():
            self._store = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("vector store loaded from %s", self.store_path)

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        if self._store is None:
            self._store = FAISS.from_documents(docs, self.embeddings)
        else:
            self._store.add_documents(docs)
        self._store.save_local(str(self.store_path))

    def get_retriever(self):
        if self._store is None:
            return EmptyRetriever()
        top_k = int(self.config.get("top_k", 6))
        return self._store.as_retriever(search_kwargs={"k": top_k})
