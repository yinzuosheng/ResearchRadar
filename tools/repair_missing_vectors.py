from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from langchain_core.documents import Document

from rag.vector_store import VectorStoreService
from storage.database import ResearchDatabase
from storage.paths import default_database_path


db = ResearchDatabase(default_database_path())
store = VectorStoreService(database=db)
chunks = db.list_chunks()
vector_ids: set[str] = set()
faiss_store = getattr(store, "_store", None)
mapping = getattr(faiss_store, "index_to_docstore_id", {})
docstore = getattr(faiss_store, "docstore", None)
if isinstance(mapping, dict) and docstore is not None:
    for doc_id in mapping.values():
        document = docstore.search(doc_id)
        if document is not None and isinstance(getattr(document, "metadata", None), dict):
            vector_ids.add(str(document.metadata.get("chunk_id")))

missing = [chunk for chunk in chunks if chunk.chunk_id not in vector_ids]
by_paper: dict[str, list] = {}
for chunk in missing:
    by_paper.setdefault(chunk.paper_id, []).append(chunk)
print("missing_chunks", len(missing), "papers", sorted(by_paper))
# VectorStoreService replaces all vectors for a paper by default. These IDs are
# already known to be absent, so append only the missing documents explicitly.
for paper_id, items in by_paper.items():
    for start in range(0, len(items), 32):
        documents = [
            Document(
                page_content=item.text,
                metadata={
                    "chunk_id": item.chunk_id,
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "page_number": item.page_number,
                    "section": item.section,
                    "evidence_label": "摘要证据" if item.page_number == 0 else "全文证据",
                },
            )
            for item in items[start : start + 32]
        ]
        store.add_documents(documents, replace_existing=False)
print("reindexed", len(missing))
