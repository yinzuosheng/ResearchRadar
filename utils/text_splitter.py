from langchain_text_splitters import RecursiveCharacterTextSplitter

from domain.models import EvidenceChunk, PageText
from utils.config import load_rag_config


def _splitter() -> RecursiveCharacterTextSplitter:
    cfg = load_rag_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=int(cfg.get("chunk_size", 800)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        separators=["\n\n", "\n", ". ", " "],
    )


def split_text(text: str) -> list[str]:
    return _splitter().split_text(text)


def split_pages(
    paper_id: str, title: str, pages: list[PageText]
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for page in pages:
        for index, text in enumerate(_splitter().split_text(page.text)):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{paper_id}:p{page.page_number}:c{index}",
                    paper_id=paper_id,
                    title=title,
                    page_number=page.page_number,
                    text=text,
                )
            )
    return chunks
