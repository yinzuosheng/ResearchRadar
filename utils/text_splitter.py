from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import load_rag_config


def split_text(text: str) -> list[str]:
    cfg = load_rag_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(cfg.get("chunk_size", 800)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)
