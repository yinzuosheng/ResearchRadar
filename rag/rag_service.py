from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt, load_report_prompt


class RagSummarizeService:
    def __init__(self) -> None:
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.summary_prompt = PromptTemplate.from_template(load_rag_prompt())
        self.report_prompt = PromptTemplate.from_template(load_report_prompt())
        self.summary_chain = self.summary_prompt | chat_model | StrOutputParser()
        self.report_chain = self.report_prompt | chat_model | StrOutputParser()

    def _build_context(self, docs: list[Document]) -> str:
        if not docs:
            return ""
        parts = []
        for idx, doc in enumerate(docs, start=1):
            parts.append(
                f"[Source {idx}]: {doc.page_content} | metadata: {doc.metadata}"
            )
        return "\n".join(parts)

    def _retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        docs = self._retrieve(query)
        if not docs:
            return "Knowledge base is empty. Run collection first."
        context = self._build_context(docs)
        return self.summary_chain.invoke({"input": query, "context": context})

    def rag_report(self, query: str) -> str:
        docs = self._retrieve(query)
        if not docs:
            return "Knowledge base is empty. Run collection first."
        context = self._build_context(docs)
        return self.report_chain.invoke({"input": query, "context": context})
