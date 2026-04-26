"""RAG chain orchestration built with LangChain Core primitives.

This project pins a lightweight `langchain` meta-package which may not include
high-level chain helpers (e.g., ConversationalRetrievalChain) in some versions.
To keep the RAG layer runnable, this module implements a small retrieval+prompt
pipeline using LCEL (LangChain Expression Language).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from retriever.exceptions import LLMError


logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "你是一个严谨的中文问答助手。请只基于提供的上下文回答问题。"
    "如果上下文不足以回答，请明确说明“无法从已索引文档中确定”，并给出需要补充的信息。"
)


class RAGChain:
    """High-level RAG chain wrapper for front-end integration."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        memory: Optional[Any] = None,
    ) -> None:
        """Initialize RAGChain.

        Args:
            retriever: LangChain retriever instance.
            llm: LangChain chat model instance.
            memory: Optional conversation memory. If not provided, you can pass chat_history to run().
        """
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _DEFAULT_SYSTEM),
                ("human", "【对话历史】\n{chat_history}\n\n【上下文】\n{context}\n\n【问题】\n{question}"),
            ]
        )

    def run(self, query: str, chat_history: Optional[list[Any]] = None) -> dict[str, Any]:
        """Run RAG query and return answer with sources.

        Args:
            query: User question.
            chat_history: Optional external chat history. If provided, it is forwarded to the chain.

        Returns:
            dict: {"answer": str, "source_documents": List[Document], ...}

        Raises:
            LLMError: If chain execution fails.
        """
        try:
            history_text = self._format_chat_history(chat_history)

            docs: list[Document]
            try:
                docs = self._retriever.invoke(query)  # type: ignore[arg-type]
            except TypeError:
                docs = self._retriever.invoke({"query": query})  # pragma: no cover

            context = self._format_context(docs)

            chain = self._prompt | self._llm
            message = chain.invoke(
                {"question": query, "chat_history": history_text, "context": context}
            )
            answer = getattr(message, "content", str(message))

            if self._memory is not None:
                self._save_to_memory(query=query, answer=str(answer))

            return {"answer": str(answer), "source_documents": docs, "raw": {"message": message}}
        except Exception as exc:
            raise LLMError(f"RAGChain failed: {exc}") from exc

    @staticmethod
    def _format_context(docs: list[Document], *, max_chars_per_doc: int = 1500) -> str:
        """Format retrieved documents into a context string."""
        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            text = doc.page_content.strip()
            if len(text) > max_chars_per_doc:
                text = text[: max_chars_per_doc - 3] + "..."
            source = str(doc.metadata.get("source") or doc.metadata.get("file_name") or "")
            parts.append(f"[{idx}] source={source}\n{text}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _format_chat_history(chat_history: Optional[list[Any]]) -> str:
        """Format chat history into text for prompt injection."""
        if not chat_history:
            return ""
        return "\n".join(str(item) for item in chat_history)

    def _save_to_memory(self, *, query: str, answer: str) -> None:
        """Save query/answer into memory if compatible with BaseMemory API."""
        try:
            load_fn = getattr(self._memory, "load_memory_variables", None)
            save_fn = getattr(self._memory, "save_context", None)
            if callable(save_fn):
                save_fn({"question": query}, {"answer": answer})
            elif callable(load_fn):
                _ = load_fn({})
        except Exception:
            return


def build_simple_qa_chain(retriever: BaseRetriever, llm: Any) -> Any:
    """Build a simple QA chain (same as RAGChain without history).

    Args:
        retriever: LangChain retriever.
        llm: LangChain LLM.

    Returns:
        Any: RAGChain instance.
    """
    return RAGChain(retriever=retriever, llm=llm, memory=None)
