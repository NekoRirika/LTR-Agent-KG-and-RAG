"""LangChain Tool examples for agent integration."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import Tool


def _format_documents(docs: list[Document], *, max_chars: int = 800) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.page_content.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        items.append({"content": text, "metadata": dict(doc.metadata)})
    return items


def build_knowledge_base_search_tool(
    retriever: BaseRetriever,
    *,
    k: int = 4,
    name: str = "knowledge_base_search",
    description: str = "Search the knowledge base and return relevant passages with metadata.",
) -> Tool:
    """Wrap a retriever as a LangChain Tool for agents.

    Args:
        retriever: LangChain retriever (e.g., RAGRetriever or HybridRetriever).
        k: Top-K docs to return.
        name: Tool name.
        description: Tool description.

    Returns:
        Tool: LangChain Tool instance.
    """

    def _run(query: str, **_kwargs: Any) -> str:
        docs: list[Document] = retriever.invoke(query, k=k)  # type: ignore[arg-type]
        payload = {"query": query, "k": k, "documents": _format_documents(docs)}
        return json.dumps(payload, ensure_ascii=False)

    return Tool(name=name, description=description, func=_run)

