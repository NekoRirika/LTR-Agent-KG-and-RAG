"""Hybrid retriever: vector retrieval + optional knowledge graph augmentation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .exceptions import RetrieverError
from .rag_retriever import RAGRetriever


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""

    vector_k: int = 8
    kg_entity_k: int = 8
    final_k: int = 8


def _tokenize(text: str) -> set[str]:
    tokens = re.split(r"[\s，。；、,.;:：!?！？()\[\]{}<>\"'“”‘’/\\]+", text)
    return {t.strip().lower() for t in tokens if t and len(t.strip()) > 1}


def _doc_key(doc: Document) -> str:
    src = str(doc.metadata.get("source") or doc.metadata.get("file_path") or doc.metadata.get("file_name") or "")
    chunk_id = str(doc.metadata.get("chunk_id") or doc.metadata.get("chunk_index") or "")
    return f"{src}::{chunk_id}::{hash(doc.page_content)}"


class HybridRetriever(BaseRetriever):
    """Combine vector retrieval with KG-derived entity expansion.

    KG integration is optional and reserved behind a callable interface:
    - kg_query_entities(query: str) -> list[str]
    """

    def __init__(
        self,
        vector_retriever: RAGRetriever,
        *,
        kg_query_entities: Optional[Callable[[str], list[str]]] = None,
        config: Optional[HybridRetrieverConfig] = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_retriever: Vector-based retriever.
            kg_query_entities: Optional KG function. Expected signature:
                kg.query_entities(query: str) -> List[str]
            config: Hybrid retriever configuration.
        """
        super().__init__()
        self._vector_retriever = vector_retriever
        self._kg_query_entities = kg_query_entities
        self._config = config or HybridRetrieverConfig()

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Compatibility wrapper for callers not using BaseRetriever.invoke()."""
        return self.invoke(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None, **kwargs: Any
    ) -> list[Document]:
        cfg = self._config
        try:
            vector_docs = self._vector_retriever.invoke(query, k=cfg.vector_k)
            kg_docs = self._retrieve_from_kg_entities(query, k=cfg.kg_entity_k)
            merged = self._deduplicate(vector_docs + kg_docs)
            reranked = self._rerank(query, merged)
            return reranked[: cfg.final_k]
        except Exception as exc:
            raise RetrieverError(f"Hybrid retrieval failed: {exc}") from exc

    def _retrieve_from_kg_entities(self, query: str, *, k: int) -> list[Document]:
        if self._kg_query_entities is None:
            return []

        try:
            entity_ids = self._kg_query_entities(query)
        except Exception as exc:
            logger.warning("kg_query_entities failed, skipping KG augmentation: %s", exc)
            return []

        if not entity_ids:
            return []

        docs: list[Document] = []
        per_entity_k = max(1, k // max(1, len(entity_ids)))
        for entity_id in entity_ids[:k]:
            try:
                docs.extend(self._vector_retriever.manager.similarity_search(str(entity_id), k=per_entity_k))
            except Exception:
                continue
        return docs

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        result: list[Document] = []
        for doc in docs:
            key = _doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            result.append(doc)
        return result

    @staticmethod
    def _rerank(query: str, docs: list[Document]) -> list[Document]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return docs

        def score(doc: Document) -> float:
            d_tokens = _tokenize(doc.page_content)
            overlap = len(q_tokens & d_tokens)
            denom = max(1, len(q_tokens))
            return overlap / denom

        return sorted(docs, key=score, reverse=True)


def default_kg_query_entities(query: str) -> list[str]:
    """Default KG hook.

    This is a reserved integration point. If your kg/ package exposes:
        kg.query_entities(query: str) -> List[str]
    you can pass it into HybridRetriever.
    """
    raise NotImplementedError("Provide kg_query_entities(query)->List[str] from kg/ module.")

