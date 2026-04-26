"""Vector-based retriever for RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from storage.vector_store import VectorStoreManager

from .exceptions import RetrieverError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGRetrieverConfig:
    """Configuration for RAGRetriever."""

    search_type: str = "similarity"
    k: int = 4
    score_threshold: Optional[float] = None
    search_kwargs: Optional[dict[str, Any]] = None


class RAGRetriever(BaseRetriever):
    """Standard vector retriever wrapping a LangChain vector store.

    Supports:
    - similarity search
    - MMR (max marginal relevance)
    - optional score_threshold filtering (when vector store exposes relevance scores)
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        *,
        config: Optional[RAGRetrieverConfig] = None,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        search_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: VectorStoreManager instance.
            config: Optional retriever configuration.
            search_type: Optional search type override ("similarity" or "mmr"). Ignored if config is provided.
            k: Optional top-k override. Ignored if config is provided.
            top_k: Optional top-k alias override. Ignored if config is provided.
            score_threshold: Optional relevance score threshold. Ignored if config is provided.
            search_kwargs: Optional extra kwargs for retrieval. Ignored if config is provided.
        """
        super().__init__()
        self._manager = vector_store
        if config is not None:
            self._config = config
            return

        effective_k: Optional[int]
        if top_k is not None:
            effective_k = int(top_k)
        elif k is not None:
            effective_k = int(k)
        else:
            effective_k = None

        self._config = RAGRetrieverConfig(
            search_type=search_type or "similarity",
            k=(effective_k if effective_k is not None else 4),
            score_threshold=score_threshold,
            search_kwargs=search_kwargs,
        )

    @property
    def manager(self) -> VectorStoreManager:
        """Underlying vector store manager."""
        return self._manager

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Compatibility wrapper for callers not using BaseRetriever.invoke()."""
        return self.invoke(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None, **kwargs: Any
    ) -> list[Document]:
        cfg = self._config
        search_kwargs: dict[str, Any] = {}
        if cfg.search_kwargs:
            search_kwargs.update(cfg.search_kwargs)
        search_kwargs.update(kwargs)

        k = int(search_kwargs.pop("k", cfg.k))
        filter_dict = search_kwargs.pop("filter", None)

        try:
            if cfg.search_type == "mmr":
                return self._mmr_search(query, k=k, filter=filter_dict, **search_kwargs)
            return self._similarity_search(query, k=k, filter=filter_dict, **search_kwargs)
        except Exception as exc:
            raise RetrieverError(f"Retrieval failed: {exc}") from exc

    def _mmr_search(self, query: str, *, k: int, filter: Optional[dict[str, Any]], **kwargs: Any) -> list[Document]:
        vs = self._manager.vectorstore
        fetch_k = int(kwargs.pop("fetch_k", max(20, k * 5)))
        lambda_mult = float(kwargs.pop("lambda_mult", 0.5))

        mmr_fn = getattr(vs, "max_marginal_relevance_search", None)
        if not callable(mmr_fn):
            logger.warning("Vector store does not support MMR; falling back to similarity search.")
            return self._similarity_search(query, k=k, filter=filter, **kwargs)

        try:
            return mmr_fn(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter)
        except TypeError:
            return mmr_fn(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    def _similarity_search(
        self, query: str, *, k: int, filter: Optional[dict[str, Any]], **kwargs: Any
    ) -> list[Document]:
        score_threshold = self._config.score_threshold
        vs = self._manager.vectorstore

        if score_threshold is None:
            return self._manager.similarity_search(query, k=k, filter=filter)

        relevance_fn = getattr(vs, "similarity_search_with_relevance_scores", None)
        if callable(relevance_fn):
            try:
                pairs = relevance_fn(query, k=k, filter=filter)
            except TypeError:
                pairs = relevance_fn(query, k=k)
            docs = [doc for doc, score in pairs if float(score) >= float(score_threshold)]
            return docs

        logger.warning(
            "score_threshold is set but vector store does not expose relevance scores; skipping threshold filtering."
        )
        return self._manager.similarity_search(query, k=k, filter=filter)
