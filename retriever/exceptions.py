"""Retriever-level exceptions for the RAG system."""

from __future__ import annotations

from storage.exceptions import RAGException


class RetrieverError(RAGException):
    """Raised when retrieval operations fail."""


class LLMError(RAGException):
    """Raised when LLM operations fail."""

