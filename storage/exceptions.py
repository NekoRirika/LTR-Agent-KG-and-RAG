"""Storage-level exceptions for the RAG system."""

from __future__ import annotations


class RAGException(Exception):
    """Base exception for RAG-related failures."""


class RAGConfigError(RAGException):
    """Raised when required configuration is missing or invalid."""


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""

