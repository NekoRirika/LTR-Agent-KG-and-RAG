"""Vector store manager for RAG retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .config import RAGSettings, get_embedding_model, get_vector_store_path
from .exceptions import RAGConfigError, VectorStoreError


logger = logging.getLogger(__name__)


def _import_chroma() -> type:
    try:
        from langchain_chroma import Chroma

        return Chroma
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma

            return Chroma
        except Exception as exc:  # pragma: no cover
            raise RAGConfigError(
                "Chroma dependencies are missing. Install: chromadb and langchain-chroma (or langchain-community)."
            ) from exc


def _import_faiss() -> type:
    try:
        from langchain_community.vectorstores import FAISS

        return FAISS
    except Exception as exc:  # pragma: no cover
        raise RAGConfigError(
            "FAISS dependencies are missing. Install: faiss-cpu and langchain-community."
        ) from exc


class VectorStoreManager:
    """Manage vector store creation, persistence, and search.

    This manager defaults to Chroma with persistence. It also provides a simple
    factory-style extension point for other providers (e.g., PGVector, Milvus).
    """

    def __init__(
        self,
        *,
        embedding: Optional[Embeddings] = None,
        provider: Optional[str] = None,
        persist_directory: str | Path | None = None,
        collection_name: str = "rag_documents",
        env_file: str | Path | None = None,
    ) -> None:
        """Initialize a VectorStoreManager.

        Args:
            embedding: Embeddings instance. If not provided, it is created from .env.
            provider: Vector store provider name (default from .env VECTOR_STORE_PROVIDER).
            persist_directory: Directory for persistence (default from .env VECTOR_STORE_PATH).
            collection_name: Collection name for persistent stores (e.g., Chroma).
            env_file: Optional .env path.
        """
        settings = RAGSettings.from_env(env_file)
        resolved_provider = (provider or settings.vector_store_provider or "chroma").strip().lower()

        if persist_directory is None:
            persist_path = get_vector_store_path(env_file)
        else:
            persist_path = Path(persist_directory)
            if not persist_path.is_absolute():
                project_root = Path(__file__).resolve().parents[1]
                persist_path = (project_root / persist_path).resolve()

        persist_path.mkdir(parents=True, exist_ok=True)

        self._provider = resolved_provider
        self._persist_path = persist_path
        self._collection_name = collection_name
        self._embedding = embedding or get_embedding_model(env_file)
        self._vectorstore: VectorStore = self._create_vectorstore()

    @property
    def persist_path(self) -> Path:
        """Persistence path for the underlying store."""
        return self._persist_path

    @property
    def provider(self) -> str:
        """Vector store provider name."""
        return self._provider

    @property
    def vectorstore(self) -> VectorStore:
        """Underlying LangChain VectorStore instance."""
        return self._vectorstore

    def _create_vectorstore(self) -> VectorStore:
        if self._provider in {"chroma", "chromadb"}:
            Chroma = _import_chroma()
            try:
                return Chroma(
                    collection_name=self._collection_name,
                    persist_directory=str(self._persist_path),
                    embedding_function=self._embedding,
                )
            except TypeError:
                return Chroma(
                    collection_name=self._collection_name,
                    persist_directory=str(self._persist_path),
                    embedding=self._embedding,  # type: ignore[arg-type]
                )
            except Exception as exc:
                logger.warning(
                    "Chroma initialization failed (provider=%s). Falling back to FAISS. error=%s",
                    self._provider,
                    exc,
                )
                self._provider = "faiss"
                return self._create_vectorstore()

        if self._provider == "faiss":
            FAISS = _import_faiss()
            try:
                if (self._persist_path / "index.faiss").exists():
                    load_fn = getattr(FAISS, "load_local", None)
                    if callable(load_fn):
                        try:
                            return load_fn(
                                str(self._persist_path),
                                self._embedding,
                                allow_dangerous_deserialization=True,
                            )
                        except TypeError:
                            return load_fn(str(self._persist_path), self._embedding)
                return self._create_empty_faiss(FAISS)
            except Exception as exc:  # pragma: no cover
                raise VectorStoreError(f"Failed to initialize FAISS: {exc}") from exc

        if self._provider in {"pgvector", "milvus"}:
            raise RAGConfigError(
                f"VECTOR_STORE_PROVIDER={self._provider} is reserved for extension but not implemented yet."
            )

        raise RAGConfigError(f"Unknown VECTOR_STORE_PROVIDER: {self._provider}")

    def add_documents(self, docs: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            docs: Documents to add.

        Returns:
            List[str]: Document IDs returned by the vector store (if supported).

        Raises:
            VectorStoreError: If the operation fails.
        """
        if not docs:
            return []
        try:
            ids = self._vectorstore.add_documents(docs)
            return list(ids) if ids is not None else []
        except Exception as exc:
            raise VectorStoreError(f"add_documents failed: {exc}") from exc

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict[str, Any]] = None
    ) -> list[Document]:
        """Run similarity search.

        Args:
            query: User query.
            k: Top-K results.
            filter: Optional metadata filter (supported by some stores like Chroma).

        Returns:
            List[Document]: Retrieved documents.

        Raises:
            VectorStoreError: If search fails.
        """
        try:
            if filter and self._provider not in {"chroma", "chromadb"}:
                logger.warning("Metadata filter is not supported by provider=%s; ignoring filter.", self._provider)
                filter = None
            return self._vectorstore.similarity_search(query, k=k, filter=filter)  # type: ignore[arg-type]
        except TypeError:
            return self._vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            raise VectorStoreError(f"similarity_search failed: {exc}") from exc

    def as_retriever(self, search_kwargs: Optional[dict[str, Any]] = None):
        """Expose a LangChain retriever.

        Args:
            search_kwargs: Keyword args forwarded to the retriever implementation.

        Returns:
            BaseRetriever: A LangChain retriever instance.
        """
        kwargs = search_kwargs or {}
        return self._vectorstore.as_retriever(search_kwargs=kwargs)

    def persist(self) -> None:
        """Persist the vector store to disk (when supported)."""
        if self._provider == "faiss":
            save_fn = getattr(self._vectorstore, "save_local", None)
            if callable(save_fn):
                try:
                    save_fn(str(self._persist_path))
                    return
                except Exception as exc:  # pragma: no cover
                    raise VectorStoreError(f"persist failed: {exc}") from exc

        persist_fn = getattr(self._vectorstore, "persist", None)
        if callable(persist_fn):
            try:
                persist_fn()
            except Exception as exc:  # pragma: no cover
                raise VectorStoreError(f"persist failed: {exc}") from exc

    def load(self, persist_directory: str | Path) -> None:
        """Load the vector store from an existing persistence directory.

        Args:
            persist_directory: Existing persistence directory.

        Raises:
            VectorStoreError: If load fails.
        """
        persist_path = Path(persist_directory)
        if not persist_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            persist_path = (project_root / persist_path).resolve()

        if not persist_path.exists():
            raise VectorStoreError(f"persist_directory does not exist: {persist_path}")

        self._persist_path = persist_path

        if self._provider == "faiss":
            FAISS = _import_faiss()
            load_fn = getattr(FAISS, "load_local", None)
            if not callable(load_fn):
                raise VectorStoreError("FAISS.load_local is not available in the installed version.")
            try:
                self._vectorstore = load_fn(
                    str(self._persist_path),
                    self._embedding,
                    allow_dangerous_deserialization=True,
                )
                return
            except TypeError:
                self._vectorstore = load_fn(str(self._persist_path), self._embedding)
                return
            except Exception as exc:  # pragma: no cover
                raise VectorStoreError(f"FAISS load failed: {exc}") from exc

        self._vectorstore = self._create_vectorstore()

    def _create_empty_faiss(self, FAISS: type) -> VectorStore:
        """Create an empty FAISS vector store.

        Args:
            FAISS: FAISS VectorStore class from langchain_community.

        Returns:
            VectorStore: An empty FAISS vector store.
        """
        try:
            import faiss  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError("faiss-cpu is required for VECTOR_STORE_PROVIDER=faiss") from exc

        try:
            from langchain_community.docstore.in_memory import InMemoryDocstore
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError("langchain-community is required for FAISS docstore") from exc

        dim = len(self._embedding.embed_query("dimension probe"))
        index = faiss.IndexFlatL2(dim)
        docstore = InMemoryDocstore({})
        index_to_docstore_id: dict[int, str] = {}

        try:
            return FAISS(
                embedding_function=self._embedding,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
        except TypeError:
            return FAISS(
                embeddings=self._embedding,  # type: ignore[arg-type]
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )

    def close(self) -> None:
        """Best-effort close for underlying resources.

        Chroma may keep file handles open on Windows. This method attempts to
        stop underlying systems so temporary directories can be cleaned up.
        """
        if self._provider not in {"chroma", "chromadb"}:
            return

        try:
            client = getattr(self._vectorstore, "_client", None)
            system = getattr(client, "_system", None) if client is not None else None
            stop_fn = getattr(system, "stop", None) if system is not None else None
            if callable(stop_fn):
                stop_fn()
        except Exception:
            return

vector_store_manager = VectorStoreManager(collection_name = "rag_documents")