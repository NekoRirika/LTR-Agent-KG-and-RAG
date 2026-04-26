"""RAG configuration utilities shared by storage and retriever modules."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from .exceptions import RAGConfigError


_ENV_LOADED = False


def ensure_env_loaded(env_file: str | Path | None = None) -> None:
    """Load .env once (idempotent).

    Args:
        env_file: Optional path to .env. If not provided, defaults to project root .env.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = Path(env_file) if env_file is not None else (project_root / ".env")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    _ENV_LOADED = True


@dataclass(frozen=True)
class RAGSettings:
    """Runtime settings loaded from environment variables."""

    embedding_provider: str
    embedding_model_name: str
    openai_api_key: str
    openai_base_url: str
    openai_embedding_model: str
    vector_store_provider: str
    vector_store_path: str

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "RAGSettings":
        """Build settings from .env and process environment variables.

        Args:
            env_file: Optional path to .env.

        Returns:
            RAGSettings: Parsed settings with defaults applied.
        """
        ensure_env_loaded(env_file)

        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip()
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()

        openai_api_key = (os.getenv("OPENAI_API_KEY", "") or os.getenv("LLM_API_KEY", "")).strip()
        openai_base_url = (os.getenv("OPENAI_BASE_URL", "") or os.getenv("LLM_BASE_URL", "")).strip()
        if not openai_base_url:
            openai_base_url = "https://api.openai.com/v1"

        openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

        vector_store_provider = os.getenv("VECTOR_STORE_PROVIDER", "chroma").strip()
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store").strip()

        return cls(
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_embedding_model=openai_embedding_model,
            vector_store_provider=vector_store_provider,
            vector_store_path=vector_store_path,
        )


def get_vector_store_path(env_file: str | Path | None = None) -> Path:
    """Get vector store persistence path.

    Args:
        env_file: Optional path to .env.

    Returns:
        Path: Resolved path for vector store persistence.
    """
    settings = RAGSettings.from_env(env_file)
    project_root = Path(__file__).resolve().parents[1]
    path = Path(settings.vector_store_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def get_embedding_model(env_file: str | Path | None = None) -> Embeddings:
    """Create an embeddings model based on environment variables.

    Supported providers:
    - openai: OpenAI-compatible embeddings API
    - huggingface: Local sentence-transformer embeddings

    Args:
        env_file: Optional path to .env.

    Returns:
        Embeddings: LangChain embeddings instance.

    Raises:
        RAGConfigError: If provider is unknown or required settings are missing.
    """
    settings = RAGSettings.from_env(env_file)
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise RAGConfigError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as exc:  # pragma: no cover
            raise RAGConfigError("langchain-openai is required for OpenAI embeddings") from exc

        kwargs: dict[str, object] = {
            "api_key": settings.openai_api_key,
            "model": settings.openai_embedding_model,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        return OpenAIEmbeddings(**kwargs)

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover
                raise RAGConfigError(
                    "langchain-huggingface (or langchain-community) is required for HuggingFace embeddings"
                ) from exc

        model_name = settings.embedding_model_name
        if not model_name:
            raise RAGConfigError("EMBEDDING_MODEL_NAME is required when EMBEDDING_PROVIDER=huggingface")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    raise RAGConfigError(f"Unknown EMBEDDING_PROVIDER: {provider}")

