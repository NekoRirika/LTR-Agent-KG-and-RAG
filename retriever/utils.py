"""Factory utilities for embeddings and LLMs used by RAG."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from storage.config import ensure_env_loaded, get_embedding_model as _get_embedding_model
from storage.exceptions import RAGConfigError

from .exceptions import LLMError


logger = logging.getLogger(__name__)


def get_embedding_model(env_file: str | Path | None = None):
    """Create an embeddings model based on .env.

    Args:
        env_file: Optional path to .env.

    Returns:
        Embeddings: LangChain embeddings instance.
    """
    return _get_embedding_model(env_file)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RAGConfigError(f"agent_config.yml not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return raw


def get_llm(
    *,
    config_path: str | Path = "agent_config.yml",
    env_file: str | Path | None = None,
) -> Any:
    """Create an LLM instance based on agent_config.yml and .env.

    Provider resolution order:
    1) agent_config.yml llm.provider
    2) .env LLM_PROVIDER
    3) default "openai"

    Supported providers:
    - openai: ChatOpenAI (OpenAI-compatible API)
    - azure: AzureChatOpenAI
    - ollama: ChatOllama (local)
    - tongyi: ChatTongyi (optional; for compatibility with existing agents)

    Args:
        config_path: Path to agent_config.yml (relative to project root by default).
        env_file: Optional path to .env.

    Returns:
        Any: LangChain chat model instance.

    Raises:
        LLMError: If configuration is missing or the provider is unsupported.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    ensure_env_loaded(env_file)
    raw = _load_yaml(cfg_path)
    llm_raw: dict[str, Any] = raw.get("llm", {}) if isinstance(raw.get("llm", {}), dict) else {}

    provider = str(llm_raw.get("provider") or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    model = str(llm_raw.get("model") or os.getenv("LLM_MODEL", "gpt-4o-mini")).strip()
    temperature = float(llm_raw.get("temperature", os.getenv("LLM_TEMPERATURE", "0")) or 0)

    try:
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except Exception as exc:  # pragma: no cover
                raise LLMError("langchain-openai is required for provider=openai") from exc

            api_key = (os.getenv("OPENAI_API_KEY", "") or os.getenv("LLM_API_KEY", "")).strip()
            if not api_key:
                raise LLMError("OPENAI_API_KEY (or LLM_API_KEY) is required when LLM_PROVIDER=openai")

            base_url = (os.getenv("OPENAI_BASE_URL", "") or os.getenv("LLM_BASE_URL", "")).strip()
            kwargs: dict[str, Any] = {"api_key": api_key, "model": model, "temperature": temperature}
            if base_url:
                kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)

        if provider == "azure":
            try:
                from langchain_openai import AzureChatOpenAI
            except Exception as exc:  # pragma: no cover
                raise LLMError("langchain-openai is required for provider=azure") from exc

            api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()

            if not api_key or not endpoint:
                raise LLMError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required when LLM_PROVIDER=azure")

            deployment = str(
                llm_raw.get("azure_deployment")
                or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
                or model
            ).strip()
            return AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                azure_deployment=deployment,
                temperature=temperature,
            )

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
            ollama_model = os.getenv("OLLAMA_MODEL", model or "llama3").strip()

            ChatOllama: Optional[type] = None
            try:
                from langchain_ollama import ChatOllama as _ChatOllama

                ChatOllama = _ChatOllama
            except Exception:
                try:
                    from langchain_community.chat_models import ChatOllama as _ChatOllama

                    ChatOllama = _ChatOllama
                except Exception as exc:  # pragma: no cover
                    raise LLMError(
                        "ChatOllama dependency is missing. Install: langchain-ollama (or langchain-community)."
                    ) from exc

            return ChatOllama(base_url=base_url, model=ollama_model, temperature=temperature)

        if provider == "tongyi":
            try:
                from langchain_community.chat_models import ChatTongyi
            except Exception as exc:  # pragma: no cover
                raise LLMError("langchain-community is required for provider=tongyi") from exc

            api_key = (str(llm_raw.get("api_key") or os.getenv("AGENT_LLM_API_KEY", ""))).strip()
            if not api_key:
                raise LLMError("AGENT_LLM_API_KEY is required when LLM_PROVIDER=tongyi")

            return ChatTongyi(api_key=api_key, model=model, temperature=temperature)
    except RAGConfigError as exc:
        raise LLMError(str(exc)) from exc

    raise LLMError(f"Unsupported LLM provider: {provider}")

