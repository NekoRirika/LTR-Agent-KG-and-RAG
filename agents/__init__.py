"""Agent 配置加载与 LLM 构建，独立于 extractor/ingestor/connection.py。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import yaml
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi


@dataclass(frozen=True)
class AgentLLMConfig:
    api_key: str
    model: str
    temperature: float


@dataclass(frozen=True)
class AgentNeo4jConfig:
    uri: str
    username: str
    password: str
    database: str


@dataclass(frozen=True)
class RouterConfig:
    min_confidence: float
    strategies: tuple[dict, ...]


@dataclass(frozen=True)
class GlobalSearchConfig:
    min_confidence: float
    limit: int


@dataclass(frozen=True)
class LocalSearchConfig:
    min_confidence: float
    max_hops: int
    neighbor_limit: int
    two_hop_confidence: float


@dataclass(frozen=True)
class SearchConfig:
    global_: GlobalSearchConfig
    local: LocalSearchConfig


@dataclass(frozen=True)
class AnswerConfig:
    low_confidence_threshold: float
    max_evidence_items: int


@dataclass(frozen=True)
class VectorSearchConfig:
    enabled: bool
    k: int = 6
    search_type: str = "similarity"
    score_threshold: float | None = None


@dataclass(frozen=True)
class AgentConfig:
    llm: AgentLLMConfig
    neo4j: AgentNeo4jConfig
    router: RouterConfig
    search: SearchConfig
    answer: AnswerConfig
    vector_search: VectorSearchConfig


def load_agent_config(config_path: str | Path = "agent_config.yml") -> AgentConfig:
    """从 YAML 加载智能体配置，空值回退到 .env 环境变量。"""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    with path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    load_dotenv(dotenv_path=path.parent / ".env", override=False)

    llm_raw = raw.get("llm", {})
    neo4j_raw = raw.get("neo4j", {})
    router_raw = raw.get("router", {})
    search_raw = raw.get("search", {})
    answer_raw = raw.get("answer", {})
    vector_raw = raw.get("vector_search", {})

    global_raw = search_raw.get("global", {})
    local_raw = search_raw.get("local", {})

    llm_api_key = llm_raw.get("api_key", "") or os.getenv("AGENT_LLM_API_KEY", "")
    neo4j_password = neo4j_raw.get("password", "") or os.getenv("AGENT_NEO4J_PASSWORD", "")

    return AgentConfig(
        llm=AgentLLMConfig(
            api_key=llm_api_key,
            model=llm_raw.get("model", "qwen-plus"),
            temperature=float(llm_raw.get("temperature", 0)),
        ),
        neo4j=AgentNeo4jConfig(
            uri=neo4j_raw.get("uri", "bolt://localhost:7687"),
            username=neo4j_raw.get("username", "neo4j"),
            password=neo4j_password,
            database=neo4j_raw.get("database", "neo4j"),
        ),
        router=RouterConfig(
            min_confidence=float(router_raw.get("min_confidence", 0.7)),
            strategies=tuple(router_raw.get("strategies", [])),
        ),
        search=SearchConfig(
            global_=GlobalSearchConfig(
                min_confidence=float(global_raw.get("min_confidence", 0.75)),
                limit=int(global_raw.get("limit", 200)),
            ),
            local=LocalSearchConfig(
                min_confidence=float(local_raw.get("min_confidence", 0.75)),
                max_hops=int(local_raw.get("max_hops", 2)),
                neighbor_limit=int(local_raw.get("neighbor_limit", 50)),
                two_hop_confidence=float(local_raw.get("two_hop_confidence", 0.8)),
            ),
        ),
        answer=AnswerConfig(
            low_confidence_threshold=float(answer_raw.get("low_confidence_threshold", 0.4)),
            max_evidence_items=int(answer_raw.get("max_evidence_items", 30)),
        ),
        vector_search=VectorSearchConfig(
            enabled=bool(vector_raw.get("enabled", False)),
            k=int(vector_raw.get("k", 6)),
            search_type=str(vector_raw.get("search_type", "similarity")),
            score_threshold=vector_raw.get("score_threshold") or None,
        ),
    )


def build_agent(config: AgentConfig) -> ChatTongyi:
    """根据 AgentConfig 构建 ChatTongyi 实例，独立于 extractor 的 build_llm。"""
    if not config.llm.api_key:
        raise ValueError("Agent LLM api_key 未配置（agent_config.yml 或 .env AGENT_LLM_API_KEY）")
    return ChatTongyi(
        api_key=config.llm.api_key,
        model=config.llm.model,
    )
