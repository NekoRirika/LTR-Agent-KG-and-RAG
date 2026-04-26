"""对 agents/graph.py 的调用适配层，隔离底层 AgentState 结构。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class QueryResult:
    query: str
    strategy: str = ""
    route_confidence: float = 0.0
    route_reasoning: str = ""
    target_entities: list[str] = field(default_factory=list)
    answer: str = ""
    answer_confidence: float = 0.0
    kg_results: list[dict] = field(default_factory=list)
    rag_results: list[dict] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    error: str | None = None


def _load_agent():
    """延迟导入并缓存，避免每次查询重复初始化 Neo4j/LLM。"""
    try:
        import streamlit as st  # noqa: PLC0415

        @st.cache_resource(show_spinner=False)
        def _cached():
            from agents.__init__ import load_agent_config  # noqa: PLC0415
            from agents.graph import build_graph  # noqa: PLC0415
            return build_graph(load_agent_config()).compile()

        return _cached()
    except ImportError:
        from agents.__init__ import load_agent_config  # noqa: PLC0415
        from agents.graph import build_graph  # noqa: PLC0415
        return build_graph(load_agent_config()).compile()


def run_query(query: str) -> QueryResult:
    """执行一次完整的 route → search → answer 查询，返回 QueryResult。"""
    try:
        agent = _load_agent()
        state = agent.invoke({"query": query})
        return QueryResult(
            query=query,
            strategy=state.get("strategy", ""),
            route_confidence=float(state.get("route_confidence", 0.0)),
            route_reasoning=state.get("route_reasoning", ""),
            target_entities=state.get("target_entities", []),
            answer=state.get("answer", ""),
            answer_confidence=float(state.get("answer_confidence", 0.0)),
            kg_results=state.get("kg_results", []),
            rag_results=state.get("rag_results", []),
            source_docs=state.get("source_docs", []),
        )
    except Exception as exc:  # noqa: BLE001
        return QueryResult(query=query, error=str(exc))
