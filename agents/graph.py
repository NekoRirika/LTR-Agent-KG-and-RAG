"""LangGraph 状态定义与节点图：路由 → 搜索 → 答案生成。"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from storage.vector_store import vector_store_manager

# 确保 project root 在 sys.path，使 storage/retriever 可导入
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.__init__ import AgentConfig, build_agent
from agents.kg_retriever import KGRetriever


# ── 状态定义 ──────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    query: str
    strategy: str                   # global_search | local_search | hybrid
    target_entities: list[str]
    route_confidence: float
    route_reasoning: str
    kg_results: list[dict]          # KG 三元组（local_search / hybrid）
    rag_results: list[dict]         # 向量检索文档片段（global_search / hybrid）
    search_results: list[dict]      # kg_results + rag_results 合并，供 answer_node 使用
    evidence_spans: list[str]
    source_docs: list[str]
    answer: str
    answer_confidence: float


# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvidenceItem:
    subject: str
    relation: str
    object: str
    source_span: str
    source_doc: str
    confidence: float


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    strategy_used: str
    evidence: tuple[EvidenceItem, ...]
    source_docs: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 路由节点 ──────────────────────────────────────────────────────────────────


_ROUTER_SYSTEM = """你是一个知识图谱查询路由器。根据用户问题判断应走哪种检索策略。

领域：AI+法律/学术出版知识图谱
实体类型：技术、场景、风险、治理措施、主体、政策文件、目标
关系类型：应用于、引发、聚焦于、属于、包括、提出、影响、需要、助力

路由规则：
- global_search：问题涉及广泛概述、跨多种类型的模式、聚合统计（如"有哪些风险""所有技术"），未指定焦点实体 → 走向量语义检索（RAG）
- local_search：问题提及特定命名实体（如"AIGC""版权归属争议"），询问其直接关系 → 走知识图谱关系查询
- hybrid：需要同时了解宏观背景和具体实体关系，或路由不确定时回退 → 两者都走

输出严格 JSON：
{{"strategy": "global_search|local_search|hybrid", "target_entities": ["实体1", ...], "confidence": 0.0-1.0, "reasoning": "简短理由"}}
"""


def _parse_json_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last < 0:
        raise ValueError("LLM 输出不含 JSON 对象")
    return json.loads(text[first : last + 1])


def route_node(state: AgentState, *, config: AgentConfig) -> dict:
    """路由节点：LLM 判断查询应走 global(RAG) / local(KG) / hybrid。"""
    print(f"[route] 输入问题: {state['query']}", file=sys.stderr)
    llm = build_agent(config)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _ROUTER_SYSTEM), ("human", "{query}")]
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    content = getattr(response, "content", str(response))
    print(f"[route] LLM 原始输出: {content[:500]}", file=sys.stderr)

    try:
        parsed = _parse_json_output(content)
        strategy = parsed.get("strategy", "global_search")
        if strategy not in ("global_search", "local_search", "hybrid"):
            strategy = "global_search"
        confidence = float(parsed.get("confidence", 0.5))
        if confidence < config.router.min_confidence:
            strategy = "hybrid"
        target_entities = parsed.get("target_entities", [])
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError, KeyError):
        strategy = "global_search"
        confidence = 0.0
        target_entities = []
        reasoning = "LLM 输出解析失败，回退全局搜索"

    print(
        f"[route] 路由结果: strategy={strategy}, entities={target_entities}, "
        f"confidence={confidence:.2f}, reasoning={reasoning}",
        file=sys.stderr,
    )
    return {
        "strategy": strategy,
        "target_entities": target_entities,
        "route_confidence": confidence,
        "route_reasoning": reasoning,
    }


# ── 搜索节点 ──────────────────────────────────────────────────────────────────


def search_node(state: AgentState, *, config: AgentConfig) -> dict:
    """搜索节点：global_search→RAG，local_search→KG，hybrid→两者并联。"""
    strategy = state.get("strategy", "global_search")
    query = state["query"]
    print(f"[search] 策略: {strategy}, 问题: {query}", file=sys.stderr)

    kg_results: list[dict] = []
    rag_results: list[dict] = []

    # local_search / hybrid → KG 关系图谱
    if strategy in ("local_search", "hybrid"):
        entities = state.get("target_entities", [])
        print(f"[search] KG 局部搜索目标实体: {entities}", file=sys.stderr)
        retriever = KGRetriever(config.neo4j)
        try:
            kg_results = _local_search(retriever, entities, config)
        finally:
            retriever.close()
        print(f"[search] KG 检索: {len(kg_results)} 条三元组", file=sys.stderr)

    # global_search / hybrid → RAG 向量检索
    if strategy in ("global_search", "hybrid"):
        if config.vector_search.enabled:
            try:
                rag_results = _rag_search(query, config)
                print(f"[search] RAG 检索: {len(rag_results)} 条", file=sys.stderr)
            except Exception as exc:
                print(f"[search] RAG 检索失败，跳过: {exc}", file=sys.stderr)
        else:
            # vector_search 未启用时 global_search 降级为 KG 全局遍历
            print("[search] vector_search 未启用，global_search 降级为 KG 全局遍历", file=sys.stderr)
            retriever = KGRetriever(config.neo4j)
            try:
                kg_results.extend(_global_search(retriever, query, config))
            finally:
                retriever.close()
            print(f"[search] KG 全局遍历: {len(kg_results)} 条三元组", file=sys.stderr)

    # 合并，KG 在前（结构化证据优先）
    all_results = kg_results + rag_results

    evidence_spans: list[str] = []
    source_docs: list[str] = []
    for item in all_results:
        if item.get("source_span"):
            evidence_spans.append(item["source_span"])
        doc = item.get("source_doc", "")
        if doc and doc not in source_docs:
            source_docs.append(doc)

    print(
        f"[search] 汇总: KG={len(kg_results)}, RAG={len(rag_results)}, "
        f"来源文档={len(source_docs)}",
        file=sys.stderr,
    )
    return {
        "kg_results": kg_results,
        "rag_results": rag_results,
        "search_results": all_results,
        "evidence_spans": evidence_spans,
        "source_docs": source_docs,
    }


def _local_search(retriever: KGRetriever, entities: list[str], config: AgentConfig) -> list[dict]:
    """实体中心邻域查询。"""
    triples: list[dict] = []
    for name in entities:
        rows = retriever.get_entity_neighborhood(
            entity_name=name,
            max_hops=config.search.local.max_hops,
            neighbor_limit=config.search.local.neighbor_limit,
            min_confidence=config.search.local.min_confidence,
            two_hop_confidence=config.search.local.two_hop_confidence,
        )
        triples.extend(rows)
    return _deduplicate(triples)


def _global_search(retriever: KGRetriever, query: str, config: AgentConfig) -> list[dict]:
    """KG 全局遍历（vector_search 未启用时的降级路径）。"""
    triples: list[dict] = []

    for relation in _infer_relations(query):
        triples.extend(
            retriever.get_relations_by_type(
                relation=relation,
                min_confidence=config.search.global_.min_confidence,
                limit=config.search.global_.limit,
            )
        )
    for etype in _infer_entity_types(query):
        triples.extend(
            retriever.get_triples_by_entity_type(
                subject_type=etype,
                min_confidence=config.search.global_.min_confidence,
                limit=config.search.global_.limit,
            )
        )
    return _deduplicate(triples)


def _rag_search(query: str, config: AgentConfig) -> list[dict]:
    """向量语义检索，返回统一格式的 dict 列表。"""
    from storage.vector_store import VectorStoreManager
    from retriever import RAGRetriever

    vs_cfg = config.vector_search
    vsm = vector_store_manager
    rag_retriever = RAGRetriever(
        vsm,
        k=vs_cfg.k,
        search_type=vs_cfg.search_type,
        score_threshold=vs_cfg.score_threshold,
    )
    docs = rag_retriever.invoke(query)
    return [
        {
            "source_type": "rag",
            "subject": doc.metadata.get("source", ""),
            "relation": "语义相关",
            "object": query,
            "source_span": doc.page_content[:300],
            "source_doc": doc.metadata.get("file_name") or doc.metadata.get("source", ""),
            "confidence": doc.metadata.get("score", 1.0),
        }
        for doc in docs
    ]


def _infer_relations(query: str) -> list[str]:
    mapping = {
        "应用于": "应用于", "用于": "应用于",
        "引发": "引发", "带来": "引发", "导致": "引发",
        "聚焦": "聚焦于", "关注": "聚焦于",
        "属于": "属于", "归属": "属于",
        "包括": "包括", "包含": "包括",
        "提出": "提出", "强调": "提出",
        "影响": "影响", "作用": "影响",
        "需要": "需要", "依赖": "需要",
        "助力": "助力", "促进": "助力", "推动": "助力",
    }
    return list({v for k, v in mapping.items() if k in query}) or [
        "应用于", "引发", "影响", "包括"
    ]


def _infer_entity_types(query: str) -> list[str]:
    """从查询文本推断可能涉及的实体类型。"""
    mapping = {
        "技术": "技术", "AIGC": "技术", "人工智能": "技术",
        "场景": "场景", "出版": "场景",
        "风险": "风险", "伦理": "风险",
        "治理": "治理措施", "规制": "治理措施",
        "主体": "主体", "学者": "主体",
        "政策": "政策文件", "文件": "政策文件",
        "目标": "目标",
    }
    return list({v for k, v in mapping.items() if k in query})


def _deduplicate(triples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for t in triples:
        key = json.dumps(t, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ── 答案节点 ──────────────────────────────────────────────────────────────────


_ANSWER_SYSTEM = """你是一个融合知识图谱与语义检索的问答系统。你必须：
1. 综合两类证据回答用户问题：
   - 【KG三元组】：结构化关系事实，格式为 (主体) [关系] (客体)，来自知识图谱
   - 【RAG文档片段】：原文语义片段，来自向量检索
2. 优先使用 KG 三元组作为结构化依据，用 RAG 片段补充上下文细节。
3. 引用源文档，如果证据不足请明确说明，不要猜测。

输出严格 JSON：
{{"answer": "你的回答", "confidence": 0.0-1.0}}
"""


def answer_node(state: AgentState, *, config: AgentConfig) -> dict:
    """答案节点：区分 KG 和 RAG 证据，生成最终答案。"""
    kg_results = state.get("kg_results", [])
    rag_results = state.get("rag_results", [])
    print(f"[answer] KG证据: {len(kg_results)} 条, RAG证据: {len(rag_results)} 条", file=sys.stderr)

    if not kg_results and not rag_results:
        print("[answer] 无证据，返回默认回答", file=sys.stderr)
        return {
            "answer": "未找到相关证据，无法回答该问题。请尝试换个方式提问。",
            "answer_confidence": 0.0,
        }

    # 按比例分配 max_evidence_items 配额
    max_items = config.answer.max_evidence_items
    total = len(kg_results) + len(rag_results)
    if total > 0:
        kg_quota = round(max_items * len(kg_results) / total)
        rag_quota = max_items - kg_quota
    else:
        kg_quota = rag_quota = 0

    kg_text = "\n".join(
        f"  - ({t.get('subject', '')}) [{t.get('relation', '')}] ({t.get('object', '')}) "
        f"| 来源: {t.get('source_doc', '')} | 原文: {t.get('source_span', '')} "
        f"| 置信度: {t.get('confidence', 0)}"
        for t in kg_results[:kg_quota]
    )
    rag_text = "\n".join(
        f"  - 片段: {r.get('source_span', '')} | 来源: {r.get('source_doc', '')}"
        for r in rag_results[:rag_quota]
    )

    evidence_text = ""
    if kg_text:
        evidence_text += f"【KG三元组】\n{kg_text}\n"
    if rag_text:
        evidence_text += f"\n【RAG文档片段】\n{rag_text}\n"

    print(f"[answer] 送入 LLM: KG={min(len(kg_results), kg_quota)}, RAG={min(len(rag_results), rag_quota)}", file=sys.stderr)

    llm = build_agent(config)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _ANSWER_SYSTEM), ("human", "用户问题：{query}\n\n证据：\n{evidence}")]
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "evidence": evidence_text})
    content = getattr(response, "content", str(response))
    print(f"[answer] LLM 原始输出: {content[:500]}", file=sys.stderr)

    try:
        parsed = _parse_json_output(content)
        answer = parsed.get("answer", content)
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, KeyError):
        answer = content
        confidence = 0.5

    print(f"[answer] 答案置信度: {confidence:.2f}", file=sys.stderr)
    return {"answer": answer, "answer_confidence": confidence}


# ── 构建图 ────────────────────────────────────────────────────────────────────


def build_graph(config: AgentConfig) -> StateGraph:
    """构建 LangGraph StateGraph：route → search → answer → END。"""
    graph = StateGraph(AgentState)

    graph.add_node("route", lambda state: route_node(state, config=config))
    graph.add_node("search", lambda state: search_node(state, config=config))
    graph.add_node("answer", lambda state: answer_node(state, config=config))

    graph.set_entry_point("route")
    graph.add_edge("route", "search")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)

    return graph


def compile_agent(config: AgentConfig):
    """编译并返回可执行的 LangGraph agent。"""
    graph = build_graph(config)
    return graph.compile()


# 模块级默认 agent，直接 from graph import agent 即可使用
from agents.__init__ import load_agent_config as _load_agent_config

agent = build_graph(_load_agent_config()).compile()


if __name__ == "__main__":
    result = agent.invoke({"query":"AIGC技术应用于哪些场景？AIGC会带来哪些风险?"})
    print("answer:" ,  result["answer"])
    print("strategy " , result["strategy"])  # 路由策略
    print("search_results" , result["search_results"])  # 证据三元组
    print("source_docs" , result["source_docs"])  # 来源文档




