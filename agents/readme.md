# agents

该目录提供面向知识图谱问答的 Agent 层能力，当前主流程为：

1. 路由（global/local/hybrid）
2. Neo4j 图谱检索
3. 基于证据三元组生成回答

## 目录说明

- [__init__.py](__init__.py)：加载 `agent_config.yml` 并构建 LLM 客户端（当前使用 `ChatTongyi`）。
- [graph.py](graph.py)：LangGraph 状态图与节点实现（route/search/answer）。
- [kg_retriever.py](kg_retriever.py)：Neo4j 只读查询层。
- [query_cli.py](query_cli.py)：查询入口，支持单次问答和交互式连续问答。
- [vector_retriever.py](vector_retriever.py)：向量检索占位模块，当前未实现。

## 当前已实现能力

- 基于问题语义进行策略路由：`global_search` / `local_search` / `hybrid`
- 基于关系类型与实体类型的全局搜索
- 基于实体邻域（支持 1~2 跳）的局部搜索
- 证据三元组去重与来源文档聚合
- 基于证据三元组的答案生成
- CLI 交互问答（可多轮提问直到主动退出）

## 当前未实现能力

- 向量检索（`vector_retriever.py` 为占位）
- 图谱+向量真正融合检索

## 运行前准备

1. 安装依赖（在项目根目录）

```bash
pip install -r requirements.txt
```

2. 准备配置文件 `agent_config.yml`（位于项目根目录）

说明：`agents/__init__.py` 会读取该文件，并从 `.env` 回填部分敏感项（如 API Key、Neo4j 密码）。

3. 确保 Neo4j 中已有图谱数据

可先运行 ingestor 入库命令（项目根目录）：

```bash
python -m extractor.ingestor.ingest_file_cli
```

## 命令行用法

### 1) 单次问答

```bash
python -m agents.query_cli -q "AIGC技术应用于哪些场景？"
```

### 2) 指定配置文件

```bash
python -m agents.query_cli -q "AIGC有哪些风险？" --config agent_config.yml
```

### 3) 交互式多轮问答

```bash
python -m agents.query_cli
```

退出方式：输入 `exit` / `quit` / `q`，或按 `Ctrl+C`。

## Python 调用示例

```python
from agents.query_cli import ask

result = ask("AIGC技术应用于哪些场景？")
print(result["answer"])
print(result["strategy"])
print(result["search_results"])
print(result["source_docs"])
```

## 返回结果字段（query_cli）

- `strategy`：检索策略
- `route_reasoning`：路由理由
- `answer`：最终回答
- `answer_confidence`：回答置信度
- `search_results`：证据三元组列表
- `source_docs`：来源文档列表

## 调试建议

- 如果回答为空：先检查 Neo4j 中是否已有 `Entity`、`Document`、`RELATED_TO` 数据。
- 如果路由不稳定：调整 `agent_config.yml` 中 `router.min_confidence` 与策略提示词。
- 如果证据过多：降低 `search.global.limit` 或 `answer.max_evidence_items`。
